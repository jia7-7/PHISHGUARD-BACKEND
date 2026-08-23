"""HTML、表单与导航规则.

- HTML_PASSWORD_INPUT
- PASSWORD_FORM_UNTRUSTED_TARGET
- FORM_NON_HTTPS_ACTION
- FORM_COLLECTS_SENSITIVE_INFO
- FORM_COLLECTS_SECRET
- META_REFRESH_REDIRECT
- JS_LOCATION_REDIRECT
- HIDDEN_FORM_OR_LINK
- IFRAME_LOGIN_PAGE
- LINK_TEXT_HREF_MISMATCH
- EXTERNAL_LINK_DOMINANCE
- PORTAL_BRAND_EXTERNAL_DOMAIN
"""
from __future__ import annotations

import re
from typing import cast
from urllib.parse import urljoin, urlsplit

from bs4.element import Tag

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.models import SEVERITY_LEVELS
from phishing_rule_detector.rules.common import (
    RuleContext,
    check_field_matches_keywords,
    extract_form_fields,
    get_registered_domain,
    is_official_domain,
    make_evidence,
    register_rule,
)
from phishing_rule_detector.trust_policy import match_trusted_service


# ── JS 跳转模式 ──
_JS_REDIRECT_RE = re.compile(
    r"(?:location\.href|window\.location|location\.assign|location\.replace)\s*[=\(]",
    re.IGNORECASE,
)


def _attr_text(tag: Tag, name: str) -> str:
    """以稳定字符串形式读取 HTML 属性."""
    value = tag.get(name, "")
    return value if isinstance(value, str) else str(value or "")


def _is_hidden(tag) -> bool:
    """检查标签是否被隐藏."""
    style = (tag.get("style") or "").lower()
    if "display:none" in style.replace(" ", "") or "display: none" in style:
        return True
    if "visibility:hidden" in style.replace(" ", "") or "visibility: hidden" in style:
        return True
    if tag.get("hidden") is not None:
        return True
    if tag.get("type") == "hidden":
        return True
    # 检查尺寸
    if tag.get("width") in ("0", "1") and tag.get("height") in ("0", "1"):
        return True
    return False


def _collect_brand_signals(
    soup, normalized_text: str, brand_kws: list[str]
) -> list[str]:
    """收集页面中的学校品牌信号（供多规则复用）."""
    signals: list[str] = []

    # 1. title 中的品牌词
    if soup:
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True).lower()
            for kw in brand_kws:
                if kw.lower() in title_text:
                    signals.append(f"title_contains_{kw}")
                    break

    # 2. 正文中 >=2 个品牌词
    text_lower = normalized_text.lower()
    brand_hits = [kw for kw in brand_kws if kw.lower() in text_lower]
    if len(brand_hits) >= 2:
        signals.append("body_brand_keywords")

    # 3. form label 含学工号/统一身份
    if soup:
        forms_data = extract_form_fields(soup)
        for fd in forms_data:
            for field in fd["fields"]:
                combined = " ".join(
                    v for k, v in field.items()
                    if k in ("name", "id", "placeholder", "label", "aria_label")
                ).lower()
                if "学工号" in combined or "统一身份" in combined:
                    signals.append("form_label_brand")
                    break

    # 4. 链接文本模仿学校服务
    if soup:
        official_links = 0
        for a_tag in cast(list[Tag], soup.find_all("a")):
            link_text = a_tag.get_text(strip=True).lower()
            if any(kw.lower() in link_text for kw in brand_kws):
                official_links += 1
        if official_links >= 2:
            signals.append("link_text_brand")

    return signals


# ── HTML_PASSWORD_INPUT ──

@register_rule
def html_password_input(ctx: RuleContext) -> list | None:
    """检测页面包含密码输入框."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    evidence_list = []

    forms = cast(list[Tag], soup.find_all("form"))
    if forms:
        for fi, form_tag in enumerate(forms):
            pw_inputs = form_tag.find_all("input", type="password")
            if pw_inputs:
                subject_id = f"form:{fi}"
                evidence_list.append(make_evidence(
                    rule_id="HTML_PASSWORD_INPUT",
                    title="页面包含密码输入框",
                    group="credential",
                    severity="low",
                    confidence=0.90,
                    base_score=4,
                    reason=f"表单 form:{fi} 包含 {len(pw_inputs)} 个密码输入框",
                    matched_content=_attr_text(form_tag, "action")[:160],
                    source="normalized",
                    tags=["password_input"],
                    subject_id=subject_id,
                ))
    else:
        # 无 form 标签但有密码框 → page 级别
        pw_inputs = soup.find_all("input", type="password")
        if pw_inputs:
            evidence_list.append(make_evidence(
                rule_id="HTML_PASSWORD_INPUT",
                title="页面包含密码输入框",
                group="credential",
                severity="low",
                confidence=0.90,
                base_score=4,
                reason=f"页面含 {len(pw_inputs)} 个密码输入框（无表单标签）",
                source="normalized",
                tags=["password_input"],
                subject_id="page:0",
            ))

    return evidence_list or None


# ── PASSWORD_FORM_UNTRUSTED_TARGET ──

@register_rule
def password_form_untrusted_target(ctx: RuleContext) -> list | None:
    """检测密码表单提交到非可信目标.

    仅在以下任一条件成立时命中（四条件逻辑）：
    (a) 品牌冒充 + 非官方目标
    (b) 跨注册域提交到非可信目标
    (c) 官方页面提交到非官方目标
    (d) 外部目标且有品牌信号
    """
    soup = ctx.parsed_html
    if soup is None:
        return None

    config = load_config()
    brand_kws = config.rules.brand_keywords
    evidence_list = []

    # 确定页面域名
    page_host = ""
    page_reg_domain = ""
    page_is_official = False
    if ctx.base_url:
        try:
            page_host = urlsplit(ctx.base_url).hostname or ""
            page_reg_domain = get_registered_domain(page_host)
            page_is_official = is_official_domain(page_host)
        except Exception:
            pass

    # 收集品牌信号（延迟计算，仅在有密码表单时）
    brand_signals: list[str] | None = None

    forms = cast(list[Tag], soup.find_all("form"))
    for fi, form_tag in enumerate(forms):
        pw_inputs = form_tag.find_all("input", type="password")
        if not pw_inputs:
            continue

        action = _attr_text(form_tag, "action").strip()

        # 解析 action 为绝对 URL
        if not action:
            # action 为空 → 提交到当前页面
            if ctx.base_url:
                resolved_action = ctx.base_url
            else:
                continue
        else:
            try:
                if ctx.base_url:
                    resolved_action = urljoin(ctx.base_url, action)
                else:
                    resolved_action = action
            except (ValueError, AttributeError):
                continue

        # 解析 action 的 hostname 和注册域
        try:
            action_parsed = urlsplit(resolved_action)
            action_host = (action_parsed.hostname or "").lower()
            if not action_host:
                continue
            action_reg_domain = get_registered_domain(action_host)
            action_is_official = is_official_domain(action_host)
        except Exception:
            continue

        # 官方 action → 可信，跳过
        if action_is_official:
            continue

        # 可信第三方 → 跳过
        svc = match_trusted_service(action_host)
        if svc is not None:
            continue

        # 评估四条件
        triggered = False
        trigger_reason = ""

        # 条件 (a)：品牌冒充 + 非官方目标
        if brand_signals is None:
            brand_signals = _collect_brand_signals(soup, ctx.normalized_text, brand_kws)
        if len(brand_signals) >= 2 and not action_is_official:
            triggered = True
            trigger_reason = (
                f"品牌冒充（{', '.join(brand_signals[:3])}）且提交到外部域名 {action_host}"
            )

        # 条件 (b)：跨注册域提交到非可信目标
        if not triggered and page_reg_domain and action_reg_domain:
            if page_reg_domain != action_reg_domain and not action_is_official:
                triggered = True
                trigger_reason = (
                    f"跨注册域提交：{page_reg_domain} → {action_reg_domain}"
                )

        # 条件 (c)：官方页面提交到非官方目标（不应发生）
        if not triggered and page_is_official and not action_is_official:
            triggered = True
            trigger_reason = (
                f"官方页面 {page_host} 提交到外部域名 {action_host}"
            )

        # 条件 (d)：外部目标且有品牌信号
        if not triggered and brand_signals and len(brand_signals) >= 1:
            triggered = True
            trigger_reason = (
                f"外部目标 {action_host} 存在品牌信号: {', '.join(brand_signals[:3])}"
            )

        if triggered:
            subject_id = f"form:{fi}"
            evidence_list.append(make_evidence(
                rule_id="PASSWORD_FORM_UNTRUSTED_TARGET",
                title="密码表单提交到非可信目标",
                group="credential",
                severity="high",
                confidence=0.95,
                base_score=30,
                reason=f"表单 form:{fi} {trigger_reason}",
                matched_content=action[:160],
                source="normalized",
                tags=["password", "external_target"],
                subject_id=subject_id,
            ))

    return evidence_list or None


# ── FORM_NON_HTTPS_ACTION ──

@register_rule
def form_non_https_action(ctx: RuleContext) -> list | None:
    """检测表单通过 HTTP 提交."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    evidence_list = []
    forms = cast(list[Tag], soup.find_all("form"))
    for fi, form_tag in enumerate(forms):
        action = _attr_text(form_tag, "action").strip()
        if not action:
            continue
        if action.lower().startswith("http://"):
            evidence_list.append(make_evidence(
                rule_id="FORM_NON_HTTPS_ACTION",
                title="表单通过 HTTP 提交",
                group="transport",
                severity="medium",
                confidence=0.80,
                base_score=12,
                reason=f"表单 form:{fi} 的 action 使用 HTTP 协议",
                matched_content=action[:160],
                source="normalized",
                tags=["http", "form"],
                subject_id=f"form:{fi}",
            ))

    return evidence_list or None


# ── FORM_COLLECTS_SENSITIVE_INFO ──

@register_rule
def form_collects_sensitive_info(ctx: RuleContext) -> list | None:
    """检测表单收集敏感个人信息."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    config = load_config()
    sensitive_kws = config.rules.sensitive_info_fields
    forms_data = extract_form_fields(soup)
    evidence_list = []

    for fd in forms_data:
        fi = fd["form_index"]
        matched_fields = []
        for field in fd["fields"]:
            if check_field_matches_keywords(field, sensitive_kws):
                matched_fields.append(field.get("name") or field.get("id") or field.get("type"))

        if matched_fields:
            # 检查可信第三方
            svc = None
            action_host = ""
            form_tags = cast(list[Tag], soup.find_all("form"))
            form_tag = form_tags[fi] if fi < len(form_tags) else None
            if form_tag:
                action = _attr_text(form_tag, "action").strip()
                try:
                    parsed = urlsplit(action) if action else None
                    if parsed and parsed.hostname:
                        action_host = parsed.hostname
                        from phishing_rule_detector.trust_policy import match_trusted_service
                        svc = match_trusted_service(action_host)
                except Exception:
                    pass

            cf = 1.0
            confidence = 0.75
            if svc:
                # 检查是否全部字段都在 allowed_fields 中
                from phishing_rule_detector.trust_policy import check_field_allowed
                all_allowed = all(
                    check_field_allowed(svc, f) for f in matched_fields
                )
                if all_allowed:
                    cf = 0.35
                    confidence = 0.55

            evidence_list.append(make_evidence(
                rule_id="FORM_COLLECTS_SENSITIVE_INFO",
                title="表单收集敏感个人信息",
                group="credential",
                severity="medium",
                confidence=confidence,
                base_score=12,
                reason=f"表单 form:{fi} 收集敏感字段: {', '.join(matched_fields[:5])}",
                matched_content=f"fields: {', '.join(matched_fields[:5])}"[:160],
                source="normalized",
                tags=["sensitive_info"],
                subject_id=f"form:{fi}",
                context_factor=cf,
            ))

    return evidence_list or None


# ── FORM_COLLECTS_SECRET ──

@register_rule
def form_collects_secret(ctx: RuleContext) -> list | None:
    """检测表单收集密码、验证码、银行卡等机密信息."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    config = load_config()
    secret_kws = config.rules.secret_fields
    forms_data = extract_form_fields(soup)
    evidence_list = []

    for fd in forms_data:
        fi = fd["form_index"]
        matched_fields = []
        for field in fd["fields"]:
            if check_field_matches_keywords(field, secret_kws):
                matched_fields.append(field.get("name") or field.get("id") or field.get("type"))
            # 额外检查：input[type=password] 无论 name 是什么都应命中
            elif field.get("type") == "password":
                matched_fields.append(field.get("name") or "password")

        if matched_fields:
            form_tags = cast(list[Tag], soup.find_all("form"))
            form_tag = form_tags[fi]
            action = _attr_text(form_tag, "action").strip()
            action_host = ""
            try:
                resolved_action = urljoin(ctx.base_url, action) if ctx.base_url else action
                action_host = (urlsplit(resolved_action).hostname or "").lower()
            except Exception:
                pass

            svc = match_trusted_service(action_host) if action_host else None
            forbidden_fields: list[str] = []
            if svc is not None:
                for field in fd["fields"]:
                    canonical_values = {
                        str(field.get("name", "")).lower(),
                        str(field.get("id", "")).lower(),
                        str(field.get("type", "")).lower(),
                    }
                    if field.get("type") == "password":
                        canonical_values.add("password")
                    for forbidden in svc.forbidden_fields:
                        forbidden_lower = forbidden.lower()
                        if forbidden_lower in canonical_values or check_field_matches_keywords(
                            field, [forbidden_lower]
                        ):
                            forbidden_fields.append(forbidden_lower)

            if forbidden_fields:
                unique_forbidden = sorted(set(forbidden_fields))
                evidence_list.append(make_evidence(
                    rule_id="TRUSTED_SERVICE_FORBIDDEN_FIELD",
                    title="可信第三方收集禁止字段",
                    group="credential",
                    severity="high",
                    confidence=0.95,
                    base_score=30,
                    reason=(
                        f"可信服务 {action_host} 的表单 form:{fi} 收集禁止字段: "
                        f"{', '.join(unique_forbidden)}"
                    ),
                    matched_content=f"fields: {', '.join(unique_forbidden)}",
                    source="normalized",
                    tags=["trusted_service", "forbidden_field"],
                    subject_id=f"form:{fi}",
                ))
                continue

            # 判断是否为纯密码字段（不含银行卡、验证码、支付密码等高危字段）
            high_value_secrets = {
                "token", "bank_card", "银行卡", "sms_code", "验证码",
                "payment_password", "支付密码",
            }
            has_high_value = False
            for field in fd["fields"]:
                search_text = " ".join(
                    str(field.get(k, "")) for k in ("name", "id", "placeholder", "label", "aria_label")
                ).lower()
                for hvs in high_value_secrets:
                    if hvs.lower() in search_text:
                        has_high_value = True
                        break
                if has_high_value:
                    break

            password_only = not has_high_value

            if password_only:
                severity: SEVERITY_LEVELS = "low"
                confidence = 0.50
                base_score = 5
            else:
                severity = "high"
                confidence = 0.92
                base_score = 28

            evidence_list.append(make_evidence(
                rule_id="FORM_COLLECTS_SECRET",
                title="表单收集机密信息",
                group="credential",
                severity=severity,
                confidence=confidence,
                base_score=base_score,
                reason=f"表单 form:{fi} 收集机密字段: {', '.join(matched_fields[:5])}",
                matched_content=f"fields: {', '.join(matched_fields[:5])}"[:160],
                source="normalized",
                tags=["secret_fields"],
                subject_id=f"form:{fi}",
            ))

    return evidence_list or None


# ── META_REFRESH_REDIRECT ──

@register_rule
def meta_refresh_redirect(ctx: RuleContext) -> list | None:
    """检测 meta refresh 跳转."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    evidence_list = []
    for tag in cast(
        list[Tag],
        soup.find_all(
            "meta",
            attrs={
                "http-equiv": lambda value: bool(
                    value and str(value).lower() == "refresh"
                )
            },
        ),
    ):
        content = _attr_text(tag, "content")
        if "url" in content.lower():
            evidence_list.append(make_evidence(
                rule_id="META_REFRESH_REDIRECT",
                title="Meta Refresh 自动跳转",
                group="navigation",
                severity="medium",
                confidence=0.80,
                base_score=10,
                reason="页面使用 meta refresh 自动跳转",
                matched_content=content[:160],
                source="normalized",
                tags=["redirect", "meta"],
                subject_id="page:0",
            ))
            break

    return evidence_list or None


# ── JS_LOCATION_REDIRECT ──

@register_rule
def js_location_redirect(ctx: RuleContext) -> list | None:
    """检测静态 JS 跳转代码."""
    text = ctx.normalized_text
    if _JS_REDIRECT_RE.search(text):
        return [make_evidence(
            rule_id="JS_LOCATION_REDIRECT",
            title="JavaScript 跳转代码",
            group="navigation",
            severity="medium",
            confidence=0.65,
            base_score=10,
            reason="页面包含静态 JavaScript 跳转代码",
            matched_content="location.href/window.location 等跳转模式"[:160],
            source="normalized",
            tags=["redirect", "javascript"],
            subject_id="page:0",
        )]
    return None


# ── HIDDEN_FORM_OR_LINK ──

@register_rule
def hidden_form_or_link(ctx: RuleContext) -> list | None:
    """检测隐藏表单或链接."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    evidence_list = []
    hidden_count = 0

    for tag_name in ("form", "a"):
        for tag in cast(list[Tag], soup.find_all(tag_name)):
            if _is_hidden(tag):
                hidden_count += 1

    if hidden_count > 0:
        evidence_list.append(make_evidence(
            rule_id="HIDDEN_FORM_OR_LINK",
            title="页面包含隐藏表单或链接",
            group="navigation",
            severity="medium",
            confidence=0.60,
            base_score=10,
            reason=f"发现 {hidden_count} 个隐藏的表单或链接",
            matched_content=f"hidden elements: {hidden_count}"[:160],
            source="normalized",
            tags=["hidden", "obfuscation"],
            subject_id="page:0",
        ))

    return evidence_list or None


# ── IFRAME_LOGIN_PAGE ──

@register_rule
def iframe_login_page(ctx: RuleContext) -> list | None:
    """检测 iframe 嵌套登录或认证页面."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    config = load_config()
    login_kws = config.rules.login_keywords
    evidence_list = []

    for iframe in cast(list[Tag], soup.find_all("iframe", src=True)):
        src = _attr_text(iframe, "src").lower()
        if any(kw.lower() in src for kw in login_kws):
            evidence_list.append(make_evidence(
                rule_id="IFRAME_LOGIN_PAGE",
                title="iframe 嵌套认证页面",
                group="navigation",
                severity="medium",
                confidence=0.75,
                base_score=12,
                reason=f"iframe 引用可能包含登录/认证的页面: {src[:80]}",
                matched_content=src[:160],
                source="normalized",
                tags=["iframe", "login"],
                subject_id="page:0",
            ))

    return evidence_list or None


# ── LINK_TEXT_HREF_MISMATCH ──

@register_rule
def link_text_href_mismatch(ctx: RuleContext) -> list | None:
    """检测链接文本显示官方域名但实际 href 指向外部域名."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    config = load_config()
    official_domains = config.trusted.official_domains
    evidence_list = []

    for a_tag in cast(list[Tag], soup.find_all("a", href=True)):
        href = _attr_text(a_tag, "href").strip()
        text = a_tag.get_text(strip=True) or ""
        if not href or not text:
            continue

        # 检查锚文本中是否显示官方域名
        text_lower = text.lower()
        shows_official = any(od.lower() in text_lower for od in official_domains)

        if not shows_official:
            continue

        # 解析 href 的注册域
        try:
            parsed = urlsplit(href)
            href_host = (parsed.hostname or "").lower()
            if not href_host:
                # 相对 URL，相对于 base_url 解析
                if ctx.base_url:
                    href_host = urlsplit(ctx.base_url).hostname or ""
            href_reg = get_registered_domain(href_host)
        except Exception:
            continue

        # 检查是否指向不同注册域
        for od in official_domains:
            od_reg = get_registered_domain(od)
            if href_reg and href_reg != od_reg:
                evidence_list.append(make_evidence(
                    rule_id="LINK_TEXT_HREF_MISMATCH",
                    title="链接文本与目标地址不一致",
                    group="navigation",
                    severity="high",
                    confidence=0.90,
                    base_score=22,
                    reason=f"链接文本显示 {text[:30]} 但 href 指向 {href_reg}",
                    matched_content=f"text: {text[:60]} href: {href[:80]}"[:160],
                    source="normalized",
                    tags=["link_mismatch", "spoofing"],
                    subject_id="page:0",
                ))
                break

    return evidence_list or None


# ── EXTERNAL_LINK_DOMINANCE ──

@register_rule
def external_link_dominance(ctx: RuleContext) -> list | None:
    """检测外域链接占比过高（使用注册域比较）."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    all_links = cast(list[Tag], soup.find_all("a", href=True))
    if len(all_links) < 3:
        return None

    # 没有页面来源时无法定义“外部”，不能据此提升风险等级。
    if not ctx.base_url:
        return None

    # 确定页面的注册域
    page_reg_domain = ""
    if ctx.base_url:
        try:
            page_host = urlsplit(ctx.base_url).hostname or ""
            page_reg_domain = get_registered_domain(page_host)
        except Exception:
            pass

    external_count = 0
    for a_tag in all_links:
        href = _attr_text(a_tag, "href").strip()
        try:
            parsed = urlsplit(href)
            host = (parsed.hostname or "").lower()
            if not host and ctx.base_url:
                host = urlsplit(ctx.base_url).hostname or ""
            if host:
                link_reg = get_registered_domain(host)
                # 相同注册域不计为外部链接
                if link_reg and page_reg_domain and link_reg != page_reg_domain:
                    external_count += 1
        except Exception:
            pass

    total = len(all_links)
    ratio = external_count / total if total > 0 else 0

    if ratio >= 0.70:
        return [make_evidence(
            rule_id="EXTERNAL_LINK_DOMINANCE",
            title="外域链接占比过高",
            group="navigation",
            severity="medium",
            confidence=min(0.85, ratio),
            base_score=12,
            reason=f"页面 {total} 个链接中 {external_count} 个指向外部注册域（{ratio:.0%}）",
            matched_content=f"external ratio: {ratio:.0%}"[:160],
            source="normalized",
            tags=["external_links"],
            subject_id="page:0",
        )]
    return None


# ── PORTAL_BRAND_EXTERNAL_DOMAIN ──

@register_rule
def portal_brand_external_domain(ctx: RuleContext) -> list | None:
    """检测外域页面冒充学校品牌门户."""
    soup = ctx.parsed_html
    base_url = ctx.base_url

    # 确定页面域名
    page_host = ""
    if base_url:
        try:
            page_host = urlsplit(base_url).hostname or ""
        except Exception:
            pass

    # 官方域名页面不触发
    if page_host and is_official_domain(page_host):
        return None

    config = load_config()
    brand_kws = config.rules.brand_keywords

    # 使用共享品牌信号收集器
    signals = _collect_brand_signals(soup, ctx.normalized_text, brand_kws)

    # 至少两个独立信号
    if len(signals) >= 2:
        return [make_evidence(
            rule_id="PORTAL_BRAND_EXTERNAL_DOMAIN",
            title="外域页面冒充学校品牌门户",
            group="identity",
            severity="high",
            confidence=0.85,
            base_score=25,
            reason=f"外部域名页面集中使用学校品牌特征: {', '.join(signals)}",
            matched_content=f"signals: {', '.join(signals)}"[:160],
            source="normalized",
            tags=["brand_impersonation", "portal"],
            subject_id="page:0",
        )]

    return None
