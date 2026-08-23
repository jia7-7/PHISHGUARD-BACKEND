"""URL 与传输规则.

- URL_IP_DOMAIN
- URL_SHORT_LINK
- URL_NON_HTTPS
- URL_SUSPICIOUS_PORT
- URL_LONG_SUBDOMAIN
- URL_AT_SYMBOL
- URL_NESTED_OFFICIAL_DOMAIN
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.rules.common import (
    RuleContext,
    make_evidence,
    register_rule,
)
from phishing_rule_detector.trust_policy import (
    _normalize_hostname,
    get_registered_domain,
    is_official_domain,
    is_short_link_domain,
)


def _split_port(hostname: str) -> tuple[str, int | None]:
    """从 hostname 中分离端口."""
    if ":" in hostname and not hostname.startswith("["):
        host, port_str = hostname.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            return hostname, None
    return hostname, None


def _count_subdomain_levels(hostname: str) -> int:
    """计算子域名层级（不含注册域）."""
    host = _normalize_hostname(hostname)
    try:
        reg = get_registered_domain(host)
    except Exception:
        reg = host
    if not reg or reg == host:
        return 0
    suffix = "." + reg
    if host.endswith(suffix):
        prefix = host[: -len(suffix)]
        return prefix.count(".") + 1
    return 0


# ── URL_IP_DOMAIN ──

@register_rule
def url_ip_domain(ctx: RuleContext) -> list | None:
    """检测公网 IP 作为域名用于登录/认证场景."""
    config = load_config()
    login_kws = config.rules.login_keywords
    evidence_list = []
    url_index = 0

    all_urls = list(ctx.extracted_urls)
    for u in all_urls:
        hostname = u.hostname
        if not hostname:
            continue
        host, _ = _split_port(hostname)

        # 检查是否为 IP 地址
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            continue

        # 排除 localhost 和私网
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            continue

        # 检查是否在登录/认证上下文
        url_text = (u.path + u.query).lower()
        is_auth_context = any(
            kw.lower() in url_text for kw in login_kws
        )

        if is_auth_context:
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_IP_DOMAIN",
                title="IP 地址作为登录域名",
                group="transport",
                severity="medium",
                confidence=0.80,
                base_score=12,
                reason=f"URL 使用公网 IP {host} 作为域名且在认证上下文中",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["ip", "auth"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None


# ── URL_SHORT_LINK ──

@register_rule
def url_short_link(ctx: RuleContext) -> list | None:
    """检测短链接服务."""
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue
        host, _ = _split_port(hostname)

        if is_short_link_domain(host):
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_SHORT_LINK",
                title="使用短链接服务",
                group="navigation",
                severity="medium",
                confidence=0.70,
                base_score=8,
                reason=f"URL 使用已知短链接服务 {host}",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["short_link"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None


# ── URL_NON_HTTPS ──

@register_rule
def url_non_https(ctx: RuleContext) -> list | None:
    """检测非 HTTPS URL 在认证上下文中的使用."""
    config = load_config()
    login_kws = config.rules.login_keywords
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        if u.scheme == "https":
            url_index += 1
            continue

        url_text = (u.path + u.query).lower()
        has_auth = any(kw.lower() in url_text for kw in login_kws)

        # 检查是否有表单存在（从 HTML 规则上下文推断，这里检查是否有 action=http）
        has_form_http = any(
            eu.scheme == "http" and eu.element_type == "form"
            for eu in ctx.extracted_urls
        )

        if has_auth or has_form_http:
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_NON_HTTPS",
                title="认证场景使用 HTTP",
                group="transport",
                severity="low",
                confidence=0.75 if has_auth else 0.55,
                base_score=6,
                reason="URL 在认证或表单上下文中使用 HTTP 而非 HTTPS",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["http", "transport"],
                subject_id=subject_id,
            ))
        elif u.scheme == "http":
            # 普通 HTTP 页面：仅 low 弱证据
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_NON_HTTPS",
                title="使用 HTTP 协议",
                group="transport",
                severity="low",
                confidence=0.30,
                base_score=6,
                reason="页面使用 HTTP 协议（非认证上下文）",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["http"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None


# ── URL_SUSPICIOUS_PORT ──

@register_rule
def url_suspicious_port(ctx: RuleContext) -> list | None:
    """检测非标准端口."""
    config = load_config()
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        port = u.port
        hostname = u.hostname
        if not hostname or port is None:
            url_index += 1
            continue

        host, _ = _split_port(hostname)

        # 标准端口放行
        if port in (80, 443):
            url_index += 1
            continue

        # 检查校内例外端口
        is_allowed = False
        for entry in config.trusted.allowed_non_standard_ports:
            pattern = entry.get("host", "")
            ports = entry.get("ports", [])
            if port in ports:
                if pattern.startswith("*."):
                    suffix = pattern[1:]  # .sdu.edu.cn
                    if host.endswith(suffix) or host == suffix[1:]:
                        is_allowed = True
                        break
                elif host == pattern:
                    is_allowed = True
                    break

        if is_allowed:
            url_index += 1
            continue

        subject_id = u.subject_id or f"url:{url_index}"
        evidence_list.append(make_evidence(
            rule_id="URL_SUSPICIOUS_PORT",
            title="使用非标准端口",
            group="transport",
            severity="low",
            confidence=0.50,
            base_score=5,
            reason=f"URL 使用非标准端口 {port}",
            matched_content=u.raw[:160],
            source=u.source,
            tags=["port"],
            subject_id=subject_id,
        ))
        url_index += 1

    return evidence_list or None


# ── URL_LONG_SUBDOMAIN ──

@register_rule
def url_long_subdomain(ctx: RuleContext) -> list | None:
    """检测超长子域名层级."""
    config = load_config()
    max_levels = config.rules.max_subdomain_levels
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        levels = _count_subdomain_levels(hostname)
        if levels > max_levels:
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_LONG_SUBDOMAIN",
                title="超长子域名层级",
                group="identity",
                severity="medium",
                confidence=0.65,
                base_score=8,
                reason=f"子域名层级 {levels} 超过阈值 {max_levels}",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["subdomain"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None


# ── URL_AT_SYMBOL ──

@register_rule
def url_at_symbol(ctx: RuleContext) -> list | None:
    """检测 URL 中使用 @ 符号进行欺骗.

    authority 中存在 userinfo，且 userinfo 部分看起来像官方域名/学校品牌，
    而实际 hostname 为外部域名。
    """
    config = load_config()
    brand_kws = config.rules.brand_keywords
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        raw = u.raw
        try:
            split = urlsplit(raw)
        except Exception:
            url_index += 1
            continue

        hostname = (split.hostname or "").lower()
        if not hostname or "@" not in split.netloc:
            url_index += 1
            continue

        # 分离 userinfo 和 hostport
        if "@" in split.netloc:
            userinfo = split.netloc.rsplit("@", 1)[0]
        else:
            url_index += 1
            continue

        if not userinfo:
            url_index += 1
            continue

        # 检查 userinfo 是否模仿官方域名或品牌
        userinfo_lower = userinfo.lower()
        looks_official = any(
            kw.lower() in userinfo_lower for kw in brand_kws
        )
        # 检查是否包含常见官方子域名特征
        looks_official = looks_official or any(
            pattern in userinfo_lower
            for pattern in ["sdu.edu", "sdu-edu", "pass.sdu", "mail.sdu"]
        )

        # hostname 是外部域名
        if looks_official and not is_official_domain(hostname):
            subject_id = u.subject_id or f"url:{url_index}"
            safe_userinfo = userinfo
            if ":" in safe_userinfo:
                username, _password = safe_userinfo.split(":", 1)
                safe_userinfo = f"{username}:[REDACTED]"
            evidence_list.append(make_evidence(
                rule_id="URL_AT_SYMBOL",
                title="URL @ 符号欺骗结构",
                group="navigation",
                severity="high",
                confidence=0.90,
                base_score=20,
                reason=f"URL 的 userinfo '{safe_userinfo[:40]}' 模仿官方域名，实际目标为外部域名 {hostname}",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["at_symbol", "spoofing"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None


# ── URL_NESTED_OFFICIAL_DOMAIN ──

@register_rule
def url_nested_official_domain(ctx: RuleContext) -> list | None:
    """检测外部域名中嵌套官方域名.

    检测模式：
    - sdu.edu.cn.evil.com（外部子域中嵌入）
    - evil.com/path/sdu.edu.cn/login（路径中嵌入）
    """
    config = load_config()
    official_domains = config.trusted.official_domains
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        host, _ = _split_port(hostname)
        is_nested = False
        nested_reason = ""

        # 检查 hostname 中是否嵌套官方域名（非官方子域名）
        for od in official_domains:
            od_lower = od.lower()
            # 检查域名嵌套：xxx.sdu.edu.cn.xxx 模式（包含点前缀）或 host 以官方域名.开头
            if (f".{od_lower}." in host) or (host.startswith(f"{od_lower}.")):
                if not is_official_domain(host):
                    is_nested = True
                    nested_reason = f"外部域名 {host} 子域中嵌入官方域名 {od}"
                    break
            # 检查路径中嵌入
            if od_lower in u.path:
                is_nested = True
                nested_reason = f"URL 路径中嵌入官方域名 {od}: {u.path[:80]}"
                break

        if is_nested:
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="URL_NESTED_OFFICIAL_DOMAIN",
                title="外部域名嵌套官方域名",
                group="identity",
                severity="high",
                confidence=0.90,
                base_score=22,
                reason=nested_reason,
                matched_content=u.raw[:160],
                source=u.source,
                tags=["nested_domain", "spoofing"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None
