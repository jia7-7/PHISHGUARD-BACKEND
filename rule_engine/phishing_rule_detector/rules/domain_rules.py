"""域名与身份规则.

- DOMAIN_SIMILAR_TO_OFFICIAL
- DOMAIN_PUNYCODE_SUSPICIOUS
- DOMAIN_HOMOGRAPH_ATTACK
- DOMAIN_KEYWORD_IMPERSONATION
"""
from __future__ import annotations


from rapidfuzz.distance import Levenshtein

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.rules.common import (
    RuleContext,
    get_confusable_skeleton,
    has_mixed_script,
    make_evidence,
    register_rule,
)
from phishing_rule_detector.trust_policy import (
    _normalize_hostname,
    get_registered_domain,
    is_official_domain,
)


def _core_domain_label(hostname: str) -> str:
    """提取注册域的核心标签（如 sdu.edu.cn → sdu）."""
    try:
        reg = get_registered_domain(hostname)
    except Exception:
        reg = hostname
    if not reg:
        return hostname.split(".")[0] if "." in hostname else hostname
    return reg.split(".")[0]


def _is_punycode(hostname: str) -> bool:
    """判断 hostname 是否为 Punycode 编码."""
    return "xn--" in hostname.lower()


# ── DOMAIN_SIMILAR_TO_OFFICIAL ──

@register_rule
def domain_similar_to_official(ctx: RuleContext) -> list | None:
    """使用编辑距离检测与官方域名相似的域名."""
    config = load_config()
    official_domains = config.trusted.official_domains
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        host = _normalize_hostname(hostname)

        # 官方域名不触发
        if is_official_domain(host):
            url_index += 1
            continue

        reg_domain = u.registered_domain or host
        core = _core_domain_label(host)

        for od in official_domains:
            od_core = _core_domain_label(od.lower())
            od_reg = od.lower()

            # 比较注册域核心标签
            if len(core) <= 8 and Levenshtein.distance(core, od_core) == 1:
                subject_id = u.subject_id or f"url:{url_index}"
                evidence_list.append(make_evidence(
                    rule_id="DOMAIN_SIMILAR_TO_OFFICIAL",
                    title="域名与官方域名高度相似",
                    group="identity",
                    severity="high",
                    confidence=0.90,
                    base_score=25,
                    reason=f"域名 {host} 的核心标签与 {od} 编辑距离仅为 1",
                    matched_content=u.raw[:160],
                    source=u.source,
                    tags=["similar_domain", "levenshtein"],
                    subject_id=subject_id,
                ))
                break

            # 归一化相似度
            max_len = max(len(reg_domain), len(od_reg))
            if max_len > 0:
                dist = Levenshtein.distance(reg_domain, od_reg)
                similarity = 1.0 - (dist / max_len)
                if similarity >= 0.85:
                    subject_id = u.subject_id or f"url:{url_index}"
                    evidence_list.append(make_evidence(
                        rule_id="DOMAIN_SIMILAR_TO_OFFICIAL",
                        title="域名与官方域名高度相似",
                        group="identity",
                        severity="high",
                        confidence=0.85,
                        base_score=25,
                        reason=f"域名 {host} 与官方域名 {od} 的归一化相似度为 {similarity:.2f}",
                        matched_content=u.raw[:160],
                        source=u.source,
                        tags=["similar_domain", "similarity"],
                        subject_id=subject_id,
                    ))
                    break

            # 添加连字符、前缀或替换顶级域
            dash_variant = od_core.replace(".", "-")
            if dash_variant in host or (od_core + "-") in host:
                subject_id = u.subject_id or f"url:{url_index}"
                evidence_list.append(make_evidence(
                    rule_id="DOMAIN_SIMILAR_TO_OFFICIAL",
                    title="域名结构与官方域名近似",
                    group="identity",
                    severity="high",
                    confidence=0.80,
                    base_score=25,
                    reason=f"域名 {host} 包含官方域名 {od} 的连字符变体",
                    matched_content=u.raw[:160],
                    source=u.source,
                    tags=["similar_domain", "structure"],
                    subject_id=subject_id,
                ))
                break

        url_index += 1

    return evidence_list or None


# ── DOMAIN_PUNYCODE_SUSPICIOUS ──

@register_rule
def domain_punycode_suspicious(ctx: RuleContext) -> list | None:
    """检测 Punycode 域名，解码后与官方域名视觉近似."""
    config = load_config()
    official_domains = config.trusted.official_domains
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        host = _normalize_hostname(hostname)

        if not _is_punycode(host):
            url_index += 1
            continue

        # 官方域名不触发（正常国际化域名也能是 Punycode）
        if is_official_domain(host):
            url_index += 1
            continue

        # 尝试 IDNA 解码（通过检查 normalized_text 或原始 URL）
        try:
            import idna
            decoded = idna.decode(host)
        except Exception:
            url_index += 1
            continue

        if decoded == host:
            # 未实际解码
            url_index += 1
            continue

        # 检查解码后是否与官方域名近似
        decoded_normalized = _normalize_hostname(decoded)
        for od in official_domains:
            od_lower = od.lower()
            # 视觉骨架对比
            decoded_skeleton = get_confusable_skeleton(decoded_normalized)
            od_skeleton = get_confusable_skeleton(od_lower)
            if od_skeleton in decoded_skeleton or decoded_skeleton in od_skeleton:
                subject_id = u.subject_id or f"url:{url_index}"
                evidence_list.append(make_evidence(
                    rule_id="DOMAIN_PUNYCODE_SUSPICIOUS",
                    title="Punycode 域名仿冒官方域名",
                    group="identity",
                    severity="high",
                    confidence=0.95,
                    base_score=28,
                    reason=f"Punycode 域名 {host} 解码为 {decoded}，与官方域名 {od} 视觉近似",
                    matched_content=u.raw[:160],
                    source=u.source,
                    tags=["punycode", "spoofing"],
                    subject_id=subject_id,
                ))
                break

            # 编辑距离检查
            decoded_reg = get_registered_domain(decoded_normalized)
            if decoded_reg:
                dist = Levenshtein.distance(decoded_reg, od_lower)
                max_len = max(len(decoded_reg), len(od_lower))
                if max_len > 0 and (dist <= 2 or (1.0 - dist / max_len) >= 0.80):
                    subject_id = u.subject_id or f"url:{url_index}"
                    evidence_list.append(make_evidence(
                        rule_id="DOMAIN_PUNYCODE_SUSPICIOUS",
                        title="Punycode 域名仿冒官方域名",
                        group="identity",
                        severity="high",
                        confidence=0.85,
                        base_score=28,
                        reason=f"Punycode 域名解码后 {decoded_reg} 与 {od} 编辑距离为 {dist}",
                        matched_content=u.raw[:160],
                        source=u.source,
                        tags=["punycode", "similar"],
                        subject_id=subject_id,
                    ))
                    break

        url_index += 1

    return evidence_list or None


# ── DOMAIN_HOMOGRAPH_ATTACK ──

@register_rule
def domain_homograph_attack(ctx: RuleContext) -> list | None:
    """检测 Unicode 同形字符攻击."""
    config = load_config()
    official_domains = config.trusted.official_domains
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        host = _normalize_hostname(hostname)

        # 官方域名不触发
        if is_official_domain(host):
            url_index += 1
            continue

        # 检测混合脚本
        has_mixed_script(host)

        # 生成视觉骨架
        skeleton = get_confusable_skeleton(host)

        for od in official_domains:
            od_lower = od.lower()
            od_skeleton = get_confusable_skeleton(od_lower)

            # 视觉骨架一致但原始码点不同 → Homograph
            if skeleton == od_skeleton and host != od_lower:
                subject_id = u.subject_id or f"url:{url_index}"
                evidence_list.append(make_evidence(
                    rule_id="DOMAIN_HOMOGRAPH_ATTACK",
                    title="Unicode 同形字符攻击",
                    group="identity",
                    severity="high",
                    confidence=1.0,
                    base_score=30,
                    reason=f"域名 {host} 的 Unicode 视觉骨架与官方域名 {od} 一致",
                    matched_content=u.raw[:160],
                    source=u.source,
                    tags=["homograph", "unicode"],
                    subject_id=subject_id,
                ))
                break

        url_index += 1

    return evidence_list or None


# ── DOMAIN_KEYWORD_IMPERSONATION ──

@register_rule
def domain_keyword_impersonation(ctx: RuleContext) -> list | None:
    """检测外部域名包含学校品牌词和认证词."""
    config = load_config()
    brand_kws = config.rules.brand_keywords
    login_kws = config.rules.login_keywords
    evidence_list = []
    url_index = 0

    for u in ctx.extracted_urls:
        hostname = u.hostname
        if not hostname:
            url_index += 1
            continue

        host = _normalize_hostname(hostname)

        # 官方域名不触发
        if is_official_domain(host):
            url_index += 1
            continue

        has_brand = any(kw.lower() in host.lower() for kw in brand_kws)
        has_login = any(kw.lower() in host.lower() for kw in login_kws)

        # 必须同时包含品牌词和认证词
        if has_brand and has_login:
            subject_id = u.subject_id or f"url:{url_index}"
            evidence_list.append(make_evidence(
                rule_id="DOMAIN_KEYWORD_IMPERSONATION",
                title="域名包含学校品牌和认证关键词",
                group="identity",
                severity="medium",
                confidence=0.78,
                base_score=12,
                reason=f"外部域名 {host} 同时包含学校品牌词和登录认证词",
                matched_content=u.raw[:160],
                source=u.source,
                tags=["keyword_impersonation"],
                subject_id=subject_id,
            ))
        url_index += 1

    return evidence_list or None
