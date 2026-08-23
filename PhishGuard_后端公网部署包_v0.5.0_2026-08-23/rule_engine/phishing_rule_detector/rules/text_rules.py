"""诱导话术规则.

- URGENT_VERIFY_WORDING
- TIME_LIMIT_PRESSURE
- ACCOUNT_DISABLE_THREAT
- REWARD_OR_REFUND_BAIT
- SECURITY_UPGRADE_WORDING
- CREDENTIAL_REQUEST_TEXT
"""
from __future__ import annotations

import re

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.rules.common import (
    RuleContext,
    _build_flexible_pattern,
    make_evidence,
    register_rule,
)
from phishing_rule_detector.trust_policy import is_official_domain


def _match_wording_pattern(
    text: str, keywords: list[str]
) -> tuple[bool, str]:
    """在文本中匹配话术关键词（支持空格/零宽插入）."""
    text_lower = text.lower()
    for kw in keywords:
        pattern = _build_flexible_pattern(kw)
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, kw
    return False, ""


def _check_official_context(ctx: RuleContext) -> bool:
    """检查是否官方上下文（发件域名+链接均为官方）."""
    sender_is_official = False
    if ctx.sender:
        try:
            from email.utils import parseaddr
            _, addr = parseaddr(ctx.sender)
            if addr and "@" in addr:
                domain = addr.rsplit("@", 1)[-1]
                sender_is_official = is_official_domain(domain)
        except Exception:
            pass

    links_all_official = True
    for u in ctx.extracted_urls:
        if u.hostname and not u.is_official:
            links_all_official = False
            break

    return sender_is_official and links_all_official


def _sentence_subject_id(text: str, keyword: str) -> str:
    """Return a stable sentence subject for a matched wording keyword."""
    sentences = [
        match.group(0)
        for match in re.finditer(r"[^。！？!?\n]+[。！？!?\n]*", text)
        if match.group(0).strip()
    ]
    for index, sentence in enumerate(sentences):
        matched, _ = _match_wording_pattern(sentence, [keyword])
        if matched:
            return f"sentence:{index}"
    return "sentence:0"


# ── URGENT_VERIFY_WORDING ──

@register_rule
def urgent_verify_wording(ctx: RuleContext) -> list | None:
    """检测紧急验证话术."""
    config = load_config()
    wp = config.rules.wording_patterns.get("urgent_verify", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.65)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    cf = 0.2 if _check_official_context(ctx) else 1.0

    return [make_evidence(
        rule_id="URGENT_VERIFY_WORDING",
        title="紧急验证话术",
        group="social",
        severity="medium",
        confidence=default_conf,
        base_score=8,
        reason=f"文本包含紧急验证关键词: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "urgency"],
        subject_id=_sentence_subject_id(text, hit_kw),
        context_factor=cf,
    )]


# ── TIME_LIMIT_PRESSURE ──

@register_rule
def time_limit_pressure(ctx: RuleContext) -> list | None:
    """检测时间限制压力话术."""
    config = load_config()
    wp = config.rules.wording_patterns.get("time_limit", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.70)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    cf = 0.2 if _check_official_context(ctx) else 1.0

    return [make_evidence(
        rule_id="TIME_LIMIT_PRESSURE",
        title="时间限制压力话术",
        group="social",
        severity="medium",
        confidence=default_conf,
        base_score=10,
        reason=f"文本包含时间限制关键词: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "time_limit"],
        subject_id=_sentence_subject_id(text, hit_kw),
        context_factor=cf,
    )]


# ── ACCOUNT_DISABLE_THREAT ──

@register_rule
def account_disable_threat(ctx: RuleContext) -> list | None:
    """检测账号停用威胁话术."""
    config = load_config()
    wp = config.rules.wording_patterns.get("account_disable", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.70)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    cf = 0.2 if _check_official_context(ctx) else 1.0

    return [make_evidence(
        rule_id="ACCOUNT_DISABLE_THREAT",
        title="账号停用威胁话术",
        group="social",
        severity="medium",
        confidence=default_conf,
        base_score=10,
        reason=f"文本包含账号停用威胁关键词: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "threat"],
        subject_id=_sentence_subject_id(text, hit_kw),
        context_factor=cf,
    )]


# ── REWARD_OR_REFUND_BAIT ──

@register_rule
def reward_or_refund_bait(ctx: RuleContext) -> list | None:
    """检测奖励/退款诱导话术."""
    config = load_config()
    wp = config.rules.wording_patterns.get("reward_bait", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.65)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    cf = 0.2 if _check_official_context(ctx) else 1.0

    return [make_evidence(
        rule_id="REWARD_OR_REFUND_BAIT",
        title="奖励或退款诱导话术",
        group="social",
        severity="medium",
        confidence=default_conf,
        base_score=8,
        reason=f"文本包含利益诱导关键词: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "bait"],
        subject_id=_sentence_subject_id(text, hit_kw),
        context_factor=cf,
    )]


# ── SECURITY_UPGRADE_WORDING ──

@register_rule
def security_upgrade_wording(ctx: RuleContext) -> list | None:
    """检测安全升级话术."""
    config = load_config()
    wp = config.rules.wording_patterns.get("security_upgrade", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.55)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    cf = 0.2 if _check_official_context(ctx) else 1.0

    return [make_evidence(
        rule_id="SECURITY_UPGRADE_WORDING",
        title="安全升级话术",
        group="social",
        severity="low",
        confidence=default_conf,
        base_score=5,
        reason=f"文本包含安全升级关键词: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "security"],
        subject_id=_sentence_subject_id(text, hit_kw),
        context_factor=cf,
    )]


# ── CREDENTIAL_REQUEST_TEXT ──

@register_rule
def credential_request_text(ctx: RuleContext) -> list | None:
    """检测文本中直接要求提交凭据."""
    config = load_config()
    wp = config.rules.wording_patterns.get("credential_request", {})
    keywords = wp.get("keywords", [])
    default_conf = wp.get("confidence", 0.85)
    if not keywords:
        return None

    text = ctx.normalized_text
    matched, hit_kw = _match_wording_pattern(text, keywords)
    if not matched:
        return None

    # 官方上下文不降权（要求提交凭据本身可疑）
    return [make_evidence(
        rule_id="CREDENTIAL_REQUEST_TEXT",
        title="文本要求提交凭据",
        group="credential",
        severity="high",
        confidence=default_conf,
        base_score=25,
        reason=f"文本直接要求提交密码/验证码/银行卡信息: {hit_kw}",
        matched_content=hit_kw[:160],
        source="normalized",
        tags=["wording", "credential_request"],
        subject_id=_sentence_subject_id(text, hit_kw),
    )]
