"""邮件与身份规则.

- EMAIL_SENDER_IMPERSONATION
- EMAIL_SENDER_LINK_MISMATCH
"""
from __future__ import annotations

from email.utils import parseaddr

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.rules.common import (
    RuleContext,
    get_registered_domain,
    is_official_domain,
    make_evidence,
    register_rule,
)


def _extract_sender_domain(sender: str) -> str:
    """从 sender 字段提取域名."""
    if not sender:
        return ""
    try:
        _, addr = parseaddr(sender)
        if addr and "@" in addr:
            return addr.rsplit("@", 1)[-1].lower()
    except Exception:
        pass
    return ""


# ── EMAIL_SENDER_IMPERSONATION ──

@register_rule
def email_sender_impersonation(ctx: RuleContext) -> list | None:
    """非官方发件人自称学校部门."""
    sender = ctx.sender
    if not sender:
        return None

    sender_domain = _extract_sender_domain(sender)
    if not sender_domain:
        return None

    # 官方发件人不触发
    if is_official_domain(sender_domain):
        return None

    # 检查文本中是否自称学校部门
    config = load_config()
    brand_kws = config.rules.brand_keywords
    text_lower = ctx.normalized_text.lower()

    has_brand = any(kw.lower() in text_lower for kw in brand_kws)
    has_department = any(
        kw in text_lower
        for kw in ["信息化", "信息技术中心", "网络中心", "教务处", "学生处", "财务处", "信息化工作办公室"]
    )

    if has_brand and has_department:
        return [make_evidence(
            rule_id="EMAIL_SENDER_IMPERSONATION",
            title="非官方发件人冒充学校部门",
            group="identity",
            severity="medium",
            confidence=0.85,
            base_score=15,
            reason=f"发件域名 {sender_domain} 非官方，但正文自称学校部门",
            matched_content=sender[:160],
            source="raw",
            tags=["sender_impersonation"],
            subject_id="sender:0",
        )]

    return None


# ── EMAIL_SENDER_LINK_MISMATCH ──

@register_rule
def email_sender_link_mismatch(ctx: RuleContext) -> list | None:
    """发件域名与正文主要链接域名的注册域不同."""
    sender = ctx.sender
    if not sender:
        return None

    sender_domain = _extract_sender_domain(sender)
    if not sender_domain:
        return None

    try:
        sender_reg = get_registered_domain(sender_domain)
    except Exception:
        sender_reg = sender_domain

    if not sender_reg:
        return None

    # 收集正文中链接的注册域
    link_regs: set[str] = set()
    for u in ctx.extracted_urls:
        if u.hostname:
            try:
                reg = u.registered_domain or get_registered_domain(u.hostname)
                if reg:
                    link_regs.add(reg)
            except Exception:
                pass

    if not link_regs:
        return None

    # 如果所有链接的注册域都与发件注册域不同，且有品牌/认证上下文
    if all(reg != sender_reg for reg in link_regs):
        config = load_config()
        brand_kws = config.rules.brand_keywords
        login_kws = config.rules.login_keywords
        text_lower = ctx.normalized_text.lower()

        has_brand = any(kw.lower() in text_lower for kw in brand_kws)
        has_login = any(kw.lower() in text_lower for kw in login_kws)

        if has_brand or has_login:
            return [make_evidence(
                rule_id="EMAIL_SENDER_LINK_MISMATCH",
                title="发件域名与链接域名不一致",
                group="identity",
                severity="medium",
                confidence=0.75 if (has_brand and has_login) else 0.60,
                base_score=12,
                reason=f"发件注册域 {sender_reg} 与正文链接注册域不一致",
                matched_content=f"sender: {sender_reg}, links: {', '.join(link_regs)}"[:160],
                source="raw",
                tags=["sender_link_mismatch"],
                subject_id="sender:0",
            )]

    return None
