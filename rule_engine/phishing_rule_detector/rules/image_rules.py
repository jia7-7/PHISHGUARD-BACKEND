"""附件、图片与二维码规则.

- ATTACHMENT_EXECUTABLE
- ATTACHMENT_DOUBLE_EXTENSION
- HTML_BASE64_IMAGE_HEAVY
- HTML_IMAGE_ONLY_CONTENT
- QR_CODE_EXTERNAL_URL
- QR_CREDENTIAL_URL
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.rules.common import (
    RuleContext,
    get_file_extensions,
    is_official_domain,
    make_evidence,
    normalize_filename,
    register_rule,
)


# ── 图片阈值（来自 YAML 或默认）──
_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,", re.IGNORECASE)
_BASE64_THRESHOLD = 2000    # base64 图片累计长度阈值
_VISIBLE_TEXT_THRESHOLD = 100  # 可见正文长度阈值
_IMAGE_COUNT_THRESHOLD = 5    # 图片数量阈值


# ── ATTACHMENT_EXECUTABLE ──

@register_rule
def attachment_executable(ctx: RuleContext) -> list | None:
    """检测危险附件后缀."""
    config = load_config()
    dangerous_exts = [e.lower() for e in config.rules.dangerous_extensions]
    evidence_list = []

    for ai, att in enumerate(ctx.attachments):
        name = normalize_filename(att)
        exts = get_file_extensions(name)
        for ext in exts:
            if ext.lower() in dangerous_exts:
                evidence_list.append(make_evidence(
                    rule_id="ATTACHMENT_EXECUTABLE",
                    title="危险附件类型",
                    group="payload",
                    severity="high",
                    confidence=0.95,
                    base_score=25,
                    reason=f"附件 {att[:80]} 为可执行/脚本类型 ({ext})",
                    matched_content=att[:160],
                    source="attachment",
                    tags=["attachment", "executable"],
                    subject_id=f"attachment:{ai}",
                ))
                break

    return evidence_list or None


# ── ATTACHMENT_DOUBLE_EXTENSION ──

@register_rule
def attachment_double_extension(ctx: RuleContext) -> list | None:
    """检测双重后缀附件."""
    config = load_config()
    dangerous_exts = [e.lower() for e in config.rules.dangerous_extensions]
    evidence_list = []

    for ai, att in enumerate(ctx.attachments):
        name = normalize_filename(att)
        exts = get_file_extensions(name)
        # 双重后缀：多个后缀且最后一层是危险类型
        if len(exts) >= 2:
            last_ext = exts[0].lower()  # 最外层后缀
            if last_ext in dangerous_exts:
                evidence_list.append(make_evidence(
                    rule_id="ATTACHMENT_DOUBLE_EXTENSION",
                    title="双重后缀附件",
                    group="payload",
                    severity="high",
                    confidence=0.95,
                    base_score=30,
                    reason=f"附件 {att[:80]} 使用双重后缀 ({' | '.join(exts[:3])})",
                    matched_content=att[:160],
                    source="attachment",
                    tags=["attachment", "double_extension"],
                    subject_id=f"attachment:{ai}",
                ))

    return evidence_list or None


# ── HTML_BASE64_IMAGE_HEAVY ──

@register_rule
def html_base64_image_heavy(ctx: RuleContext) -> list | None:
    """检测 HTML 中大量使用内嵌 base64 图片."""
    text = ctx.normalized_text
    if not text:
        return None

    total_base64_len = 0
    for m in _BASE64_IMAGE_RE.finditer(text):
        # 从 data URI 前缀后开始，找到闭合引号或空白
        data_start = m.end()
        end_pos = data_start
        quote_char = None
        if data_start > 0 and text[data_start - 1] in ('"', "'"):
            quote_char = text[data_start - 1]
        # 搜索 base64 数据结束位置
        for i in range(data_start, min(len(text), data_start + 100000)):
            ch = text[i]
            if quote_char and ch == quote_char:
                end_pos = i
                break
            if not quote_char and ch in ('"', "'", ' ', '>', '\n', '\r'):
                end_pos = i
                break
        else:
            end_pos = min(len(text), data_start + 100000)
        total_base64_len += (end_pos - data_start)

    if total_base64_len > _BASE64_THRESHOLD:
        # 检查可见正文长度
        visible_text = re.sub(r"<[^>]+>", " ", text).strip()
        visible_len = len(visible_text)
        if visible_len < _VISIBLE_TEXT_THRESHOLD:
            return [make_evidence(
                rule_id="HTML_BASE64_IMAGE_HEAVY",
                title="大量内嵌 Base64 图片",
                group="payload",
                severity="low",
                confidence=0.45,
                base_score=5,
                reason=f"HTML 包含大量 base64 图片（约 {total_base64_len} 字符），可见正文仅 {visible_len} 字符",
                matched_content=f"base64 images: ~{total_base64_len} chars"[:160],
                source="normalized",
                tags=["base64", "image"],
                subject_id="page:0",
            )]

    return None


# ── HTML_IMAGE_ONLY_CONTENT ──

@register_rule
def html_image_only_content(ctx: RuleContext) -> list | None:
    """检测页面主要由图片组成且缺少可读文本."""
    soup = ctx.parsed_html
    if soup is None:
        return None

    # 统计图片数量
    img_tags = soup.find_all("img")
    img_count = len(img_tags)

    if img_count < _IMAGE_COUNT_THRESHOLD:
        return None

    # 检查可见文本
    visible_text = soup.get_text(separator=" ", strip=True)
    if len(visible_text) < _VISIBLE_TEXT_THRESHOLD and img_count >= _IMAGE_COUNT_THRESHOLD:
        return [make_evidence(
            rule_id="HTML_IMAGE_ONLY_CONTENT",
            title="页面主要由图片组成",
            group="payload",
            severity="low",
            confidence=0.40,
            base_score=5,
            reason=f"页面包含 {img_count} 张图片，但仅有 {len(visible_text)} 字符可见文本",
            matched_content=f"images: {img_count}, text: {len(visible_text)} chars"[:160],
            source="normalized",
            tags=["image_only"],
            subject_id="page:0",
        )]

    return None


# ── QR_CODE_EXTERNAL_URL ──

@register_rule
def qr_code_external_url(ctx: RuleContext) -> list | None:
    """检测二维码 URL 指向外部域名."""
    evidence_list = []

    for qi, qr_url in enumerate(ctx.qr_urls):
        try:
            parsed = urlsplit(qr_url)
            hostname = (parsed.hostname or "").lower()
        except Exception:
            continue

        if not hostname:
            continue

        if not is_official_domain(hostname):
            evidence_list.append(make_evidence(
                rule_id="QR_CODE_EXTERNAL_URL",
                title="二维码指向外部域名",
                group="payload",
                severity="medium",
                confidence=0.70,
                base_score=12,
                reason=f"二维码 URL 指向外部域名 {hostname}",
                matched_content=qr_url[:160],
                source="qr",
                tags=["qr", "external"],
                subject_id=f"qr:{qi}",
            ))

    return evidence_list or None


# ── QR_CREDENTIAL_URL ──

@register_rule
def qr_credential_url(ctx: RuleContext) -> list | None:
    """检测二维码 URL 指向外部登录/凭据页面."""
    evidence_list = []

    for qi, qr_url in enumerate(ctx.qr_urls):
        try:
            parsed = urlsplit(qr_url)
            hostname = (parsed.hostname or "").lower()
            path_query = (parsed.path + parsed.query).lower()
        except Exception:
            continue

        if not hostname:
            continue

        if is_official_domain(hostname):
            continue

        # 检查凭据线索
        credential_clues = ["login", "auth", "verify", "password", "token", "sms", "验证码", "登录", "认证"]
        has_credential = any(clue in path_query for clue in credential_clues)

        if has_credential:
            evidence_list.append(make_evidence(
                rule_id="QR_CREDENTIAL_URL",
                title="二维码指向外域凭据页面",
                group="credential",
                severity="high",
                confidence=0.90,
                base_score=25,
                reason=f"二维码指向外域登录/认证/验证码页面: {hostname}",
                matched_content=qr_url[:160],
                source="qr",
                tags=["qr", "credential"],
                subject_id=f"qr:{qi}",
            ))

    return evidence_list or None
