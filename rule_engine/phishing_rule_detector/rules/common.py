"""规则公共模块：RuleContext、URL 提取、confusable 加载与通用工具."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from bs4.element import Tag

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.models import (
    EVIDENCE_GROUPS,
    EVIDENCE_SOURCE,
    SEVERITY_LEVELS,
    EvidenceItem,
)
from phishing_rule_detector.scoring import compute_effective_score
from phishing_rule_detector.trust_policy import (
    get_registered_domain,
    is_official_domain,
)

_logger = logging.getLogger(__name__)

# ── 尾随标点正则（与 normalizer 保持一致）──
_TRAILING_PUNCTUATION_RE = re.compile(r"[.,);!?\]}。，；！？）】〉》\"']+$")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── 空壳关键词模式（插空格/零宽字符）──
_SPACE_OR_ZW = re.compile(r"[\s​‌‍﻿­]*")


def _build_flexible_pattern(keyword: str) -> str:
    """为关键词生成允许零宽和空格插入的正则片段."""
    chars = [_SPACE_OR_ZW.pattern + re.escape(ch) for ch in keyword]
    return "".join(chars) + _SPACE_OR_ZW.pattern


# ── Confusable 加载 ──

_confusable_map: dict[str, str] | None = None


def _load_confusable_map() -> dict[str, str]:
    """加载 confusables.txt 并缓存映射（码点 → 骨架码点）."""
    global _confusable_map
    if _confusable_map is not None:
        return _confusable_map

    mapping: dict[str, str] = {}
    path = _DATA_DIR / "confusables.txt"
    if not path.exists():
        _logger.warning("confusables.txt 未找到: %s", path)
        _confusable_map = mapping
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 格式: 05AD ;  0596 ;  MA  # ...
            # 多码点目标: 1E96 ; 0068 0331 ;  MA  # (ẖ → ẖ)
            parts = line.split(";", 2)
            if len(parts) < 2:
                continue
            source = parts[0].strip()
            target = parts[1].strip()
            try:
                source_cp = int(source, 16)
                # 解析目标：支持多码点（空格分隔）和单码点
                target_cps = [int(t, 16) for t in target.split()]
                mapping[chr(source_cp)] = "".join(chr(cp) for cp in target_cps)
            except (ValueError, OverflowError):
                continue

    _confusable_map = mapping
    return mapping


def get_confusable_skeleton(text: str) -> str:
    """将文本中所有可混淆字符替换为其骨架字符."""
    mapping = _load_confusable_map()
    if not mapping:
        return text
    return "".join(mapping.get(ch, ch) for ch in text)


def has_mixed_script(text: str) -> bool:
    """检测文本是否混合使用拉丁、西里尔、希腊等脚本."""
    scripts_found: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            scripts_found.add("latin")
        elif 0x0400 <= cp <= 0x04FF:
            scripts_found.add("cyrillic")
        elif 0x0370 <= cp <= 0x03FF:
            scripts_found.add("greek")
        elif 0x0530 <= cp <= 0x058F:
            scripts_found.add("armenian")
    return len(scripts_found) >= 2


# ── URL 提取 ──

@dataclass
class ExtractedURL:
    """从输入中提取的 URL 信息."""
    raw: str                    # 原始 URL
    normalized: str             # 规范化后的 URL（IDNA 解码、URL decode 等）
    scheme: str
    hostname: str
    port: int | None
    path: str
    query: str
    fragment: str
    registered_domain: str
    is_official: bool
    source: EVIDENCE_SOURCE = "raw"
    element_type: str = ""      # a | form | iframe | meta | img
    original_href: str = ""     # HTML 中的原始 href/action 值
    subject_id: str = ""        # 由 detector 汇总去重后统一分配
    parse_warning: str | None = None


# ── URL 正则（匹配 http/https scheme）──
_URL_RE = re.compile(r"https?://[^\s<>\"'　]+", re.IGNORECASE)


def _strip_trailing_punctuation(url: str) -> str:
    """移除 URL 末尾的自然语言标点."""
    return _TRAILING_PUNCTUATION_RE.sub("", url)


def _canonical_url_key(url: str) -> str:
    """生成 URL 的规范化键（用于去重）.

    - 小写化 scheme 和 hostname
    - 移除默认端口（80/443）
    - IDNA 解码 Punycode hostname
    """
    try:
        split = urlsplit(url)
        scheme = split.scheme.lower()
        hostname = (split.hostname or "").lower()
        port = split.port

        # 移除默认端口
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        # IDNA 解码 Punycode hostname
        if "xn--" in hostname:
            try:
                import idna
                hostname = idna.decode(hostname).lower()
            except Exception:
                pass

        # 重建规范化 URL
        netloc = hostname
        if port is not None:
            netloc = f"{hostname}:{port}"
        elif split.username or split.password:
            userinfo = ""
            if split.username:
                userinfo = split.username
                if split.password:
                    userinfo += ":[REDACTED]"
            netloc = f"{userinfo}@{hostname}"
            if port is not None:
                netloc += f":{port}"

        canonical = split._replace(scheme=scheme, netloc=netloc).geturl()
        return canonical
    except Exception:
        return url


def extract_urls_from_text(
    text: str,
    base_url: str | None = None,
    source: EVIDENCE_SOURCE = "raw",
    element_type: str = "",
    original_href: str = "",
) -> list[ExtractedURL]:
    """从文本中提取所有 http/https URL."""
    results: list[ExtractedURL] = []
    seen: set[str] = set()

    for match in _URL_RE.finditer(text):
        try:
            raw = match.group(0)
            # 剥离尾部标点
            clean = _strip_trailing_punctuation(raw)
            canon_key = _canonical_url_key(clean)
            if canon_key in seen:
                continue
            seen.add(canon_key)

            extracted = _parse_url(clean, base_url, source, element_type, original_href)
            if extracted:
                results.append(extracted)
        except Exception:
            continue

    return results


def extract_urls_from_html(
    soup: BeautifulSoup,
    base_url: str | None = None,
) -> list[ExtractedURL]:
    """从 BeautifulSoup DOM 中提取所有 URL（href、src、action）."""
    results: list[ExtractedURL] = []
    seen: set[str] = set()

    def _try_add(href: str, element_type: str) -> None:
        """安全提取并添加单个 URL（单条异常不影响批量）."""
        try:
            resolved = _resolve_url(href, base_url)
            if not resolved:
                return
            canon_key = _canonical_url_key(resolved)
            if canon_key in seen:
                return
            seen.add(canon_key)
            extracted = _parse_url(
                resolved, base_url, source="raw", element_type=element_type,
                original_href=href,
            )
            if extracted:
                extracted.raw = resolved
                extracted.original_href = href
                results.append(extracted)
        except Exception:
            pass

    # a[href]
    for tag in cast(list[Tag], soup.find_all("a", href=True)):
        _try_add(str(tag.get("href", "")), "a")

    # form[action]
    for tag in cast(list[Tag], soup.find_all("form", action=True)):
        _try_add(str(tag.get("action", "")), "form")

    # iframe[src]
    for tag in cast(list[Tag], soup.find_all("iframe", src=True)):
        _try_add(str(tag.get("src", "")), "iframe")

    # meta[content] with refresh
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
        content = str(tag.get("content", ""))
        url_match = re.search(r"url[=]\s*([^;,\s]+)", content, re.IGNORECASE)
        if url_match:
            _try_add(url_match.group(1).strip("\"'"), "meta")

    # img[src] — 仅用于图片分析，不作为导航 URL
    # (由 image_rules 单独处理)

    return results


def _resolve_url(href: str, base_url: str | None) -> str | None:
    """解析相对 URL."""
    if not href or not href.strip():
        return None
    href = href.strip()
    if base_url:
        try:
            resolved = urljoin(base_url, href)
        except (ValueError, AttributeError):
            resolved = href
    else:
        resolved = href
    return resolved if resolved.startswith(("http://", "https://")) else None


def _parse_url(
    url: str,
    base_url: str | None = None,
    source: EVIDENCE_SOURCE = "raw",
    element_type: str = "",
    original_href: str = "",
) -> ExtractedURL | None:
    """解析单个 URL 为 ExtractedURL."""
    try:
        split = urlsplit(url)
    except (ValueError, AttributeError):
        return None

    hostname = (split.hostname or "").lower()
    if not hostname:
        return None

    try:
        reg_domain = get_registered_domain(hostname)
    except Exception:
        reg_domain = hostname

    parse_warning = None
    try:
        port = split.port
    except ValueError:
        port = None
        parse_warning = "invalid_port"

    return ExtractedURL(
        raw=url,
        normalized=url,
        scheme=split.scheme.lower(),
        hostname=hostname,
        port=port,
        path=split.path or "",
        query=split.query or "",
        fragment=split.fragment or "",
        registered_domain=reg_domain,
        is_official=is_official_domain(hostname),
        source=source,
        element_type=element_type,
        original_href=original_href,
        parse_warning=parse_warning,
    )


# ── RuleContext ──

@dataclass
class RuleContext:
    """规则检测上下文，供所有规则模块使用."""
    input_text: str
    normalized_text: str
    input_type: str
    base_url: str | None
    raw_text: str
    sender: str = ""
    attachments: list[str] = field(default_factory=list)
    qr_urls: list[str] = field(default_factory=list)
    ocr_text: str = ""
    parsed_html: BeautifulSoup | None = None
    extracted_urls: list[ExtractedURL] = field(default_factory=list)
    trace_id: str = ""


# ── 规则注册表 ──

RuleCallable = Callable[[RuleContext], EvidenceItem | list[EvidenceItem] | None]
_RULE_REGISTRY: list[RuleCallable] = []


def register_rule(
    func: RuleCallable | None = None,
    *,
    rule_id: str | None = None,
) -> Any:
    """装饰器：将规则函数注册到全局注册表.

    用法:
        @register_rule
        def my_rule(ctx): ...

        @register_rule(rule_id="CUSTOM_ID")
        def my_rule(ctx): ...

    显式 rule_id 优先于函数名，确保跨重构稳定。
    """
    def _decorator(f: RuleCallable) -> RuleCallable:
        setattr(f, "rule_id", rule_id or f.__name__.upper())
        _RULE_REGISTRY.append(f)
        return f

    if func is not None:
        return _decorator(func)
    return _decorator


def get_registered_rules() -> list[RuleCallable]:
    """返回所有已注册的规则函数."""
    return list(_RULE_REGISTRY)


def run_all_rules(ctx: RuleContext) -> tuple[list[EvidenceItem], list[str]]:
    """按注册顺序执行所有规则，隔离异常.

    尊重 rule_definitions 中的 enabled 开关。

    Returns:
        (evidence_list, warnings_list)
    """
    all_evidence: list[EvidenceItem] = []
    all_warnings: list[str] = []

    config = load_config()
    rule_defs = config.rules.rule_definitions

    for rule_func in _RULE_REGISTRY:
        rule_id = str(
            getattr(rule_func, "rule_id", rule_func.__name__.upper())
        )

        # 检查规则是否启用
        rule_def = rule_defs.get(rule_id)
        if rule_def is not None and not rule_def.enabled:
            continue

        try:
            evidence = rule_func(ctx)
            if evidence:
                if isinstance(evidence, list):
                    all_evidence.extend(evidence)
                else:
                    all_evidence.append(evidence)
        except Exception as exc:
            _logger.error(
                "规则 %s 执行异常 [error_type=%s]",
                rule_id,
                type(exc).__name__,
            )
            all_warnings.append(f"RULE_ERROR:{rule_id}")

    return all_evidence, all_warnings


# ── 脱敏工具 ──

_SENSITIVE_PARAMS_RE = re.compile(
    r"((?:^|[?&#])(?:password|token|code|key|secret|pwd|pass)=)([^&#\s]*)",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{3})(\d{13})([\dXx]{2})(?![\dXx])")
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)

# 匹配 URL userinfo 中的密码部分：scheme://user:password@host
_USERINFO_PASSWORD_RE = re.compile(
    r"(https?://[^:@/\s]+:)([^@/\s]+)(@)",
    re.IGNORECASE,
)

# Evidence reasons sometimes contain a quoted userinfo fragment without a URL
# scheme, for example: "userinfo 'user.example:secret'".
_USERINFO_FRAGMENT_RE = re.compile(
    r"((?:[A-Za-z0-9._-]+\.)+[A-Za-z0-9._-]+:)([^@\s'\"/]+)(?=@|['\"])",
)


def _redact_sensitive_params(query: str) -> str:
    """脱敏 URL query 中的敏感参数."""
    if not query:
        return query
    return _SENSITIVE_PARAMS_RE.sub(r"\1[REDACTED]", query)


def _redact_userinfo_password(url: str) -> str:
    """脱敏 URL userinfo 中的密码."""
    redacted = _USERINFO_PASSWORD_RE.sub(r"\1[REDACTED]\3", url)
    return _USERINFO_FRAGMENT_RE.sub(r"\1[REDACTED]", redacted)


def _redact_personal_data(content: str) -> str:
    """按隐私规范掩码身份证号、手机号和邮箱本地部分."""
    content = _ID_CARD_RE.sub(
        lambda match: match.group(1) + "*" * 13 + match.group(3),
        content,
    )
    content = _PHONE_RE.sub(r"\1****\3", content)

    def _mask_email(match: re.Match[str]) -> str:
        local = match.group(1)
        if len(local) == 1:
            masked_local = local + "*"
        elif len(local) == 2:
            masked_local = local[0] + "*" + local[-1]
        else:
            masked_local = local[0] + "***" + local[-1]
        return f"{masked_local}@{match.group(2)}"

    return _EMAIL_RE.sub(_mask_email, content)


def redact_url(url: str) -> str:
    """脱敏 URL：替换 query 中敏感参数值和 userinfo 密码."""
    try:
        url = _redact_userinfo_password(url)
        split = urlsplit(url)
        clean_query = _redact_sensitive_params(split.query)
        clean_fragment = _redact_sensitive_params(split.fragment)
        result = split._replace(
            query=clean_query,
            fragment=clean_fragment,
        ).geturl()
        return _redact_personal_data(result)
    except Exception:
        return url


def redact_matched_content(content: str, max_len: int = 160) -> str:
    """脱敏并截断命中内容."""
    if not content:
        return ""
    # 移除换行
    content = content.replace("\n", " ").replace("\r", " ")
    # 脱敏 URL userinfo 密码
    content = _redact_userinfo_password(content)
    # 脱敏 query 和 fragment 敏感参数
    content = _SENSITIVE_PARAMS_RE.sub(r"\1[REDACTED]", content)
    content = _redact_personal_data(content)
    if len(content) > max_len:
        content = content[:max_len - 3] + "..."
    return content


def make_evidence(
    rule_id: str,
    title: str,
    group: EVIDENCE_GROUPS,
    severity: SEVERITY_LEVELS,
    confidence: float,
    base_score: int,
    reason: str,
    matched_content: str = "",
    source: EVIDENCE_SOURCE = "raw",
    tags: list[str] | None = None,
    subject_id: str | None = None,
    context_factor: float = 1.0,
) -> EvidenceItem:
    """构造 EvidenceItem，自动计算 effective_score 并脱敏."""
    rule_definition = load_config().rules.rule_definitions.get(rule_id)
    if rule_definition is not None:
        if rule_definition.title:
            title = rule_definition.title
        if rule_definition.tags:
            tags = list(rule_definition.tags)
        group = rule_definition.group
        if not rule_definition.dynamic_scoring:
            severity = rule_definition.severity
            base_score = rule_definition.base_score
            if rule_definition.confidence is not None:
                confidence = rule_definition.confidence

    effective_score = compute_effective_score(base_score, confidence, context_factor)
    return EvidenceItem(
        rule_id=rule_id,
        title=title,
        group=group,
        severity=severity,
        confidence=confidence,
        context_factor=context_factor,
        base_score=base_score,
        effective_score=effective_score,
        reason=redact_matched_content(reason, max_len=500),
        matched_content=redact_matched_content(matched_content),
        source=source,
        tags=tags or [],
        subject_id=subject_id,
    )


# ── 表单字段识别 ──

def extract_form_fields(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """从 DOM 中提取表单字段信息（name、id、placeholder、label、type）."""
    fields: list[dict[str, Any]] = []
    form_tags = cast(list[Tag], soup.find_all("form")) if soup else []
    for fi, form_tag in enumerate(form_tags):
        form_fields: list[dict[str, Any]] = []
        input_tags = cast(
            list[Tag],
            form_tag.find_all(["input", "select", "textarea"]),
        )
        for input_tag in input_tags:
            field_info = {
                "tag": input_tag.name,
                "name": str(input_tag.get("name", "")).lower(),
                "id": str(input_tag.get("id", "")).lower(),
                "type": str(input_tag.get("type", "")).lower(),
                "placeholder": str(input_tag.get("placeholder", "")).lower(),
                "aria_label": str(input_tag.get("aria-label", "")).lower(),
            }
            # 查找关联 label
            label_text = ""
            label_for = input_tag.get("id", "")
            if label_for:
                label_tag = soup.find("label", attrs={"for": label_for})
                if label_tag:
                    label_text = label_tag.get_text(strip=True).lower()
            # 也检查父级 label
            parent_label = input_tag.find_parent("label")
            if parent_label:
                label_text = label_text or parent_label.get_text(strip=True).lower()
            field_info["label"] = label_text
            form_fields.append(field_info)
        fields.append({"form_index": fi, "fields": form_fields})
    return fields


def check_field_matches_keywords(
    field: dict[str, Any], keywords: list[str]
) -> bool:
    """检查表单字段是否匹配关键词列表（在 name/id/placeholder/label 中搜索）."""
    search_text = " ".join(
        v for k, v in field.items()
        if k in ("name", "id", "placeholder", "label", "aria_label")
    )
    search_text = search_text.lower()
    for kw in keywords:
        if kw.lower() in search_text:
            return True
    return False


def normalize_filename(filename: str) -> str:
    """规范化文件名：去除首尾空格和点，小写化."""
    return filename.strip(" .").lower()


def get_file_extensions(filename: str) -> list[str]:
    """获取文件名的所有后缀层级（如 .pdf.exe → [.exe, .pdf.exe]）."""
    name = normalize_filename(filename)
    exts: list[str] = []
    while "." in name:
        _, ext = name.rsplit(".", 1)
        if not ext:
            break
        exts.append("." + ext)
        name = name.rsplit(".", 1)[0]
    return exts
