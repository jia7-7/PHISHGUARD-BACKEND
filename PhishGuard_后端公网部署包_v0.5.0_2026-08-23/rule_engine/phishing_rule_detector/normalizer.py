"""输入规范化与反混淆管线.

处理顺序（遵循设计文档 §4.1）：
  原始输入
  → HTML 实体解码
  → Unicode NFKC 规范化
  → 删除零宽和方向控制字符
  → URL Percent Decode（最多两轮）
  → URL 与域名结构化解析
  → IDNA/Punycode 解码
  → HTML DOM 解析或纯文本降级
"""
from __future__ import annotations

import html as html_mod
import logging
import re
import unicodedata
from urllib.parse import unquote, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from phishing_rule_detector.models import MAX_INPUT_BYTES

_logger = logging.getLogger(__name__)


class PayloadTooLargeError(ValueError):
    """输入字节数超过规范化管线限制."""

    def __init__(self, actual_bytes: int, max_bytes: int = MAX_INPUT_BYTES):
        self.actual_bytes = actual_bytes
        self.max_bytes = max_bytes
        super().__init__(f"总输入超过 {max_bytes} 字节限制: {actual_bytes}")


# 零宽和方向控制字符集合
ZERO_WIDTH_CHARS = re.compile(
    "[​‌‍⁠﻿‎‏‪-‮­]"
)


def decode_html_entities(text: str) -> tuple[str, bool]:
    """解码 HTML 实体（&#x61; / &#97; / &colon; 等）."""
    decoded = html_mod.unescape(text)
    changed = decoded != text
    return decoded, changed


def normalize_unicode(text: str) -> tuple[str, list[str]]:
    """NFKC 规范化 Unicode 文本."""
    result = unicodedata.normalize("NFKC", text)
    ops = []
    if result != text:
        ops.append("unicode_nfkc")
    return result, ops


def remove_zero_width(text: str) -> tuple[str, bool]:
    """删除零宽字符、方向控制字符和软连字符."""
    result = ZERO_WIDTH_CHARS.sub("", text)
    changed = result != text
    return result, changed


def url_decode_text(text: str, max_rounds: int = 2) -> tuple[str, int]:
    """对文本执行 URL Percent Decode，最多 max_rounds 轮.

    使用 unquote（不把 + 转空格），避免 unquote_plus 的副作用。
    """
    rounds = 0
    current = text
    for _ in range(max_rounds):
        decoded = unquote(current, errors="replace")
        if decoded == current:
            break
        current = decoded
        rounds += 1
    return current, rounds


def normalize_domain(hostname: str) -> str:
    """域名规范化：小写化并移除末尾点."""
    hostname = hostname.lower().strip()
    hostname = hostname.rstrip(".")
    return hostname


def parse_html_dom(
    html: str,
) -> tuple[BeautifulSoup | None, dict]:
    """使用 BeautifulSoup 解析 HTML.

    解析器优先级：lxml → html.parser.
    返回 (soup, meta_dict).
    meta_dict 包含:
        - parser: 实际使用的解析器名称
        - parser_fallback: 是否因 lxml 不可用而降级
        - error: 解析失败时的错误信息（可选）
    """
    meta: dict = {"parser": "lxml", "parser_fallback": False}

    if not html or not html.strip():
        meta["parser"] = "html.parser"
        return BeautifulSoup("", "html.parser"), meta

    # 首选 lxml
    try:
        soup = BeautifulSoup(html, "lxml")
        return soup, meta
    except Exception:
        pass

    # 降级为 html.parser
    meta["parser"] = "html.parser"
    meta["parser_fallback"] = True
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup, meta
    except Exception as e:
        meta["error"] = str(e)
        return None, meta


def _decode_idna_hostname(hostname: str) -> tuple[str, bool]:
    """使用 idna 包对 IDNA 2008 域名进行解码.

    Returns:
        (decoded_hostname, success) — success 为 False 表示解码失败.
    """
    import idna

    try:
        # IDNA 2008 解码：将 Punycode/ACE 域名转为 Unicode
        decoded = idna.decode(hostname)
        return decoded, True
    except (idna.IDNAError, UnicodeError, LookupError):
        return hostname, False


# 自然语言标点符号（不在 URL hostname 中合法出现），需从匹配尾部剥离
_TRAILING_PUNCTUATION_RE = re.compile(
    r"[.,:);!?\]}。，；：！？）】〉》\"']+$"
)


def _strip_trailing_punctuation(url: str) -> tuple[str, str]:
    """移除 URL 末尾的自然语言标点符号.

    仅剥离在 URL hostname 中不合法的尾部标点；
    不修改 URL 内部的合法标点（如路径中的逗号）。

    Returns:
        (clean_url, stripped_suffix) — 剥离后的 URL 与被剥离的尾部字符串.
    """
    match = _TRAILING_PUNCTUATION_RE.search(url)
    if match:
        stripped = match.group(0)
        return url[:match.start()], stripped
    return url, ""


def _extract_and_normalize_domains(text: str) -> tuple[str, bool]:
    """从文本中识别 URL 并对域名部分做 IDNA 解码.

    使用正则定位 URL 范围，urlsplit 解析 URL 结构，只对 hostname 做 IDNA 2008 解码，
    最后 urlunsplit 重建 URL。

    保留：username/password、端口、IPv6、path、query、fragment。

    Returns:
        (processed_text, any_success) — any_success 表示是否有成功解码的域名.
    """
    any_success = False

    # 正则只负责定位 URL 范围（hostname 部分，不含路径），URL 内部结构由 urlsplit 解析
    url_pattern = re.compile(r"https?://([^/\"'\s<>]+)", re.IGNORECASE)

    def _decode_url(match: re.Match) -> str:
        nonlocal any_success
        full_url = match.group(0)

        # 剥离尾部自然语言标点（审查 Fix 2）
        clean_url, trailing = _strip_trailing_punctuation(full_url)

        # 用 urlsplit 解析完整 URL，提取纯净的 hostname
        try:
            split = urlsplit(clean_url)
        except Exception:
            return full_url

        hostname = split.hostname
        if hostname is None:
            # 无合法 hostname，保留原 URL（含标点）
            return full_url

        # 尝试对 hostname 做 IDNA 解码
        decoded_hostname, ok = _decode_idna_hostname(hostname)
        if not ok:
            # 解码失败，保留原 URL（含标点）
            return full_url

        # idna.decode() 对普通 ASCII hostname 也会成功，只有 hostname 确实
        # 从 ACE/Punycode 变为 Unicode 时才记录操作并重建 URL。
        if decoded_hostname.casefold() == hostname.casefold():
            return full_url

        # port 是延迟校验属性；畸形或越界端口应保留原 URL，而不是逃逸管线。
        try:
            parsed_port = split.port
        except ValueError:
            return full_url

        any_success = True

        # 保留原始 userinfo 和端口文本，只替换结构化解析出的 hostname。
        # 这样可保留空密码分隔符、userinfo 内的 @ 以及带前导零的端口。
        if "@" in split.netloc:
            userinfo, hostport = split.netloc.rsplit("@", 1)
            userinfo_prefix = f"{userinfo}@"
        else:
            hostport = split.netloc
            userinfo_prefix = ""

        port_suffix = ""
        if parsed_port is not None:
            _raw_host, raw_port = hostport.rsplit(":", 1)
            port_suffix = f":{raw_port}"

        new_netloc = f"{userinfo_prefix}{decoded_hostname}{port_suffix}"

        rebuilt = urlunsplit((
            split.scheme,
            new_netloc,
            split.path,
            split.query,
            split.fragment,
        ))
        # 将剥离的标点符号追加回结果中，保留原文本结构（审查 Fix 2）
        return rebuilt + trailing

    result = url_pattern.sub(_decode_url, text)
    return result, any_success


def normalize(
    input_text: str,
    input_type: str,
    base_url: str | None = None,
    ocr_text: str = "",
) -> tuple[str, dict]:
    """执行完整的输入规范化管线.

    Args:
        input_text: 原始输入文本.
        input_type: url / html / email / sms / text.
        base_url: 页面来源 URL（仅 HTML 类型有意义）.
        ocr_text: 上游 OCR 结果.

    Returns:
        (normalized_text, meta_dict).

    Raises:
        ValueError: 输入为空或超过大小限制.
    """
    # ── 大小校验 ──
    total_bytes = len(input_text.encode("utf-8"))
    if ocr_text:
        total_bytes += len(ocr_text.encode("utf-8"))

    if total_bytes > MAX_INPUT_BYTES:
        raise PayloadTooLargeError(total_bytes)

    text = input_text
    if not text or not text.strip():
        raise ValueError("input_text 不能为空")

    operations: list[str] = []
    input_bytes = len(input_text.encode("utf-8"))

    # ── 1. HTML 实体解码 ──
    text, changed = decode_html_entities(text)
    if changed:
        operations.append("html_entity_decode")

    # ── 2. Unicode NFKC 规范化 ──
    text, nfkc_ops = normalize_unicode(text)
    operations.extend(nfkc_ops)

    # ── 3. 删除零宽字符 ──
    text, changed = remove_zero_width(text)
    if changed:
        operations.append("zero_width_removed")

    # ── 4. URL Percent Decode（最多两轮）──
    # 仅对 URL 类型或包含 URL 特征的内容执行
    if "url" in input_type or "%" in text:
        text, rounds = url_decode_text(text)
        if rounds > 0:
            operations.append("url_decode")

    # ── 5. IDNA/Punycode 解码（审查 Fix 4）──
    if "xn--" in text.lower():
        text, idna_ok = _extract_and_normalize_domains(text)
        if idna_ok:
            operations.append("idna_decode")

    # ── 6. HTML DOM 解析 ──（不在此处执行，由调用方决定）
    # 对于 'html' 类型，规范化后的结果将传入 rules/html_rules.py 做 DOM 分析.
    # 原始 HTML 保留在 meta["raw_text"] 中供 DOM 解析（审查 Fix 5）.
    # 如果 OCR 文本存在，将其附加到结果中供后续分析.
    if ocr_text:
        ocr, changed = decode_html_entities(ocr_text)
        if changed:
            operations.append("ocr_html_entity_decode")

        ocr, nfkc_ops = normalize_unicode(ocr)
        if nfkc_ops:
            operations.append("ocr_unicode_nfkc")

        ocr, changed = remove_zero_width(ocr)
        if changed:
            operations.append("ocr_zero_width_removed")

        if "%" in ocr:
            ocr, rounds = url_decode_text(ocr)
            if rounds > 0:
                operations.append("ocr_url_decode")

        if "xn--" in ocr.lower():
            ocr, idna_ok = _extract_and_normalize_domains(ocr)
            if idna_ok:
                operations.append("ocr_idna_decode")

        text = text + "\n" + ocr

    meta = {
        "operations": operations,
        "input_bytes": input_bytes,
        "input_truncated": total_bytes > MAX_INPUT_BYTES,
        "raw_text": input_text,  # 保留原始文本用于 DOM 解析等（审查 Fix 5）
    }

    return text, meta
