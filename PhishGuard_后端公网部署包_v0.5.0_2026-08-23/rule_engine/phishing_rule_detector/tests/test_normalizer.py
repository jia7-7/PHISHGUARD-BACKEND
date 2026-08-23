"""测试输入规范化与反混淆管线."""
import pytest
from phishing_rule_detector.normalizer import (
    normalize,
    decode_html_entities,
    normalize_unicode,
    remove_zero_width,
    url_decode_text,
    normalize_domain,
    parse_html_dom,
)


# ── HTML Entity Decode ──────────────────────────────────────

class TestHtmlEntityDecode:
    def test_named_entity(self):
        result, changed = decode_html_entities("&colon;")
        assert ":" in result
        assert changed is True

    def test_decimal_entity(self):
        result, changed = decode_html_entities("&#97;")
        assert result == "a"
        assert changed is True

    def test_hex_entity(self):
        result, changed = decode_html_entities("&#x61;")
        assert result == "a"
        assert changed is True

    def test_no_entity_unchanged(self):
        result, changed = decode_html_entities("hello")
        assert result == "hello"
        assert changed is False

    def test_mixed_entities(self):
        result, changed = decode_html_entities("&#x6C;&#111;gin")
        assert result == "login"
        assert changed is True


# ── Unicode NFKC ────────────────────────────────────────────

class TestUnicodeNormalize:
    def test_nfkc_fullwidth(self):
        """全角字符转为半角."""
        result, ops = normalize_unicode("Ｈｅｌｌｏ")
        assert result == "Hello"

    def test_nfkc_no_change(self):
        result, ops = normalize_unicode("hello")
        assert result == "hello"

    def test_ops_recorded(self):
        _, ops = normalize_unicode("Ｈｅｌｌｏ")
        assert "unicode_nfkc" in ops


# ── Zero-Width Removal ──────────────────────────────────────

class TestZeroWidthRemoval:
    def test_zero_width_space_removed(self):
        result, changed = remove_zero_width("hel​lo")
        assert result == "hello"
        assert changed is True

    def test_multiple_zero_width(self):
        result, changed = remove_zero_width(
            "​x‌‍⁠﻿y"
        )
        assert result == "xy"
        assert changed is True

    def test_no_zero_width(self):
        result, changed = remove_zero_width("hello")
        assert result == "hello"
        assert changed is False


# ── URL Decode ──────────────────────────────────────────────

class TestUrlDecode:
    def test_single_decode(self):
        result, rounds = url_decode_text("hello%20world")
        assert result == "hello world"
        assert rounds == 1

    def test_double_decode(self):
        """%2530 → %30 → 0（两轮解码）."""
        result, rounds = url_decode_text("%2530")
        assert result == "0"
        assert rounds == 2

    def test_no_decode_needed(self):
        result, rounds = url_decode_text("hello")
        assert result == "hello"
        assert rounds == 0

    def test_plus_not_decoded(self):
        """+ 不应被解码为空格（区别于 unquote_plus）."""
        result, rounds = url_decode_text("a+b")
        # 使用 unquote，+ 保持原样
        assert "+" in result  # +
        assert " " not in result

    def test_max_two_rounds(self):
        """三轮编码不应无限解码."""
        # %25253x → %253x → %3x（两轮后停止）
        result, rounds = url_decode_text("%2525%2530")
        # 两轮后仍保留残留 %25 或部分解码结果
        assert rounds <= 2

    def test_partial_percent_preserved(self):
        """不完整的 % 编码不被破坏."""
        result, _ = url_decode_text("100%")
        # 保留原始值
        assert "%" in result


# ── Domain Normalization ────────────────────────────────────

class TestDomainNormalize:
    def test_lowercase(self):
        result = normalize_domain("EXAMPLE.COM")
        assert result == "example.com"

    def test_trailing_dot_removed(self):
        result = normalize_domain("example.com.")
        assert result == "example.com"

    def test_combined(self):
        result = normalize_domain("EXAMPLE.COM.")
        assert result == "example.com"


# ── HTML DOM Parsing ────────────────────────────────────────

class TestHtmlDomParse:
    def test_parse_simple_html(self):
        soup, meta = parse_html_dom("<html><body><p>hello</p></body></html>")
        assert soup is not None
        assert meta.get("error") is None
        assert soup.find("p").text == "hello"

    def test_parse_malformed_html(self):
        """畸形 HTML 应能容错解析."""
        soup, meta = parse_html_dom("<p>hello<br>world</div>")
        assert soup is not None
        assert meta.get("error") is None

    def test_non_html_text(self):
        soup, error = parse_html_dom("just plain text")
        assert soup is not None  # BeautifulSoup 会包装为文档

    def test_empty_html(self):
        soup, error = parse_html_dom("")
        assert soup is not None


# ── Full Normalization Pipeline ─────────────────────────────

class TestNormalizePipeline:
    def test_normalize_simple_text(self):
        result, meta = normalize("hello world", "text")
        assert result == "hello world"
        assert "input_bytes" in meta
        assert "operations" in meta

    def test_normalize_records_operations(self):
        _, meta = normalize("hello&#32;world", "html")
        assert "html_entity_decode" in meta["operations"]

    def test_normalize_zero_width_url(self):
        """零宽字符在 URL 中应被清除."""
        result, meta = normalize("https://sdu​.edu.cn", "url")
        assert "​" not in result
        assert "zero_width_removed" in meta["operations"]

    def test_normalize_url_with_entities(self):
        """URL 中 HTML 实体 → 解码 → 零宽清除."""
        result, meta = normalize(
            "https://&#115;du&#46;edu&#46;cn", "url"
        )
        assert "operations" in meta

    def test_input_bytes_recorded(self):
        _, meta = normalize("hello", "text")
        assert meta["input_bytes"] == 5

    def test_input_truncated_false(self):
        _, meta = normalize("hello", "text")
        assert meta["input_truncated"] is False

    def test_normalize_refuses_payload_too_large(self):
        """超过 200 KiB 应返回错误."""
        with pytest.raises(ValueError):
            normalize("x" * 204801, "text")

    def test_normalize_empty_input(self):
        with pytest.raises(ValueError):
            normalize("", "text")

    def test_normalize_nfkc_before_zero_width(self):
        """NFKC 可能产生零宽字符，必须先 NFKC 后清除零宽."""
        # 验证处理顺序正确：NFKC 先执行
        text = "hello&#x200B;world"  # 零宽字符通过实体编码
        result, meta = normalize(text, "html")
        # 实体先解码 → NFKC → 清除零宽
        assert "​" not in result

    def test_ocr_uses_full_anti_evasion_pipeline(self):
        result, meta = normalize(
            "image content",
            "text",
            ocr_text="https://ｓｄｕ-login.example.com/%6c%6f%67%69%6e",
        )

        assert "https://sdu-login.example.com/login" in result
        assert "ocr_unicode_nfkc" in meta["operations"]
        assert "ocr_url_decode" in meta["operations"]


# ── Punycode / IDNA ─────────────────────────────────────────

class TestIdnaHandling:
    def test_punycode_domain_decoded(self):
        """xn-- 前缀的 Punycode 域名应被解码为 Unicode."""
        # xn--fiq228c.com → 中文.com (Punycode for 中文)
        result, meta = normalize(
            "https://xn--fiq228c.com/path", "url"
        )
        assert "xn--fiq228c.com" not in result
        assert "中文.com" in result
        assert "idna_decode" in meta["operations"]

    def test_ascii_domain_unchanged(self):
        result, _ = normalize("https://example.com", "url")
        assert "example.com" in result

    def test_idna_decode_uses_urlsplit(self):
        """IDNA 解码应使用 urlsplit 提取 hostname，而非正则（审查 Fix 4）."""
        # xn--fiq228c.com → 中文.com (Punycode for 中文)
        result, meta = normalize(
            "https://xn--fiq228c.com/path?q=1", "url"
        )
        # hostname 部分被解码
        assert "xn--fiq228c.com" not in result

    def test_idna_failure_not_recorded(self):
        """IDNA 解码失败时不应记录 idna_decode 操作（审查 Fix 4）."""
        # 无效的 Punycode
        result, meta = normalize(
            "https://xn--invalid-punycode---.example.com/path", "url"
        )
        # 解码失败不应把 idna_decode 加到 operations
        assert "idna_decode" not in meta["operations"]

    def test_idna_success_recorded(self):
        """IDNA 解码成功时应记录 idna_decode（审查 Fix 4）."""
        result, meta = normalize(
            "https://xn--fiq228c.com/path", "url"
        )
        assert "idna_decode" in meta["operations"]

    def test_simple_punycode_domain_decoded_to_unicode(self):
        """简单 punycode 域名应被解码为实际 Unicode 输出."""
        result, meta = normalize("https://xn--fiq228c.com/path", "url")
        assert "xn--fiq228c.com" not in result
        # xn--fiq228c → 中文 (U+4E2D U+6587)
        assert "中文.com" in result
        assert "idna_decode" in meta["operations"]

    def test_punycode_url_with_username(self):
        """带 username 的 punycode URL: https://user@xn--fiq228c.com/path."""
        result, meta = normalize("https://user@xn--fiq228c.com/path", "url")
        assert "xn--fiq228c.com" not in result
        assert "user@中文.com" in result
        assert "/path" in result

    def test_punycode_url_with_username_password(self):
        """带 username:password 的 punycode URL."""
        result, meta = normalize("https://user:pass@xn--fiq228c.com/path", "url")
        assert "xn--fiq228c.com" not in result
        assert "user:pass@中文.com" in result

    def test_punycode_url_with_nonstandard_port(self):
        """带非标准端口的 punycode URL: https://xn--fiq228c.com:8443/path."""
        result, meta = normalize("https://xn--fiq228c.com:8443/path", "url")
        assert "xn--fiq228c.com" not in result
        assert "中文.com:8443" in result
        assert "/path" in result

    def test_ipv6_url_unchanged(self):
        """IPv6 URL 不应被错误修改."""
        result, meta = normalize("https://[::1]:8080/path", "url")
        assert "[::1]:8080" in result
        assert "/path" in result

    def test_url_with_query_only_no_slash(self):
        """无显式 /、只有 query 的 URL."""
        result, meta = normalize("https://xn--fiq228c.com?q=test", "url")
        assert "xn--fiq228c.com" not in result
        assert "中文.com" in result
        assert "?q=test" in result

    def test_invalid_punycode_keeps_original_url(self):
        """非法 punycode 域名应保留原 URL."""
        url = "https://xn--invalid-punycode---.example.com/path"
        result, meta = normalize(url, "url")
        # 域名应保持原样（解码失败）
        assert "xn--invalid-punycode---.example.com" in result
        assert "idna_decode" not in meta["operations"]

    def test_multiple_urls_in_text(self):
        """多个 URL 出现在同一段文本中，各自独立解码."""
        text = "Visit https://xn--fiq228c.com/page or https://example.com/other"
        result, meta = normalize(text, "text")
        assert "中文.com/page" in result
        assert "example.com/other" in result

    def test_plain_text_with_xn_not_in_url_not_modified(self):
        """非 URL 中的 xn-- 前缀文本不应被修改."""
        text = "The prefix xn--test is not a URL"
        result, meta = normalize(text, "text")
        assert "xn--test" in result

    # ── Fix 1: uppercase punycode ──

    def test_uppercase_punycode_decoded(self):
        """大写 Punycode (XN--FIQ228C.COM) 应被正确解码（审查 Fix 1）."""
        result, meta = normalize("https://XN--FIQ228C.COM/path", "url")
        assert "XN--FIQ228C.COM" not in result
        assert "xn--fiq228c.com" not in result
        assert "中文.com" in result
        assert "idna_decode" in meta["operations"]

    def test_uppercase_punycode_with_port(self):
        """大写 Punycode 带端口: https://XN--FIQ228C.COM:8443/path."""
        result, meta = normalize("https://XN--FIQ228C.COM:8443/path", "url")
        assert "XN--FIQ228C.COM" not in result
        assert "中文.com:8443" in result

    def test_mixed_case_punycode_decoded(self):
        """混合大小写 Punycode 应被正确解码."""
        result, meta = normalize("https://Xn--FiQ228c.CoM/path", "url")
        assert "中文.com" in result
        assert "idna_decode" in meta["operations"]

    # ── Fix 2: trailing punctuation ──

    @pytest.mark.parametrize(
        "url,expected_host",
        [
            ("https://xn--fiq228c.com,", "中文.com"),
            ("https://xn--fiq228c.com).", "中文.com"),
            ("https://xn--fiq228c.com;", "中文.com"),
            ("https://xn--fiq228c.com!", "中文.com"),
            ("https://xn--fiq228c.com?", "中文.com"),
            ("https://xn--fiq228c.com。", "中文.com"),
            ("https://xn--fiq228c.com，", "中文.com"),
            ("https://xn--fiq228c.com；", "中文.com"),
            ("https://xn--fiq228c.com！", "中文.com"),
            ("https://xn--fiq228c.com？", "中文.com"),
            ("https://xn--fiq228c.com）", "中文.com"),
            ("https://xn--fiq228c.com】", "中文.com"),
        ],
    )
    def test_trailing_punctuation_stripped(self, url, expected_host):
        """URL 后的标点符号不应被吸入 hostname（审查 Fix 2）."""
        result, meta = normalize(url, "text")
        assert expected_host in result
        assert "idna_decode" in meta["operations"]

    def test_bracketed_url_preserves_parens(self):
        """括号包裹的 URL: (https://xn--fiq228c.com) 的 () 应保留在文本中（审查 Fix 2）."""
        result, meta = normalize("(https://xn--fiq228c.com)", "text")
        assert "中文.com" in result
        assert "(" in result
        assert ")" in result
        assert "idna_decode" in meta["operations"]

    def test_bracketed_url_with_path(self):
        """括号包裹带路径 URL: (https://xn--fiq228c.com/path) 的 () 应保留."""
        result, meta = normalize("(https://xn--fiq228c.com/path)", "text")
        assert "中文.com" in result
        assert "(" in result
        assert ")" in result

    def test_url_with_path_comma_not_stripped_from_path(self):
        """URL 路径中的逗号不应被去除（仅在 hostname 尾部处理）."""
        # 路径中的逗号是正常的 URL 路径字符
        result, meta = normalize(
            "https://xn--fiq228c.com/page,view", "text"
        )
        assert "中文.com" in result
        # 路径中的逗号保留（不在 hostname 尾部）
        assert "page,view" in result

    def test_comma_after_url_preserved_in_text(self):
        """URL 后的逗号应保留在原文本中（审查 Fix 2）."""
        result, meta = normalize("Visit https://xn--fiq228c.com, thanks", "text")
        assert result == "Visit https://中文.com, thanks"

    def test_ascii_colon_after_url_is_preserved(self):
        """无端口数字的尾随冒号是正文标点，规范化后必须原位保留."""
        result, meta = normalize("Visit https://xn--fiq228c.com: now", "text")
        assert result == "Visit https://中文.com: now"
        assert "idna_decode" in meta["operations"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://xn--fiq228c.com:bad/path",
            "https://xn--fiq228c.com:99999/path",
        ],
    )
    def test_invalid_port_keeps_original_url(self, url):
        """畸形端口不得逃逸规范化管线；应保留原 URL 并降级."""
        result, meta = normalize(url, "url")
        assert result == url
        assert "idna_decode" not in meta["operations"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/path/xn--not-a-host",
            "https://xn--label@example.com/path",
        ],
    )
    def test_idna_operation_requires_hostname_change(self, url):
        """xn-- 仅出现在路径或 userinfo 时不得报告 IDNA 域名解码."""
        result, meta = normalize(url, "url")
        assert result == url
        assert "idna_decode" not in meta["operations"]

    def test_empty_password_delimiter_is_preserved(self):
        """结构化重建 netloc 时应完整保留空密码分隔符."""
        result, meta = normalize(
            "https://user:@xn--fiq228c.com/path", "url"
        )
        assert result == "https://user:@中文.com/path"
        assert "idna_decode" in meta["operations"]


# ── Raw HTML Preservation ────────────────────────────────────

class TestRawHtmlPreservation:
    def test_normalize_keeps_raw_html_for_dom(self):
        """规范化后的文本应保留足够信息用于 DOM 解析（审查 Fix 5）."""
        html = "<html><body><p>hello&#32;world</p></body></html>"
        result, meta = normalize(html, "html")
        # 实体解码后的文本可被 BS4 解析
        soup, _ = parse_html_dom(result)
        assert soup is not None

    def test_html_normalize_preserves_structure(self):
        """HTML 规范化后应保留标签结构（审查 Fix 5）."""
        html = "<form action='https://evil.com/login'><input type='password'></form>"
        result, meta = normalize(html, "html")
        assert "<form" in result.lower() or "form" in result.lower()
        assert "password" in result.lower()

    def test_raw_text_preserves_entity_encoded_html(self):
        """raw_text 必须保持原始转义形式，不能解码 HTML 实体（审查 Fix 6）."""
        html = "&lt;input type=&quot;password&quot;&gt;"
        result, meta = normalize(html, "html")
        # normalized_text 可用于文本规则（已解码）
        assert "<input" in result or "input" in result.lower()
        # meta["raw_text"] 必须保持原始转义形式
        assert meta["raw_text"] == html
        assert "&lt;input" in meta["raw_text"]
        assert "&quot;password&quot;" in meta["raw_text"]

    def test_raw_text_dom_parse_no_input_node(self):
        """从 raw_text 解析时不应出现 input DOM 节点（审查 Fix 6）.

        Phase 2 HTML 规则必须解析 raw_text，而非 normalized_text。
        """
        html = "&lt;input type=&quot;password&quot;&gt;"
        _, meta = normalize(html, "html")
        raw = meta["raw_text"]
        # 从 raw_text 解析：由于实体未被解码，BS4 不会看到 <input> 标签
        soup, _ = parse_html_dom(raw)
        input_nodes = soup.find_all("input")
        assert len(input_nodes) == 0, (
            "raw_text 中的实体未被解码，不应出现 DOM 节点"
        )
