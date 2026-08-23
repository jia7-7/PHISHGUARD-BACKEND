"""测试 URL 与传输规则."""
from phishing_rule_detector.rules.common import (
    RuleContext,
    extract_urls_from_text,
)
from phishing_rule_detector.rules.url_rules import (
    url_ip_domain,
    url_short_link,
    url_non_https,
    url_suspicious_port,
    url_long_subdomain,
    url_at_symbol,
    url_nested_official_domain,
    _count_subdomain_levels,
)


def _make_ctx(
    text: str,
    base_url: str | None = None,
    input_type: str = "url",
    normalized: str | None = None,
    sender: str = "",
    qr_urls: list[str] | None = None,
) -> RuleContext:
    """快捷构造 RuleContext."""
    urls = extract_urls_from_text(text, base_url)
    if qr_urls:
        for q in qr_urls:
            urls.extend(extract_urls_from_text(q, source="qr"))
    return RuleContext(
        input_text=text,
        normalized_text=normalized or text,
        input_type=input_type,
        base_url=base_url,
        raw_text=text,
        sender=sender,
        extracted_urls=urls,
        qr_urls=qr_urls or [],
    )


# ── URL_IP_DOMAIN ──

class TestUrlIpDomain:
    def test_ip_auth_url_hits(self):
        ctx = _make_ctx("http://192.168.1.1/login")
        result = url_ip_domain(ctx)
        # 私网地址不命中
        assert result is None

    def test_public_ip_auth_hits(self):
        ctx = _make_ctx("http://93.184.216.34/login")
        result = url_ip_domain(ctx)
        assert result is not None
        assert len(result) == 1
        assert result[0].rule_id == "URL_IP_DOMAIN"

    def test_public_ip_no_auth_no_hit(self):
        ctx = _make_ctx("http://93.184.216.34/page")
        result = url_ip_domain(ctx)
        # 无认证上下文不命中
        assert result is None

    def test_loopback_not_hit(self):
        ctx = _make_ctx("http://127.0.0.1/login")
        result = url_ip_domain(ctx)
        assert result is None

    def test_private_not_hit(self):
        ctx = _make_ctx("http://10.0.0.1/login")
        result = url_ip_domain(ctx)
        assert result is None


# ── URL_SHORT_LINK ──

class TestUrlShortLink:
    def test_short_link_hits(self):
        ctx = _make_ctx("https://bit.ly/abc123")
        result = url_short_link(ctx)
        assert result is not None
        assert result[0].rule_id == "URL_SHORT_LINK"

    def test_normal_domain_no_hit(self):
        ctx = _make_ctx("https://example.com/page")
        result = url_short_link(ctx)
        assert result is None

    def test_short_link_subdomain_hits(self):
        ctx = _make_ctx("https://goo.suo.im/test")
        result = url_short_link(ctx)
        assert result is not None


# ── URL_NON_HTTPS ──

class TestUrlNonHttps:
    def test_http_with_auth_hits(self):
        ctx = _make_ctx("http://example.com/login")
        result = url_non_https(ctx)
        assert result is not None
        assert any(e.rule_id == "URL_NON_HTTPS" for e in result)

    def test_https_no_hit(self):
        ctx = _make_ctx("https://example.com/login")
        result = url_non_https(ctx)
        # https 不命中
        if result:
            assert all(e.rule_id != "URL_NON_HTTPS" for e in result)

    def test_plain_http_low_confidence(self):
        ctx = _make_ctx("http://example.com/news")
        result = url_non_https(ctx)
        assert result is not None
        e = [r for r in result if r.rule_id == "URL_NON_HTTPS"][0]
        assert e.confidence <= 0.35  # 普通 HTTP 低置信度


# ── URL_SUSPICIOUS_PORT ──

class TestUrlSuspiciousPort:
    def test_nonstandard_port_hits(self):
        ctx = _make_ctx("http://example.com:8081/path")
        result = url_suspicious_port(ctx)
        assert result is not None
        assert result[0].rule_id == "URL_SUSPICIOUS_PORT"

    def test_standard_port_80_no_hit(self):
        ctx = _make_ctx("http://example.com:80/path")
        result = url_suspicious_port(ctx)
        assert result is None

    def test_standard_port_443_no_hit(self):
        ctx = _make_ctx("https://example.com:443/path")
        result = url_suspicious_port(ctx)
        assert result is None

    def test_allowed_port_no_hit(self):
        ctx = _make_ctx("https://qdxq.sdu.edu.cn:8443/path")
        result = url_suspicious_port(ctx)
        assert result is None

    def test_no_port_no_hit(self):
        ctx = _make_ctx("https://example.com/path")
        result = url_suspicious_port(ctx)
        assert result is None


# ── URL_LONG_SUBDOMAIN ──

class TestUrlLongSubdomain:
    def test_long_subdomain_hits(self):
        ctx = _make_ctx("https://a.b.c.d.e.f.example.com/login")
        result = url_long_subdomain(ctx)
        assert result is not None
        assert result[0].rule_id == "URL_LONG_SUBDOMAIN"

    def test_normal_subdomain_no_hit(self):
        ctx = _make_ctx("https://mail.example.com/login")
        result = url_long_subdomain(ctx)
        assert result is None

    def test_subdomain_level_count(self):
        # a.b.c.sdu.edu.cn: 注册域=sdu.edu.cn, 前缀=a.b.c (3 labels) → 3 级
        assert _count_subdomain_levels("a.b.c.sdu.edu.cn") == 3
        # a.b.c.d.sdu.edu.cn → 4 级
        assert _count_subdomain_levels("a.b.c.d.sdu.edu.cn") == 4


# ── URL_AT_SYMBOL ──

class TestUrlAtSymbol:
    def test_at_symbol_spoof_hits(self):
        ctx = _make_ctx("https://sdu.edu.cn@evil.com/login")
        result = url_at_symbol(ctx)
        assert result is not None
        assert result[0].rule_id == "URL_AT_SYMBOL"

    def test_normal_userinfo_no_spoof(self):
        ctx = _make_ctx("https://user:pass@example.com/path")
        result = url_at_symbol(ctx)
        # 不含官方品牌特征不命中
        assert result is None

    def test_no_at_no_hit(self):
        ctx = _make_ctx("https://example.com/path")
        result = url_at_symbol(ctx)
        assert result is None


# ── URL_NESTED_OFFICIAL_DOMAIN ──

class TestUrlNestedOfficialDomain:
    def test_nested_in_subdomain_hits(self):
        ctx = _make_ctx("https://sdu.edu.cn.evil.com/login")
        result = url_nested_official_domain(ctx)
        assert result is not None
        assert result[0].rule_id == "URL_NESTED_OFFICIAL_DOMAIN"

    def test_nested_in_path_hits(self):
        ctx = _make_ctx("https://evil.com/path/sdu.edu.cn/login")
        result = url_nested_official_domain(ctx)
        assert result is not None
        assert any(e.rule_id == "URL_NESTED_OFFICIAL_DOMAIN" for e in result)

    def test_official_subdomain_no_hit(self):
        ctx = _make_ctx("https://pass.sdu.edu.cn/login")
        result = url_nested_official_domain(ctx)
        if result:
            assert all(e.rule_id != "URL_NESTED_OFFICIAL_DOMAIN" for e in result)
