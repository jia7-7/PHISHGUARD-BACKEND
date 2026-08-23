"""测试官方域名判断、可信第三方策略和 tldextract 离线模式."""
from phishing_rule_detector.trust_policy import (
    is_official_domain,
    is_official_subdomain,
    match_trusted_service,
    check_field_allowed,
    check_field_forbidden,
    is_short_link_domain,
    get_registered_domain,
    _get_tldextract,
)


class TestOfficialDomain:
    # ── 精确匹配 ──
    def test_exact_match(self):
        assert is_official_domain("sdu.edu.cn") is True

    # ── 合法子域名（用户确认的全部场景）──
    def test_subdomain_qdxq(self):
        assert is_official_domain("qdxq.sdu.edu.cn") is True

    def test_subdomain_wh(self):
        assert is_official_domain("wh.sdu.edu.cn") is True

    def test_subdomain_ehall(self):
        assert is_official_domain("ehall.sdu.edu.cn") is True

    def test_subdomain_pass(self):
        assert is_official_domain("pass.sdu.edu.cn") is True

    def test_subdomain_mail(self):
        assert is_official_domain("mail.sdu.edu.cn") is True

    def test_deep_subdomain(self):
        assert is_official_domain("a.b.c.sdu.edu.cn") is True

    # ── 必须拒绝的外部域名 ──
    def test_nested_external_rejected(self):
        """sdu.edu.cn.evil.com 的注册域为 evil.com."""
        assert is_official_domain("sdu.edu.cn.evil.com") is False

    def test_partial_suffix_rejected(self):
        """fakesdu.edu.cn 不以 .sdu.edu.cn 结尾."""
        assert is_official_domain("fakesdu.edu.cn") is False

    def test_dash_variant_rejected(self):
        assert is_official_domain("sdu-edu.cn") is False

    def test_ends_with_but_not_label_boundary(self):
        """xsdu.edu.cn 既不是 sdu.edu.cn 也不以 .sdu.edu.cn 结尾."""
        assert is_official_domain("xsdu.edu.cn") is False

    # ── 边界处理 ──
    def test_trailing_dot_handled(self):
        assert is_official_domain("sdu.edu.cn.") is True

    def test_case_insensitive(self):
        assert is_official_domain("PASS.SDU.EDU.CN") is True


class TestOfficialSubdomain:
    def test_is_subdomain(self):
        assert is_official_subdomain("pass.sdu.edu.cn") is True

    def test_root_is_not_subdomain(self):
        """根域名本身不是子域名."""
        assert is_official_subdomain("sdu.edu.cn") is False

    def test_external_is_not(self):
        assert is_official_subdomain("evil.com") is False


class TestTrustedService:
    def test_match_wjx(self):
        svc = match_trusted_service("www.wjx.cn")
        assert svc is not None
        assert svc.service == "survey"

    def test_match_wjx_com(self):
        svc = match_trusted_service("wjx.com")
        assert svc is not None

    def test_non_trusted_domain_returns_none(self):
        assert match_trusted_service("evil.com") is None

    def test_not_trusted_sdu(self):
        """sdu.edu.cn 不是可信第三方."""
        assert match_trusted_service("sdu.edu.cn") is None


class TestFieldChecks:
    def test_allowed_field_student_id(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_allowed(svc, "student_id") is True

    def test_allowed_field_phone(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_allowed(svc, "phone") is True

    def test_allowed_field_email(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_allowed(svc, "email") is True

    def test_not_allowed_field_password(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_allowed(svc, "password") is False

    def test_not_allowed_field_name(self):
        """name 不在 allowed_fields 中."""
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_allowed(svc, "name") is False

    # ── Forbidden ──
    def test_forbidden_field_password(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_forbidden(svc, "password") is True

    def test_forbidden_field_sms_code(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_forbidden(svc, "sms_code") is True

    def test_forbidden_field_bank_card(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_forbidden(svc, "bank_card") is True

    def test_forbidden_field_token(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_forbidden(svc, "token") is True

    def test_not_forbidden_field_email(self):
        svc = match_trusted_service("www.wjx.cn")
        assert check_field_forbidden(svc, "email") is False

    def test_none_service_safe(self):
        """None 传入不应崩溃."""
        assert check_field_allowed(None, "student_id") is False
        assert check_field_forbidden(None, "password") is False


class TestShortLink:
    def test_short_link_detected(self):
        assert is_short_link_domain("bit.ly") is True
        assert is_short_link_domain("t.cn") is True
        assert is_short_link_domain("url.cn") is True
        assert is_short_link_domain("dwz.cn") is True
        assert is_short_link_domain("suo.im") is True
        assert is_short_link_domain("shorturl.at") is True

    def test_normal_domain_not_short_link(self):
        assert is_short_link_domain("sdu.edu.cn") is False
        assert is_short_link_domain("example.com") is False

    def test_subdomain_of_short_link(self):
        """子域名也视为短链接."""
        assert is_short_link_domain("goo.suo.im") is True


class TestTldextractOffline:
    def test_registered_domain_sdu(self):
        ext = _get_tldextract()
        result = ext("pass.sdu.edu.cn")
        assert result.registered_domain == "sdu.edu.cn"

    def test_registered_domain_qdxq(self):
        ext = _get_tldextract()
        result = ext("qdxq.sdu.edu.cn")
        assert result.registered_domain == "sdu.edu.cn"

    def test_registered_domain_nested(self):
        """sdu.edu.cn.evil.com 的注册域是 evil.com."""
        ext = _get_tldextract()
        result = ext("sdu.edu.cn.evil.com")
        assert result.registered_domain == "evil.com"

    def test_registered_domain_simple(self):
        ext = _get_tldextract()
        result = ext("evil.com")
        assert result.registered_domain == "evil.com"

    def test_no_http_requests(self, monkeypatch):
        """确认 tldextract 不会发起 HTTP 请求（审查 Fix 5）.

        拦截 urllib 和 requests 两个入口，确保离线模式下零网络访问.
        """
        import urllib.request

        urllib_calls = []
        requests_calls = []

        def fake_urlopen(*args, **kwargs):
            urllib_calls.append(True)
            raise RuntimeError("不应该发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        # 也尝试拦截 requests 库（如果已安装）
        try:
            import requests
            def fake_get(*args, **kwargs):
                requests_calls.append(True)
                raise RuntimeError("不应该发起网络请求")
            monkeypatch.setattr(requests, "get", fake_get)
            monkeypatch.setattr(requests, "head", fake_get)
        except ImportError:
            pass

        # 重置单例以确保使用 suffix_list_urls=()
        import phishing_rule_detector.trust_policy as tp
        monkeypatch.setattr(tp, "_tld_extract", None)

        ext = _get_tldextract()
        # 验证配置为离线模式
        assert ext.suffix_list_urls == ()

        result = ext("qdxq.sdu.edu.cn")
        assert result.registered_domain == "sdu.edu.cn"
        assert len(urllib_calls) == 0
        assert len(requests_calls) == 0

    def test_get_registered_domain(self):
        assert get_registered_domain("pass.sdu.edu.cn") == "sdu.edu.cn"
        assert get_registered_domain("sdu.edu.cn.evil.com") == "evil.com"
