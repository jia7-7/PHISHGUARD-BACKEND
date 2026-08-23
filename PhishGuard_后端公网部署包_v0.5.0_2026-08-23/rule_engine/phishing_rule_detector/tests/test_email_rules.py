"""测试邮件与身份规则."""
from phishing_rule_detector.rules.common import (
    RuleContext,
    extract_urls_from_text,
)
from phishing_rule_detector.rules.email_rules import (
    email_sender_impersonation,
    email_sender_link_mismatch,
    _extract_sender_domain,
)


def _email_ctx(text: str, sender: str = "", urls_text: str = "") -> RuleContext:
    urls = extract_urls_from_text(urls_text or text)
    return RuleContext(
        input_text=text,
        normalized_text=text,
        input_type="email",
        base_url=None,
        raw_text=text,
        sender=sender,
        extracted_urls=urls,
    )


class TestEmailSenderImpersonation:
    def test_external_sender_impersonates(self):
        r = email_sender_impersonation(_email_ctx(
            "您好，这里是山东大学信息技术中心，请点击链接验证",
            sender="fake@evil.com",
        ))
        assert r is not None
        assert r[0].rule_id == "EMAIL_SENDER_IMPERSONATION"

    def test_official_sender_no_hit(self):
        r = email_sender_impersonation(_email_ctx(
            "您好，这里是山东大学信息技术中心",
            sender="service@sdu.edu.cn",
        ))
        assert r is None

    def test_no_sender_no_hit(self):
        r = email_sender_impersonation(_email_ctx("hello"))
        assert r is None

    def test_extract_sender_domain(self):
        assert _extract_sender_domain("test@example.com") == "example.com"
        assert _extract_sender_domain("Test <test@sdu.edu.cn>") == "sdu.edu.cn"
        assert _extract_sender_domain("") == ""


class TestEmailSenderLinkMismatch:
    def test_mismatch_hits(self):
        r = email_sender_link_mismatch(_email_ctx(
            "请登录统一认证系统验证您的账号",
            sender="admin@sdu.edu.cn",
            urls_text="https://evil.com/login https://evil.com/verify",
        ))
        assert r is not None
        assert r[0].rule_id == "EMAIL_SENDER_LINK_MISMATCH"

    def test_no_links_no_hit(self):
        r = email_sender_link_mismatch(_email_ctx(
            "hello world",
            sender="admin@sdu.edu.cn",
        ))
        assert r is None
