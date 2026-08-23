"""测试诱导话术规则."""
from phishing_rule_detector.rules.common import RuleContext
from phishing_rule_detector.rules.text_rules import (
    urgent_verify_wording,
    time_limit_pressure,
    account_disable_threat,
    reward_or_refund_bait,
    security_upgrade_wording,
    credential_request_text,
)


def _text_ctx(text: str, sender: str = "") -> RuleContext:
    return RuleContext(
        input_text=text,
        normalized_text=text,
        input_type="text",
        base_url=None,
        raw_text=text,
        sender=sender,
    )


class TestUrgentVerify:
    def test_urgent_verify_hits(self):
        r = urgent_verify_wording(_text_ctx("请立即验证您的账号"))
        assert r is not None
        assert r[0].rule_id == "URGENT_VERIFY_WORDING"

    def test_no_match(self):
        r = urgent_verify_wording(_text_ctx("hello world"))
        assert r is None

    def test_official_sender_reduces_confidence(self):
        r = urgent_verify_wording(_text_ctx(
            "请立即验证您的账号",
            sender="service@sdu.edu.cn",
        ))
        assert r is not None
        assert r[0].context_factor == 0.2


class TestTimeLimit:
    def test_time_limit_hits(self):
        r = time_limit_pressure(_text_ctx("请在12小时内完成验证"))
        assert r is not None
        assert r[0].rule_id == "TIME_LIMIT_PRESSURE"


class TestAccountDisable:
    def test_account_disable_hits(self):
        r = account_disable_threat(_text_ctx("您的账号停用通知"))
        assert r is not None
        assert r[0].rule_id == "ACCOUNT_DISABLE_THREAT"


class TestRewardBait:
    def test_reward_bait_hits(self):
        r = reward_or_refund_bait(_text_ctx("请领取您的奖学金"))
        assert r is not None
        assert r[0].rule_id == "REWARD_OR_REFUND_BAIT"


class TestSecurityUpgrade:
    def test_security_upgrade_hits(self):
        r = security_upgrade_wording(_text_ctx("请进行安全升级"))
        assert r is not None
        assert r[0].rule_id == "SECURITY_UPGRADE_WORDING"
        assert r[0].severity == "low"


class TestCredentialRequest:
    def test_credential_request_hits(self):
        r = credential_request_text(_text_ctx("请回复密码进行验证"))
        assert r is not None
        assert r[0].rule_id == "CREDENTIAL_REQUEST_TEXT"
        assert r[0].severity == "high"


class TestFlexibleMatching:
    def test_spaces_in_keyword(self):
        """插空格形式 '立 即 验 证' 应匹配."""
        r = urgent_verify_wording(_text_ctx("请立 即 验 证您的账号"))
        assert r is not None

    def test_zero_width_in_keyword(self):
        """零宽字符 '立​即验证' 应匹配."""
        r = urgent_verify_wording(_text_ctx("请立​即验证您的账号"))
        assert r is not None
