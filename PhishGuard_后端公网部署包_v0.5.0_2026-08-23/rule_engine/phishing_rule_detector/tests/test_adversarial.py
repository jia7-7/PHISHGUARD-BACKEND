"""对抗性测试：验证规则对常见绕过手法的鲁棒性."""
from __future__ import annotations


from phishing_rule_detector.detector import detect


# ── 空格/零宽字符绕过 ──

class TestWhitespaceBypass:
    def test_spaces_in_keyword_still_matched(self):
        """关键字插空格仍应匹配."""
        result = detect(
            input_text="请 立 即 验 证 您 的 账 号",
            input_type="email",
            context={"sender": "fake@evil.com"},
        )
        evidence_ids = [e["rule_id"] for e in result["evidence"]]
        assert "URGENT_VERIFY_WORDING" in evidence_ids

    def test_zero_width_in_keyword_still_matched(self):
        """零宽字符在关键字中仍应匹配."""
        result = detect(
            input_text="请立​即验证您的账号",
            input_type="email",
            context={"sender": "fake@evil.com"},
        )
        evidence_ids = [e["rule_id"] for e in result["evidence"]]
        assert "URGENT_VERIFY_WORDING" in evidence_ids

    def test_newlines_in_text_still_processed(self):
        """文本包含换行不影响检测."""
        result = detect(
            input_text="您好\n紧急通知\n请立即验证账号\nhttps://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        )
        assert result["success"]


# ── Unicode 同形异义 ──

class TestHomographBypass:
    def test_cyrillic_a_in_domain(self):
        """域名中使用西里尔字母 а 代替拉丁 a."""
        result = detect(
            input_text="https://sdu-verify.com/login",
            input_type="url",
        )
        assert result["success"]

    def test_greek_omicron_in_domain(self):
        """域名混用希腊字母."""
        result = detect(
            input_text="https://sdυ-edu.com/login",
            input_type="url",
        )
        assert result["success"]


# ── URL 编码绕过 ──

class TestUrlEncodingBypass:
    def test_percent_encoded_path(self):
        """URL 路径百分号编码不应绕过检测."""
        result = detect(
            input_text="https://evil.com/%6C%6F%67%69%6E",
            input_type="url",
        )
        assert result["success"]

    def test_double_encoded_url(self):
        """双重 URL 编码."""
        result = detect(
            input_text="https://evil.com/%256c%256f%2567%2569%256e",
            input_type="url",
        )
        assert result["success"]


# ── HTML 绕过 ──

class TestHtmlBypass:
    def test_entity_encoded_password_not_parsed(self):
        """HTML 实体编码的 password 输入不应被解析为真实 DOM."""
        result = detect(
            input_text="&lt;input type=&quot;password&quot;&gt;",
            input_type="html",
        )
        assert result["success"]

    def test_css_hidden_form(self):
        """CSS 方式隐藏表单."""
        result = detect(
            input_text='<html><body><form style="position:absolute;left:-9999px"><input type="password" name="pwd"></form></body></html>',
            input_type="html",
        )
        assert result["success"]

    def test_visibility_hidden_form(self):
        """visibility:hidden 隐藏表单."""
        result = detect(
            input_text='<html><body><form style="visibility:hidden"><input type="password" name="pwd"></form></body></html>',
            input_type="html",
        )
        assert result["success"]

    def test_opacity_zero_form(self):
        """opacity:0 隐藏表单."""
        result = detect(
            input_text='<html><body><form style="opacity:0"><input type="password" name="pwd"></form></body></html>',
            input_type="html",
        )
        assert result["success"]


# ── URL 超长子域名 ──

class TestSubdomainBypass:
    def test_excessive_subdomain_levels(self):
        """超多级子域名."""
        result = detect(
            input_text="https://a.b.c.d.e.f.g.h.i.j.k.evil.com/login",
            input_type="url",
        )
        assert result["success"]


# ── HTTP vs HTTPS ──

class TestProtocolBypass:
    def test_http_in_auth_context(self):
        """认证上下文中 HTTP 明文链接."""
        result = detect(
            input_text="请登录 http://evil.com/login 验证账号",
            input_type="text",
        )
        assert result["success"]

    def test_https_normal(self):
        """HTTPS 链接正常."""
        result = detect(
            input_text="https://sdu.edu.cn/login",
            input_type="url",
        )
        assert result["risk"]["level"] in ("safe", "low")


# ── 混合内容 ──

class TestMixedContent:
    def test_official_domain_in_path_only(self):
        """官方域名仅在外部域名的路径中出现."""
        result = detect(
            input_text="https://evil.com/sdu.edu.cn/login",
            input_type="url",
        )
        assert result["success"]

    def test_official_domain_as_parameter(self):
        """官方域名出现在查询参数中."""
        result = detect(
            input_text="https://evil.com/login?redirect=sdu.edu.cn",
            input_type="url",
        )
        assert result["success"]


# ── 发件人欺骗 ──

class TestSenderBypass:
    def test_display_name_spoofing(self):
        """显示名冒充官方但实际邮箱不同."""
        result = detect(
            input_text="您好，这里是山东大学信息技术中心，请验证您的统一认证账号",
            input_type="email",
            context={"sender": "山东大学信息技术中心 <fake@evil.com>"},
        )
        assert result["success"]
        rule_ids = [e["rule_id"] for e in result["evidence"]]
        assert "EMAIL_SENDER_IMPERSONATION" in rule_ids

    def test_no_sender_no_crash(self):
        """无发件人字段不应崩溃."""
        result = detect(
            input_text="请验证您的账号",
            input_type="email",
        )
        assert result["success"]

    def test_empty_sender_no_crash(self):
        """空发件人不崩溃."""
        result = detect(
            input_text="请验证您的账号",
            input_type="email",
            context={"sender": ""},
        )
        assert result["success"]
