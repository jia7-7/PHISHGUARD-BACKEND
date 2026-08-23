"""测试附件、图片与二维码规则."""
from phishing_rule_detector.rules.common import (
    RuleContext,
)
from phishing_rule_detector.normalizer import parse_html_dom
from phishing_rule_detector.rules.image_rules import (
    attachment_executable,
    attachment_double_extension,
    html_base64_image_heavy,
    html_image_only_content,
    qr_code_external_url,
    qr_credential_url,
)


def _ctx(**kwargs) -> RuleContext:
    defaults = {
        "input_text": "",
        "normalized_text": "",
        "input_type": "email",
        "base_url": None,
        "raw_text": "",
        "attachments": [],
        "qr_urls": [],
        "parsed_html": None,
    }
    defaults.update(kwargs)
    return RuleContext(**defaults)


# ── ATTACHMENT_EXECUTABLE ──

class TestAttachmentExecutable:
    def test_exe_detected(self):
        r = attachment_executable(_ctx(attachments=["invoice.exe"]))
        assert r is not None
        assert r[0].rule_id == "ATTACHMENT_EXECUTABLE"
        assert r[0].severity == "high"
        assert r[0].subject_id == "attachment:0"

    def test_bat_detected(self):
        r = attachment_executable(_ctx(attachments=["script.bat"]))
        assert r is not None
        assert r[0].rule_id == "ATTACHMENT_EXECUTABLE"

    def test_ps1_detected(self):
        r = attachment_executable(_ctx(attachments=["run.ps1"]))
        assert r is not None

    def test_safe_pdf_not_detected(self):
        r = attachment_executable(_ctx(attachments=["document.pdf"]))
        assert r is None

    def test_safe_docx_not_detected(self):
        r = attachment_executable(_ctx(attachments=["report.docx"]))
        assert r is None

    def test_no_attachments(self):
        r = attachment_executable(_ctx())
        assert r is None

    def test_multiple_attachments_mixed(self):
        r = attachment_executable(_ctx(attachments=["doc.pdf", "payload.exe", "notes.txt"]))
        assert r is not None
        assert len(r) == 1
        assert r[0].matched_content and "payload.exe" in r[0].matched_content


# ── ATTACHMENT_DOUBLE_EXTENSION ──

class TestAttachmentDoubleExtension:
    def test_double_exe_detected(self):
        r = attachment_double_extension(_ctx(attachments=["report.pdf.exe"]))
        assert r is not None
        assert r[0].rule_id == "ATTACHMENT_DOUBLE_EXTENSION"
        assert r[0].severity == "high"
        assert r[0].subject_id == "attachment:0"

    def test_double_scr_detected(self):
        r = attachment_double_extension(_ctx(attachments=["photo.jpg.scr"]))
        assert r is not None

    def test_single_safe_not_detected(self):
        r = attachment_double_extension(_ctx(attachments=["document.pdf"]))
        assert r is None

    def test_double_safe_not_detected(self):
        """两个安全后缀（如 .tar.gz）不触发."""
        r = attachment_double_extension(_ctx(attachments=["archive.tar.gz"]))
        assert r is None

    def test_no_attachments(self):
        r = attachment_double_extension(_ctx())
        assert r is None

    def test_triple_extension_detected(self):
        r = attachment_double_extension(_ctx(attachments=["evil.txt.pdf.exe"]))
        assert r is not None


# ── HTML_BASE64_IMAGE_HEAVY ──

class TestHtmlBase64ImageHeavy:
    def test_heavy_base64_triggers(self):
        html = (
            '<html><body>'
            + '<img src="data:image/png;base64,' + 'A' * 2500 + '" />'
            + '</body></html>'
        )
        r = html_base64_image_heavy(_ctx(normalized_text=html))
        assert r is not None
        assert r[0].rule_id == "HTML_BASE64_IMAGE_HEAVY"
        assert r[0].subject_id == "page:0"

    def test_light_base64_no_trigger(self):
        html = (
            '<html><body>'
            + '<img src="data:image/png;base64,' + 'A' * 50 + '" />'
            + '</body></html>'
        )
        r = html_base64_image_heavy(_ctx(normalized_text=html))
        assert r is None

    def test_no_base64(self):
        r = html_base64_image_heavy(_ctx(normalized_text="<html><body>hello</body></html>"))
        assert r is None

    def test_empty_text(self):
        r = html_base64_image_heavy(_ctx(normalized_text=""))
        assert r is None

    def test_heavy_with_visible_text_no_trigger(self):
        """大量base64但有足够可见文本时不触发."""
        visible = "hello world " * 20  # ~240 chars
        html = (
            '<html><body><p>' + visible + '</p>'
            + '<img src="data:image/png;base64,' + 'A' * 2500 + '" />'
            + '</body></html>'
        )
        r = html_base64_image_heavy(_ctx(normalized_text=html))
        # visible_text > 100 chars → 不触发
        assert r is None


# ── HTML_IMAGE_ONLY_CONTENT ──

class TestHtmlImageOnlyContent:
    def test_image_only_triggers(self):
        imgs = ''.join(f'<img src="img{i}.png" />' for i in range(6))
        html = f"<html><body>{imgs}</body></html>"
        soup, _ = parse_html_dom(html)
        r = html_image_only_content(_ctx(normalized_text=html, parsed_html=soup))
        assert r is not None
        assert r[0].rule_id == "HTML_IMAGE_ONLY_CONTENT"
        assert r[0].subject_id == "page:0"

    def test_enough_text_no_trigger(self):
        imgs = ''.join(f'<img src="img{i}.png" />' for i in range(6))
        text = "This is a normal page with readable content. " * 10
        html = f"<html><body>{imgs}<p>{text}</p></body></html>"
        soup, _ = parse_html_dom(html)
        r = html_image_only_content(_ctx(normalized_text=html, parsed_html=soup))
        assert r is None

    def test_few_images_no_trigger(self):
        html = '<html><body><img src="a.png" /><img src="b.png" /></body></html>'
        soup, _ = parse_html_dom(html)
        r = html_image_only_content(_ctx(normalized_text=html, parsed_html=soup))
        assert r is None

    def test_no_html_no_trigger(self):
        r = html_image_only_content(_ctx())
        assert r is None


# ── QR_CODE_EXTERNAL_URL ──

class TestQRCodeExternalUrl:
    def test_external_qr_detected(self):
        r = qr_code_external_url(_ctx(qr_urls=["https://evil.com/phish"]))
        assert r is not None
        assert r[0].rule_id == "QR_CODE_EXTERNAL_URL"
        assert r[0].severity == "medium"
        assert r[0].subject_id == "qr:0"

    def test_official_qr_not_detected(self):
        r = qr_code_external_url(_ctx(qr_urls=["https://sdu.edu.cn/info"]))
        assert r is None

    def test_no_qr_urls(self):
        r = qr_code_external_url(_ctx())
        assert r is None

    def test_multiple_qr_mixed(self):
        r = qr_code_external_url(_ctx(qr_urls=["https://sdu.edu.cn", "https://evil.com"]))
        assert r is not None
        assert len(r) == 1

    def test_invalid_url_skipped(self):
        r = qr_code_external_url(_ctx(qr_urls=[":::bad-url"]))
        assert r is None


# ── QR_CREDENTIAL_URL ──

class TestQrCredentialUrl:
    def test_external_login_qr_detected(self):
        r = qr_credential_url(_ctx(qr_urls=["https://evil.com/login?token=abc"]))
        assert r is not None
        assert r[0].rule_id == "QR_CREDENTIAL_URL"
        assert r[0].severity == "high"
        assert r[0].subject_id == "qr:0"

    def test_external_auth_qr_detected(self):
        r = qr_credential_url(_ctx(qr_urls=["https://fake.com/auth/verify"]))
        assert r is not None

    def test_percent_encoded_login_qr_detected_by_entrypoint(self):
        from phishing_rule_detector.detector import detect

        result = detect(
            "scan",
            "text",
            context={"qr_urls": ["https://evil.com/%6c%6f%67%69%6e"]},
        )

        assert result["risk"]["level"] == "high"
        assert any(e["rule_id"] == "QR_CREDENTIAL_URL" for e in result["evidence"])

    def test_punycode_homograph_qr_runs_domain_rules(self):
        from phishing_rule_detector.detector import detect

        result = detect(
            "scan",
            "text",
            context={"qr_urls": ["https://xn--du-doc.edu.cn/news"]},
        )

        assert result["risk"]["level"] == "high"
        assert any(
            e["rule_id"] == "DOMAIN_PUNYCODE_SUSPICIOUS"
            for e in result["evidence"]
        )

    def test_official_login_not_detected(self):
        r = qr_credential_url(_ctx(qr_urls=["https://sdu.edu.cn/login"]))
        assert r is None

    def test_no_login_clue_not_detected(self):
        r = qr_credential_url(_ctx(qr_urls=["https://evil.com/about"]))
        assert r is None

    def test_no_qr_urls(self):
        r = qr_credential_url(_ctx())
        assert r is None
