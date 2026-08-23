"""测试顶层入口 detect()."""
import pytest
from phishing_rule_detector.detector import detect


class TestDetectErrors:
    def test_empty_input(self):
        result = detect("", "text")
        assert result["success"] is False
        assert result["error"]["code"] == "EMPTY_INPUT"

    def test_whitespace_input(self):
        result = detect("   ", "text")
        assert result["success"] is False
        assert result["error"]["code"] == "EMPTY_INPUT"

    def test_invalid_input_type(self):
        result = detect("hello", "image")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_INPUT_TYPE"

    def test_payload_too_large(self):
        result = detect("x" * 204801, "text")
        assert result["success"] is False
        assert result["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_invalid_context_attachments(self):
        result = detect(
            "test",
            "email",
            context={"attachments": "not_a_list"},
        )
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"


class TestDetectSuccess:
    def test_normal_text_returns_result(self):
        result = detect("hello world", "text")
        assert result["success"] is True
        assert "risk" in result
        assert "evidence" in result
        assert "rule_version" in result
        assert result["rule_version"] == "3.0.0"

    def test_trace_id_present(self):
        result = detect("hello", "text")
        assert len(result["trace_id"]) == 16
        # 十六进制校验
        int(result["trace_id"], 16)

    def test_duration_ms_present(self):
        result = detect("hello", "text")
        assert result["duration_ms"] >= 0

    def test_normalization_meta_present(self):
        result = detect("hello", "text")
        assert "normalization" in result
        assert "operations" in result["normalization"]
        assert "input_bytes" in result["normalization"]

    def test_warnings_default_empty(self):
        result = detect("hello", "text")
        assert result["warnings"] == []

    def test_summary_structure(self):
        result = detect("hello", "text")
        summary = result["summary"]
        assert "high_count" in summary
        assert "medium_count" in summary
        assert "low_count" in summary
        assert "evidence_groups" in summary

    def test_risk_structure(self):
        result = detect("hello", "text")
        risk = result["risk"]
        assert "score" in risk
        assert "raw_score" in risk
        assert "level" in risk
        assert "level_floor" in risk
        assert "confidence" in risk
        assert "critical_lock" in risk
        assert risk["score"] >= 0
        assert risk["score"] <= 100

    def test_safe_url_returns_low(self):
        """安全 URL（官方域名）必须返回 low 且分数在 [0, 29] 范围."""
        result = detect("https://www.sdu.edu.cn/", "url")
        assert result["success"] is True
        assert result["risk"]["level"] == "low"
        assert result["risk"]["score"] >= 0
        assert result["risk"]["score"] <= 29
        assert result["risk"]["critical_lock"] is False

    def test_bare_url_is_normalized_before_detection(self):
        result = detect("sdu-login.example.com", "url")

        assert result["risk"]["level"] == "high"
        assert "url_scheme_inferred" in result["normalization"]["operations"]
        assert any(
            item["rule_id"] == "DOMAIN_KEYWORD_IMPERSONATION"
            for item in result["evidence"]
        )

    def test_obfuscated_ocr_url_is_detected(self):
        result = detect(
            "image content",
            "text",
            context={
                "ocr_text": "https://ｓｄｕ-login.example.com/%6c%6f%67%69%6e"
            },
        )

        assert result["risk"]["level"] == "high"
        assert any(
            item["rule_id"] == "DOMAIN_KEYWORD_IMPERSONATION"
            for item in result["evidence"]
        )

    def test_input_types_all_work(self):
        for t in ("url", "html", "email", "sms", "text"):
            result = detect("test content", t)
            assert result["success"] is True

    def test_base_url_accepted(self):
        result = detect(
            "<html>test</html>",
            "html",
            base_url="https://example.com/page",
        )
        assert result["success"] is True

    def test_context_sender(self):
        result = detect(
            "test email",
            "email",
            context={"sender": "service@sdu.edu.cn"},
        )
        assert result["success"] is True

    def test_context_ocr_text(self):
        result = detect(
            "test with ocr",
            "text",
            context={"ocr_text": "扫描得到的文字"},
        )
        assert result["success"] is True

    def test_debug_mode(self):
        result = detect("test", "text", context={"debug": True})
        assert result["success"] is True
        # debug 模式不改变基本结构
        assert "risk" in result

    def test_trace_ids_unique(self):
        ids = {detect("test", "text")["trace_id"] for _ in range(10)}
        assert len(ids) == 10

    def test_scoring_failure_returns_internal_rule_error(self):
        """评分管线异常 → INTERNAL_RULE_ERROR，不泄露 traceback（审查 Fix 4）."""
        from unittest.mock import patch

        def _raise(*args, **kwargs):
            raise RuntimeError("sensitive secret 12345")

        with patch(
            "phishing_rule_detector.detector.score_pipeline",
            side_effect=_raise,
        ):
            result = detect("test content", "text")
            assert result["success"] is False
            assert result["error"]["code"] == "INTERNAL_RULE_ERROR"
            # 响应中不包含 Traceback
            assert "Traceback" not in str(result)
            # 响应中不包含本机路径
            import sys
            for p in sys.path:
                if p and len(p) > 4:  # 跳过空路径和短路径
                    assert p not in str(result)
            # 响应中不包含敏感异常原文
            assert "sensitive secret 12345" not in str(result)
            assert "scoring exploded" not in str(result)
            # 响应包含 trace_id
            assert "trace_id" in result

    def test_rule_orchestrator_failure_returns_internal_rule_error(self):
        """规则编排层整体失败时不得返回 low 安全结果."""
        from unittest.mock import patch

        with patch(
            "phishing_rule_detector.detector.run_all_rules",
            side_effect=RuntimeError("sensitive rule failure"),
        ):
            result = detect("https://evil.example/login", "url")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_RULE_ERROR"
        assert "sensitive rule failure" not in str(result)

    def test_rule_orchestrator_failure_log_redacts_exception_message(self, caplog):
        from unittest.mock import patch

        secret = "token=LogSecretValue"
        with patch(
            "phishing_rule_detector.detector.run_all_rules",
            side_effect=RuntimeError(secret),
        ):
            detect("https://evil.example/login", "url")

        assert secret not in caplog.text

    def test_individual_rule_failure_log_redacts_exception_message(
        self,
        caplog,
        monkeypatch,
    ):
        import phishing_rule_detector.rules.common as common

        secret = "password=IndividualRuleSecret"

        def failing_rule(_ctx):
            raise RuntimeError(secret)

        failing_rule.rule_id = "TEST_FAILING_RULE"
        monkeypatch.setattr(common, "_RULE_REGISTRY", [failing_rule])
        result = detect("ordinary text", "text")

        assert result["success"] is True
        assert "RULE_ERROR:TEST_FAILING_RULE" in result["warnings"]
        assert secret not in caplog.text

    def test_detector_uses_post_scoring_evidence(self):
        """detector 返回的 evidence 是评分后的 active evidence（审查 Fix 3）."""
        result = detect("https://www.sdu.edu.cn/", "url")
        assert result["success"] is True
        assert isinstance(result["evidence"], list)
        if result.get("suppressed_evidence") is not None:
            assert isinstance(result["suppressed_evidence"], list)

    # ── Fix 7: 输出契约 ──

    def test_non_debug_response_excludes_suppressed_evidence(self):
        """非 debug 响应不应出现 suppressed_evidence: null（审查 Fix 7）."""
        result = detect("hello", "text")
        assert result["success"] is True
        # suppressed_evidence 不应出现在输出字典中
        assert "suppressed_evidence" not in result

    def test_context_must_be_dict(self):
        """context 必须为 dict，[] 应返回错误（审查 Fix 7）."""
        result = detect("test", "text", context=[])
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_debug_must_be_bool(self):
        """context.debug 必须严格为 bool，字符串应返回错误（审查 Fix 7）."""
        result = detect("test", "text", context={"debug": "false"})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_debug_must_be_bool_int(self):
        """context.debug 必须严格为 bool，数字 1 应返回错误（审查 Fix 7）."""
        result = detect("test", "text", context={"debug": 1})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_sender_must_be_string(self):
        """context.sender 如果提供，必须是 str，0 应返回错误（审查 Fix 7）."""
        result = detect("test", "text", context={"sender": 0})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_sender_none_rejected(self):
        """context.sender=None 应被拒绝（审查 Fix 7）."""
        result = detect("test", "text", context={"sender": None})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_ocr_text_must_be_string(self):
        """context.ocr_text 如果提供，必须是 str（审查 Fix 7）."""
        result = detect("test", "text", context={"ocr_text": 123})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_ocr_text_none_rejected(self):
        """context.ocr_text=None 应被拒绝（审查 Fix 7）."""
        result = detect("test", "text", context={"ocr_text": None})
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_CONTEXT"

    def test_sender_empty_string_allowed(self):
        """context.sender='' 空字符串允许（审查 Fix 7）."""
        result = detect("test", "text", context={"sender": ""})
        assert result["success"] is True

    def test_ocr_text_empty_string_allowed(self):
        """context.ocr_text='' 空字符串允许（审查 Fix 7）."""
        result = detect("test", "text", context={"ocr_text": ""})
        assert result["success"] is True

    def test_payload_too_large_has_details(self):
        """PAYLOAD_TOO_LARGE 错误应将 max_bytes/actual_bytes 嵌套在 details 中（审查 Fix 4）."""
        result = detect("x" * 204801, "text")
        assert result["success"] is False
        assert result["error"]["code"] == "PAYLOAD_TOO_LARGE"
        assert "details" in result["error"]
        assert result["error"]["details"]["max_bytes"] == 204800
        assert result["error"]["details"]["actual_bytes"] >= 204801
        # 旧版扁平字段不应存在
        assert "max_bytes" not in result["error"]
        assert "actual_bytes" not in result["error"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://xn--fiq228c.com:bad/path",
            "https://xn--fiq228c.com:99999/path",
        ],
    )
    def test_invalid_url_port_is_not_payload_too_large(self, url):
        """小型畸形 URL 不得被错误分类为超大输入."""
        result = detect(url, "url")
        assert result["success"] is True
        assert result["risk"]["level"] == "low"

    def test_unexpected_normalization_value_error_is_internal_error(self):
        """普通规范化 ValueError 不得冒充 PAYLOAD_TOO_LARGE."""
        from unittest.mock import patch

        with patch(
            "phishing_rule_detector.detector.normalize",
            side_effect=ValueError("sensitive normalization detail"),
        ):
            result = detect("small input", "text")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_RULE_ERROR"
        assert "sensitive normalization detail" not in str(result)
