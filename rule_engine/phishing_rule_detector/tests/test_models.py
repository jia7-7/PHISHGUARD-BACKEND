"""测试数据模型的校验与序列化."""
import pytest
from phishing_rule_detector.models import (
    DetectionInput,
    EvidenceItem,
    RiskResult,
    DetectionResult,
    ErrorResult,
    ErrorCode,
    generate_trace_id,
    MAX_INPUT_BYTES,
)


class TestDetectionInput:
    def test_valid_input_minimal(self):
        inp = DetectionInput(input_text="hello", input_type="text")
        assert inp.input_text == "hello"
        assert inp.input_type == "text"

    def test_empty_input_raises(self):
        with pytest.raises(Exception):
            DetectionInput(input_text="", input_type="text")

    def test_whitespace_only_input_raises(self):
        with pytest.raises(Exception):
            DetectionInput(input_text="   ", input_type="text")

    def test_invalid_input_type_raises(self):
        with pytest.raises(Exception):
            DetectionInput(input_text="hello", input_type="image")

    def test_all_valid_input_types(self):
        for t in ("url", "html", "email", "sms", "text"):
            inp = DetectionInput(input_text="test", input_type=t)
            assert inp.input_type == t

    def test_payload_too_large(self):
        big = "x" * (MAX_INPUT_BYTES + 1)
        with pytest.raises(Exception):
            DetectionInput(input_text=big, input_type="text")

    def test_payload_at_limit_passes(self):
        ok = "x" * MAX_INPUT_BYTES
        inp = DetectionInput(input_text=ok, input_type="text")
        assert inp.input_text == ok

    def test_ocr_text_counts_toward_limit(self):
        big = "x" * (MAX_INPUT_BYTES - 10)
        with pytest.raises(Exception):
            DetectionInput(
                input_text=big,
                input_type="text",
                context={"ocr_text": "y" * 20},
            )

    def test_attachments_exceed_max(self):
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="email",
                context={"attachments": ["a.txt"] * 51},
            )

    def test_attachment_name_too_long(self):
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="email",
                context={"attachments": ["x" * 256]},
            )

    def test_qr_urls_exceed_max(self):
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="email",
                context={"qr_urls": ["https://x.com"] * 21},
            )

    def test_qr_url_too_long(self):
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="email",
                context={"qr_urls": ["https://x.com/" + "y" * 4097]},
            )

    def test_base_url_optional(self):
        inp = DetectionInput(input_text="test", input_type="html")
        assert inp.base_url is None

    def test_base_url_provided(self):
        inp = DetectionInput(
            input_text="test", input_type="html", base_url="https://example.com"
        )
        assert inp.base_url == "https://example.com"

    # ── Fix 7: strict sender/ocr_text validation ──

    def test_sender_must_be_str_reject_none(self):
        """sender 为 None 时拒绝（审查 Fix 7）."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"sender": None},
            )

    def test_sender_must_be_str_reject_int(self):
        """sender 为数字时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"sender": 0},
            )

    def test_sender_must_be_str_reject_list(self):
        """sender 为列表时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"sender": ["a@b.com"]},
            )

    def test_sender_must_be_str_reject_dict(self):
        """sender 为字典时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"sender": {"email": "a@b.com"}},
            )

    def test_sender_empty_string_allowed(self):
        """sender 为空字符串时允许（审查 Fix 7）."""
        inp = DetectionInput(
            input_text="test",
            input_type="text",
            context={"sender": ""},
        )
        assert inp.context["sender"] == ""

    def test_sender_missing_allowed(self):
        """sender 键缺失时允许."""
        inp = DetectionInput(
            input_text="test",
            input_type="text",
            context={"debug": True},
        )
        assert "sender" not in inp.context

    def test_ocr_text_must_be_str_reject_none(self):
        """ocr_text 为 None 时拒绝（审查 Fix 7）."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"ocr_text": None},
            )

    def test_ocr_text_must_be_str_reject_int(self):
        """ocr_text 为数字时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"ocr_text": 0},
            )

    def test_ocr_text_must_be_str_reject_list(self):
        """ocr_text 为列表时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"ocr_text": ["text"]},
            )

    def test_ocr_text_must_be_str_reject_dict(self):
        """ocr_text 为字典时拒绝."""
        with pytest.raises(Exception):
            DetectionInput(
                input_text="test",
                input_type="text",
                context={"ocr_text": {"text": "value"}},
            )

    def test_ocr_text_empty_string_allowed(self):
        """ocr_text 为空字符串时允许."""
        inp = DetectionInput(
            input_text="test",
            input_type="text",
            context={"ocr_text": ""},
        )
        assert inp.context["ocr_text"] == ""

    def test_ocr_text_missing_allowed(self):
        """ocr_text 键缺失时允许."""
        inp = DetectionInput(
            input_text="test",
            input_type="text",
            context={"debug": True},
        )
        assert "ocr_text" not in inp.context


class TestEvidenceItem:
    def test_minimal_evidence(self):
        ev = EvidenceItem(
            rule_id="TEST_RULE",
            title="Test rule",
            group="identity",
            severity="medium",
            confidence=0.8,
            base_score=10,
            effective_score=8,
            reason="test reason",
            matched_content="test match",
            source="raw",
            tags=["test"],
        )
        data = ev.model_dump()
        assert data["rule_id"] == "TEST_RULE"
        assert data["group"] == "identity"
        assert data["context_factor"] == 1.0

    def test_invalid_group_raises(self):
        """group 必须是 EVIDENCE_GROUPS 中的一个."""
        with pytest.raises(Exception):
            EvidenceItem(
                rule_id="TEST",
                title="Test",
                group="invalid_group",
                severity="low",
                confidence=0.5,
                base_score=5,
                effective_score=5,
                reason="test",
            )

    def test_all_valid_groups(self):
        valid_groups = [
            "identity", "credential", "navigation",
            "social", "transport", "payload",
        ]
        for g in valid_groups:
            ev = EvidenceItem(
                rule_id="TEST", title="T", group=g,
                severity="low", confidence=0.5,
                base_score=5, effective_score=5, reason="r",
            )
            assert ev.group == g

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            EvidenceItem(
                rule_id="TEST",
                title="Test",
                group="identity",
                severity="low",
                confidence=1.5,
                base_score=5,
                effective_score=5,
                reason="test",
            )

    def test_context_factor_default(self):
        ev = EvidenceItem(
            rule_id="TEST",
            title="Test",
            group="identity",
            severity="low",
            confidence=0.5,
            base_score=5,
            effective_score=5,
            reason="test",
        )
        assert ev.context_factor == 1.0


class TestDetectionResult:
    def test_success_structure(self):
        result = DetectionResult(
            success=True,
            trace_id="abc123",
            rule_version="2.0.0",
            risk=RiskResult(
                score=20,
                raw_score=20,
                level="low",
                level_floor="low",
                confidence=0.0,
                critical_lock=False,
            ),
            evidence=[],
            summary={
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "evidence_groups": [],
            },
            normalization={
                "operations": [],
                "input_bytes": 5,
                "input_truncated": False,
            },
            warnings=[],
            duration_ms=10,
        )
        data = result.model_dump()
        assert data["success"] is True
        assert data["trace_id"] == "abc123"
        assert "risk" in data

    def test_suppressed_evidence_optional(self):
        result = DetectionResult(
            success=True,
            trace_id="abc123",
            rule_version="2.0.0",
            risk=RiskResult(
                score=20,
                raw_score=20,
                level="low",
                level_floor="low",
                confidence=0.0,
                critical_lock=False,
            ),
            evidence=[],
            summary={},
            normalization={},
        )
        assert result.suppressed_evidence is None


class TestErrorResult:
    def test_error_structure(self):
        result = ErrorResult(
            success=False,
            trace_id="abc123",
            error={"code": "EMPTY_INPUT", "message": "输入为空"},
        )
        data = result.model_dump()
        assert data["success"] is False
        assert data["error"]["code"] == "EMPTY_INPUT"

    def test_all_error_codes(self):
        for code in ErrorCode:
            assert code.value in (
                "EMPTY_INPUT",
                "INVALID_INPUT_TYPE",
                "PAYLOAD_TOO_LARGE",
                "INVALID_CONTEXT",
                "INTERNAL_RULE_ERROR",
            )


class TestTraceId:
    def test_trace_id_length(self):
        tid = generate_trace_id()
        assert len(tid) == 16

    def test_trace_id_hex(self):
        tid = generate_trace_id()
        int(tid, 16)  # 不抛异常即为合法的十六进制

    def test_trace_id_unique(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100
