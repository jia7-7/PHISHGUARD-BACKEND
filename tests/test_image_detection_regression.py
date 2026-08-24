from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"
sys.path.insert(0, str(AI_SERVICE_DIR))

from ocr_module import OCREngine  # noqa: E402

from app.config import Settings  # noqa: E402
from app.detectors.remote_ai import RemoteAIDetector  # noqa: E402
from app.models import AnalysisPayload, DetectorResult, InputKind  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.analyzer import AnalysisService, AnalysisUnavailableError  # noqa: E402
from app.services.fusion import RiskFusionService  # noqa: E402
from app.services.image_ocr import (  # noqa: E402
    ImageOCRInsufficientError,
    ImageOCRResult,
)


class _FakeTesseractEngine:
    @staticmethod
    def image_to_data(*_args, **_kwargs):
        return {
            "text": ["", "山东大学", "立即验证"],
            "conf": ["-1", "96.514236", "82.25"],
            "left": [0, 10, 10],
            "top": [0, 10, 35],
            "width": [0, 120, 100],
            "height": [0, 20, 20],
        }


def test_tesseract_decimal_confidence_does_not_abort_image_ocr():
    engine = OCREngine.__new__(OCREngine)
    engine.engine_name = "tesseract"
    engine.config = {"languages": "chi_sim+eng"}
    engine._engine = _FakeTesseractEngine()
    fake_pytesseract = SimpleNamespace(Output=SimpleNamespace(DICT="dict"))

    with patch.dict(sys.modules, {"pytesseract": fake_pytesseract}):
        blocks = engine._do_ocr(Image.new("RGB", (320, 180), "white"))

    assert [block["text"] for block in blocks] == ["山东大学", "立即验证"]
    assert blocks[0]["confidence"] == 0.96514236
    assert blocks[1]["confidence"] == 0.8225


def test_image_insufficient_evidence_has_actionable_reason():
    detector = RemoteAIDetector(Settings(ai_service_url="http://127.0.0.1:8100"))

    result = detector._translate_response(
        {"status": "insufficient_evidence"},
        use_pipeline_score=True,
    )

    assert result is not None
    assert "OCR could not extract reliable text" in result.metadata["skip_reason"]


def test_reliable_image_ocr_preserves_strong_semantic_risk_floor():
    detector = RemoteAIDetector(Settings(ai_service_url="http://127.0.0.1:8100"))
    result = detector._translate_response(
        {
            "status": "success",
            "llm_sample_status": "usable_first_attempt",
            "verified_evidence": [
                {
                    "source": "llm",
                    "verification": "matched",
                    "type": evidence_type,
                    "severity": "high",
                    "quote": quote,
                    "explanation": "可疑语义证据",
                }
                for evidence_type, quote in [
                    ("credential_request", "输入账号密码"),
                    ("urgency", "立即验证"),
                ]
            ],
            "score_result": {"confidence": 0.243, "risk_score": 95},
        },
        source_metadata={
            "ocr_confidence": 0.93,
            "ocr_quality_status": "reliable",
        },
    )

    assert result is not None
    assert result.confidence >= 0.65

    fused = RiskFusionService(Settings()).fuse(
        [
            DetectorResult(
                detector="security-rules-v3",
                family="rules",
                score=0,
                confidence=0,
            ),
            result,
        ]
    )
    assert fused.score >= 60
    assert fused.level.value == "high"


def test_image_insufficient_evidence_is_surfaced_by_analysis_service():
    class _RulesDetector:
        name = "security-rules-v3"
        family = "rules"

        async def analyze(self, _payload):
            return None

    class _AIDetector:
        name = "ai-semantic-v5"
        family = "ai"

        async def analyze(self, _payload):
            return DetectorResult(
                detector=self.name,
                family=self.family,
                score=0,
                confidence=0,
                metadata={
                    "skip_reason": (
                        "OCR could not extract reliable text from the image; "
                        "please upload a clearer screenshot"
                    )
                },
            )

    service = AnalysisService([_RulesDetector(), _AIDetector()], fusion=None)

    try:
        asyncio.run(service.analyze(AnalysisPayload(kind=InputKind.IMAGE)))
    except AnalysisUnavailableError as error:
        assert "clearer screenshot" in str(error)
    else:
        raise AssertionError("Expected an actionable image OCR error")


def test_public_image_endpoint_runs_server_ocr_then_both_detectors():
    class _FakeOCR:
        received = b""

        def extract(self, image_bytes):
            self.received = image_bytes
            return ImageOCRResult(
                text="山东大学统一身份认证，请立即输入账号密码完成验证",
                confidence=0.91,
                quality_status="reliable",
                processing_ms=12,
            )

    class _Detector:
        def __init__(self, name, family, score):
            self.name = name
            self.family = family
            self.score = score

        async def analyze(self, payload):
            assert payload.kind == InputKind.TEXT
            assert "账号密码" in (payload.text or "")
            return DetectorResult(
                detector=self.name,
                family=self.family,
                score=self.score,
                confidence=0.8,
            )

    fake_ocr = _FakeOCR()
    app = create_app(
        settings=Settings(),
        detectors=[
            _Detector("security-rules-v3", "rules", 75),
            _Detector("ai-semantic-v5", "ai", 80),
        ],
        image_ocr=fake_ocr,
    )

    response = TestClient(app).post(
        "/api/v1/analyze/file?filename=test.png",
        content=b"\x89PNG\r\n\x1a\nserver-ocr-test",
        headers={"Content-Type": "image/png"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["input_type"] == "image"
    assert [item["status"] for item in data["detector_statuses"]] == [
        "completed",
        "completed",
    ]
    assert fake_ocr.received.endswith(b"server-ocr-test")
    assert "服务器完成文字识别" in data["warnings"][0]


def test_public_image_endpoint_returns_actionable_low_quality_error():
    class _InsufficientOCR:
        def extract(self, _image_bytes):
            raise ImageOCRInsufficientError(
                "OCR could not extract reliable text from the image; "
                "please upload a clearer screenshot"
            )

    app = create_app(
        settings=Settings(),
        detectors=[],
        image_ocr=_InsufficientOCR(),
    )

    response = TestClient(app).post(
        "/api/v1/analyze/file?filename=test.png",
        content=b"\x89PNG\r\n\x1a\nlow-quality-test",
        headers={"Content-Type": "image/png"},
    )

    assert response.status_code == 422
    assert "clearer screenshot" in response.json()["detail"]
