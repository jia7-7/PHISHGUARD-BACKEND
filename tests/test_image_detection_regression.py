from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"
sys.path.insert(0, str(AI_SERVICE_DIR))

from ocr_module import OCREngine  # noqa: E402

from app.config import Settings  # noqa: E402
from app.detectors.remote_ai import RemoteAIDetector  # noqa: E402
from app.models import AnalysisPayload, DetectorResult, InputKind  # noqa: E402
from app.services.analyzer import AnalysisService, AnalysisUnavailableError  # noqa: E402


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
