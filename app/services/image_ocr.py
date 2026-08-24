from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path


class ImageOCRUnavailableError(RuntimeError):
    pass


class ImageOCRInsufficientError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ImageOCRResult:
    text: str
    confidence: float
    quality_status: str
    processing_ms: int
    warnings: tuple[str, ...] = ()


class ImageOCRService:
    """Runs the bundled OCR engine in the public API process."""

    def __init__(self) -> None:
        self._engine = None
        self._load_lock = threading.Lock()

    def extract(self, image_bytes: bytes) -> ImageOCRResult:
        try:
            result = self._get_engine().extract_text_with_details(image_bytes)
        except Exception as exc:
            raise ImageOCRUnavailableError("Server OCR is temporarily unavailable") from exc

        text = str(result.get("corrected_text") or result.get("full_text") or "").strip()
        quality_status = str(result.get("ocr_quality_status") or "unknown")
        if not text or quality_status == "insufficient_evidence":
            raise ImageOCRInsufficientError(
                "OCR could not extract reliable text from the image; "
                "please upload a clearer screenshot"
            )

        confidence = result.get("ocr_confidence_avg", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        processing_ms = result.get("processing_time_ms", 0)
        if not isinstance(processing_ms, (int, float)):
            processing_ms = 0
        warnings = tuple(
            str(item)[:300]
            for item in result.get("ocr_warnings", [])
            if isinstance(item, str) and item.strip()
        )
        return ImageOCRResult(
            text=text,
            confidence=max(0.0, min(1.0, float(confidence))),
            quality_status=quality_status,
            processing_ms=max(0, round(float(processing_ms))),
            warnings=warnings,
        )

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        with self._load_lock:
            if self._engine is not None:
                return self._engine
            ai_service_dir = Path(__file__).resolve().parents[2] / "ai_service"
            ai_service_path = str(ai_service_dir)
            if ai_service_path not in sys.path:
                sys.path.insert(0, ai_service_path)
            try:
                from ocr_module import OCREngine
            except Exception as exc:
                raise ImageOCRUnavailableError("Server OCR could not be loaded") from exc
            self._engine = OCREngine()
            return self._engine
