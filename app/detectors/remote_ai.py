from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from app.config import Settings
from app.models import AnalysisPayload, DetectorResult, InputKind, Signal


_ACTIVE_LLM_STATUSES = {
    "usable_first_attempt",
    "recovered_after_validation_retry",
    "recovered_after_conservative_enum_fallback",
}
_SEMANTIC_SEVERITY = {"low": 25, "medium": 55, "high": 80}


class RemoteAIDetector:
    """Adapter for the delivered AI/OCR FastAPI service."""

    name = "ai-semantic-v5"
    family = "ai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def analyze(self, payload: AnalysisPayload) -> DetectorResult | None:
        if not self.settings.ai_service_url:
            return None
        return await asyncio.to_thread(self._request, payload)

    def _request(self, payload: AnalysisPayload) -> DetectorResult | None:
        data = self._post_image(payload) if payload.kind == InputKind.IMAGE else self._post_text(payload)
        return self._translate_response(
            data,
            use_pipeline_score=payload.kind == InputKind.IMAGE,
        )

    def _post_text(self, payload: AnalysisPayload) -> dict:
        content, content_type = self._text_content(payload)
        truncated = len(content) > 10_000
        metadata = dict(payload.metadata)
        if truncated:
            metadata["integration_truncated_to_chars"] = 10_000
        body = {
            "content": content[:10_000],
            "content_type": content_type,
            "source_channel": payload.source_type or "unknown",
            "metadata": metadata or None,
            "scoring_strategy": "weighted",
        }
        return self._send(
            self._endpoint("text"),
            json.dumps(body, ensure_ascii=False).encode("utf-8"),
            "application/json",
        )

    def _post_image(self, payload: AnalysisPayload) -> dict:
        if payload.raw_bytes is None:
            raise RuntimeError("Image payload is empty")
        boundary = f"phishguard-{uuid4().hex}"
        filename = self._safe_filename(payload.filename or "image.png")
        content_type = payload.content_type or ""
        if not content_type.startswith("image/"):
            content_type = self._image_content_type(filename)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8") + payload.raw_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
        query = urlencode({"source_channel": "screenshot", "scoring_strategy": "weighted"})
        return self._send(
            f"{self._endpoint('image')}?{query}",
            body,
            f"multipart/form-data; boundary={boundary}",
        )

    def _send(self, url: str, body: bytes, content_type: str) -> dict:
        headers = {"Content-Type": content_type, "Accept": "application/json"}
        if self.settings.ai_service_token:
            headers["Authorization"] = f"Bearer {self.settings.ai_service_token}"
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.settings.ai_timeout_seconds) as response:
                response_body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError("AI service request failed") from exc
        try:
            data = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("AI service returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("AI service returned an invalid response")
        return data

    def _translate_response(
        self,
        data: dict,
        *,
        use_pipeline_score: bool = False,
    ) -> DetectorResult | None:
        if data.get("status") != "success":
            if use_pipeline_score and data.get("status") == "insufficient_evidence":
                return self._skipped(
                    "OCR could not extract reliable text from the image; "
                    "please upload a clearer screenshot"
                )
            return self._skipped("AI service did not return sufficient evidence")
        if use_pipeline_score:
            return self._translate_image_response(data)

        llm_status = str(data.get("llm_sample_status") or "not_requested")
        if llm_status not in _ACTIVE_LLM_STATUSES:
            return self._skipped(
                "LLM is not active; duplicate deterministic AI score was ignored"
            )

        llm_evidence = [
            item
            for item in data.get("verified_evidence", [])
            if (
                isinstance(item, dict)
                and item.get("source") == "llm"
                and item.get("verification") == "matched"
            )
        ]
        signals = [self._to_signal(item) for item in llm_evidence]
        score_result = data.get("score_result") or {}
        confidence = score_result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        return DetectorResult(
            detector=self.name,
            family="ai",
            score=self._semantic_score(llm_evidence),
            confidence=max(0.0, min(1.0, float(confidence))),
            signals=signals,
            metadata={
                "score_kind": "llm_semantic_evidence",
                "semantic_evidence_count": len(llm_evidence),
                "ai_analysis_id": data.get("analysis_id"),
                "ai_pipeline_score": score_result.get("risk_score"),
                "ai_pipeline_level": score_result.get("risk_level"),
                "llm_sample_status": llm_status,
                "model_used": data.get("model_used"),
                "warnings": data.get("warnings", []),
            },
        )

    def _translate_image_response(self, data: dict) -> DetectorResult:
        score_result = data.get("score_result") or {}
        score = score_result.get("risk_score")
        confidence = score_result.get("confidence", 0.0)
        if not isinstance(score, (int, float)):
            return self._skipped("OCR completed without a usable risk score")
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        evidence = [
            item
            for item in data.get("verified_evidence", [])
            if (
                isinstance(item, dict)
                and item.get("verification") in {"matched", "deterministic"}
            )
        ]
        return DetectorResult(
            detector=self.name,
            family="ai",
            score=max(0.0, min(100.0, float(score))),
            confidence=max(0.0, min(1.0, float(confidence))),
            signals=[self._to_signal(item) for item in evidence],
            metadata={
                "score_kind": "image_ocr_pipeline",
                "ai_analysis_id": data.get("analysis_id"),
                "ai_pipeline_level": score_result.get("risk_level"),
                "llm_sample_status": data.get("llm_sample_status"),
                "model_used": data.get("model_used"),
                "ocr_audit": data.get("ocr_audit", {}),
                "ocr_scoring_reliable": data.get("ocr_scoring_reliable"),
                "warnings": data.get("warnings", []),
            },
        )

    def _skipped(self, reason: str) -> DetectorResult:
        return DetectorResult(
            detector=self.name,
            family="ai",
            score=0,
            confidence=0,
            metadata={"skip_reason": reason},
        )

    def _endpoint(self, kind: str) -> str:
        base = self.settings.ai_service_url.rstrip("/")
        marker = "/api/v1/analyze/"
        if marker in base:
            base = base.split(marker, 1)[0]
        if base.endswith("/api/v1"):
            return f"{base}/analyze/{kind}"
        return f"{base}/api/v1/analyze/{kind}"

    @staticmethod
    def _text_content(payload: AnalysisPayload) -> tuple[str, str]:
        if payload.kind == InputKind.URL:
            return payload.url or "", "link"
        if payload.kind == InputKind.HTML:
            return payload.html or "", "html"
        content_type = {
            "email": "email",
            "sms": "sms",
            "notice": "notification",
        }.get(payload.source_type, "email")
        return payload.text or "", content_type

    @staticmethod
    def _semantic_score(evidence: list[dict]) -> int:
        if not evidence:
            return 5
        strongest = max(_SEMANTIC_SEVERITY.get(str(item.get("severity")), 25) for item in evidence)
        distinct_types = {str(item.get("type")) for item in evidence if item.get("type")}
        return min(95, strongest + max(0, len(distinct_types) - 1) * 5)

    @staticmethod
    def _to_signal(item: dict) -> Signal:
        evidence_type = str(item.get("type") or "semantic")
        source = str(item.get("source") or "ai")
        title_prefix = "AI语义证据" if source == "llm" else "OCR管线证据"
        return Signal(
            category=evidence_type,
            title=f"{title_prefix}：{evidence_type}",
            description=str(item.get("explanation") or "AI识别到可疑语义"),
            severity=_SEMANTIC_SEVERITY.get(str(item.get("severity")), 25),
            evidence=str(item.get("quote") or "") or None,
            location={
                "source": source,
                "verification": item.get("verification"),
                "ocr_block_id": item.get("ocr_block_id"),
                "bbox": item.get("bbox"),
            },
        )

    @staticmethod
    def _image_content_type(filename: str) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(Path(filename).suffix.lower(), "application/octet-stream")

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name
        return name.replace('"', "_").replace("\r", "_").replace("\n", "_")
