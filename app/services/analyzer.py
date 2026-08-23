from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from app.detectors.base import Detector
from app.models import (
    AnalysisPayload,
    AnalysisResponse,
    DetectorResult,
    DetectorStatus,
    InputKind,
)
from app.services.fusion import RiskFusionService


class AnalysisUnavailableError(RuntimeError):
    pass


class AnalysisService:
    def __init__(self, detectors: list[Detector], fusion: RiskFusionService) -> None:
        self.detectors = detectors
        self.fusion = fusion

    async def analyze(self, payload: AnalysisPayload) -> AnalysisResponse:
        started = time.perf_counter()
        request_id = uuid4().hex
        raw_results = await asyncio.gather(
            *(detector.analyze(payload) for detector in self.detectors),
            return_exceptions=True,
        )

        completed: list[DetectorResult] = []
        statuses: list[DetectorStatus] = []
        warnings: list[str] = []
        for detector, result in zip(self.detectors, raw_results):
            if isinstance(result, BaseException):
                warnings.append(f"{detector.name}: detector unavailable")
                statuses.append(
                    DetectorStatus(
                        detector=detector.name,
                        family=detector.family,  # type: ignore[arg-type]
                        status="failed",
                        detail="detector unavailable",
                    )
                )
            elif result is None:
                statuses.append(
                    DetectorStatus(
                        detector=detector.name,
                        family=detector.family,  # type: ignore[arg-type]
                        status="skipped",
                        detail="input type not supported or service not configured",
                    )
                )
            elif result.metadata.get("skip_reason"):
                warnings.append(f"{result.detector}: {result.metadata['skip_reason']}")
                statuses.append(
                    DetectorStatus(
                        detector=result.detector,
                        family=result.family,
                        status="skipped",
                        detail=str(result.metadata["skip_reason"]),
                    )
                )
            else:
                completed.append(result)
                warnings.extend(self._metadata_warnings(result))
                statuses.append(
                    DetectorStatus(
                        detector=result.detector,
                        family=result.family,
                        status="completed",
                        score=round(result.score),
                        confidence=round(result.confidence, 3),
                    )
                )

        if not completed:
            raise AnalysisUnavailableError("No configured detector supports this input")

        fused = self.fusion.fuse(completed)
        if payload.kind == InputKind.IMAGE and fused.confidence < 0.4:
            warnings.append("图片检测置信度较低，结果需要人工复核")
        processing_ms = round((time.perf_counter() - started) * 1000)
        return AnalysisResponse(
            request_id=request_id,
            input_type=payload.kind,
            risk_score=fused.score,
            risk_level=fused.level,
            confidence=fused.confidence,
            summary=fused.summary,
            evidence=fused.evidence,
            recommendations=fused.recommendations,
            warnings=list(dict.fromkeys(warnings))[:20],
            detector_statuses=statuses,
            processing_ms=processing_ms,
        )

    @staticmethod
    def _metadata_warnings(result: DetectorResult) -> list[str]:
        messages: list[str] = []
        for item in result.metadata.get("warnings") or []:
            if isinstance(item, str):
                message = item
            elif isinstance(item, dict):
                message = str(item.get("message") or item.get("type") or "")
            else:
                continue
            if message:
                messages.append(f"{result.detector}: {message[:300]}")
        return messages
