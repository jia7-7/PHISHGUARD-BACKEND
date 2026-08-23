from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import AnalysisPayload, DetectorResult, InputKind, Signal


_SEVERITY = {"low": 30, "medium": 60, "high": 85}


class IntegratedRuleDetector:
    """Adapter for the security teammate's offline v3 rule engine."""

    name = "security-rules-v3"
    family = "rules"

    def __init__(
        self,
        settings: Settings,
        detect_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._detect = detect_fn or self._load_detector(settings.rule_engine_path)

    async def analyze(self, payload: AnalysisPayload) -> DetectorResult | None:
        if self._detect is None or payload.kind == InputKind.IMAGE:
            return None

        input_text, input_type, base_url = self._translate_input(payload)
        context = self._build_context(payload)
        result = await asyncio.to_thread(
            self._detect,
            input_text=input_text,
            input_type=input_type,
            base_url=base_url,
            context=context,
        )
        if not result.get("success"):
            error = result.get("error") or {}
            code = error.get("code", "UNKNOWN_RULE_ERROR")
            raise RuntimeError(f"Rule engine failed: {code}")
        return self._to_detector_result(result)

    @staticmethod
    def _load_detector(
        engine_path: str,
    ) -> Callable[..., dict[str, Any]] | None:
        root = Path(engine_path).resolve()
        if not (root / "phishing_rule_detector" / "detector.py").is_file():
            return None
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from phishing_rule_detector.detector import detect

        return detect

    @staticmethod
    def _translate_input(payload: AnalysisPayload) -> tuple[str, str, str | None]:
        if payload.kind == InputKind.URL:
            return payload.url or "", "url", None
        if payload.kind == InputKind.HTML:
            return payload.html or "", "html", payload.url

        input_type = payload.source_type if payload.source_type in {"email", "sms"} else "text"
        return payload.text or "", input_type, None

    @staticmethod
    def _build_context(payload: AnalysisPayload) -> dict[str, Any]:
        metadata = payload.metadata
        return {
            "sender": metadata.get("sender") or "",
            "attachments": list(metadata.get("attachments") or []),
            "qr_urls": list(metadata.get("qr_urls") or []),
            "ocr_text": metadata.get("ocr_text") or "",
            "debug": False,
        }

    def _to_detector_result(self, result: dict[str, Any]) -> DetectorResult:
        risk = result["risk"]
        signals = [self._to_signal(item) for item in result.get("evidence", [])]
        return DetectorResult(
            detector=self.name,
            family="rules",
            score=risk["score"],
            confidence=risk["confidence"],
            signals=signals,
            metadata={
                "rule_version": result.get("rule_version"),
                "trace_id": result.get("trace_id"),
                "raw_score": risk.get("raw_score"),
                "rule_level": risk.get("level"),
                "level_floor": risk.get("level_floor", "low"),
                "critical_lock": bool(risk.get("critical_lock")),
                "warnings": result.get("warnings", []),
                "duration_ms": result.get("duration_ms"),
            },
        )

    @staticmethod
    def _to_signal(item: dict[str, Any]) -> Signal:
        rule_id = str(item.get("rule_id") or "UNKNOWN_RULE")
        group = str(item.get("group") or "rules")
        if rule_id in {"PASSWORD_FORM_UNTRUSTED_TARGET", "HTML_PASSWORD_INPUT"}:
            category = "credential-form"
        elif rule_id.startswith("FORM_"):
            category = "form-action"
        elif group == "identity":
            category = "impersonation"
        else:
            category = {
                "credential": "credential-theft",
                "navigation": "navigation",
                "social": "social-engineering",
                "transport": "transport",
                "payload": "payload",
            }.get(group, group)

        return Signal(
            category=category,
            title=str(item.get("title") or rule_id),
            description=str(item.get("reason") or "规则引擎检测到可疑特征"),
            severity=_SEVERITY.get(str(item.get("severity")), 30),
            evidence=str(item.get("matched_content") or "") or None,
            location={
                "rule_id": rule_id,
                "group": group,
                "source": item.get("source"),
                "subject_id": item.get("subject_id"),
                "confidence": item.get("confidence"),
            },
        )
