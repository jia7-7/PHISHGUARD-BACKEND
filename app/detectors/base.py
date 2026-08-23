from __future__ import annotations

from typing import Protocol

from app.models import AnalysisPayload, DetectorResult


class Detector(Protocol):
    name: str
    family: str

    async def analyze(self, payload: AnalysisPayload) -> DetectorResult | None:
        """Return a result, or None when this detector does not support the input."""

