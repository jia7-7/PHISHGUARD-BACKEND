from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InputKind(StrEnum):
    TEXT = "text"
    URL = "url"
    HTML = "html"
    IMAGE = "image"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TextAnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1)
    source_type: Literal["email", "sms", "message", "notice", "other"] = "other"
    sender: str | None = Field(default=None, max_length=320)
    attachments: list[str] = Field(default_factory=list, max_length=50)
    qr_urls: list[str] = Field(default_factory=list, max_length=20)


class UrlAnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(min_length=4, max_length=4096)


class HtmlAnalysisRequest(BaseModel):
    html: str = Field(min_length=1)
    source_url: str | None = Field(default=None, max_length=4096)
    sender: str | None = Field(default=None, max_length=320)


class AnalysisPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: InputKind
    text: str | None = None
    url: str | None = None
    html: str | None = None
    filename: str | None = None
    content_type: str | None = None
    source_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_bytes: bytes | None = Field(default=None, repr=False, exclude=True)


class Signal(BaseModel):
    category: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    severity: int = Field(ge=0, le=100)
    evidence: str | None = Field(default=None, max_length=500)
    location: dict[str, Any] | None = None


class DetectorResult(BaseModel):
    detector: str = Field(min_length=1, max_length=64)
    family: Literal["rules", "ai"]
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    signals: list[Signal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    category: str
    title: str
    description: str
    severity: int
    evidence: str | None = None
    location: dict[str, Any] | None = None
    sources: list[str]


class DetectorStatus(BaseModel):
    detector: str
    family: Literal["rules", "ai"]
    status: Literal["completed", "skipped", "failed"]
    score: int | None = None
    confidence: float | None = None
    detail: str | None = None


class AnalysisResponse(BaseModel):
    request_id: str
    input_type: InputKind
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    summary: str
    evidence: list[Evidence]
    recommendations: list[str]
    warnings: list[str] = Field(default_factory=list)
    detector_statuses: list[DetectorStatus]
    processing_ms: int = Field(ge=0)
    retention_policy: Literal["not_stored"] = "not_stored"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    configured_detectors: list[str]
    raw_input_storage: Literal["disabled"] = "disabled"


class PublicConfigResponse(BaseModel):
    max_text_chars: int
    max_upload_bytes: int
    supported_file_types: list[str]
    risk_thresholds: dict[str, int]
