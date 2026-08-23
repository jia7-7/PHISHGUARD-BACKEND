"""Pydantic v2 数据模型 — 输入/输出/证据/错误."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ErrorCode(str, Enum):
    EMPTY_INPUT = "EMPTY_INPUT"
    INVALID_INPUT_TYPE = "INVALID_INPUT_TYPE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INVALID_CONTEXT = "INVALID_CONTEXT"
    INTERNAL_RULE_ERROR = "INTERNAL_RULE_ERROR"


INPUT_TYPES = Literal["url", "html", "email", "sms", "text"]
SEVERITY_LEVELS = Literal["low", "medium", "high"]
RISK_LEVELS = Literal["low", "medium", "high", "critical"]
EVIDENCE_GROUPS = Literal[
    "identity", "credential", "navigation", "social", "transport", "payload"
]
EVIDENCE_SOURCE = Literal[
    "raw", "normalized", "ocr", "qr", "attachment", "online"
]

MAX_INPUT_BYTES = 204800  # 200 KiB
MAX_ATTACHMENTS = 50
MAX_ATTACHMENT_NAME_LEN = 255
MAX_QR_URLS = 20
MAX_QR_URL_LEN = 4096
MAX_MATCHED_CONTENT_LEN = 160


class DetectionInput(BaseModel):
    """detect() 函数的输入模型."""

    input_text: str = Field(..., min_length=1, description="待检测内容")
    input_type: INPUT_TYPES = Field(..., description="输入类型")
    base_url: str | None = Field(default=None, description="页面来源 URL")
    context: dict[str, Any] = Field(
        default_factory=dict, description="扩展上下文"
    )

    @field_validator("input_text")
    @classmethod
    def check_not_empty_after_strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input_text 不能为空或仅含空白字符")
        return v

    @field_validator("context")
    @classmethod
    def check_context_types(cls, v: dict[str, Any]) -> dict[str, Any]:
        # context 本身必须是 dict（审查 Fix 7）
        if not isinstance(v, dict):
            raise ValueError("context 必须是 dict")

        attachments = v.get("attachments", [])
        if not isinstance(attachments, list):
            raise ValueError("context.attachments 必须是 list[str]")
        if len(attachments) > MAX_ATTACHMENTS:
            raise ValueError(f"attachments 最多 {MAX_ATTACHMENTS} 项")
        for a in attachments:
            if not isinstance(a, str):
                raise ValueError("attachments 每项必须是 str")
            if len(a) > MAX_ATTACHMENT_NAME_LEN:
                raise ValueError(
                    f"附件名最长 {MAX_ATTACHMENT_NAME_LEN} 字符"
                )

        qr_urls = v.get("qr_urls", [])
        if not isinstance(qr_urls, list):
            raise ValueError("context.qr_urls 必须是 list[str]")
        if len(qr_urls) > MAX_QR_URLS:
            raise ValueError(f"qr_urls 最多 {MAX_QR_URLS} 项")
        for q in qr_urls:
            if not isinstance(q, str):
                raise ValueError("qr_urls 每项必须是 str")
            if len(q) > MAX_QR_URL_LEN:
                raise ValueError(f"二维码 URL 最长 {MAX_QR_URL_LEN} 字符")

        # debug 必须严格为 bool（审查 Fix 7）
        if "debug" in v:
            if not isinstance(v["debug"], bool):
                raise ValueError("context.debug 必须是 bool")

        # sender 如果提供必须严格为 str（审查 Fix 7）
        # None / 0 / list / dict 均需拒绝，仅允许缺失键
        if "sender" in v:
            if not isinstance(v["sender"], str):
                raise ValueError("context.sender 必须是 str")

        # ocr_text 如果提供必须严格为 str（审查 Fix 7）
        if "ocr_text" in v:
            if not isinstance(v["ocr_text"], str):
                raise ValueError("context.ocr_text 必须是 str")

        return v

    @model_validator(mode="after")
    def check_total_size(self) -> "DetectionInput":
        total = self.input_text.encode("utf-8")
        ocr = self.context.get("ocr_text", "")
        if ocr:
            total += ocr.encode("utf-8") if isinstance(ocr, str) else b""
        if len(total) > MAX_INPUT_BYTES:
            raise ValueError(f"总输入超过 {MAX_INPUT_BYTES} 字节限制")
        return self


class EvidenceItem(BaseModel):
    """单条证据结构."""

    rule_id: str
    title: str
    group: EVIDENCE_GROUPS
    severity: SEVERITY_LEVELS
    confidence: float = Field(..., ge=0.0, le=1.0)
    context_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    base_score: int = Field(..., ge=0)
    effective_score: int = Field(..., ge=0)
    reason: str
    matched_content: str = ""
    source: EVIDENCE_SOURCE = "raw"
    tags: list[str] = Field(default_factory=list)
    subject_id: str | None = Field(
        default=None,
        description="证据所属对象标识（如 url:0, form:1, sentence:2）。规则抑制仅在同一 subject_id 内生效。",
    )


class RiskResult(BaseModel):
    """风险评级结果."""

    score: int = Field(..., ge=0, le=100)
    raw_score: int = Field(..., ge=0, le=100)
    level: RISK_LEVELS
    level_floor: RISK_LEVELS
    confidence: float = Field(..., ge=0.0, le=1.0)
    critical_lock: bool = False


class DetectionResult(BaseModel):
    """成功返回顶层结构."""

    success: Literal[True] = True
    trace_id: str
    rule_version: str
    risk: RiskResult
    evidence: list[EvidenceItem]
    summary: dict[str, Any]
    normalization: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    suppressed_evidence: list[EvidenceItem] | None = Field(default=None)


class ErrorResult(BaseModel):
    """错误返回顶层结构."""

    success: Literal[False] = False
    trace_id: str
    error: dict[str, Any]


def generate_trace_id() -> str:
    """生成 16 字符十六进制 trace_id."""
    return uuid.uuid4().hex[:16]
