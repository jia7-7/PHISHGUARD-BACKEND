"""
=============================================================================
Pydantic 数据模型 — 严格特征信任边界
=============================================================================
extra="forbid" / 严格枚举 / 长度限制
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceType(str, Enum):
    suspicious_link = "suspicious_link"
    sender_anomaly = "sender_anomaly"
    domain_anomaly = "domain_anomaly"
    info_request = "info_request"
    urgency = "urgency"
    attachment_risk = "attachment_risk"
    language_issue = "language_issue"
    # v5.11.5: Extended semantic types for common phishing patterns
    credential_request = "credential_request"
    payment_request = "payment_request"
    impersonation = "impersonation"
    secrecy = "secrecy"
    reward_lure = "reward_lure"
    other = "other"

# v5.11.5: Allowlist alias normalization map for EvidenceType (case/whitespace-insensitive)
# Unknown values NOT in this map → rejected (enters schema retry)
EVIDENCE_TYPE_ALIASES: Dict[str, str] = {
    # Canonical → canonical (identity)
    "suspicious_link": "suspicious_link",
    "sender_anomaly": "sender_anomaly",
    "domain_anomaly": "domain_anomaly",
    "info_request": "info_request",
    "urgency": "urgency",
    "attachment_risk": "attachment_risk",
    "language_issue": "language_issue",
    "credential_request": "credential_request",
    "payment_request": "payment_request",
    "impersonation": "impersonation",
    "secrecy": "secrecy",
    "reward_lure": "reward_lure",
    "other": "other",
    # Common case/spacing variants → canonical
    "suspicious link": "suspicious_link",
    "Suspicious Link": "suspicious_link",
    "sender anomaly": "sender_anomaly",
    "Sender Anomaly": "sender_anomaly",
    "domain anomaly": "domain_anomaly",
    "Domain Anomaly": "domain_anomaly",
    "info request": "info_request",
    "Info Request": "info_request",
    "attachment risk": "attachment_risk",
    "Attachment Risk": "attachment_risk",
    "language issue": "language_issue",
    "Language Issue": "language_issue",
    "credential request": "credential_request",
    "Credential Request": "credential_request",
    "payment request": "payment_request",
    "Payment Request": "payment_request",
    "reward lure": "reward_lure",
    "Reward Lure": "reward_lure",
}


def normalize_evidence_type(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """v5.11.5: Normalize evidence type strings via allowlist alias map.
    Returns (canonical_value, original_value_if_normalized_or_None).
    - Case/whitespace normalization is safe and silent.
    - If raw is already canonical → (canonical, None).
    - If raw maps via alias → (canonical, from_value).
    - If raw is None → (None, None) — upstream must reject.
    - If raw is unmapped unknown → (None, None) — upstream must reject.
    Never returns the raw value as canonical if it's not in the alias map."""
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, None
    # Direct match (canonical or known alias)
    stripped = raw.strip()
    if stripped in EVIDENCE_TYPE_ALIASES:
        canonical = EVIDENCE_TYPE_ALIASES[stripped]
        if canonical == stripped:
            return canonical, None  # Already canonical
        return canonical, stripped  # Normalized alias
    # Case-insensitive match against canonical values
    lowered = stripped.lower()
    for canonical in EvidenceType:
        if canonical.value.lower() == lowered:
            if canonical.value == stripped:
                return canonical.value, None
            return canonical.value, stripped
    # Case-insensitive match against alias keys
    for alias, canonical in EVIDENCE_TYPE_ALIASES.items():
        if alias.lower() == lowered:
            return canonical, stripped
    # Unknown — reject
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# v5.11.5.7: Conservative enum fallback — unknown EvidenceType → "other"
# ═══════════════════════════════════════════════════════════════════════════

_HIGH_RISK_TYPES = {"credential_request", "payment_request", "impersonation",
                     "secrecy", "urgency", "suspicious_link", "domain_anomaly",
                     "sender_anomaly", "info_request", "attachment_risk",
                     "reward_lure"}
_SENSITIVE_PATTERNS = re.compile(r'(sk-|api[_-]?key|token|bearer|secret|authorization)', re.IGNORECASE)
_CANDIDATE_PATTERN = re.compile(r'^[a-z][a-z0-9_ -]{0,31}$')


def _sanitize_enum_candidate(raw_value: Any) -> str:
    """v5.11.5.7: Security-audit-safe recording of unknown enum candidate values.
    Returns either the sanitized value or a redacted placeholder."""
    if not isinstance(raw_value, str):
        return f"<redacted:non_string:{len(str(raw_value))}:sha256:...>"
    stripped = raw_value.strip()
    lowered = stripped.lower()
    # Reject sensitive keywords
    if _SENSITIVE_PATTERNS.search(lowered):
        import hashlib
        h = hashlib.sha256(lowered.encode()).hexdigest()[:12]
        return f"<redacted:sensitive:{h}:len{len(lowered)}>"
    # Reject non-ASCII, empty, or too long
    if len(lowered) > 31 or len(lowered) == 0:
        import hashlib
        h = hashlib.sha256(lowered.encode()).hexdigest()[:12]
        return f"<redacted:length:{h}:len{len(lowered)}>"
    if not lowered.isascii():
        import hashlib
        h = hashlib.sha256(lowered.encode()).hexdigest()[:12]
        return f"<redacted:non_ascii:{h}:len{len(lowered)}>"
    # Must match safe pattern
    if not _CANDIDATE_PATTERN.match(lowered):
        import hashlib
        h = hashlib.sha256(lowered.encode()).hexdigest()[:12]
        return f"<redacted:pattern:{h}:len{len(lowered)}>"
    return lowered


def _is_pure_enum_validation_error(errors: list, parsed: Optional[dict] = None) -> bool:
    """v5.11.5.9: Check if ALL Pydantic validation errors are strictly
    enum type errors on raw_evidence[*].type paths with valid unknown candidates.

    STRICT CONDITIONS (all must pass for every error):
    1. error type == "enum" (strict equality)
    2. len(loc) == 3 (exactly 3, not >= 3)
    3. loc[0] == "raw_evidence"
    4. loc[1] is int (not bool, not float)
    5. loc[2] == "type"
    6. If parsed dict provided: original type at that index must be a non-empty
       non-whitespace string, not int/float/bool/list/dict/None, AND
       normalize_evidence_type() must return (None, None) — truly unknown.

    Returns True only if there's at least one such error and NO other errors."""
    if not errors:
        return False
    raw_evidence = None
    if parsed is not None:
        raw_evidence = parsed.get("raw_evidence")
        if not isinstance(raw_evidence, list):
            raw_evidence = None
    has_enum_error = False
    for err in errors:
        etype = err.get("type", "")
        loc = tuple(err.get("loc", []))
        # Condition 1-3: Must be enum error on raw_evidence[index].type
        if etype != "enum":
            return False
        if len(loc) != 3:
            return False
        if loc[0] != "raw_evidence":
            return False
        if loc[2] != "type":
            return False
        # Condition 4: loc[1] must be int (not bool — isinstance(True,int)==True)
        idx = loc[1]
        if type(idx) is not int:
            return False
        # Condition 5-6: Validate candidate value from parsed payload
        if raw_evidence is not None and 0 <= idx < len(raw_evidence):
            ev = raw_evidence[idx]
            if isinstance(ev, dict):
                raw_type = ev.get("type")
                # Reject non-string types: int, float, bool, list, dict, None
                if not isinstance(raw_type, str):
                    return False
                stripped = raw_type.strip()
                if not stripped:
                    return False
                # normalize_evidence_type must return (None, None) — truly unknown
                canonical, _ = normalize_evidence_type(stripped)
                if canonical is not None:
                    return False
        has_enum_error = True
    return has_enum_error


def _extract_unknown_enum_paths_and_candidates(errors: list, parsed: dict) -> Tuple[List[str], List[str]]:
    """v5.11.5.9: Extract (path_strings, candidate_strings) from pure enum validation errors.
    Candidate values come from the parsed JSON payload at the error location."""
    paths = []
    candidates = []
    raw_evidence = parsed.get("raw_evidence")
    if not isinstance(raw_evidence, list):
        return paths, candidates
    for err in errors:
        etype = err.get("type", "")
        loc = tuple(err.get("loc", []))
        if etype == "enum" and len(loc) == 3 and loc[0] == "raw_evidence" and loc[2] == "type":
            idx = loc[1]
            if type(idx) is int and 0 <= idx < len(raw_evidence):
                ev = raw_evidence[idx]
                if isinstance(ev, dict):
                    raw_type = ev.get("type")
                    path_str = f"enum:raw_evidence.{idx}.type"
                    paths.append(path_str)
                    candidates.append(_sanitize_enum_candidate(raw_type))
    return paths, candidates


def attempt_conservative_enum_fallback(parsed: dict, errors: list) -> Tuple[Optional[Any], List[str], List[str]]:
    """v5.11.5.9: Conservative enum fallback — convert unknown EvidenceType values
    to 'other' under strict conditions.

    Returns (validated_response, unknown_paths, unknown_candidates) or (None, [], []).

    STRICT CONDITIONS (all must pass):
    1. ALL Pydantic errors are purely enum:raw_evidence[*].type (via _is_pure_enum_validation_error)
       - error type == "enum" (strict equality)
       - len(loc) == 3 (exact)
       - loc[0] == "raw_evidence", loc[1] is int, loc[2] == "type"
       - Original type must be non-empty string, not int/float/bool/list/dict/None
       - normalize_evidence_type() must return (None, None) — truly unknown
    2. raw_evidence is a valid array
    3. No other schema errors exist (severity, quote, extra, null, type mismatches)
    4. After conversion to 'other', the FULL response passes RawLLMResponse.model_validate()

    CRITICAL RULES:
    - Unknown types → 'other' ONLY (never credential_request, payment_request, etc.)
    - 'other' evidence MUST NOT trigger promotion, rule score, or high-risk upgrade
    - Works on a deep copy — never mutates the original parsed dict
    - quote must still pass original text matching (unverified if unmatched)
    """
    import copy
    # v5.11.5.9: Pass parsed dict for candidate value validation
    if not _is_pure_enum_validation_error(errors, parsed):
        return None, [], []

    raw_evidence = parsed.get("raw_evidence")
    if not isinstance(raw_evidence, list) or len(raw_evidence) == 0:
        return None, [], []

    # Extract paths and candidates BEFORE modification (security audit)
    unknown_paths, unknown_candidates = _extract_unknown_enum_paths_and_candidates(errors, parsed)

    # Deep copy and convert unknown types to "other"
    fixed = copy.deepcopy(parsed)
    fixed_evidence = fixed.get("raw_evidence", [])
    for err in errors:
        loc = tuple(err.get("loc", []))
        if len(loc) == 3 and loc[0] == "raw_evidence" and loc[2] == "type":
            idx = loc[1]
            if type(idx) is int and 0 <= idx < len(fixed_evidence):
                ev = fixed_evidence[idx]
                if isinstance(ev, dict):
                    # v5.11.5.9: _is_pure_enum_validation_error already confirmed
                    # the original type is a non-empty string AND normalize_evidence_type
                    # returns (None, None). But double-check as defense-in-depth.
                    original = ev.get("type", "")
                    if not isinstance(original, str) or not original.strip():
                        continue  # Not a valid string candidate — skip
                    canonical, _ = normalize_evidence_type(original)
                    if canonical is not None:
                        # Already known — shouldn't reach here; skip as safety
                        continue
                    ev["type"] = "other"

    # Re-validate with Pydantic
    try:
        validated = RawLLMResponse.model_validate(fixed)
        return validated, unknown_paths, unknown_candidates
    except Exception:
        return None, [], []


class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class EvidenceSource(str, Enum):
    llm = "llm"
    deterministic_url = "deterministic_url"
    deterministic_header = "deterministic_header"
    keyword = "keyword"
    html_parser = "html_parser"


class RawLLMEvidence(BaseModel):
    """LLM输出的原始证据 — 仅LLM可输出字段，不含source/sources/verification等系统字段"""
    quote: str = Field(..., min_length=1, max_length=500)
    type: EvidenceType
    severity: Severity
    explanation: str = Field(..., min_length=1, max_length=500)
    model_config = {"extra": "forbid"}

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_evidence_type(cls, value: Any) -> Any:
        """v5.11.5: Allowlist alias normalization for evidence type.
        - None: strictly reject (must remain strict boundary).
        - Canonical: pass through.
        - Known alias (case/whitespace): normalize to canonical.
        - Unknown value: pass to Pydantic enum validation → rejected."""
        if value is None:
            return value  # Pydantic will reject missing required field
        canonical, _from = normalize_evidence_type(value)
        if canonical is not None:
            return canonical
        # Unknown — let Pydantic enum validation reject it
        return value

    def to_evidence(self) -> "Evidence":
        """由可信代码转换为正式Evidence（source由系统注入）"""
        return Evidence(
            quote=self.quote, type=self.type, severity=self.severity,
            explanation=self.explanation, source=EvidenceSource.llm,
            sources=[EvidenceSource.llm], verification="pending")


class Evidence(BaseModel):
    """单条证据 — 严格验证（系统完整证据，含source/sources/verification等）"""
    quote: str = Field(..., min_length=1, max_length=500)
    type: EvidenceType
    severity: Severity
    explanation: str = Field(..., min_length=1, max_length=500)
    source: EvidenceSource
    sources: List[EvidenceSource] = Field(default_factory=list)
    verification: str = Field(default="pending")  # matched|partial|unmatched|deterministic
    ocr_block_id: Optional[int] = None
    bbox: Optional[List[List[float]]] = None
    ocr_confidence: Optional[float] = None
    ocr_uncertain: bool = False
    rule_ids: List[str] = Field(default_factory=list)
    dedup_key: str = Field(default="")

    model_config = {"extra": "forbid"}

    @field_validator("bbox")
    @classmethod
    def bbox_max_8(cls, v):
        if v is not None and len(v) > 8:
            raise ValueError("bbox must have at most 8 points")
        return v


class SenderFeature(BaseModel):
    display_name: str = Field(default="unknown", max_length=200)
    address: str = Field(default="unknown", max_length=200)
    reply_to: str = Field(default="unknown", max_length=200)
    claimed_identity: str = Field(default="unknown", max_length=200)
    domain_mismatch: bool = False
    notes: str = Field(default="", max_length=500)
    verified: bool = False  # 确定性校验后才为True
    model_config = {"extra": "forbid"}

    @field_validator("display_name", "address", "reply_to", "claimed_identity", mode="before")
    @classmethod
    def _null_to_unknown(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"unknown" for descriptive sender strings."""
        return "unknown" if value is None else value

    @field_validator("notes", mode="before")
    @classmethod
    def _null_notes_to_empty(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"" for optional sender notes."""
        return "" if value is None else value


class URLFeature(BaseModel):
    url: str = Field(..., max_length=2000)
    text: str = Field(default="", max_length=500)
    registered_domain: str = Field(default="unknown", max_length=200)
    issues: List[str] = Field(default_factory=list, max_length=20)
    is_llm_output: bool = True  # LLM输出的URL为True，确定性重新分析后为False
    model_config = {"extra": "forbid"}

    @field_validator("text", mode="before")
    @classmethod
    def _null_text_to_empty(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"" for URL display text."""
        return "" if value is None else value

    @field_validator("registered_domain", mode="before")
    @classmethod
    def _null_domain_to_unknown(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"unknown" for URL registered_domain."""
        return "unknown" if value is None else value


class ContentFeature(BaseModel):
    urgency_indicators: List[str] = Field(default_factory=list, max_length=20)
    threats: List[str] = Field(default_factory=list, max_length=20)
    info_requests: List[str] = Field(default_factory=list, max_length=20)
    greeting: str = Field(default="unknown", max_length=200)
    signature: str = Field(default="unknown", max_length=200)
    model_config = {"extra": "forbid"}

    @field_validator("greeting", "signature", mode="before")
    @classmethod
    def _null_to_unknown(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"unknown" for content greeting/signature."""
        return "unknown" if value is None else value


class LanguageFeature(BaseModel):
    errors: List[str] = Field(default_factory=list, max_length=20)
    inconsistencies: List[str] = Field(default_factory=list, max_length=20)
    translation_quality: str = Field(default="unknown", max_length=50)
    model_config = {"extra": "forbid"}

    @field_validator("translation_quality", mode="before")
    @classmethod
    def _null_translation_to_unknown(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"unknown" for translation_quality."""
        return "unknown" if value is None else value


class AttachmentFeature(BaseModel):
    filename: str = Field(default="unknown", max_length=200)
    extension: str = Field(default="", max_length=20)
    risk_factors: List[str] = Field(default_factory=list, max_length=20)
    model_config = {"extra": "forbid"}

    @field_validator("filename", mode="before")
    @classmethod
    def _null_filename_to_unknown(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"unknown" for attachment filename."""
        return "unknown" if value is None else value

    @field_validator("extension", mode="before")
    @classmethod
    def _null_extension_to_empty(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"" for attachment extension."""
        return "" if value is None else value


class RawFeatures(BaseModel):
    """LLM输出的原始特征 — 不能直接评分"""
    sender: SenderFeature = Field(default_factory=SenderFeature)
    urls: List[URLFeature] = Field(default_factory=list, max_length=50)
    content: ContentFeature = Field(default_factory=ContentFeature)
    language: LanguageFeature = Field(default_factory=LanguageFeature)
    attachments: List[AttachmentFeature] = Field(default_factory=list, max_length=20)
    overall_impression: str = Field(default="", max_length=500)
    model_config = {"extra": "forbid"}

    @field_validator("overall_impression", mode="before")
    @classmethod
    def _null_impression_to_empty(cls, value: Any) -> Any:
        """v5.11.4: Allowlist null→"" for overall_impression."""
        return "" if value is None else value


class ValidatedFeatureSet(BaseModel):
    """经过确定性校验后的特征集 — 可以进入RiskScorer"""
    sender_credibility_score: Optional[int] = Field(default=None, ge=0, le=100)
    sender_credibility_notes: List[str] = Field(default_factory=list)
    link_safety_score: Optional[int] = Field(default=None, ge=0, le=100)
    link_safety_notes: List[str] = Field(default_factory=list)
    content_urgency_score: Optional[int] = Field(default=None, ge=0, le=100)
    information_request_score: Optional[int] = Field(default=None, ge=0, le=100)
    language_quality_score: Optional[int] = Field(default=None, ge=0, le=100)
    attachment_risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    model_config = {"extra": "forbid"}


class ScoreResultModel(BaseModel):
    risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    risk_level: Optional[str] = None
    risk_level_label: Optional[str] = None
    is_phishing: Optional[bool] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: str = Field(default="weighted")
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    # v5.11.5.1: promotion tracking
    promotion_applied: bool = Field(default=False)
    promotion_rule: str = Field(default="")
    pre_promotion_score: Optional[int] = Field(default=None)
    promotion_score: Optional[int] = Field(default=None)
    promotion_evidence_types: List[str] = Field(default_factory=list)
    # v5.11.5.2: deterministic anchor tracking
    promotion_anchor_source: str = Field(default="")
    promotion_anchor_rule_ids: List[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}


class AnalysisStatus(str, Enum):
    success = "success"
    insufficient_evidence = "insufficient_evidence"
    error = "error"


class LLMSampleStatus(str, Enum):
    """v5.11.1: Per-sample LLM outcome — read from actual response, never hardcoded from use_llm."""
    not_requested = "not_requested"
    not_configured = "not_configured"
    usable_first_attempt = "usable_first_attempt"
    recovered_after_validation_retry = "recovered_after_validation_retry"
    recovered_after_conservative_enum_fallback = "recovered_after_conservative_enum_fallback"
    terminal_validation_failed = "terminal_validation_failed"
    terminal_transport_failed = "terminal_transport_failed"


@dataclass(frozen=True)
class LLMCallTelemetry:
    """v5.11.2: Per-logical-call telemetry returned by chat() on success,
    carried by LLMTransportError on terminal transport failure.
    All counts are from a SINGLE logical LLM call (attempt1 or schema retry)."""
    text: str = ""
    request_attempt_count: int = 0
    http_success_count: int = 0
    http_error_count: int = 0
    transport_retry_count: int = 0
    json_mode_attempted: bool = False
    json_mode_fallback_triggered: bool = False
    json_mode_fallback_count: int = 0
    latencies_ms: Tuple[float, ...] = ()


class RawLLMResponse(BaseModel):
    """LLM完整响应 — 一次性严格验证，evidence不含source字段"""
    raw_features: RawFeatures = Field(default_factory=RawFeatures)
    raw_evidence: List[RawLLMEvidence] = Field(default_factory=list, max_length=100)
    uncertainties: List[str] = Field(default_factory=list, max_length=20)
    model_config = {"extra": "forbid"}


class AnalysisResponse(BaseModel):
    """统一API响应"""
    analysis_id: str
    status: AnalysisStatus
    score_result: ScoreResultModel = Field(default_factory=ScoreResultModel)
    verified_evidence: List[Evidence] = Field(default_factory=list)
    unverified_evidence: List[Evidence] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    handling_suggestion: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_used: str = ""
    warnings: List[Dict[str, str]] = Field(default_factory=list)
    # v5.11.2: per-sample LLM telemetry from real LLMCallTelemetry summation
    llm_sample_status: str = Field(default="not_requested")
    llm_request_count_for_sample: int = Field(default=0, ge=0)
    llm_http_success_count_for_sample: int = Field(default=0, ge=0)
    llm_http_error_count_for_sample: int = Field(default=0, ge=0)
    llm_transport_retry_count_for_sample: int = Field(default=0, ge=0)
    llm_schema_retry_count_for_sample: int = Field(default=0, ge=0)
    llm_invalid_response_count_for_sample: int = Field(default=0, ge=0)
    llm_json_mode_fallback_count_for_sample: int = Field(default=0, ge=0)
    llm_validation_error_types: List[str] = Field(default_factory=list)
    llm_total_latency_ms: float = Field(default=0.0, ge=0.0)
    fallback_used: bool = False
    fallback_reason: str = Field(default="")
    # v5.11.4: null normalization audit fields
    llm_null_normalization_count_for_sample: int = Field(default=0, ge=0)
    llm_normalized_field_paths: List[str] = Field(default_factory=list)
    # v5.11.5: enum normalization audit fields (EvidenceType alias/case normalization)
    llm_enum_normalization_count_for_sample: int = Field(default=0, ge=0)
    llm_enum_normalized_paths: List[str] = Field(default_factory=list)
    # v5.11.5: deterministic fallback tracking
    llm_fallback_to_deterministic: bool = Field(default=False)
    # v5.11.5.7: conservative enum fallback and unknown enum security audit
    llm_unknown_enum_fallback_count_for_sample: int = Field(default=0, ge=0)
    llm_unknown_enum_paths: List[str] = Field(default_factory=list)
    llm_unknown_enum_candidates: List[str] = Field(default_factory=list)
    llm_conservative_enum_fallback_used: bool = Field(default=False)
    # v5.11.5.12: OCR URL trust boundary audit fields
    deterministic_url_candidates_seen: int = Field(default=0, ge=0)
    deterministic_url_accepted_count: int = Field(default=0, ge=0)
    deterministic_url_rejected_count: int = Field(default=0, ge=0)
    deterministic_url_rejected_reasons: List[str] = Field(default_factory=list)
    deterministic_urls_used: List[str] = Field(default_factory=list)
    ocr_url_trust_policy: str = Field(default="")
    model_config = {"extra": "forbid"}
