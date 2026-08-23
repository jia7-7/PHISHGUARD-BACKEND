"""Unified AnalysisPipeline — deterministic parsing, LLM extraction, validation, scoring"""
import hashlib, json, logging, os, re, threading, time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ANALYSIS_PROMPT_TEMPLATE, API_RETRY_CONFIG, LLM_CONFIG,
    LLM_SCHEMA_RETRY_MAX, LLM_JSON_MODE_ENABLED,
    MAX_HTTP_ATTEMPTS_PER_SAMPLE,
    PROMPT_HASH, PROMPT_VERSION, PROMPT_BUNDLE_HASH,
    SCHEMA_REPAIR_PROMPT_TEMPLATE, SCHEMA_REPAIR_PROMPT_VERSION, SCHEMA_REPAIR_PROMPT_HASH,
    SYSTEM_PROMPT, check_llm_configured,
    analyze_url_safety, extract_urls, check_email_alignment,
    parse_auth_results, parse_html_links, check_display_url_mismatch,
    get_registered_domain,
    # v5.11.5: LLM fail-fast + schema repair v3
    _validate_llm_config_strict, LLMConfigError, build_field_hints,
)
from models import (
    ValidatedFeatureSet, RawFeatures, RawLLMResponse, Evidence,
    EvidenceType, Severity, EvidenceSource, ScoreResultModel,
    AnalysisStatus, AnalysisResponse, LLMCallTelemetry,
    # v5.11.5: evidence type alias normalization
    normalize_evidence_type,
)
from risk_scorer import RiskScorer, canonical_evidence_key
from config import is_sdu_official_hostname

logger = logging.getLogger("pipeline")

EVIDENCE_SOURCE_LLM = "llm"
EVIDENCE_SOURCE_DET_URL = "deterministic_url"
EVIDENCE_SOURCE_DET_HDR = "deterministic_header"
EVIDENCE_SOURCE_KW = "keyword"
EVIDENCE_SOURCE_HTML = "html_parser"

SCORABLE_SOURCES = {EVIDENCE_SOURCE_LLM, EVIDENCE_SOURCE_DET_URL,
                    EVIDENCE_SOURCE_DET_HDR, EVIDENCE_SOURCE_KW, EVIDENCE_SOURCE_HTML}
RULE_ONLY_SOURCES = {EVIDENCE_SOURCE_DET_URL, EVIDENCE_SOURCE_DET_HDR,
                     EVIDENCE_SOURCE_KW, EVIDENCE_SOURCE_HTML}

# v5.11.5.12: Strict URL syntax validator for OCR trust boundary
# Only URLs matching this pattern are eligible for deterministic scoring from OCR
_STRICT_URL_RE = re.compile(
    r'^https?://[a-zA-Z0-9](?:[-a-zA-Z0-9]*[a-zA-Z0-9])?'
    r'(?:\.[a-zA-Z0-9](?:[-a-zA-Z0-9]*[a-zA-Z0-9])?)*'
    r'\.[a-zA-Z]{2,}'
    r'(?:/[\w\-._~:/?#\[\]@!$&()*+,;=%]*)?$',
    re.IGNORECASE,
)

SOURCE_PRIORITY = {"deterministic_url": 5, "deterministic_header": 5,
                   "html_parser": 4, "keyword": 2, "llm": 1}


# ═══════════════════════════════════════════════════════════════════════════════
# v5.11.3: Structured result from unified parse+validate helper
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class LLMParseResult:
    """Result of parsing and validating LLM response text.
    Exactly one outcome: success, empty, json_parse_error, or validation_error."""
    success: bool = False
    raw_features: Dict[str, Any] = None
    raw_llm_evidence: List[Dict] = None
    uncertainties: List[str] = None
    error_type: str = ""            # "empty_response" | "json_parse_error" | "validation_error" | ""
    validation_error_types: List[str] = None
    # v5.11.4: null normalization audit
    normalized_field_paths: List[str] = None
    null_normalization_count: int = 0
    # v5.11.5: enum normalization audit (EvidenceType alias/case normalization)
    enum_normalized_paths: List[str] = None
    enum_normalization_count: int = 0
    # v5.11.5.7: Parsed JSON payload preserved for conservative enum fallback
    # Only set on validation_error; never on empty_response or json_parse_error.
    # Never logged, cached, serialized to manifest/raw predictions, or shared across threads.
    parsed_payload: Optional[Dict[str, Any]] = None
    # v5.11.5.9: Raw Pydantic ValidationError.errors() list for conservative enum fallback.
    # Only set on validation_error. Each element is a dict with "type","loc","msg","input".
    # Never logged, cached, serialized, or shared across threads.
    pydantic_errors: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.raw_features is None:
            self.raw_features = {}
        if self.raw_llm_evidence is None:
            self.raw_llm_evidence = []
        if self.uncertainties is None:
            self.uncertainties = []
        if self.validation_error_types is None:
            self.validation_error_types = []
        if self.normalized_field_paths is None:
            self.normalized_field_paths = []
        if self.enum_normalized_paths is None:
            self.enum_normalized_paths = []


def _norm(text: str) -> str:
    t = text.strip(); t = re.sub(r'\s+', '', t)
    for f,h in [('：',':'),('（','('),('）',')'),('，',','),('。','.'),('！','!'),('？','?'),('；',';'),('＠','@'),('．','.')]:
        t = t.replace(f,h)
    return t.lower()


def _clean_validation_errors(e: Exception) -> List[str]:
    """v5.11.1: Extract clean error type/field path from Pydantic ValidationError.
    Never leaks full exceptions, stacks, API keys, or response content."""
    result: List[str] = []
    if hasattr(e, 'errors') and callable(getattr(e, 'errors', None)):
        try:
            for err in e.errors()[:3]:
                etype = err.get('type', 'unknown')
                loc = '.'.join(str(x) for x in err.get('loc', []))
                result.append(f"{etype}:{loc}" if loc else etype)
        except Exception:
            result.append(f"{type(e).__name__}")
    else:
        result.append(f"{type(e).__name__}")
    return result


def _detect_null_normalizations(data: dict) -> Tuple[List[str], int]:
    """v5.11.4: Pre-scan raw JSON at allowlist paths to record which fields
    will be null-normalized by Pydantic field_validators.
    Read-only — does NOT modify the input dict. Returns (sorted paths, count)."""
    paths: List[str] = []
    rf = data.get("raw_features")
    if not isinstance(rf, dict):
        return [], 0

    # Sender fields
    sender = rf.get("sender")
    if isinstance(sender, dict):
        for field in ["display_name", "address", "reply_to", "claimed_identity"]:
            if sender.get(field) is None:
                paths.append(f"raw_features.sender.{field}")
        if sender.get("notes") is None:
            paths.append("raw_features.sender.notes")

    # URL array items
    urls = rf.get("urls")
    if isinstance(urls, list):
        for i, u in enumerate(urls):
            if isinstance(u, dict):
                if u.get("text") is None:
                    paths.append(f"raw_features.urls[{i}].text")
                if u.get("registered_domain") is None:
                    paths.append(f"raw_features.urls[{i}].registered_domain")

    # Content fields
    content = rf.get("content")
    if isinstance(content, dict):
        for field in ["greeting", "signature"]:
            if content.get(field) is None:
                paths.append(f"raw_features.content.{field}")

    # Language
    language = rf.get("language")
    if isinstance(language, dict):
        if language.get("translation_quality") is None:
            paths.append("raw_features.language.translation_quality")

    # Attachment array items
    attachments = rf.get("attachments")
    if isinstance(attachments, list):
        for i, a in enumerate(attachments):
            if isinstance(a, dict):
                if a.get("filename") is None:
                    paths.append(f"raw_features.attachments[{i}].filename")
                if a.get("extension") is None:
                    paths.append(f"raw_features.attachments[{i}].extension")

    # Overall impression
    if rf.get("overall_impression") is None:
        paths.append("raw_features.overall_impression")

    return sorted(paths), len(paths)


def _detect_enum_normalizations(data: dict) -> Tuple[List[str], int]:
    """v5.11.5: Pre-scan raw JSON evidence array to record which type values
    will be normalized by the EvidenceType alias map (case/whitespace only).
    Read-only — does NOT modify the input dict. Returns (sorted paths, count)."""
    paths: List[str] = []
    raw_evidence = data.get("raw_evidence")
    if not isinstance(raw_evidence, list):
        return [], 0

    for i, ev in enumerate(raw_evidence):
        if not isinstance(ev, dict):
            continue
        raw_type = ev.get("type")
        if raw_type is None:
            continue
        canonical, from_val = normalize_evidence_type(raw_type)
        if canonical is not None and from_val is not None:
            # Was normalized from a non-canonical alias
            paths.append(f"raw_evidence[{i}].type: '{from_val}' -> '{canonical}'")

    return sorted(paths), len(paths)


def _dedup_cross_source(evidence_list: List[Dict]) -> List[Dict]:
    """跨来源去重：SHA-256规范化quote+type，确定性优先级/severity"""
    groups = {}
    for ev in evidence_list:
        key = canonical_evidence_key(ev)
        if key not in groups:
            groups[key] = {"merged": dict(ev), "sources": [], "det_sev": None, "llm_sev": None}
        g = groups[key]
        src = ev.get("source",""); g["sources"].append(src)
        sev = ev.get("severity","low")
        sev_order = {"high":3,"medium":2,"low":1}
        if src in ("deterministic_url","deterministic_header","html_parser"):
            cur = sev_order.get(g["det_sev"],0) if g["det_sev"] else 0
            if sev_order.get(sev,0) > cur: g["det_sev"] = sev
        elif src == "llm":
            cur = sev_order.get(g["llm_sev"],0) if g["llm_sev"] else 0
            if sev_order.get(sev,0) > cur: g["llm_sev"] = sev
    result = []
    for g in groups.values():
        merged = g["merged"]
        merged["sources"] = list(dict.fromkeys(g["sources"]))
        # 确定性severity优先，LLM不得升级
        if g["det_sev"]:
            merged["severity"] = g["det_sev"]
            merged["source"] = next((s for s in merged["sources"] if s in ("deterministic_url","deterministic_header","html_parser","keyword")), merged["sources"][0])
        elif g["llm_sev"]:
            merged["severity"] = g["llm_sev"]
            merged["source"] = "llm"
        result.append(merged)
    return result


class LLMTransportError(RuntimeError):
    """v5.11.2: Terminal transport failure carrying telemetry for the failed logical call."""
    def __init__(self, message: str, telemetry: LLMCallTelemetry):
        super().__init__(message)
        self.telemetry = telemetry


class LLMClient:
    def __init__(self, provider=None):
        self.provider = provider or LLM_CONFIG["provider"]
        cfg = LLM_CONFIG.get(self.provider,{})
        self.config = cfg; self.retry = API_RETRY_CONFIG; self._client = None
        self.vendor = LLM_CONFIG.get("vendor", os.getenv("LLM_VENDOR", "unknown"))
        self.model = cfg.get("model", "unknown")
        # v5.11: per-run statistics (kept for backward compat)
        self.http_success_count = 0
        self.usable_response_count = 0
        self.failure_count = 0
        self.validation_failure_count = 0
        self.cache_hit_count = 0
        self.latencies: list = []
        self.base_host = ""
        # v5.11.1: redefined LLM statistics (see docs for invariant definitions)
        self.sample_requested_count = 0
        self.request_attempt_count = 0
        self.transport_retry_count = 0
        self.schema_retry_count = 0
        self.invalid_response_count = 0
        self.recovered_after_validation_count = 0
        self.terminal_validation_failure_count = 0
        self.terminal_transport_failure_count = 0
        # v5.11.1: JSON mode support tracking
        self.json_mode_used: bool = False
        self.json_mode_fallback_triggered: bool = False
        # v5.11.2: per-run JSON mode fallback count + capability cache
        self.json_mode_fallback_count: int = 0
        self._json_mode_supported: Optional[bool] = None  # None=unknown, True=supported, False=unsupported
        # v5.11.4: null normalization audit (thread-safe via _safe_incr)
        self.null_normalization_count: int = 0
        self.samples_with_null_normalization_count: int = 0
        # v5.11.5: enum normalization audit (EvidenceType alias/case normalization)
        self.enum_normalization_count: int = 0
        self.samples_with_enum_normalization_count: int = 0
        # v5.11.5: pipeline success vs LLM usable separation
        self.pipeline_success_count: int = 0
        self.deterministic_fallback_count: int = 0
        # v5.11.5.7: conservative enum fallback counters (thread-safe via _safe_incr)
        self.unknown_enum_fallback_count: int = 0
        self.samples_with_unknown_enum_fallback_count: int = 0
        self.unknown_enum_candidate_counts: Dict[str, int] = {}
        # v5.11.2: thread-safe aggregation lock
        self._lock = threading.Lock()
        try:
            from urllib.parse import urlparse as _up
            bu = cfg.get("base_url", "")
            if bu: self.base_host = _up(bu).hostname or ""
        except: pass

    def _get_client(self):
        if self._client: return self._client
        if self.provider == "anthropic":
            import anthropic
            kw = {"api_key": self.config.get("api_key",""), "timeout": self.retry["timeout"]}
            if self.config.get("base_url"): kw["base_url"] = self.config["base_url"]
            self._client = anthropic.Anthropic(**kw)
        elif self.provider in ("openai","local"):
            from openai import OpenAI
            kw = {"api_key": self.config.get("api_key",""), "timeout": self.retry["timeout"]}
            if self.config.get("base_url"): kw["base_url"] = self.config["base_url"]
            self._client = OpenAI(**kw)
        return self._client

    @staticmethod
    def _is_json_mode_rejection(error: Exception) -> bool:
        """v5.11.2: Check if error is specifically about response_format being unsupported.
        Must be a 400-level BadRequestError referencing response_format — never 401/403/429/5xx/connection/timeout."""
        try:
            from openai import BadRequestError
        except ImportError:
            return False
        if not isinstance(error, BadRequestError):
            return False
        status_code = getattr(error, 'status_code', None)
        if status_code is not None and status_code != 400:
            return False
        msg = str(error).lower()
        # Specific: response_format referenced with rejection language
        if 'response_format' not in msg:
            return False
        rejection_terms = ('unknown', 'unsupported', 'invalid', 'not supported', 'unrecognized')
        return any(term in msg for term in rejection_terms)

    def _safe_incr(self, **kwargs):
        """v5.11.8: Thread-safe increment of run-level counters.
        REJECTS negative deltas — counters must never decrease.
        Usage: _safe_incr(sample_requested_count=1, invalid_response_count=1, validation_failure_count=1)"""
        with self._lock:
            for attr, delta in kwargs.items():
                d = int(delta)
                if d < 0:
                    raise ValueError(
                        f"_safe_incr does not accept negative deltas: {attr}={delta}. "
                        f"Counters must only increase, never decrease."
                    )
                setattr(self, attr, getattr(self, attr, 0) + d)

    def _json_mode_is_supported(self) -> Optional[bool]:
        """v5.11.3: Thread-safe read of JSON mode capability cache."""
        with self._lock:
            return self._json_mode_supported

    def _json_mode_set_unsupported(self):
        """v5.11.3: Thread-safe write of JSON mode capability cache (None→False only)."""
        with self._lock:
            if self._json_mode_supported is None:
                self._json_mode_supported = False

    def _json_mode_set_supported(self):
        """v5.11.3: Thread-safe write of JSON mode capability cache (None→True only)."""
        with self._lock:
            if self._json_mode_supported is None:
                self._json_mode_supported = True

    def _accumulate(self, t: LLMCallTelemetry):
        """v5.11.2: Thread-safe accumulation of per-call telemetry into run-level stats."""
        with self._lock:
            self.http_success_count += t.http_success_count
            self.request_attempt_count += t.request_attempt_count
            self.transport_retry_count += t.transport_retry_count
            if t.json_mode_attempted:
                self.json_mode_used = True
            if t.json_mode_fallback_triggered:
                self.json_mode_fallback_triggered = True
            self.json_mode_fallback_count += t.json_mode_fallback_count
            self.latencies.extend(t.latencies_ms)

    def chat(self, system_prompt, user_message, use_json_mode: bool = False) -> LLMCallTelemetry:
        """v5.11.3: Transport retry + JSON mode real capability fallback.
        Returns LLMCallTelemetry on success.
        Raises LLMTransportError(telemetry) on terminal transport failure.
        Each logical call bounded by (max_retries + 1) HTTP attempts.
        v5.11.3 P0 fixes:
          - Defect 1: Terminal telemetry accumulated before raise
          - Defect 3: JSON rejection counted as http_error with independent latency"""
        temp = self.config.get("temperature",0.0); mt = self.config.get("max_tokens",4096)
        model = self.config.get("model",""); client = self._get_client()
        max_attempts = self.retry["max_retries"] + 1  # 4

        last_error = None
        request_attempt_count = 0
        http_success_count = 0
        http_error_count = 0
        transport_retry_count = 0
        json_mode_attempted = False
        json_mode_fallback_triggered = False
        json_mode_fallback_count = 0
        latencies = []
        retry_delay_index = 0

        should_try_json = use_json_mode and self.provider in ("openai", "local")

        while request_attempt_count < max_attempts:
            try:
                t_start = time.time()
                request_attempt_count += 1

                if self.provider == "anthropic":
                    resp = client.messages.create(model=model, max_tokens=mt, temperature=temp,
                        system=system_prompt, messages=[{"role":"user","content":user_message}])
                    text = "".join(b.text for b in resp.content if hasattr(b,"text"))
                else:
                    kwargs = {"model": model,
                        "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
                        "temperature": temp, "max_tokens": mt}

                    # v5.11.3: Real JSON mode capability detection + fallback with per-request accounting
                    json_attempted_this_time = False
                    json_supported = self._json_mode_is_supported()
                    if should_try_json and json_supported is not False:
                        kwargs["response_format"] = {"type": "json_object"}
                        json_mode_attempted = True
                        json_attempted_this_time = True

                    try:
                        resp = client.chat.completions.create(**kwargs)
                        # JSON mode succeeded — cache capability
                        if json_attempted_this_time and json_supported is None:
                            self._json_mode_set_supported()
                    except Exception as api_error:
                        # Check if this is a JSON mode capability rejection
                        if (json_attempted_this_time
                                and json_supported is None
                                and self._is_json_mode_rejection(api_error)):
                            # v5.11.3 FIX C: JSON capability rejection IS a real HTTP error
                            # Must count as http_error + independent latency, but NOT transport retry
                            self._json_mode_set_unsupported()
                            json_mode_fallback_triggered = True
                            json_mode_fallback_count += 1
                            # Account for the rejected request
                            http_error_count += 1
                            latencies.append((time.time() - t_start) * 1000)
                            if request_attempt_count >= max_attempts:
                                # No budget left for fallback — terminal
                                last_error = api_error
                                break  # exit while loop → terminal transport
                            # Fallback: retry without response_format within attempt budget
                            del kwargs["response_format"]
                            t_start = time.time()      # Reset latency for fallback request
                            request_attempt_count += 1  # Fallback counts as an independent HTTP attempt
                            resp = client.chat.completions.create(**kwargs)
                        else:
                            raise

                    text = resp.choices[0].message.content or ""

                http_success_count += 1
                latencies.append((time.time() - t_start) * 1000)

                telemetry = LLMCallTelemetry(
                    text=text,
                    request_attempt_count=request_attempt_count,
                    http_success_count=http_success_count,
                    http_error_count=http_error_count,
                    transport_retry_count=transport_retry_count,
                    json_mode_attempted=json_mode_attempted,
                    json_mode_fallback_triggered=json_mode_fallback_triggered,
                    json_mode_fallback_count=json_mode_fallback_count,
                    latencies_ms=tuple(latencies),
                )
                self._accumulate(telemetry)
                return telemetry

            except Exception as e:
                last_error = e
                http_error_count += 1
                latencies.append((time.time() - t_start) * 1000)
                if request_attempt_count < max_attempts:
                    transport_retry_count += 1
                    retry_delay_index += 1
                    time.sleep(self.retry["retry_delay"] * (self.retry["retry_backoff"] ** (retry_delay_index - 1)))

        # Terminal transport failure — build telemetry, accumulate, and raise
        # v5.11.3 FIX A: _accumulate BEFORE raise so telemetry enters global stats
        telemetry = LLMCallTelemetry(
            text="",
            request_attempt_count=request_attempt_count,
            http_success_count=http_success_count,
            http_error_count=http_error_count,
            transport_retry_count=transport_retry_count,
            json_mode_attempted=json_mode_attempted,
            json_mode_fallback_triggered=json_mode_fallback_triggered,
            json_mode_fallback_count=json_mode_fallback_count,
            latencies_ms=tuple(latencies),
        )
        self._accumulate(telemetry)
        self._safe_incr(failure_count=1, terminal_transport_failure_count=1)
        raise LLMTransportError(f"LLM fail: {last_error}", telemetry)


class AnalysisPipeline:
    """Unified pipeline — single entry for API/evaluate"""

    def __init__(self, use_llm: bool = True, provider=None,
                 allow_deterministic_fallback: bool = False):
        self.llm_requested = use_llm
        self.llm_configured = check_llm_configured()
        # v5.11.5: LLM fail-fast — if use_llm=true but config is incomplete, crash
        # unless --allow-deterministic-fallback is explicitly passed.
        if use_llm and not allow_deterministic_fallback:
            config_errors = _validate_llm_config_strict()
            if config_errors:
                raise LLMConfigError(
                    f"LLM configuration incomplete: {'; '.join(config_errors)}. "
                    f"Fix configuration or pass --allow-deterministic-fallback to continue with rule-only scoring."
                )
        # v5.11.2: use_llm requires BOTH requested AND configured; llm_requested alone
        # controls whether not_configured state is reachable.
        self.use_llm = use_llm and self.llm_configured
        self.allow_deterministic_fallback = allow_deterministic_fallback
        self.llm_client = LLMClient(provider=provider) if self.use_llm else None
        from config import APP_VERSION as _APP_VER
        self.version = _APP_VER
        self.prompt_version = PROMPT_VERSION
        self.prompt_hash = PROMPT_HASH
        # v5.11.2: prompt bundle covers full runtime prompts
        self.prompt_bundle_hash = PROMPT_BUNDLE_HASH

    # ═══════════════════════════════════════════════════════════════════════
    # v5.11.3: Unified parse+validate helper for LLM response text
    # Used by BOTH first-attempt AND schema-retry paths — no duplicated branches.
    # ═══════════════════════════════════════════════════════════════════════
    def _parse_and_validate_llm_text(self, text: Optional[str]) -> LLMParseResult:
        """v5.11.3: Single entry point for parsing and Pydantic-validating LLM output.
        Handles empty/whitespace/None uniformly (Defect 2 fix).
        Returns LLMParseResult with success=True only when parse AND validation pass."""
        # Empty/whitespace/None → "empty_response"
        if text is None or not text.strip():
            return LLMParseResult(
                success=False,
                error_type="empty_response",
            )

        # Attempt JSON parse
        parsed = self._parse_json(text)
        if not parsed:
            return LLMParseResult(
                success=False,
                error_type="json_parse_error",
                validation_error_types=["json_parse_error"],
            )

        # v5.11.4: Pre-scan allowlist paths for JSON null before Pydantic normalizes them
        norm_paths, norm_count = _detect_null_normalizations(parsed)
        # v5.11.5: Pre-scan evidence type aliases before Pydantic normalizes them
        enum_paths, enum_count = _detect_enum_normalizations(parsed)

        # Attempt Pydantic validation (field_validators normalize allowlist null→default + type aliases)
        try:
            validated_resp = RawLLMResponse.model_validate(parsed)
            return LLMParseResult(
                success=True,
                raw_features=validated_resp.raw_features.model_dump(),
                raw_llm_evidence=[e.to_evidence().model_dump() for e in validated_resp.raw_evidence],
                uncertainties=validated_resp.uncertainties,
                normalized_field_paths=norm_paths,
                null_normalization_count=norm_count,
                enum_normalized_paths=enum_paths,
                enum_normalization_count=enum_count,
            )
        except Exception as e:
            # v5.11.5.7: Preserve parsed payload for conservative enum fallback
            # Only keeps the JSON dict in memory — never logged, cached, or serialized
            # v5.11.5.9: Also preserve raw Pydantic errors() list — never reparsed
            pydantic_errs = None
            if hasattr(e, 'errors') and callable(getattr(e, 'errors', None)):
                pydantic_errs = list(e.errors())
            return LLMParseResult(
                success=False,
                error_type="validation_error",
                validation_error_types=_clean_validation_errors(e),
                parsed_payload=parsed,  # v5.11.5.7: for conservative enum recovery
                pydantic_errors=pydantic_errs,  # v5.11.5.9: raw errors for fallback
            )

    def get_llm_stats(self) -> dict:
        """v5.11.3: return live LLM statistics for manifest/report with redefined counts"""
        client = self.llm_client
        if client is None:
            return {
                "llm_requested": self.llm_requested,
                "llm_configured": self.llm_configured,
                "llm_vendor": "unknown",
                "llm_client_adapter": "none",
                "llm_api_type": "none",
                "llm_base_host": "",
                "llm_model": "not_configured",
                "llm_call_attempted": False,
                # v5.11 backward compat
                "llm_http_success_count": 0,
                "llm_usable_response_count": 0,
                "llm_failure_count": 0,
                "llm_validation_failure_count": 0,
                "llm_cache_hit_count": 0,
                "llm_latency_p50_ms": 0,
                "llm_latency_p95_ms": 0,
                # v5.11.1 redefined statistics
                "llm_sample_requested_count": 0,
                "llm_request_attempt_count": 0,
                "llm_transport_retry_count": 0,
                "llm_schema_retry_count": 0,
                "llm_invalid_response_count": 0,
                "llm_recovered_after_validation_count": 0,
                "llm_terminal_validation_failure_count": 0,
                "llm_terminal_transport_failure_count": 0,
                "llm_json_mode_used": False,
                "llm_json_mode_fallback_triggered": False,
                # v5.11.2 new telemetry
                "llm_json_mode_fallback_count": 0,
                "llm_http_error_count": 0,
                # v5.11.3: latency sample count for invariant validation
                "llm_latency_sample_count": 0,
                # v5.11.4: null normalization audit
                "llm_null_normalization_count": 0,
                "llm_samples_with_null_normalization_count": 0,
                # v5.11.5: enum normalization audit
                "llm_enum_normalization_count": 0,
                "llm_samples_with_enum_normalization_count": 0,
                # v5.11.5: pipeline success vs LLM usable separation
                "llm_pipeline_success_count": 0,
                "llm_deterministic_fallback_count": 0,
                # v5.11.5.7: conservative enum fallback + unknown enum audit
                "llm_unknown_enum_fallback_count": 0,
                "llm_samples_with_unknown_enum_fallback_count": 0,
                "llm_unknown_enum_candidate_counts": {},
                # v5.11.2: prompt bundle traceability
                "prompt_bundle_hash": self.prompt_bundle_hash,
                "schema_repair_prompt_version": SCHEMA_REPAIR_PROMPT_VERSION,
                "schema_repair_prompt_hash": SCHEMA_REPAIR_PROMPT_HASH,
                "max_http_attempts_per_sample": MAX_HTTP_ATTEMPTS_PER_SAMPLE,
            }
        lats = sorted(client.latencies)
        api_type = "openai_compatible" if client.provider in ("openai","local") else client.provider
        return {
            "llm_requested": self.llm_requested,
            "llm_configured": self.llm_configured,
            "llm_vendor": client.vendor,
            "llm_client_adapter": client.provider,
            "llm_api_type": api_type,
            "llm_base_host": client.base_host,
            "llm_model": client.model,
            "llm_call_attempted": (client.http_success_count + client.failure_count) > 0,
            # v5.11 backward compat
            "llm_http_success_count": client.http_success_count,
            "llm_usable_response_count": client.usable_response_count,
            "llm_failure_count": client.failure_count,
            "llm_validation_failure_count": client.validation_failure_count,
            "llm_cache_hit_count": client.cache_hit_count,
            "llm_latency_p50_ms": round(lats[len(lats)//2], 1) if lats else 0,
            "llm_latency_p95_ms": round(lats[int(len(lats)*0.95)], 1) if lats else 0,
            # v5.11.1 redefined statistics
            "llm_sample_requested_count": client.sample_requested_count,
            "llm_request_attempt_count": client.request_attempt_count,
            "llm_transport_retry_count": client.transport_retry_count,
            "llm_schema_retry_count": client.schema_retry_count,
            "llm_invalid_response_count": client.invalid_response_count,
            "llm_recovered_after_validation_count": client.recovered_after_validation_count,
            "llm_terminal_validation_failure_count": client.terminal_validation_failure_count,
            "llm_terminal_transport_failure_count": client.terminal_transport_failure_count,
            "llm_json_mode_used": client.json_mode_used,
            "llm_json_mode_fallback_triggered": client.json_mode_fallback_triggered,
            # v5.11.2 new telemetry
            "llm_json_mode_fallback_count": client.json_mode_fallback_count,
            "llm_http_error_count": (client.request_attempt_count - client.http_success_count),
            # v5.11.3: latency sample count = number of individual HTTP request latencies
            "llm_latency_sample_count": len(client.latencies),
            # v5.11.4: null normalization audit
            "llm_null_normalization_count": client.null_normalization_count,
            "llm_samples_with_null_normalization_count": client.samples_with_null_normalization_count,
            # v5.11.5: enum normalization audit
            "llm_enum_normalization_count": client.enum_normalization_count,
            "llm_samples_with_enum_normalization_count": client.samples_with_enum_normalization_count,
            # v5.11.5: pipeline success vs LLM usable separation
            "llm_pipeline_success_count": client.pipeline_success_count,
            "llm_deterministic_fallback_count": client.deterministic_fallback_count,
            # v5.11.5.7: conservative enum fallback + unknown enum audit
            "llm_unknown_enum_fallback_count": client.unknown_enum_fallback_count,
            "llm_samples_with_unknown_enum_fallback_count": client.samples_with_unknown_enum_fallback_count,
            "llm_unknown_enum_candidate_counts": dict(client.unknown_enum_candidate_counts),
            # v5.11.2: prompt bundle traceability
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "schema_repair_prompt_version": SCHEMA_REPAIR_PROMPT_VERSION,
            "schema_repair_prompt_hash": SCHEMA_REPAIR_PROMPT_HASH,
            "max_http_attempts_per_sample": MAX_HTTP_ATTEMPTS_PER_SAMPLE,
        }

    # ═══════════════════════════════════════════════════════════════════
    # v5.11.5.12: Deterministic URL trust boundary resolver
    # ═══════════════════════════════════════════════════════════════════
    def _resolve_deterministic_urls(
        self, content: str, metadata: Optional[Dict],
        ocr_result: Optional[Dict],
    ) -> Dict[str, Any]:
        """v5.11.5.12: Resolve the single trusted set of deterministic URLs.

        For non-OCR inputs (text/email/sms): extract URLs from content + metadata.
        For OCR image inputs: ONLY use URLs from ocr_result.url_audits that pass
        the trust boundary check (verified + deterministic_eligible + strict syntax).

        Returns dict with:
          - urls: List[str] — the trusted URL set (deduplicated, stable order)
          - accepted_count: int
          - rejected_count: int
          - rejected_reasons: List[str]
          - trust_policy: str — "ocr_trust_boundary" or "standard_extraction"
          - candidates_seen: int — total URL candidates considered from audits
        """
        is_ocr_input = (
            ocr_result is not None
            and ocr_result.get("url_audits") is not None
            and len(ocr_result.get("url_audits", [])) > 0
        )
        # Also check metadata signal
        if not is_ocr_input and metadata:
            is_ocr_input = metadata.get("input_type") == "image_ocr"

        if not is_ocr_input:
            # Standard path: text/email/sms — extract URLs from content + metadata
            urls = extract_urls(content or "")
            if metadata:
                for k in ("links", "urls", "link"):
                    v = metadata.get(k)
                    if isinstance(v, str):
                        urls.extend(extract_urls(v))
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, str):
                                urls.extend(extract_urls(x))
            # Deduplicate, keep order
            seen = set()
            deduped = []
            for u in urls:
                u = u.strip().rstrip(".,;:!?)】")
                if u and u not in seen:
                    seen.add(u)
                    deduped.append(u)
            return {
                "urls": deduped,
                "accepted_count": len(deduped),
                "rejected_count": 0,
                "rejected_reasons": [],
                "trust_policy": "standard_extraction",
                "candidates_seen": len(deduped),
            }

        # OCR trust boundary path
        url_audits = ocr_result.get("url_audits", [])
        candidates_seen = len(url_audits)
        # v5.11.5.12.3: Track accepted candidates (pre-dedup) separately
        # from deduplicated URL list, so counts close properly.
        accepted_urls = []       # deduplicated URL strings
        accepted_count = 0       # count of candidates that passed
        rejected_count = 0       # count of candidates rejected
        rejected_reasons = []    # per-candidate rejection entries

        for audit in url_audits:
            url_verification = audit.get("url_verification", "unverified")
            deterministic_eligible = audit.get("deterministic_eligible", False)
            selected_text = (audit.get("selected_text") or "").strip()
            block_id = audit.get("block_id", "?")

            # Check each rejection condition, collecting all applicable reasons
            candidate_reasons = []

            if url_verification != "verified":
                candidate_reasons.append(
                    f"block_{block_id}: url_verification={url_verification} (not verified)"
                )

            if not deterministic_eligible:
                candidate_reasons.append(
                    f"block_{block_id}: deterministic_eligible=False"
                )

            if not selected_text:
                candidate_reasons.append(
                    f"block_{block_id}: selected_text empty"
                )

            if selected_text and not _STRICT_URL_RE.match(selected_text):
                candidate_reasons.append(
                    f"block_{block_id}: selected_text fails strict URL syntax: "
                    f"{selected_text[:80]}"
                )

            # v5.11.5.12: Also verify isn't in the malformed category
            if audit.get("malformed_candidate"):
                candidate_reasons.append(
                    f"block_{block_id}: malformed candidate excluded"
                )

            if candidate_reasons:
                rejected_count += 1
                # Prepend candidate label, then list reasons
                rejected_reasons.append(
                    f"block_{block_id}: "
                    + "; ".join(candidate_reasons)
                )
                continue

            # Passes all checks
            accepted_count += 1
            if selected_text not in accepted_urls:
                accepted_urls.append(selected_text)

        # v5.11.5.12.3: insufficient_evidence moves ALL candidates to rejected.
        # Each accepted candidate gets an additional quality reason
        # and is transferred to rejected. Count closure is maintained.
        ocr_quality = ocr_result.get("ocr_quality_status", "unknown")
        if ocr_quality == "insufficient_evidence":
            for i in range(len(rejected_reasons)):
                rejected_reasons[i] += " [quality: insufficient_evidence]"
            if accepted_count > 0:
                for url in accepted_urls:
                    rejected_reasons.append(
                        f"url:{url[:60]}: was accepted but quality=insufficient_evidence "
                        f"— removed from deterministic set"
                    )
                rejected_count += accepted_count
                accepted_count = 0
                accepted_urls = []

        return {
            "urls": accepted_urls,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "rejected_reasons": rejected_reasons,
            "trust_policy": "ocr_trust_boundary",
            "candidates_seen": candidates_seen,
        }

    def analyze(self, content: str, content_type: str = "email",
                source_channel: str = "未知", metadata: Optional[Dict] = None,
                ocr_result: Optional[Dict] = None,
                strategy: str = "weighted") -> AnalysisResponse:
        """统一分析入口 — strategy参数进入流水线"""
        if not content or not content.strip():
            return AnalysisResponse(
                analysis_id=f"PHISH-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                status=AnalysisStatus.insufficient_evidence,
                summary="No content", model_used="none")

        content = content.strip()
        aid = f"PHISH-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        t0 = time.time()
        validation_errors = []; warnings = []

        # v5.11.5.12: Resolve deterministic URL trust boundary FIRST
        url_trust = self._resolve_deterministic_urls(content, metadata, ocr_result)
        trusted_urls = url_trust["urls"]

        # v5.11.5.12: Trust boundary audit warnings
        if url_trust["trust_policy"] == "ocr_trust_boundary":
            if url_trust["rejected_count"] > 0:
                warnings.append({
                    "type": "ocr_url_trust_boundary_rejections",
                    "message": (
                        f"{url_trust['rejected_count']}/{url_trust['candidates_seen']} "
                        f"OCR URL candidate(s) rejected by trust boundary — "
                        f"not used as deterministic evidence"
                    ),
                })
            if url_trust["accepted_count"] == 0 and url_trust["candidates_seen"] > 0:
                warnings.append({
                    "type": "ocr_no_deterministic_urls",
                    "message": "All OCR URL candidates rejected — zero trusted deterministic URLs",
                })

        # === Step 1: 确定性解析 ===
        # v5.11.5.12: _det_url() accepts optional pre-filtered URL list for OCR trust boundary
        det_url_ev = self._det_url(content, metadata, urls=trusted_urls)
        det_hdr_ev = self._det_header(metadata)
        html_ev = self._det_html(content, metadata)
        kw_ev = self._keyword_check(content)

        # OCR metadata (来自真实引擎，不受LLM影响)
        ocr_meta = {}
        if ocr_result:
            blocks = ocr_result.get("blocks",[])
            ocr_conf = sum(b.get("confidence",0) for b in blocks)/len(blocks) if blocks else 0.0
            ocr_meta = {"blocks":blocks,"uncertain_block_ids":ocr_result.get("uncertain_block_ids",[]),
                        "ocr_confidence_avg":round(ocr_conf,4),"engine":ocr_result.get("engine","unknown"),
                        "processing_time_ms":ocr_result.get("processing_time_ms",0),
                        # v5.11.5.12: Audit fields
                        "preprocessing_mode":ocr_result.get("preprocessing_mode","unknown"),
                        "ocr_quality_status":ocr_result.get("ocr_quality_status","unknown"),
                        "ocr_warnings":ocr_result.get("ocr_warnings",[]),
                        "url_uncertain_count":ocr_result.get("url_uncertain_count",0),
                        "url_audits":ocr_result.get("url_audits",[]),
                        "preprocessing_selected_reason":ocr_result.get("preprocessing_selected_reason",""),
                        "candidate_quality_scores":ocr_result.get("candidate_quality_scores",None),
                        # v5.11.5.12: Hard gate + trust boundary fields
                        "hard_gate_applied":ocr_result.get("hard_gate_applied",False),
                        "hard_gate_reason":ocr_result.get("hard_gate_reason",""),
                        "malformed_url_candidate_count":ocr_result.get("malformed_url_candidate_count",0),
                        "deterministic_url_eligible_count":ocr_result.get("deterministic_url_eligible_count",0)}
            # v5.11.5.12: Only verified URLs enter deterministic evidence
            verified_url_texts = []
            for a in ocr_result.get("url_audits", []):
                if a.get("deterministic_eligible", False) and a.get("selected_text"):
                    verified_url_texts.append(a["selected_text"])
            ocr_meta["verified_url_texts_for_deterministic"] = verified_url_texts

            if ocr_conf < 0.7 and blocks:
                warnings.append({"type":"low_ocr_confidence","message":f"OCR conf {ocr_conf:.0%}"})
            # v5.11.5.12: uncertain/malformed URLs — flag as evidence uncertainty
            ocr_quality = ocr_result.get("ocr_quality_status","unknown")
            if ocr_quality in ("insufficient_evidence",):
                warnings.append({
                    "type":"ocr_insufficient_evidence",
                    "message":"OCR quality insufficient — extracted text may be unreliable. "
                              "Provide a clearer screenshot."
                })
            if ocr_result.get("url_uncertain_count", 0) > 0:
                warnings.append({
                    "type":"uncertain_url_evidence",
                    "message":f"{ocr_result['url_uncertain_count']} URL(s) unverified — "
                              f"not used as deterministic evidence"
                })
            if ocr_result.get("malformed_url_candidate_count", 0) > 0:
                warnings.append({
                    "type":"malformed_url_candidate",
                    "message":f"{ocr_result['malformed_url_candidate_count']} malformed URL "
                              f"candidate(s) detected — not used as evidence"
                })

        # === Step 2: LLM原始提取 (Pydantic严格验证 + v5.11.3 unified parse+validate helper) ===
        raw_features = {}
        raw_llm_evidence = []
        uncertainties = []
        llm_used = "rule_only"
        # v5.11.2: per-sample LLM tracking (summed from LLMCallTelemetry)
        llm_sample_status = "not_requested"
        llm_request_count_for_sample = 0
        llm_http_success_count_for_sample = 0
        llm_http_error_count_for_sample = 0
        llm_transport_retry_count_for_sample = 0
        llm_schema_retry_count_for_sample = 0
        llm_invalid_response_count_for_sample = 0
        llm_json_mode_fallback_count_for_sample = 0
        llm_validation_error_types: List[str] = []
        llm_total_latency_ms = 0.0
        fallback_used = False
        fallback_reason = ""
        # v5.11.4: per-sample null normalization tracking
        llm_null_normalization_count_for_sample = 0
        llm_normalized_field_paths: List[str] = []
        # v5.11.5: per-sample enum normalization tracking
        llm_enum_normalization_count_for_sample = 0
        llm_enum_normalized_paths: List[str] = []
        # v5.11.5: deterministic fallback tracking
        llm_fallback_to_deterministic = False
        # v5.11.5.7: conservative enum fallback and security audit
        llm_unknown_enum_fallback_count_for_sample = 0
        llm_unknown_enum_paths: List[str] = []
        llm_unknown_enum_candidates: List[str] = []
        llm_conservative_enum_fallback_used = False

        if self.llm_requested:
            if not self.llm_configured:
                llm_sample_status = "not_configured"
            elif self.llm_client:
                # v5.11.3: Thread-safe increment
                self.llm_client._safe_incr(sample_requested_count=1)
                use_json_mode = (LLM_JSON_MODE_ENABLED and self.llm_client.provider in ("openai", "local"))
                t_llm_start = time.time()

                proc = content[:3000] + ("\n[truncated]" if len(content)>3000 else "")
                md_sec = self._md_section(metadata)
                url_sec = self._url_section(content, metadata)
                html_sec = self._html_section(content, metadata)
                user_msg = ANALYSIS_PROMPT_TEMPLATE.format(
                    content_type=content_type, source_channel=source_channel,
                    metadata_section=md_sec, content=proc,
                    url_section=url_sec, html_section=html_sec)

                # --- Attempt 1: transport retry only (returns LLMCallTelemetry) ---
                call1_telemetry: Optional[LLMCallTelemetry] = None
                attempt1_transport_failed = False
                try:
                    call1_telemetry = self.llm_client.chat(SYSTEM_PROMPT, user_msg, use_json_mode=use_json_mode)
                    llm_used = self.llm_client.config.get("model", "unknown")
                except LLMTransportError as e:
                    attempt1_transport_failed = True
                    call1_telemetry = e.telemetry
                    llm_sample_status = "terminal_transport_failed"
                    fallback_used = True
                    fallback_reason = "transport_exhausted"
                    llm_fallback_to_deterministic = True
                    logger.error(f"LLM transport fail: {e}")
                    uncertainties.append(f"LLM transport error: {e}")
                    warnings.append({"type": "llm_transport_failed", "message": str(e)[:200]})

                # v5.11.3 FIX B: Unified parse+validate for first attempt (handles ""/"  "/None)
                if call1_telemetry is not None and not attempt1_transport_failed:
                    parse_result = self._parse_and_validate_llm_text(call1_telemetry.text)

                    if parse_result.success:
                        raw_features = parse_result.raw_features
                        raw_llm_evidence = parse_result.raw_llm_evidence
                        uncertainties = parse_result.uncertainties
                        # v5.11.4: capture null normalization audit data
                        llm_normalized_field_paths = parse_result.normalized_field_paths
                        llm_null_normalization_count_for_sample = parse_result.null_normalization_count
                        if llm_null_normalization_count_for_sample > 0:
                            self.llm_client._safe_incr(
                                null_normalization_count=llm_null_normalization_count_for_sample,
                                samples_with_null_normalization_count=1)
                        # v5.11.5: capture enum normalization audit data
                        llm_enum_normalized_paths = parse_result.enum_normalized_paths
                        llm_enum_normalization_count_for_sample = parse_result.enum_normalization_count
                        if llm_enum_normalization_count_for_sample > 0:
                            self.llm_client._safe_incr(
                                enum_normalization_count=llm_enum_normalization_count_for_sample,
                                samples_with_enum_normalization_count=1)
                        self.llm_client._safe_incr(usable_response_count=1)
                        llm_sample_status = "usable_first_attempt"
                    else:
                        # Any failure (empty/json_parse/validation) → invalid
                        self.llm_client._safe_incr(invalid_response_count=1, validation_failure_count=1)
                        llm_invalid_response_count_for_sample += 1
                        llm_validation_error_types = parse_result.validation_error_types
                        if parse_result.error_type == "empty_response":
                            validation_errors.append("LLM response: empty response")
                        elif parse_result.error_type == "json_parse_error":
                            validation_errors.append("LLM response: JSON parse failed")
                        else:
                            validation_errors.append(f"LLM response validation: {parse_result.error_type}")

                # --- Schema repair retry (max 1, independent from transport retry) ---
                call2_telemetry: Optional[LLMCallTelemetry] = None
                if (llm_sample_status not in ("usable_first_attempt", "terminal_transport_failed")
                        and not attempt1_transport_failed):
                    if LLM_SCHEMA_RETRY_MAX >= 1:
                        self.llm_client._safe_incr(schema_retry_count=1)
                        llm_schema_retry_count_for_sample += 1

                        # v5.11.5: Schema repair v3 — field-level enum hints from build_field_hints()
                        # Safe: only _clean_validation_errors() output (type:field_path), no raw content
                        sanitized_paths = ", ".join(llm_validation_error_types[:5]) if llm_validation_error_types else "schema mismatch"
                        field_hints = build_field_hints(llm_validation_error_types) if llm_validation_error_types else ""
                        repair_prompt = SCHEMA_REPAIR_PROMPT_TEMPLATE.format(
                            sanitized_error_paths=sanitized_paths,
                            field_hints=field_hints)
                        corrected_msg = user_msg + repair_prompt

                        try:
                            call2_telemetry = self.llm_client.chat(SYSTEM_PROMPT, corrected_msg, use_json_mode=use_json_mode)
                        except LLMTransportError as e:
                            # v5.11.2 FIX A: Schema retry transport exhaustion → terminal_transport_failed
                            call2_telemetry = e.telemetry
                            llm_sample_status = "terminal_transport_failed"
                            fallback_used = True
                            fallback_reason = "schema_retry_transport_exhausted"
                            llm_fallback_to_deterministic = True
                            raw_features = {}
                            raw_llm_evidence = []
                            logger.error(f"LLM schema retry transport fail: {e}")
                            warnings.append({"type": "llm_schema_retry_failed", "message": str(e)[:200]})

                        if llm_sample_status != "terminal_transport_failed":
                            # v5.11.3: Unified parse+validate for schema retry (same helper as first attempt)
                            parse_result2 = self._parse_and_validate_llm_text(call2_telemetry.text)

                            if parse_result2.success:
                                raw_features = parse_result2.raw_features
                                raw_llm_evidence = parse_result2.raw_llm_evidence
                                uncertainties = parse_result2.uncertainties
                                # v5.11.4: capture null normalization audit data from schema retry
                                if not llm_normalized_field_paths:
                                    llm_normalized_field_paths = parse_result2.normalized_field_paths
                                    llm_null_normalization_count_for_sample = parse_result2.null_normalization_count
                                    if llm_null_normalization_count_for_sample > 0:
                                        self.llm_client._safe_incr(
                                            null_normalization_count=llm_null_normalization_count_for_sample,
                                            samples_with_null_normalization_count=1)
                                # v5.11.5: capture enum normalization from schema retry
                                if not llm_enum_normalized_paths and parse_result2.enum_normalization_count > 0:
                                    llm_enum_normalized_paths = parse_result2.enum_normalized_paths
                                    llm_enum_normalization_count_for_sample = parse_result2.enum_normalization_count
                                    self.llm_client._safe_incr(
                                        enum_normalization_count=llm_enum_normalization_count_for_sample,
                                        samples_with_enum_normalization_count=1)
                                self.llm_client._safe_incr(usable_response_count=1, recovered_after_validation_count=1)
                                llm_sample_status = "recovered_after_validation_retry"
                            else:
                                # Schema retry also failed — try conservative enum fallback FIRST
                                # v5.11.5.9: Do NOT increment terminal_validation_failure_count yet.
                                # Only increment after conservative recovery has failed.
                                self.llm_client._safe_incr(
                                    invalid_response_count=1,
                                    validation_failure_count=1)
                                llm_invalid_response_count_for_sample += 1
                                if parse_result2.validation_error_types:
                                    llm_validation_error_types.extend(parse_result2.validation_error_types)
                                elif "json_parse_error" not in str(llm_validation_error_types):
                                    llm_validation_error_types.append(parse_result2.error_type or "json_parse_error")
                                raw_features = {}
                                raw_llm_evidence = []

                                # v5.11.5.9: Conservative enum fallback on SECOND response ONLY
                                # (never fall back to first response when retry was executed)
                                # Use parse_result2.parsed_payload and .pydantic_errors directly — never re-parse
                                recovered = False
                                if (parse_result2.parsed_payload and parse_result2.pydantic_errors
                                        and parse_result2.error_type == "validation_error"):
                                    from models import attempt_conservative_enum_fallback as _try_conservative
                                    rec, enum_paths, enum_candidates = _try_conservative(
                                        parse_result2.parsed_payload, parse_result2.pydantic_errors)
                                    if rec is not None:
                                        raw_features = rec.raw_features.model_dump()
                                        raw_llm_evidence = [ev.to_evidence().model_dump()
                                                            for ev in rec.raw_evidence]
                                        uncertainties = rec.uncertainties
                                        self.llm_client._safe_incr(
                                            usable_response_count=1,
                                            recovered_after_validation_count=1,
                                            unknown_enum_fallback_count=len(enum_paths),
                                            samples_with_unknown_enum_fallback_count=1)
                                        llm_sample_status = "recovered_after_conservative_enum_fallback"
                                        fallback_used = False
                                        fallback_reason = ""
                                        llm_fallback_to_deterministic = False
                                        llm_conservative_enum_fallback_used = True
                                        llm_unknown_enum_fallback_count_for_sample = len(enum_paths)
                                        llm_unknown_enum_paths = enum_paths
                                        llm_unknown_enum_candidates = enum_candidates
                                        with self.llm_client._lock:
                                            for c in enum_candidates:
                                                self.llm_client.unknown_enum_candidate_counts[c] = \
                                                    self.llm_client.unknown_enum_candidate_counts.get(c, 0) + 1
                                        recovered = True

                                if not recovered:
                                    # v5.11.5.9: Recovery failed — NOW increment terminal counters
                                    self.llm_client._safe_incr(terminal_validation_failure_count=1)
                                    llm_sample_status = "terminal_validation_failed"
                                    fallback_used = True
                                    fallback_reason = "exhausted_schema_retry"
                                    llm_fallback_to_deterministic = True
                                    validation_errors.append(f"LLM schema retry still invalid: {parse_result2.error_type}")
                    else:
                        # v5.11.5.9: LLM_SCHEMA_RETRY_MAX=0 + invalid first response
                        # Allow conservative enum fallback on FIRST response
                        # Do NOT increment terminal_validation_failure_count yet.
                        raw_features = {}
                        raw_llm_evidence = []

                        recovered = False
                        if (parse_result.parsed_payload and parse_result.pydantic_errors
                                and parse_result.error_type == "validation_error"):
                            from models import attempt_conservative_enum_fallback as _try_conservative
                            rec, enum_paths, enum_candidates = _try_conservative(
                                parse_result.parsed_payload, parse_result.pydantic_errors)
                            if rec is not None:
                                raw_features = rec.raw_features.model_dump()
                                raw_llm_evidence = [ev.to_evidence().model_dump()
                                                    for ev in rec.raw_evidence]
                                uncertainties = rec.uncertainties
                                self.llm_client._safe_incr(
                                    usable_response_count=1,
                                    recovered_after_validation_count=1,
                                    unknown_enum_fallback_count=len(enum_paths),
                                    samples_with_unknown_enum_fallback_count=1)
                                llm_sample_status = "recovered_after_conservative_enum_fallback"
                                fallback_used = False
                                fallback_reason = ""
                                llm_fallback_to_deterministic = False
                                llm_conservative_enum_fallback_used = True
                                llm_unknown_enum_fallback_count_for_sample = len(enum_paths)
                                llm_unknown_enum_paths = enum_paths
                                llm_unknown_enum_candidates = enum_candidates
                                with self.llm_client._lock:
                                    for c in enum_candidates:
                                        self.llm_client.unknown_enum_candidate_counts[c] = \
                                            self.llm_client.unknown_enum_candidate_counts.get(c, 0) + 1
                                recovered = True

                        if not recovered:
                            # v5.11.5.9: Recovery failed — NOW increment terminal counters
                            self.llm_client._safe_incr(terminal_validation_failure_count=1)
                            llm_sample_status = "terminal_validation_failed"
                            fallback_used = True
                            fallback_reason = "schema_retry_disabled"
                            llm_fallback_to_deterministic = True
                            validation_errors.append(f"LLM response validation failed: {parse_result.error_type}")

                # Set source for LLM evidence
                for ev in raw_llm_evidence:
                    ev["source"] = EVIDENCE_SOURCE_LLM

                llm_total_latency_ms = round((time.time() - t_llm_start) * 1000, 1)

                # v5.11.2 FIX D: Per-sample counts summed from LLMCallTelemetry — no approximation
                if call1_telemetry is not None:
                    llm_request_count_for_sample += call1_telemetry.request_attempt_count
                    llm_http_success_count_for_sample += call1_telemetry.http_success_count
                    llm_http_error_count_for_sample += call1_telemetry.http_error_count
                    llm_transport_retry_count_for_sample += call1_telemetry.transport_retry_count
                    llm_json_mode_fallback_count_for_sample += call1_telemetry.json_mode_fallback_count
                if call2_telemetry is not None:
                    llm_request_count_for_sample += call2_telemetry.request_attempt_count
                    llm_http_success_count_for_sample += call2_telemetry.http_success_count
                    llm_http_error_count_for_sample += call2_telemetry.http_error_count
                    llm_transport_retry_count_for_sample += call2_telemetry.transport_retry_count
                    llm_json_mode_fallback_count_for_sample += call2_telemetry.json_mode_fallback_count

        if not self.llm_requested:
            warnings.append({"type": "rule_only", "message": "LLM disabled"})

        # === Step 3: 确定性特征推导 (无论LLM是否启用) ===
        # v5.11.5.12: Pass trusted URL set for OCR trust boundary enforcement
        validated = self._derive_deterministic_features(
            content, metadata, ocr_meta, deterministic_urls=trusted_urls)

        # v5.11.5.12: metadata URL also enters validated (standard extraction only)
        if metadata and url_trust["trust_policy"] == "standard_extraction":
            for k in ("links","urls","link"):
                v = metadata.get(k)
                if isinstance(v,str): urls_md = extract_urls(v)
                elif isinstance(v,list): urls_md = [u for x in v if isinstance(x,str) for u in extract_urls(x)]
                else: urls_md = []
                for url in urls_md:
                    info = analyze_url_safety(url.strip().rstrip(".,;:!?)】"))
                    if info["impersonates_sdu"] or info["has_suspicious_tld"]:
                        if validated.link_safety_score is None: validated.link_safety_score = 0
                        validated.link_safety_score = min(100, validated.link_safety_score + 30)

        # email auth
        if metadata:
            auth = parse_auth_results(metadata.get("auth_results") or metadata.get("authentication-results") or "")
            for m, st in auth.items():
                if st == "fail":
                    validated.sender_credibility_notes.append(f"{m.upper()}: fail")
                    if validated.sender_credibility_score is None: validated.sender_credibility_score = 0
                    validated.sender_credibility_score = min(100, validated.sender_credibility_score + 30)

        # === Step 4: LLM候选校验 (仅当LLM有输出时) ===
        if raw_features and isinstance(raw_features, dict):
            validated = self._validate_llm_candidates(raw_features, content, validated)

        # === Step 4: 证据合并+跨来源去重+quote校验 ===
        all_ev = det_hdr_ev + det_url_ev + html_ev + raw_llm_evidence + kw_ev
        all_ev = _dedup_cross_source(all_ev)

        verified, unverified = [], []
        for ev in all_ev:
            quote = ev.get("quote",""); ev_src = ev.get("source","")
            is_det = ev_src in ("deterministic_url","deterministic_header","html_parser","keyword")
            if is_det:
                ev["verification"] = "deterministic"
                verified.append(ev)
            else:
                matched, vtype = self._quote_match(quote, content)
                if vtype == "matched":
                    ev["verification"] = "matched"
                    verified.append(ev)
                else:
                    ev["verification"] = vtype
                    unverified.append(ev)

        blocks = ocr_meta.get("blocks",[])
        if blocks:
            verified = self._map_ocr(verified, blocks)
            unverified = self._map_ocr(unverified, blocks)

        # === Step 5: 评分 (唯一调用) ===
        scorer = RiskScorer()
        score_result = scorer.score(validated=validated, evidence=verified, strategy=strategy)

        # === Step 6: 组装 ===
        # v5.11.5: Track pipeline success vs LLM fallback
        if self.llm_client is not None:
            self.llm_client._safe_incr(pipeline_success_count=1)
            if llm_fallback_to_deterministic:
                self.llm_client._safe_incr(deterministic_fallback_count=1)
        processing_time = (time.time()-t0)*1000
        return AnalysisResponse(
            analysis_id=aid, status=AnalysisStatus.success,
            score_result=ScoreResultModel(
                risk_score=score_result.score, risk_level=score_result.level,
                risk_level_label=score_result.level_label,
                is_phishing=score_result.is_phishing, confidence=score_result.confidence,
                strategy=score_result.strategy, score_breakdown=score_result.score_breakdown,
                # v5.11.5.1: promotion tracking
                promotion_applied=score_result.promotion_applied,
                promotion_rule=score_result.promotion_rule,
                pre_promotion_score=score_result.pre_promotion_score if score_result.pre_promotion_score else None,
                promotion_score=score_result.promotion_score if score_result.promotion_score else None,
                promotion_evidence_types=list(score_result.promotion_evidence_types),
                # v5.11.5.2: deterministic anchor tracking
                promotion_anchor_source=score_result.promotion_anchor_source,
                promotion_anchor_rule_ids=list(score_result.promotion_anchor_rule_ids)),
            verified_evidence=[self._safe_evidence(e) for e in verified[:50]],
            unverified_evidence=[self._safe_evidence(e) for e in unverified[:50]],
            validation_errors=validation_errors,
            summary=raw_features.get("overall_impression","") if isinstance(raw_features,dict) else "",
            processing_time_ms=round(processing_time,1),
            model_used=llm_used, warnings=warnings,
            # v5.11.2: per-sample LLM telemetry from LLMCallTelemetry summation
            llm_sample_status=llm_sample_status,
            llm_request_count_for_sample=llm_request_count_for_sample,
            llm_http_success_count_for_sample=llm_http_success_count_for_sample,
            llm_http_error_count_for_sample=llm_http_error_count_for_sample,
            llm_transport_retry_count_for_sample=llm_transport_retry_count_for_sample,
            llm_schema_retry_count_for_sample=llm_schema_retry_count_for_sample,
            llm_invalid_response_count_for_sample=llm_invalid_response_count_for_sample,
            llm_json_mode_fallback_count_for_sample=llm_json_mode_fallback_count_for_sample,
            llm_validation_error_types=llm_validation_error_types,
            llm_total_latency_ms=llm_total_latency_ms,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            # v5.11.4: null normalization audit
            llm_null_normalization_count_for_sample=llm_null_normalization_count_for_sample,
            llm_normalized_field_paths=llm_normalized_field_paths,
            # v5.11.5: enum normalization + deterministic fallback audit
            llm_enum_normalization_count_for_sample=llm_enum_normalization_count_for_sample,
            llm_enum_normalized_paths=llm_enum_normalized_paths,
            llm_fallback_to_deterministic=llm_fallback_to_deterministic,
            # v5.11.5.7: conservative enum fallback and unknown enum security audit
            llm_unknown_enum_fallback_count_for_sample=llm_unknown_enum_fallback_count_for_sample,
            llm_unknown_enum_paths=llm_unknown_enum_paths,
            llm_unknown_enum_candidates=llm_unknown_enum_candidates,
            llm_conservative_enum_fallback_used=llm_conservative_enum_fallback_used,
            # v5.11.5.12: OCR URL trust boundary audit
            deterministic_url_candidates_seen=url_trust.get("candidates_seen", 0),
            deterministic_url_accepted_count=url_trust.get("accepted_count", 0),
            deterministic_url_rejected_count=url_trust.get("rejected_count", 0),
            deterministic_url_rejected_reasons=url_trust.get("rejected_reasons", []),
            deterministic_urls_used=trusted_urls,
            ocr_url_trust_policy=url_trust.get("trust_policy", ""))

    # ---- 确定性特征推导 (always runs) ----
    def _derive_deterministic_features(self, content, metadata, ocr_meta,
                                         deterministic_urls=None) -> ValidatedFeatureSet:
        v = ValidatedFeatureSet()
        # v5.11.5.12: Use pre-resolved deterministic URL set when provided (OCR trust boundary)
        # Otherwise fall back to text extraction (standard text/email/sms path)
        if deterministic_urls is not None:
            urls_found = list(deterministic_urls)
        else:
            urls_found = extract_urls(content or "")
        if urls_found:
            v.link_safety_score = 0
            seen = set()
            for url in urls_found:
                url = url.strip().rstrip(".,;:!?)】")
                if url in seen: continue
                seen.add(url)
                info = analyze_url_safety(url)
                if info["has_suspicious_tld"]:
                    v.link_safety_score = min(100, v.link_safety_score + 30)
                    v.link_safety_notes.append(f"Suspicious TLD: {info['registered_domain']}")
                if info["impersonates_sdu"]:
                    v.link_safety_score = min(100, v.link_safety_score + 50)
                    v.link_safety_notes.append(f"Impersonates SDU: {info['registered_domain']}")
                if info["is_ddns"]: v.link_safety_score = min(100, v.link_safety_score + 40)
                if info["is_shortener"]: v.link_safety_score = min(100, v.link_safety_score + 20)
                if info["is_ip_host"]: v.link_safety_score = min(100, v.link_safety_score + 30)
                if info["has_userinfo"]: v.link_safety_score = min(100, v.link_safety_score + 60)
            if v.link_safety_score == 0: v.link_safety_score = None
        # sender: metadata only
        if metadata:
            from_addr = (metadata.get("from") or metadata.get("sender") or "")
            if from_addr:
                v.sender_credibility_score = 0
                if "@" in str(from_addr):
                    domain = str(from_addr).split("@")[-1].strip().lower().rstrip(">")
                    if "sdu" in domain and not is_sdu_official_hostname(domain):
                        v.sender_credibility_score = min(100, v.sender_credibility_score + 60)
                        v.sender_credibility_notes.append(f"Non-official SDU domain: {domain}")
        # attachment: metadata only
        if metadata and metadata.get("attachments"):
            v.attachment_risk_score = 0
        return v

    # ---- LLM候选校验 (仅当LLM有输出时) ----
    def _validate_llm_candidates(self, raw_features, content, validated) -> ValidatedFeatureSet:
        """只处理需要LLM理解的候选特征，不覆盖确定性维度"""
        # content_urgency + info_requests: quote必须在原文匹配
        cf = raw_features.get("content",{})
        if cf and isinstance(cf, dict):
            urg = [q for q in cf.get("urgency_indicators",[]) if q and q in (content or "")]
            thr = [q for q in cf.get("threats",[]) if q and q in (content or "")]
            irq = [q for q in cf.get("info_requests",[]) if q and q in (content or "")]
            if urg or thr: validated.content_urgency_score = min(100, len(urg)*25 + len(thr)*30)
            if irq: validated.information_request_score = min(100, len(irq)*30)
        # language: errors必须在原文匹配
        lang = raw_features.get("language",{})
        if lang and isinstance(lang, dict):
            matched_errs = [e for e in lang.get("errors",[]) if isinstance(e,str) and e in (content or "")]
            matched_inc = [e for e in lang.get("inconsistencies",[]) if isinstance(e,str) and e in (content or "")]
            if matched_errs or matched_inc: validated.language_quality_score = min(100, len(matched_errs)*10 + len(matched_inc)*15)
        # sender LLM candidates: only free-email check needs LLM claimed_identity
        if validated.sender_credibility_score is not None:
            sender = raw_features.get("sender",{})
            claimed = sender.get("claimed_identity","") if isinstance(sender, dict) else ""
            if claimed and any(kw in str(claimed) for kw in ["山大","学校","教务","财务","网络中心","图书馆"]):
                from_addr = ""
                if validated.sender_credibility_notes:
                    pass  # Already has metadata-derived notes
        return validated

    # ---- 确定性检测 ----
    def _det_url(self, content, metadata=None, urls=None):
        # v5.11.5.12: Use pre-resolved trusted URL set when provided (OCR trust boundary).
        # When urls is not None, it is the single source of deterministic URLs —
        # skip extract_urls(content) entirely to prevent OCR text bypass.
        if urls is not None:
            pass  # use the caller-provided trusted set directly
        else:
            urls = extract_urls(content or "")
            if metadata:
                for k in ("links","urls","link"):
                    v = metadata.get(k)
                    if isinstance(v,str): urls.extend(extract_urls(v))
                    elif isinstance(v,list):
                        for x in v:
                            if isinstance(x,str): urls.extend(extract_urls(x))
        ev=[]; seen=set()
        for url in urls:
            url=url.strip().rstrip(".,;:!?)】")
            if url in seen: continue
            seen.add(url)
            info=analyze_url_safety(url)
            issues = info["issues"]
            if not issues:
                continue
            # Aggregate all issues from one URL into a SINGLE structured evidence
            HIGH_ISSUES = {"impersonates_sdu_domain","url_contains_userinfo_credentials",
                          "uses_ipv4_address","uses_ipv6_address","ddns_service","punycode_encoding"}
            MED_ISSUES = {"suspicious_tld","url_shortener","non_standard_port","no_hostname","url_parse_failed"}
            high_count = sum(1 for i in issues if any(h in i for h in HIGH_ISSUES))
            med_count = sum(1 for i in issues if any(m in i for m in MED_ISSUES))
            has_impersonation = any("impersonates_sdu" in i for i in issues)
            has_userinfo = any("url_userinfo" in i for i in issues)
            has_suspicious_tld = any("suspicious_tld" in i for i in issues)
            if has_impersonation or has_userinfo or high_count >= 2:
                sev = "high"
            elif high_count >= 1 and has_suspicious_tld:
                sev = "high"
            elif high_count >= 1 or med_count >= 2:
                sev = "medium"
            else:
                sev = "low"
            issue_summary = "; ".join(
                f"{i.split(':')[0]}:{i.split(':')[1]}" if ":" in i else i
                for i in issues[:5]
            )
            rule_ids = []
            if has_impersonation and has_suspicious_tld:
                rule_ids.append("url_sdu_impersonation_and_suspicious_tld")
            elif has_impersonation:
                rule_ids.append("url_sdu_impersonation")
            if has_userinfo:
                rule_ids.append("url_userinfo_deception")
            if info["is_ip_host"]:
                rule_ids.append("url_valid_ip_host")
            if info["has_punycode"] and has_impersonation:
                rule_ids.append("url_punycode_impersonation")
            if info["is_shortener"] and not has_impersonation:
                rule_ids.append("url_shortener_only")
            if info["is_ddns"]:
                rule_ids.append("url_ddns_service")
            if has_suspicious_tld and not has_impersonation:
                rule_ids.append("url_suspicious_tld_only")

            ev.append({
                "quote": url, "type": "domain_anomaly", "severity": sev,
                "explanation": f"URL({len(issues)} issues): {issue_summary} (domain: {info['registered_domain']})",
                "source": EVIDENCE_SOURCE_DET_URL, "rule_ids": rule_ids,
            })
        return ev

    def _det_header(self, metadata=None):
        if not metadata: return []
        ev=[]
        alignment=check_email_alignment(metadata)
        for issue in alignment["alignment_issues"]:
            rid = "from_replyto_mismatch" if "reply_to" in issue.lower() else None
            ev.append({"quote":f"From:{alignment['from']['address']} Reply-To:{alignment['reply_to']['address']}",
                       "type":"sender_anomaly","severity":"medium","explanation":issue,"source":EVIDENCE_SOURCE_DET_HDR,
                       "rule_ids":[rid] if rid else []})
        fr=alignment["from"]["address"]
        if fr and "@" in fr:
            domain=fr.split("@")[-1]
            if "sdu" in domain.lower() and not is_sdu_official_hostname(domain):
                ev.append({"quote":fr,"type":"sender_anomaly","severity":"high",
                           "explanation":f"Domain {domain} not official SDU","source":EVIDENCE_SOURCE_DET_HDR,
                           "rule_ids":["sdu_domain_impersonation"]})
        auth=parse_auth_results(metadata.get("auth_results") or metadata.get("authentication-results") or "")
        fail_count=sum(1 for m,st in auth.items() if st=="fail")
        for m,st in auth.items():
            if st=="fail":
                ev.append({"quote":f"{m.upper()}: fail","type":"sender_anomaly","severity":"high",
                           "explanation":f"Auth {m.upper()} failed","source":EVIDENCE_SOURCE_DET_HDR})
            elif st=="none":
                ev.append({"quote":f"{m.upper()}: none","type":"sender_anomaly","severity":"low",
                           "explanation":f"Auth {m.upper()} not configured","source":EVIDENCE_SOURCE_DET_HDR})
        if fail_count>=2:
            ev.append({"quote":f"{fail_count} auth failures","type":"sender_anomaly","severity":"high",
                       "explanation":f"Multiple auth failures ({fail_count})","source":EVIDENCE_SOURCE_DET_HDR,
                       "rule_ids":["auth_multiple_failures"]})
        return ev

    def _det_html(self, content, metadata=None):
        if not content: return []
        ev=[]
        mismatches=check_display_url_mismatch(content)
        for mm in mismatches:
            td=mm.get("display_text",""); hd=mm.get("actual_href","")
            from urllib.parse import urlparse as up
            try:
                rd1=get_registered_domain(td) if td.startswith("http") else get_registered_domain("https://"+td)
                rd2=get_registered_domain(hd)
            except: rd1=td; rd2=hd
            if rd1!=rd2:
                ev.append({"quote":f"display:{td} href:{hd}","type":"domain_anomaly","severity":"high",
                           "explanation":f"Display domain ({rd1}) != href domain ({rd2})",
                           "source":EVIDENCE_SOURCE_HTML})
        parsed=parse_html_links(content)
        for form in parsed.get("forms",[]):
            action=form.get("action","")
            if action and not action.startswith("/") and not action.startswith("#"):
                try:
                    rd=get_registered_domain(action)
                    if rd!="unknown": ev.append({"quote":f"form action={action}","type":"domain_anomaly",
                        "severity":"medium","explanation":f"Form action to external domain {rd}","source":EVIDENCE_SOURCE_HTML})
                except: pass
        return ev

    def _keyword_check(self, content):
        from config import RISK_SCORING_CONFIG
        kws=RISK_SCORING_CONFIG.get("high_risk_keywords",[])
        ev=[]
        for kw in kws:
            if kw.lower() in (content or "").lower():
                idx=(content or "").lower().find(kw.lower())
                q=content[max(0,idx-20):idx+len(kw)+30].strip()
                ev.append({"quote":q,"type":"other","severity":"low",
                           "explanation":f"Keyword: {kw} (weak)","source":EVIDENCE_SOURCE_KW})
        return ev

    def _safe_evidence(self, e):
        """安全转换dict→Evidence，strip非模型字段"""
        if isinstance(e, Evidence): return e
        if not isinstance(e, dict): return e
        allowed = {"quote","type","severity","explanation","source","sources","verification",
                   "ocr_block_id","bbox","ocr_confidence","ocr_uncertain","dedup_key","rule_ids"}
        cleaned = {k:v for k,v in e.items() if k in allowed}
        try: return Evidence(**cleaned)
        except Exception: return None

    def _validate_features_public(self, raw_features, content, metadata, ocr_meta):
        return self._validate_features(raw_features, content, metadata, ocr_meta)

    def _md_section(self, md):
        if not md: return ""
        lines=["## Metadata"]
        for k in ("from","sender","reply_to","reply-to","subject","return_path","return-path","auth_results","authentication-results"):
            v=md.get(k)
            if v: lines.append(f"{k}: {v}")
        return "\n".join(lines) if len(lines)>1 else ""

    def _url_section(self, content, md):
        urls=extract_urls(content or "");
        if not urls: return ""
        deduped=list(dict.fromkeys(urls)); lines=["## URL Analysis"]
        for url in deduped[:20]:
            info=analyze_url_safety(url)
            lines.append(f"- {info['url']} (domain: {info['registered_domain']})")
            if info["issues"]: lines.append(f"  Issues: {', '.join(info['issues'])}")
        return "\n".join(lines)

    def _html_section(self, content, md):
        if not content: return ""
        parsed=parse_html_links(content)
        links=parsed.get("links",[]); forms=parsed.get("forms",[]); iframes=parsed.get("iframes",[])
        if not links and not forms and not iframes: return ""
        lines=["## HTML Analysis"]
        for l in links[:10]: lines.append(f"- Link: text='{l['text'][:60]}' href='{l['href'][:100]}'")
        for f in forms[:5]: lines.append(f"- Form action: {f.get('action','')}")
        for i in iframes[:5]: lines.append(f"- Iframe src: {i.get('src','')}")
        return "\n".join(lines)

    def _parse_json(self, s):
        if not s: return None
        try: return json.loads(s)
        except:
            for pat in [r"```json\s*([\s\S]*?)\s*```",r"```\s*([\s\S]*?)\s*```"]:
                m=re.search(pat,s)
                if m:
                    try: return json.loads(m.group(1))
                    except: pass
            try: return json.loads(s[s.index("{"):s.rindex("}")+1])
            except: return None

    def _quote_match(self, quote, original):
        if not quote or not original: return False, "unmatched"
        if quote in original: return True, "matched"
        nq=_norm(quote); no=_norm(original)
        if len(nq)>=8 and nq in no: return True, "matched"
        return False, "unmatched"

    def _map_ocr(self, evidence, blocks):
        for ev in evidence:
            q=ev.get("quote","")
            if not q or not blocks: continue
            for b in blocks:
                if q in b.get("text","") or q in b.get("corrected_text",""):
                    ev["ocr_block_id"]=b.get("block_id"); ev["bbox"]=b.get("bbox")
                    ev["ocr_confidence"]=b.get("confidence"); ev["ocr_uncertain"]=b.get("status")=="uncertain"
                    break
        return evidence

    # _validate_features_public defined once above
