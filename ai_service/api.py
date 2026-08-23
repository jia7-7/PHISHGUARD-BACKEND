"""
API — unified pipeline, AnalysisResponse model, dynamic version from config.APP_VERSION
"""
import concurrent.futures, logging, time, traceback
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import (
    API_CONFIG, APP_VERSION, get_api_version, IMAGE_MAGIC_BYTES, PHISHING_APP_HOME,
    SUPPORTED_CONTENT_TYPES, SUPPORTED_IMAGE_FORMATS, SUPPORTED_IMAGE_MIMES,
    LLMConfigError, check_llm_configured,
    get_llm_enabled, get_allow_deterministic_fallback,
)
from models import AnalysisResponse, ScoreResultModel, AnalysisStatus, Evidence
from pipeline import AnalysisPipeline

API_VERSION = get_api_version()  # v5.11.1: proper semver, no blind ".0" append
logger = logging.getLogger("api")

class ScoringStrategy(str, Enum):
    weighted="weighted"; rule_only="rule_only"; hybrid="hybrid"

class TextRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    content_type: str = Field(default="email")
    source_channel: str = Field(default="unknown")
    metadata: Optional[Dict] = None
    scoring_strategy: ScoringStrategy = Field(default=ScoringStrategy.weighted)

    @field_validator("content_type")
    @classmethod
    def check_ct(cls, v):
        if v not in SUPPORTED_CONTENT_TYPES: raise ValueError(f"Invalid: {v}")
        return v

class BatchRequest(BaseModel):
    items: List[TextRequest] = Field(..., min_length=1, max_length=50)
    scoring_strategy: ScoringStrategy = Field(default=ScoringStrategy.weighted)

def _ensure_dirs():
    from config import RUNTIME_LOG_DIR, RUNTIME_REPORT_DIR, RUNTIME_CACHE_DIR, EASYOCR_MODEL_DIR, RUNTIME_TEMP_DIR
    for d in [RUNTIME_LOG_DIR,RUNTIME_REPORT_DIR,RUNTIME_CACHE_DIR,EASYOCR_MODEL_DIR,RUNTIME_TEMP_DIR]:
        d.mkdir(parents=True,exist_ok=True)

def _build_ocr_audit(ocr_result: Dict) -> Dict:
    """v5.11.5.12: Build OCR audit dict from result."""
    if not ocr_result:
        return {"preprocessing_mode": "unknown", "ocr_quality_status": "unknown"}
    audit = {
        "preprocessing_mode": ocr_result.get("preprocessing_mode", "unknown"),
        "preprocessing_attempted_modes": ocr_result.get("preprocessing_attempted_modes", []),
        "preprocessing_selected_reason": ocr_result.get("preprocessing_selected_reason", ""),
        "ocr_quality_status": ocr_result.get("ocr_quality_status", "unknown"),
        "ocr_confidence_avg": ocr_result.get("ocr_confidence_avg", 0.0),
        "selected_uncertain_ratio": ocr_result.get("selected_uncertain_ratio", 0.0),
        "ocr_warnings": ocr_result.get("ocr_warnings", []),
        # v5.11.5.12: Hard gate fields
        "hard_gate_applied": ocr_result.get("hard_gate_applied", False),
        "hard_gate_reason": ocr_result.get("hard_gate_reason", ""),
        "confident_block_count": ocr_result.get("confident_block_count", 0),
        "uncertain_block_count": ocr_result.get("uncertain_block_count", 0),
        # v5.11.5.12: URL trust boundary
        "url_uncertain_count": ocr_result.get("url_uncertain_count", 0),
        "malformed_url_candidate_count": ocr_result.get("malformed_url_candidate_count", 0),
        "deterministic_url_eligible_count": ocr_result.get("deterministic_url_eligible_count", 0),
        "url_audits": ocr_result.get("url_audits", []),
    }
    return audit

@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_dirs()
    logger.info(f"API v{APP_VERSION} starting, runtime={PHISHING_APP_HOME}")

    # v5.11.5.1: Explicit environment-based mode control
    llm_enabled = get_llm_enabled()
    allow_fallback = get_allow_deterministic_fallback()
    llm_configured = check_llm_configured()

    app.state.llm_requested = llm_enabled
    app.state.llm_configured = llm_configured
    app.state.deterministic_fallback_allowed = allow_fallback

    # v5.11.5.2: executor created AFTER pipeline init to prevent leak on config failure
    app.state.executor = None
    try:
        if not llm_enabled:
            # Explicit rule-only — not a config failure
            app.state.pipeline = AnalysisPipeline(use_llm=False)
            app.state.operating_mode = "rule_only"
            app.state.llm_active = False
            logger.info("Pipeline ready (rule_only by LLM_ENABLED=false)")
        elif llm_configured:
            app.state.pipeline = AnalysisPipeline(use_llm=True)
            app.state.operating_mode = "llm"
            app.state.llm_active = True
            logger.info("Pipeline ready (LLM active)")
        elif allow_fallback:
            app.state.pipeline = AnalysisPipeline(use_llm=False, allow_deterministic_fallback=True)
            app.state.operating_mode = "degraded"
            app.state.llm_active = False
            logger.warning("LLM not configured, starting with rule-only (deterministic fallback allowed)")
        else:
            raise LLMConfigError(
                "LLM not configured and ALLOW_DETERMINISTIC_FALLBACK=false. "
                "Set ALLOW_DETERMINISTIC_FALLBACK=true to continue with rule-only, "
                "or configure LLM credentials."
            )
        # v5.11.5.2: create executor only after pipeline init succeeds
        app.state.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    except LLMConfigError:
        raise
    except Exception as e:
        if type(e).__name__ == "LLMConfigError":
            raise
        logger.error(f"Pipeline fail: {e}"); app.state.pipeline = None
        app.state.operating_mode = "failed"
        app.state.llm_active = False
        # v5.11.5.2: create executor even in failed mode (needed for shutdown)
        app.state.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try: app.state.ocr_engine = None; logger.info("OCR deferred")
    except: pass
    app.state.start_time = time.time()
    yield
    if app.state.executor is not None:
        app.state.executor.shutdown(wait=True)

app = FastAPI(title=f"Phishing Detection API v{APP_VERSION}", version=API_VERSION, lifespan=lifespan, docs_url="/docs")

if API_CONFIG.get("enable_cors"):
    from fastapi.middleware.cors import CORSMiddleware
    origins = API_CONFIG.get("cors_origins",["http://localhost:3000"])
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET","POST"], allow_headers=["*"])

def _validate_image(file, image_bytes):
    if not file.filename: raise HTTPException(400, detail="No filename")
    ext = "."+file.filename.rsplit(".",1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_IMAGE_FORMATS: raise HTTPException(400, detail=f"Bad format: {ext}")
    if len(image_bytes) > API_CONFIG["max_upload_size"]: raise HTTPException(413)
    if file.content_type and file.content_type not in SUPPORTED_IMAGE_MIMES: raise HTTPException(400, detail="Bad MIME")
    valid_sig = any(image_bytes[:len(m)]==m for m in IMAGE_MAGIC_BYTES)
    if not valid_sig: raise HTTPException(400, detail="Bad signature")
    try:
        from PIL import Image; img = Image.open(BytesIO(image_bytes)); img.verify()
        img = Image.open(BytesIO(image_bytes)); w,h = img.size
        if w*h > API_CONFIG.get("max_image_pixels",100_000_000): raise HTTPException(400, detail="Too many pixels")
        # Verify extension, MIME, and Pillow format are consistent
        pillow_fmt = (img.format or "").lower()
        EXT_TO_FMT = {".jpg":"jpeg",".jpeg":"jpeg",".png":"png",".bmp":"bmp",".tiff":"tiff",".tif":"tiff",".webp":"webp"}
        MIME_TO_FMT = {"image/jpeg":"jpeg","image/png":"png","image/bmp":"bmp","image/tiff":"tiff","image/webp":"webp"}
        ext_fmt = EXT_TO_FMT.get(ext, "")
        mime_fmt = MIME_TO_FMT.get((file.content_type or "").lower(), "")
        if ext_fmt and pillow_fmt != ext_fmt:
            raise HTTPException(400, detail=f"Extension {ext} does not match image format {pillow_fmt}")
        if mime_fmt and pillow_fmt != mime_fmt:
            raise HTTPException(400, detail=f"MIME {file.content_type} does not match image format {pillow_fmt}")
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, detail=f"Decode fail: {e}")

@app.post("/api/v1/analyze/text", tags=["Analysis"])
async def analyze_text(request: TextRequest):
    if not app.state.pipeline: raise HTTPException(503, detail="Pipeline not ready")
    try:
        loop = __import__('asyncio').get_event_loop()
        result = await loop.run_in_executor(app.state.executor,
            lambda: app.state.pipeline.analyze(
                content=request.content, content_type=request.content_type,
                source_channel=request.source_channel, metadata=request.metadata,
                strategy=request.scoring_strategy.value))
        return result.model_dump()
    except HTTPException: raise
    except Exception as e: logger.error(f"Fail: {e}"); raise HTTPException(500, detail=str(e))

@app.post("/api/v1/analyze/image", tags=["Analysis"])
async def analyze_image(file: UploadFile = File(...), source_channel: str = "screenshot",
                         scoring_strategy: ScoringStrategy = ScoringStrategy.weighted):
    image_bytes = await file.read()
    _validate_image(file, image_bytes)
    if not app.state.pipeline: raise HTTPException(503)
    # OCR engine required
    if not hasattr(app.state, 'ocr_engine') or app.state.ocr_engine is None:
        try:
            from ocr_module import OCREngine
            app.state.ocr_engine = OCREngine()
        except Exception as e:
            raise HTTPException(503, detail=f"OCR engine unavailable: {e}")
    try:
        loop = __import__('asyncio').get_event_loop()
        try:
            ocr_result = await loop.run_in_executor(app.state.executor,
                lambda: app.state.ocr_engine.extract_text_with_details(image_bytes))
        except ImportError as e:
            raise HTTPException(503, detail=f"OCR model not installed: {e}")
        except RuntimeError as e:
            if "model" in str(e).lower() or "download" in str(e).lower():
                raise HTTPException(503, detail=f"OCR model unavailable: {e}")
            raise HTTPException(500, detail=str(e))
        except Exception as e:
            raise HTTPException(500, detail=f"OCR processing error: {e}")
        full_text = ocr_result.get("corrected_text", ocr_result.get("full_text", ""))
        ocr_quality_status = ocr_result.get("ocr_quality_status", "unknown")

        # v5.11.5.12: No text or insufficient OCR evidence → return structured insufficient_evidence
        if not full_text.strip() or ocr_quality_status == "insufficient_evidence":
            audit = _build_ocr_audit(ocr_result)
            return {
                "analysis_id": f"PHISH-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "status": "insufficient_evidence",
                "summary": (
                    "OCR could not extract reliable text from this image. "
                    "Please provide a clearer screenshot with better resolution and contrast."
                ),
                "risk_score": None,
                "risk_level": "unknown",
                "ocr_audit": audit,
                "warnings": [{
                    "type": "insufficient_ocr_quality",
                    "message": (
                        f"OCR quality: {ocr_quality_status} — "
                        f"text extraction is unreliable, no risk assessment performed"
                    )
                }],
            }
        result = await loop.run_in_executor(app.state.executor,
            lambda: app.state.pipeline.analyze(
                content=full_text, content_type="image",
                source_channel=source_channel, metadata={"input_type":"image_ocr"},
                ocr_result=ocr_result, strategy=scoring_strategy.value))
        response = result.model_dump()
        # v5.11.5.12: Include comprehensive OCR audit fields
        response["ocr_audit"] = _build_ocr_audit(ocr_result)
        # v5.11.5.12: Flag uncertain/malformed URLs — not deterministic evidence
        if ocr_result.get("url_uncertain_count", 0) > 0:
            response["warnings"] = response.get("warnings", []) + [{
                "type": "uncertain_url_evidence",
                "message": f"{ocr_result['url_uncertain_count']} URL(s) unverified by secondary OCR — not used as deterministic evidence"
            }]
        if ocr_result.get("malformed_url_candidate_count", 0) > 0:
            response["warnings"] = response.get("warnings", []) + [{
                "type": "malformed_url_candidate",
                "message": f"{ocr_result['malformed_url_candidate_count']} malformed URL candidate(s) detected — not used as evidence"
            }]
        # v5.11.5.12: Include whether OCR quality allows reliable scoring
        response["ocr_scoring_reliable"] = (
            ocr_quality_status in ("reliable", "usable_with_warnings")
        )
        return response
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, detail=str(e))

@app.post("/api/v1/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchRequest):
    if not app.state.pipeline: raise HTTPException(503)
    results=[]; success=0; errors=0
    for item in request.items:
        item.scoring_strategy = request.scoring_strategy
        try: r = await analyze_text(item); results.append(r); success+=1
        except HTTPException as e: errors+=1; results.append({"analysis_id":"ERROR","status":"error","error":str(e.detail)})
    return {"total":len(request.items),"success":success,"errors":errors,"results":results}

@app.get("/api/v1/health", tags=["System"])
async def health():
    pipeline_ok = app.state.pipeline is not None
    ocr_ok = hasattr(app.state,'ocr_engine') and app.state.ocr_engine is not None
    # v5.11.5.1: detailed health with operating mode and llm status
    llm_requested = getattr(app.state, 'llm_requested', False)
    llm_configured = getattr(app.state, 'llm_configured', False)
    llm_active = getattr(app.state, 'llm_active', False)
    operating_mode = getattr(app.state, 'operating_mode', 'unknown')
    det_fallback_allowed = getattr(app.state, 'deterministic_fallback_allowed', False)

    components = {"pipeline":"healthy" if pipeline_ok else "unavailable",
                  "ocr":"ready" if ocr_ok else "not_loaded"}
    return {
        "status": "healthy" if pipeline_ok else "degraded",
        "version": API_VERSION,
        "app_version": APP_VERSION,
        "components": components,
        "llm_requested": llm_requested,
        "llm_configured": llm_configured,
        "llm_active": llm_active,
        "deterministic_fallback_allowed": det_fallback_allowed,
        "operating_mode": operating_mode,
    }

@app.get("/api/v1/stats", tags=["System"])
async def stats(): return {"api_version":API_VERSION}

@app.exception_handler(HTTPException)
async def http_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error":"http_error","message":exc.detail,"timestamp":datetime.now().isoformat()})
