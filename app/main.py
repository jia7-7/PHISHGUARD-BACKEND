from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, load_settings
from app.detectors.base import Detector
from app.detectors.integrated_rules import IntegratedRuleDetector
from app.detectors.remote_ai import RemoteAIDetector
from app.models import (
    AnalysisPayload,
    AnalysisResponse,
    HealthResponse,
    HtmlAnalysisRequest,
    InputKind,
    PublicConfigResponse,
    TextAnalysisRequest,
    UrlAnalysisRequest,
)
from app.services.analyzer import AnalysisService, AnalysisUnavailableError
from app.services.fusion import RiskFusionService
from app.services.image_ocr import (
    ImageOCRInsufficientError,
    ImageOCRService,
    ImageOCRUnavailableError,
)


SUPPORTED_FILE_TYPES = [".txt", ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp"]
RULE_ENGINE_MAX_INPUT_BYTES = 200 * 1024
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app(
    settings: Settings | None = None,
    detectors: Sequence[Detector] | None = None,
    image_ocr: ImageOCRService | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    configured_detectors = list(detectors) if detectors is not None else [
        IntegratedRuleDetector(settings),
        RemoteAIDetector(settings),
    ]
    analyzer = AnalysisService(configured_detectors, RiskFusionService(settings))
    image_ocr = image_ocr or ImageOCRService()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Evidence-based phishing analysis API. Raw input is processed in memory and not stored.",
    )
    app.state.settings = settings
    app.state.analyzer = analyzer
    app.state.detectors = configured_detectors

    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False, response_model=None)
    async def service_index():
        frontend_index = FRONTEND_DIR / "index.html"
        if frontend_index.is_file():
            return FileResponse(frontend_index)
        return JSONResponse(
            {
                "service": settings.app_name,
                "version": settings.app_version,
                "status": "running",
                "health": "/health",
                "docs": "/docs",
            }
        )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )

    @app.exception_handler(AnalysisUnavailableError)
    async def handle_unavailable(_: Request, exc: AnalysisUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            version=settings.app_version,
            configured_detectors=[detector.name for detector in configured_detectors],
        )

    @app.get("/api/v1/config", response_model=PublicConfigResponse, tags=["system"])
    async def public_config() -> PublicConfigResponse:
        return PublicConfigResponse(
            max_text_chars=settings.max_text_chars,
            max_upload_bytes=settings.max_upload_bytes,
            supported_file_types=SUPPORTED_FILE_TYPES,
            risk_thresholds={
                "medium": settings.medium_threshold,
                "high": settings.high_threshold,
                "critical": settings.critical_threshold,
            },
        )

    @app.post("/api/v1/analyze/text", response_model=AnalysisResponse, tags=["analysis"])
    async def analyze_text(body: TextAnalysisRequest) -> AnalysisResponse:
        _validate_text_size(body.content, settings)
        return await analyzer.analyze(
            AnalysisPayload(
                kind=InputKind.TEXT,
                text=body.content,
                source_type=body.source_type,
                metadata={
                    "sender": body.sender,
                    "attachments": body.attachments,
                    "qr_urls": body.qr_urls,
                },
            )
        )

    @app.post("/api/v1/analyze/url", response_model=AnalysisResponse, tags=["analysis"])
    async def analyze_url(body: UrlAnalysisRequest) -> AnalysisResponse:
        _validate_http_url(body.url)
        return await analyzer.analyze(AnalysisPayload(kind=InputKind.URL, url=body.url))

    @app.post("/api/v1/analyze/html", response_model=AnalysisResponse, tags=["analysis"])
    async def analyze_html(body: HtmlAnalysisRequest) -> AnalysisResponse:
        _validate_text_size(body.html, settings)
        if body.source_url:
            _validate_http_url(body.source_url)
        return await analyzer.analyze(
            AnalysisPayload(
                kind=InputKind.HTML,
                html=body.html,
                url=body.source_url,
                source_type="html",
                metadata={"sender": body.sender},
            )
        )

    @app.post("/api/v1/analyze/file", response_model=AnalysisResponse, tags=["analysis"])
    async def analyze_file(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        content_type: str | None = Header(default=None),
    ) -> AnalysisResponse:
        raw = await _read_limited_body(request, settings.max_upload_bytes)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {suffix or 'none'}",
            )

        if suffix in {".txt", ".html", ".htm"}:
            text = _decode_text(raw)
            _validate_text_size(text, settings)
            kind = InputKind.TEXT if suffix == ".txt" else InputKind.HTML
            return await analyzer.analyze(
                AnalysisPayload(
                    kind=kind,
                    text=text if kind == InputKind.TEXT else None,
                    html=text if kind == InputKind.HTML else None,
                    filename=filename,
                    content_type=content_type,
                    source_type="text" if kind == InputKind.TEXT else "html",
                )
            )

        _validate_image_signature(raw, suffix)
        try:
            ocr_result = await asyncio.to_thread(image_ocr.extract, raw)
        except ImageOCRInsufficientError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except ImageOCRUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        result = await analyzer.analyze(
            AnalysisPayload(
                kind=InputKind.TEXT,
                text=ocr_result.text,
                filename=filename,
                content_type=content_type,
                source_type="other",
                metadata={
                    "ocr_text": ocr_result.text,
                    "ocr_confidence": ocr_result.confidence,
                    "ocr_quality_status": ocr_result.quality_status,
                },
            )
        )
        result.input_type = InputKind.IMAGE
        result.processing_ms += ocr_result.processing_ms
        ocr_notice = (
            "图片已在服务器完成文字识别"
            f"（平均置信度 {round(ocr_result.confidence * 100)}%，"
            f"质量状态 {ocr_result.quality_status}），请结合原图复核。"
        )
        result.warnings = list(
            dict.fromkeys([ocr_notice, *ocr_result.warnings, *result.warnings])
        )[:20]
        return result

    return app


def _validate_text_size(content: str, settings: Settings) -> None:
    if len(content) > settings.max_text_chars:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Text exceeds {settings.max_text_chars} characters",
        )
    if len(content.encode("utf-8")) > RULE_ENGINE_MAX_INPUT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"UTF-8 text exceeds {RULE_ENGINE_MAX_INPUT_BYTES} bytes",
        )


def _validate_http_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only absolute HTTP and HTTPS URLs are supported",
        )


async def _read_limited_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds {limit} bytes",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header",
            ) from None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds {limit} bytes",
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    return b"".join(chunks)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Text file encoding must be UTF-8 or GB18030",
    )


def _validate_image_signature(raw: bytes, suffix: str) -> None:
    valid = {
        ".png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": raw.startswith(b"\xff\xd8\xff"),
        ".jpeg": raw.startswith(b"\xff\xd8\xff"),
        ".webp": len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP",
    }
    if not valid.get(suffix, False):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content does not match its image extension",
        )


app = create_app()
