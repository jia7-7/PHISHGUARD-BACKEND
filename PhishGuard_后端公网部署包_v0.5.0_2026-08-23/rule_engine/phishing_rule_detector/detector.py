"""顶层入口 detect() — 编排全检测流程.

注意：本模块仅返回规则侧结果，不做 AI 融合。
AI 融合由后端综合判定模块负责。
"""
from __future__ import annotations

import logging
import time
from typing import Any, cast

from phishing_rule_detector.config_loader import load_config
from pydantic import ValidationError

from phishing_rule_detector.models import (
    DetectionInput,
    DetectionResult,
    ErrorResult,
    EvidenceItem,
    INPUT_TYPES,
    MAX_INPUT_BYTES,
    generate_trace_id,
)
from phishing_rule_detector.normalizer import PayloadTooLargeError, normalize, parse_html_dom
from phishing_rule_detector.scoring import score_pipeline
from phishing_rule_detector.rules import (  # noqa: F401 — 导入即注册所有规则
    url_rules,
    domain_rules,
    html_rules,
    text_rules,
    email_rules,
    image_rules,
)
from phishing_rule_detector.rules.common import (
    RuleContext,
    _canonical_url_key,
    _parse_url,
    extract_urls_from_html,
    extract_urls_from_text,
    run_all_rules,
)

_logger = logging.getLogger(__name__)


def detect(
    input_text: str,
    input_type: str,
    base_url: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """安全规则检测顶层入口.

    Args:
        input_text: 待检测的 URL / HTML / 邮件正文 / 短信 / 普通文本.
        input_type: "url" | "html" | "email" | "sms" | "text".
        base_url: 页面来源 URL（HTML 类型时有意义）.
        context: 扩展上下文（sender, attachments, qr_urls, ocr_text, debug）.

    Returns:
        成功时返回 DetectionResult 的 model_dump() dict.
        失败时返回 ErrorResult 的 model_dump() dict.
    """
    trace_id = generate_trace_id()
    start = time.perf_counter()
    # context 必须严格为 dict 或 None，[] 等不得被静默转换（审查 Fix 7）
    ctx = context if context is not None else {}

    # ── 输入校验 ──
    try:
        inp = DetectionInput(
            input_text=input_text,
            input_type=cast(INPUT_TYPES, input_type),
            base_url=base_url,
            context=ctx,
        )
    except ValidationError as e:
        errors = e.errors()
        err_msg = str(e.errors()[0]["msg"]) if e.errors() else str(e)

        # 检查是否由 model_validator 触发（大小校验）
        combined_msg = " ".join(err.get("msg", "") for err in errors)
        if "字节" in combined_msg or "超过" in combined_msg:
            code = "PAYLOAD_TOO_LARGE"
            actual_bytes = len(input_text.encode("utf-8"))
            ocr = ctx.get("ocr_text", "")
            if ocr and isinstance(ocr, str):
                actual_bytes += len(ocr.encode("utf-8"))
            error_result = ErrorResult(
                success=False,
                trace_id=trace_id,
                error={
                    "code": code,
                    "message": err_msg,
                    "details": {
                        "max_bytes": MAX_INPUT_BYTES,
                        "actual_bytes": actual_bytes,
                    },
                },
            )
            return error_result.model_dump()
        else:
            # 按字段和错误类型分类
            error_fields = {err["loc"][0] for err in errors if err.get("loc")}
            if "input_text" in error_fields:
                code = "EMPTY_INPUT"
            elif "input_type" in error_fields:
                code = "INVALID_INPUT_TYPE"
            else:
                code = "INVALID_CONTEXT"

        error_result = ErrorResult(
            success=False,
            trace_id=trace_id,
            error={
                "code": code,
                "message": err_msg,
            },
        )
        return error_result.model_dump()
    except ValueError as e:
        err_msg = str(e)
        # 大小校验等
        if "字节" in err_msg or "大小" in err_msg or "超过" in err_msg:
            code = "PAYLOAD_TOO_LARGE"
            actual_bytes = len(input_text.encode("utf-8"))
            ocr = ctx.get("ocr_text", "")
            if ocr and isinstance(ocr, str):
                actual_bytes += len(ocr.encode("utf-8"))
            error_result = ErrorResult(
                success=False,
                trace_id=trace_id,
                error={
                    "code": code,
                    "message": err_msg,
                    "details": {
                        "max_bytes": MAX_INPUT_BYTES,
                        "actual_bytes": actual_bytes,
                    },
                },
            )
        else:
            code = "INVALID_CONTEXT"
            error_result = ErrorResult(
                success=False,
                trace_id=trace_id,
                error={
                    "code": code,
                    "message": err_msg,
                },
            )
        return error_result.model_dump()

    # ── 规范化 ──
    try:
        normalized_text, norm_meta = normalize(
            input_text=input_text,
            input_type=input_type,
            base_url=base_url,
            ocr_text=ctx.get("ocr_text", ""),
        )
    except PayloadTooLargeError as e:
        error_result = ErrorResult(
            success=False,
            trace_id=trace_id,
            error={
                "code": "PAYLOAD_TOO_LARGE",
                "message": str(e),
                "details": {
                    "max_bytes": e.max_bytes,
                    "actual_bytes": e.actual_bytes,
                },
            },
        )
        return error_result.model_dump()
    except Exception as exc:
        _logger.error(
            "输入规范化内部错误 [trace_id=%s, error_type=%s]",
            trace_id,
            type(exc).__name__,
        )
        error_result = ErrorResult(
            success=False,
            trace_id=trace_id,
            error={
                "code": "INTERNAL_RULE_ERROR",
                "message": "输入规范化内部错误",
                "trace_id": trace_id,
            },
        )
        return error_result.model_dump()

    # URL 类型表示单个用户粘贴链接。浏览器地址栏常省略协议，
    # 推断 https 后再进入统一 URL 提取和域名规则。
    if input_type == "url":
        url_value = normalized_text.strip()
        if url_value.startswith("//"):
            normalized_text = "https:" + url_value
            norm_meta["operations"].append("url_scheme_inferred")
        elif "://" not in url_value:
            normalized_text = "https://" + url_value
            norm_meta["operations"].append("url_scheme_inferred")

    warnings: list[str] = []
    debug_mode = ctx.get("debug", False)

    # ── 规则检测 ──
    # Phase 2: 构建 RuleContext，运行所有已注册规则
    sender = ctx.get("sender", "")
    attachments = ctx.get("attachments", [])
    qr_urls = ctx.get("qr_urls", [])
    ocr_text = ctx.get("ocr_text", "")

    # 二维码 URL 使用与主输入一致的反混淆管线。原始 URL 仍会加入
    # extracted_urls，以保留 Punycode 等域名攻击的原始证据。
    normalized_qr_urls: list[str] = []
    for qr_index, qr_url in enumerate(qr_urls):
        try:
            normalized_qr_url, _ = normalize(qr_url, "url")
            normalized_qr_urls.append(normalized_qr_url)
        except Exception:
            normalized_qr_urls.append(qr_url)
            warnings.append(f"QR_URL_NORMALIZATION_ERROR:qr:{qr_index}")

    # 解析 HTML DOM（仅 HTML 类型，从 raw_text 解析）
    parsed_html = None
    if input_type == "html":
        try:
            parsed_html, parse_meta = parse_html_dom(inp.input_text)
            if parse_meta.get("parser_fallback"):
                warnings.append("HTML_PARSER_FALLBACK")
            if parse_meta.get("error"):
                warnings.append("HTML_PARSE_ERROR")
        except Exception:
            _logger.warning("HTML DOM 解析失败 [trace_id=%s]", trace_id)
            warnings.append("HTML_PARSE_ERROR")

    # 提取 URL（同时从原始和规范化文本中提取，保留 xn-- 等原始特征）
    extracted_urls: list = []
    seen_urls: set[str] = set()
    def _add_url(eu) -> None:
        key = _canonical_url_key(eu.raw)
        if key not in seen_urls:
            seen_urls.add(key)
            if not eu.subject_id:
                eu.subject_id = f"url:{len(extracted_urls)}"
            extracted_urls.append(eu)
            if eu.parse_warning:
                warnings.append(f"URL_PARSE_ERROR:{eu.subject_id}")

    try:
        if parsed_html is not None:
            for eu in extract_urls_from_html(parsed_html, base_url):
                _add_url(eu)
        # 从原始文本提取（保留 Punycode 等特征）
        for eu in extract_urls_from_text(norm_meta["raw_text"], base_url):
            _add_url(eu)
        # 也从规范化文本提取（可能包含解码后的 URL）
        if normalized_text != norm_meta["raw_text"]:
            for eu in extract_urls_from_text(normalized_text, base_url):
                _add_url(eu)
    except Exception:
        _logger.warning("URL 提取异常 [trace_id=%s]", trace_id)
        warnings.append("URL_EXTRACTION_ERROR")

    # P0 #3: 将 base_url 加入 extracted_urls，使 URL/域名规则能获取身份证据
    if base_url and base_url.startswith(("http://", "https://")):
        try:
            base_eu = _parse_url(
                base_url,
                source="raw",
                element_type="base",
            )
            if base_eu is not None:
                _add_url(base_eu)
            else:
                warnings.append("URL_PARSE_ERROR:base")
        except Exception:
            _logger.warning("base_url 解析失败 [trace_id=%s]", trace_id)
            warnings.append("URL_PARSE_ERROR:base")

    # QR URL 参与通用 URL/域名规则，同时保留 qr:N 作为抑制作用域。
    for qr_index, qr_url in enumerate(qr_urls):
        try:
            qr_eu = _parse_url(
                qr_url,
                source="qr",
                element_type="qr",
            )
            if qr_eu is not None:
                qr_eu.subject_id = f"qr:{qr_index}"
                _add_url(qr_eu)
            else:
                warnings.append(f"URL_PARSE_ERROR:qr:{qr_index}")
        except Exception:
            _logger.warning(
                "二维码 URL 解析失败 [trace_id=%s, qr_index=%s]",
                trace_id,
                qr_index,
            )
            warnings.append(f"URL_PARSE_ERROR:qr:{qr_index}")

    rule_ctx = RuleContext(
        input_text=inp.input_text,
        normalized_text=normalized_text,
        input_type=input_type,
        base_url=base_url,
        raw_text=norm_meta["raw_text"],
        sender=sender,
        attachments=attachments,
        qr_urls=normalized_qr_urls,
        ocr_text=ocr_text,
        parsed_html=parsed_html,
        extracted_urls=extracted_urls,
        trace_id=trace_id,
    )

    # 运行所有注册规则
    evidence: list[EvidenceItem] = []
    try:
        evidence, rule_warnings = run_all_rules(rule_ctx)
        warnings.extend(rule_warnings)
    except Exception as exc:
        _logger.error(
            "规则执行异常 [trace_id=%s, error_type=%s]",
            trace_id,
            type(exc).__name__,
        )
        error_result = ErrorResult(
            success=False,
            trace_id=trace_id,
            error={
                "code": "INTERNAL_RULE_ERROR",
                "message": "规则引擎内部错误",
                "trace_id": trace_id,
            },
        )
        return error_result.model_dump()

    # ── 评分与等级门控 ──
    try:
        risk, active_evidence, suppressed = score_pipeline(evidence)
    except Exception as exc:
        _logger.error(
            "评分管线内部错误 [trace_id=%s, error_type=%s]",
            trace_id,
            type(exc).__name__,
        )
        error_result = ErrorResult(
            success=False,
            trace_id=trace_id,
            error={
                "code": "INTERNAL_RULE_ERROR",
                "message": "评分管线内部错误",
                "trace_id": trace_id,
            },
        )
        return error_result.model_dump()

    # ── 统计摘要 ──
    high_count = sum(1 for e in active_evidence if e.severity == "high")
    medium_count = sum(1 for e in active_evidence if e.severity == "medium")
    low_count = sum(1 for e in active_evidence if e.severity == "low")
    evidence_groups = sorted({e.group for e in active_evidence})
    suppressed_count = len(suppressed)

    summary = {
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "evidence_groups": evidence_groups,
        "suppressed_count": suppressed_count,
    }

    # ── 构建返回 ──
    duration_ms = int((time.perf_counter() - start) * 1000)
    config = load_config()

    result = DetectionResult(
        success=True,
        trace_id=trace_id,
        rule_version=config.scoring.rule_version,
        risk=risk,
        evidence=active_evidence,
        summary=summary,
        normalization={
            "operations": norm_meta["operations"],
            "input_bytes": norm_meta["input_bytes"],
            "input_truncated": norm_meta["input_truncated"],
        },
        warnings=warnings,
        duration_ms=duration_ms,
        suppressed_evidence=suppressed if debug_mode else None,
    )

    return result.model_dump(exclude_none=True)
