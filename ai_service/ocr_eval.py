"""
=============================================================================
OCR专项评测工具 v2.0 — v5.11.5.12
=============================================================================
v5.11.5.12 改进:
  - --mode CLI: safe_base / mild_enhanced / none / adaptive
  - Raw vs corrected CER/WER (分列 raw 和 corrected)
  - URL violation tracking (推测性字符补写检测)
  - Fabricated character violation count
  - Preprocessing pipeline version in manifest

支持指标:
  - CER (Character Error Rate): 字符级编辑距离
  - WER (Word Error Rate):   词级编辑距离 (CJK感知)
  - URL Accuracy:            URL字段专项精度
  - URL Violations:          推测性字符补写违规计数
  - Block-level metrics:     逐块精度、置信度相关性
  - Batch evaluation:        目录批量评测

用法:
  python code/ocr_eval.py --images <dir> --ground-truth <dir> --output <file>.json
  python code/ocr_eval.py --images <dir> --ground-truth <dir> --mode safe_base
  python code/ocr_eval.py --images <dir> --ground-truth <dir> --mode none --engine easyocr

安全约束: 不调用真实LLM, 不读取API密钥, 不联网下载模型.
=============================================================================
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from config import APP_VERSION, SUPPORTED_IMAGE_FORMATS


# ============================================================================
# Edit Distance (Levenshtein)
# ============================================================================
def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[j] + 1,
                prev[j + 1] + 1,
                prev[j] + cost,
            ))
        prev = curr
    return prev[-1]


# ============================================================================
# Metric: CER (Character Error Rate)
# ============================================================================
def compute_cer(predicted: str, ground_truth: str) -> float:
    """Compute Character Error Rate.
    CER = edit_distance(pred, gt) / max(len(gt), 1)
    Range [0, inf), lower is better.
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    dist = _levenshtein(predicted, ground_truth)
    return dist / max(len(ground_truth), 1)


# ============================================================================
# Metric: WER (Word Error Rate)
# ============================================================================
def compute_wer(predicted: str, ground_truth: str) -> float:
    """Compute Word Error Rate.
    For CJK text, treats each character as a word.
    For Latin/whitespace-separated text, uses word-level tokens.
    """
    if any('一' <= c <= '鿿' for c in predicted + ground_truth):
        pred_tokens = list(predicted.replace(' ', '').replace('\n', ''))
        gt_tokens = list(ground_truth.replace(' ', '').replace('\n', ''))
    else:
        pred_tokens = predicted.strip().split()
        gt_tokens = ground_truth.strip().split()

    if not gt_tokens:
        return 1.0 if pred_tokens else 0.0
    dist = _levenshtein(pred_tokens, gt_tokens)
    return dist / max(len(gt_tokens), 1)


# ============================================================================
# Metric: URL Accuracy
# ============================================================================
_URL_RE = re.compile(
    r'(https?://[^\s<>"\'{}\[\]()（）一-鿿　-〿＀-￯]*)',
    re.IGNORECASE,
)


def _extract_urls_from_text(text: str) -> List[str]:
    """Extract all http/https URLs from text."""
    return [m.group(1) for m in _URL_RE.finditer(text)]


def compute_url_accuracy(predicted: str, ground_truth: str) -> Dict[str, Any]:
    """Compute URL-specific accuracy metrics.

    Returns dict with:
      - url_exact_match: bool
      - url_count_gt: int
      - url_count_pred: int
      - domain_match: bool
      - protocol_preserved: bool
      - tld_correct: bool
      - details: list of per-URL comparisons
    """
    gt_urls = _extract_urls_from_text(ground_truth)
    pred_urls = _extract_urls_from_text(predicted)

    result = {
        "url_exact_match": False,
        "url_count_gt": len(gt_urls),
        "url_count_pred": len(pred_urls),
        "domain_match": False,
        "protocol_preserved": True,
        "tld_correct": True,
        "details": [],
    }

    if not gt_urls:
        result["url_exact_match"] = True
        result["domain_match"] = True
        return result

    if len(gt_urls) != len(pred_urls):
        return result

    all_exact = True
    all_domain = True

    for gt, pred in zip(gt_urls, pred_urls):
        detail = {"gt": gt, "pred": pred, "exact": gt == pred}

        from urllib.parse import urlparse
        try:
            gt_parsed = urlparse(gt)
            pred_parsed = urlparse(pred)
            detail["gt_domain"] = (gt_parsed.hostname or "")
            detail["pred_domain"] = (pred_parsed.hostname or "")
            detail["domain_match"] = (
                detail["gt_domain"].lower() == detail["pred_domain"].lower()
            )
            if not detail["domain_match"]:
                all_domain = False

            detail["protocol_ok"] = gt_parsed.scheme == pred_parsed.scheme
            if not detail["protocol_ok"]:
                result["protocol_preserved"] = False

            gt_host = (gt_parsed.hostname or "")
            pred_host = (pred_parsed.hostname or "")
            gt_tld = gt_host.rsplit(".", 1)[-1] if "." in gt_host else gt_host
            pred_tld = pred_host.rsplit(".", 1)[-1] if "." in pred_host else pred_host
            detail["tld_match"] = gt_tld.lower() == pred_tld.lower()
            if not detail["tld_match"]:
                result["tld_correct"] = False
        except Exception:
            detail["domain_match"] = False
            detail["protocol_ok"] = False
            detail["tld_match"] = False
            all_domain = False
            result["protocol_preserved"] = False
            result["tld_correct"] = False

        if not detail["exact"]:
            all_exact = False

        result["details"].append(detail)

    result["url_exact_match"] = all_exact
    result["domain_match"] = all_domain
    return result


# ============================================================================
# v5.11.5.12: URL Violation Detection
# ============================================================================
def detect_url_violations(raw_text: str, corrected_text: str) -> Dict[str, Any]:
    """v5.11.5.12: Detect fabricated character violations in URL post-processing.

    A violation occurs when corrected_text contains URL characters that are
    NOT present in raw_text and are NOT a 1:1 fullwidth→halfwidth normalization.

    Returns dict with:
      - violation_count: int — number of URL blocks with fabricated chars
      - violations: list of per-URL violation details
      - fabricated_chars: set of characters that were fabricated
    """
    raw_urls = _extract_urls_from_text(raw_text)
    corrected_urls = _extract_urls_from_text(corrected_text)

    violations = []
    fabricated_chars = set()

    # 1:1 normalization mappings that are NOT violations
    ONE_TO_ONE_MAP = {
        '：': ':', '．': '.', '／': '/', '＼': '\\', '＠': '@',
        'ｈ': 'h', 'ｔ': 't', 'ｐ': 'p', 'ｓ': 's',  # fullwidth basic latin
    }

    for i, (raw, corr) in enumerate(zip(raw_urls, corrected_urls)):
        # Find characters in corrected that are NOT:
        # 1. In the raw string
        # 2. A 1:1 fullwidth→halfwidth normalization of a raw char
        raw_set = set(raw)
        corr_set = set(corr)

        fabricated = set()
        for c in corr:
            if c in raw_set:
                continue
            # Check if c is a halfwidth normalization of some fullwidth char in raw
            normalized_from = False
            for fw, hw in ONE_TO_ONE_MAP.items():
                if hw == c and fw in raw_set:
                    normalized_from = True
                    break
            if not normalized_from:
                fabricated.add(c)

        if fabricated:
            fabricated_chars.update(fabricated)
            violations.append({
                "url_index": i,
                "raw_url": raw,
                "corrected_url": corr,
                "fabricated_chars": sorted(fabricated),
                "violation_type": "character_fabrication",
            })

    return {
        "violation_count": len(violations),
        "violations": violations,
        "fabricated_chars": sorted(fabricated_chars),
    }


# ============================================================================
# Per-Image Evaluation
# ============================================================================
def evaluate_single_image(
    image_path: Path,
    ground_truth_path: Path,
    engine: str = "easyocr",
    mode: Optional[str] = None,
    preprocess: Optional[str] = None,  # v5.11.5.9 backward compat
) -> Dict[str, Any]:
    """Evaluate a single image against its ground truth text file.

    Args:
        image_path: Path to image file
        ground_truth_path: Path to .txt ground truth file
        engine: OCR engine name
        mode: v5.11.5.12 preprocessing mode (safe_base, mild_enhanced, none, adaptive)
        preprocess: v5.11.5.9 backward compat

    Returns dict with comprehensive evaluation metrics.
    """
    from ocr_module import OCREngine

    gt_text = ground_truth_path.read_text(encoding="utf-8").strip()

    # v5.11.5.12: mode takes precedence over preprocess
    effective_mode = mode
    if effective_mode is None and preprocess is not None:
        # Backward compat mapping
        if preprocess == "none":
            effective_mode = "none"
        elif preprocess == "baseline":
            effective_mode = "safe_base"
        elif preprocess == "standard":
            effective_mode = "legacy_enhanced"

    ocr = OCREngine(engine=engine)
    result = ocr.extract_text_with_details(image_path, mode=effective_mode)

    raw_text = result.get("full_text", "")
    corrected_text = result.get("corrected_text", "")

    # Raw CER/WER (on raw OCR text)
    raw_cer = compute_cer(raw_text, gt_text)
    raw_wer = compute_wer(raw_text, gt_text)

    # Corrected CER/WER (on post-processed text)
    corrected_cer = compute_cer(corrected_text, gt_text)
    corrected_wer = compute_wer(corrected_text, gt_text)

    # URL metrics (on corrected text)
    url_metrics = compute_url_accuracy(corrected_text, gt_text)

    # v5.11.5.12: URL violation detection
    url_violations = detect_url_violations(raw_text, corrected_text)

    return {
        "image": str(image_path.name),
        "ground_truth_file": str(ground_truth_path.name),
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "ground_truth_text": gt_text,
        "raw_cer": round(raw_cer, 6),
        "raw_wer": round(raw_wer, 6),
        "corrected_cer": round(corrected_cer, 6),
        "corrected_wer": round(corrected_wer, 6),
        "url_metrics": url_metrics,
        "url_violations": url_violations,
        "ocr_confidence_avg": result.get("ocr_confidence_avg", 0),
        "uncertain_block_count": len(result.get("uncertain_block_ids", [])),
        "total_blocks": len(result.get("blocks", [])),
        "processing_time_ms": result.get("processing_time_ms", 0),
        "engine": result.get("engine", engine),
        "preprocessing_mode": result.get("preprocessing_mode", "unknown"),
        "ocr_quality_status": result.get("ocr_quality_status", "unknown"),
        "ocr_warnings": result.get("ocr_warnings", []),
    }


# ============================================================================
# Batch Evaluation
# ============================================================================
def evaluate_directory(
    images_dir: Path,
    ground_truth_dir: Path,
    engine: str = "easyocr",
    mode: Optional[str] = None,
    preprocess: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate all images in a directory against their ground truth .txt files.

    Ground truth files must have the same stem name as the image file.
    Example: sample-001.png <-> sample-001.txt
    """
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    )

    if not image_files:
        return {
            "error": f"No supported image files found in {images_dir}",
            "total_images": 0,
        }

    per_image = []
    raw_cer_values = []
    raw_wer_values = []
    corrected_cer_values = []
    corrected_wer_values = []

    for img_path in image_files:
        gt_paths = [
            ground_truth_dir / (img_path.stem + ext)
            for ext in [".txt", ".text", ".gt.txt"]
        ]
        gt_path = None
        for p in gt_paths:
            if p.exists():
                gt_path = p
                break

        if not gt_path:
            per_image.append({
                "image": str(img_path.name),
                "error": "No ground truth file found",
                "skipped": True,
            })
            continue

        try:
            result = evaluate_single_image(
                img_path, gt_path, engine=engine, mode=mode, preprocess=preprocess,
            )
            per_image.append(result)
            raw_cer_values.append(result["raw_cer"])
            raw_wer_values.append(result["raw_wer"])
            corrected_cer_values.append(result["corrected_cer"])
            corrected_wer_values.append(result["corrected_wer"])
        except Exception as e:
            per_image.append({
                "image": str(img_path.name),
                "error": str(e),
                "skipped": True,
            })

    valid_count = len(raw_cer_values)
    total = len(image_files)

    # Aggregate URL metrics
    url_exact_count = sum(
        1 for r in per_image
        if not r.get("skipped") and r.get("url_metrics", {}).get("url_exact_match", False)
    )
    url_domain_count = sum(
        1 for r in per_image
        if not r.get("skipped") and r.get("url_metrics", {}).get("domain_match", False)
    )

    # v5.11.5.12: Aggregate URL violation metrics
    total_violations = sum(
        r.get("url_violations", {}).get("violation_count", 0)
        for r in per_image if not r.get("skipped")
    )
    images_with_violations = sum(
        1 for r in per_image
        if not r.get("skipped") and r.get("url_violations", {}).get("violation_count", 0) > 0
    )

    effective_mode = mode or preprocess or "adaptive"

    return {
        "evaluation_id": f"OCR-EVAL-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "app_version": APP_VERSION,
        "eval_tool_version": "v5.11.5.12",
        "total_images": total,
        "evaluated": valid_count,
        "skipped": total - valid_count,
        "engine": engine,
        "preprocess_mode": effective_mode,
        "metrics": {
            # v5.11.5.12: Raw text CER/WER
            "raw_cer_avg": round(sum(raw_cer_values) / valid_count, 6) if valid_count else 0,
            "raw_cer_min": round(min(raw_cer_values), 6) if valid_count else 0,
            "raw_cer_max": round(max(raw_cer_values), 6) if valid_count else 0,
            "raw_wer_avg": round(sum(raw_wer_values) / valid_count, 6) if valid_count else 0,
            "raw_wer_min": round(min(raw_wer_values), 6) if valid_count else 0,
            "raw_wer_max": round(max(raw_wer_values), 6) if valid_count else 0,
            # v5.11.5.12: Corrected text CER/WER
            "corrected_cer_avg": round(sum(corrected_cer_values) / valid_count, 6) if valid_count else 0,
            "corrected_cer_min": round(min(corrected_cer_values), 6) if valid_count else 0,
            "corrected_cer_max": round(max(corrected_cer_values), 6) if valid_count else 0,
            "corrected_wer_avg": round(sum(corrected_wer_values) / valid_count, 6) if valid_count else 0,
            "corrected_wer_min": round(min(corrected_wer_values), 6) if valid_count else 0,
            "corrected_wer_max": round(max(corrected_wer_values), 6) if valid_count else 0,
            # v5.11.5.9 backward compat (use corrected as primary)
            "cer_avg": round(sum(corrected_cer_values) / valid_count, 6) if valid_count else 0,
            "cer_min": round(min(corrected_cer_values), 6) if valid_count else 0,
            "cer_max": round(max(corrected_cer_values), 6) if valid_count else 0,
            "wer_avg": round(sum(corrected_wer_values) / valid_count, 6) if valid_count else 0,
            "wer_min": round(min(corrected_wer_values), 6) if valid_count else 0,
            "wer_max": round(max(corrected_wer_values), 6) if valid_count else 0,
            # URL metrics
            "url_exact_match_rate": round(url_exact_count / valid_count, 4) if valid_count else 0,
            "url_domain_match_rate": round(url_domain_count / valid_count, 4) if valid_count else 0,
            # v5.11.5.12: URL violation metrics
            "url_violation_count": total_violations,
            "images_with_url_violations": images_with_violations,
        },
        "per_image": per_image,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=f"OCR Evaluation Tool v{APP_VERSION} (v5.11.5.12.3) — "
                    f"CER/WER/URL accuracy + violation tracking",
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing images to evaluate",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Directory containing ground truth .txt files (same stem names as images)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--engine", default="easyocr",
        choices=["easyocr", "tesseract", "paddleocr"],
        help="OCR engine to use (default: easyocr)",
    )
    parser.add_argument(
        "--mode", default=None,
        choices=["safe_base", "mild_enhanced", "none", "adaptive", "legacy_enhanced"],
        help="v5.11.5.12: Preprocessing mode. safe_base (EXIF+RGB+resize only), "
             "mild_enhanced (safe_base+autocontrast+mild sharpen), "
             "none (no preprocessing), adaptive (auto, default).",
    )
    parser.add_argument(
        "--preprocess", default=None,
        choices=["standard", "baseline", "none"],
        help="v5.11.5.9 backward compat: Preprocessing mode. "
             "standard→legacy_enhanced, baseline→safe_base, none→none. "
             "Use --mode for v5.11.5.12.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    gt_dir = Path(args.ground_truth)

    if not images_dir.is_dir():
        print(f"Error: --images is not a directory: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not gt_dir.is_dir():
        print(f"Error: --ground-truth is not a directory: {gt_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine effective mode
    effective_mode = args.mode or args.preprocess or "adaptive"
    # Map old preprocess names
    if args.preprocess and not args.mode:
        if args.preprocess == "baseline":
            effective_mode = "safe_base"
        elif args.preprocess == "standard":
            effective_mode = "legacy_enhanced"

    print(f"OCR Evaluation Tool v{APP_VERSION} (v5.11.5.12.3)")
    print(f"  Images: {images_dir}")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Engine: {args.engine}")
    print(f"  Mode: {effective_mode}")
    print()

    report = evaluate_directory(
        images_dir, gt_dir,
        engine=args.engine,
        mode=args.mode,
        preprocess=args.preprocess if not args.mode else None,
    )

    json_output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output, encoding="utf-8")
        print(f"Report written to: {output_path}")
    else:
        print(json_output)

    metrics = report.get("metrics", {})
    print()
    print(f"=== Summary ===")
    print(f"  Images: {report['evaluated']} evaluated, {report['skipped']} skipped")
    if metrics:
        print(f"  Raw CER avg:      {metrics.get('raw_cer_avg', 0):.4f}")
        print(f"  Corrected CER avg: {metrics.get('corrected_cer_avg', 0):.4f}")
        print(f"  Raw WER avg:      {metrics.get('raw_wer_avg', 0):.4f}")
        print(f"  Corrected WER avg: {metrics.get('corrected_wer_avg', 0):.4f}")
        print(f"  URL exact:  {metrics.get('url_exact_match_rate', 0):.2%}")
        print(f"  URL domain: {metrics.get('url_domain_match_rate', 0):.2%}")
        vc = metrics.get('url_violation_count', 0)
        if vc > 0:
            print(f"  URL violations: {vc} ({metrics.get('images_with_url_violations', 0)} images)")


if __name__ == "__main__":
    main()
