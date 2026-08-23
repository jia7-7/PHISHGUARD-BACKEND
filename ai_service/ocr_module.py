"""
=============================================================================
OCR文字识别模块 v3.1 — v5.11.5.12.3
=============================================================================
v5.11.5.12 改进:
  - P0-1: 质量硬门槛 — avg_conf/uncertain_ratio 硬性条件不可被复合加权分绕过
  - P0-2: 扩大畸形URL候选检测 — 损坏前缀的http/www模式识别
  - P0-3: 禁止协议补斜杠 — 移除 https:/→https:// 推测规则
  - P0-4: 区分 secondary OCR (像素重识别) 与 heuristic correction (字符串补写)
  - P0-5: OCR→评分器信任边界 — unverified URL 不进入确定性证据
  - 新增审计字段: hard_gate, malformed_candidate, url_verification 增强
v5.11.5.12 改进:
  - safe_base 预处理: EXIF + RGB + resize ONLY, 零破坏性增强
  - 真正的 baseline-first 自适应流程 (safe_base → mild_enhanced)
  - 综合质量评分, 不再只看平均置信度
  - URL bbox 二次识别 (crop + LANCZOS upscale + allowlist OCR)
  - 完整的预处理/URL审计字段
  - 移除推测性URL字符补写 (dot fabrication)
  - 全角→半角一对一规范化保留, 无信息增量
  - _to_native() 增强: np.bool_ 支持
=============================================================================
"""

import base64
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import OCR_CONFIG, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


# ============================================================================
# v5.11.5.12: Enhanced numpy-to-native-Python converter for JSON serialization
# ============================================================================
def _to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization.
    v5.11.5.12: Added np.bool_ → bool, np.bytes_ → str, np.str_ → str handling.
    """
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bytes_,)):
        return str(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# ============================================================================
# v5.11.5.12: Preprocessing mode constants
# ============================================================================
class PreprocessMode:
    """Explicit preprocessing modes for v5.11.5.12."""
    SAFE_BASE = "safe_base"          # EXIF + RGB + safe resize ONLY
    MILD_ENHANCED = "mild_enhanced"  # safe_base + autocontrast + mild sharpen
    LEGACY_ENHANCED = "legacy_enhanced"  # old full pipeline (deprecated for production)
    NONE = "none"                    # no preprocessing at all


class ImagePreprocessor:
    """图像预处理器 — v5.11.5.12: safe_base/mild/legacy modes, true baseline-first"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or OCR_CONFIG["preprocessing"]
        self.max_size = self.config.get("max_image_size", 4096)
        self.enhance_contrast = self.config.get("enhance_contrast", True)
        self.denoise = self.config.get("denoise", True)
        self.sharpen = self.config.get("sharpen", True)
        self.correct_exif = self.config.get("correct_exif", True)
        # v5.11.5.12: production mode
        self.production_mode = self.config.get("production_mode", "adaptive")
        self.composite_quality_threshold = self.config.get("composite_quality_threshold", 0.55)
        self.mild_fallback_enabled = self.config.get("mild_fallback_enabled", True)
        # legacy backward compat
        self.adaptive_mode = self.config.get("adaptive_mode", True)
        self.baseline_only = self.config.get("baseline_only", False)

    def assess_quality(self, image: Image.Image) -> Dict:
        """v5.11.5.9: Assess image quality to decide if enhancement is needed.
        Returns dict with:
          - resolution_ok: min dimension >= 200px
          - contrast_ok: std dev of grayscale >= 30
          - blur_level: 'low'|'medium'|'high' based on Laplacian variance
          - needs_enhancement: True if any quality indicator is poor
        """
        w, h = image.size
        resolution_ok = min(w, h) >= 200

        gray = image.convert("L")
        gray_arr = np.array(gray, dtype=np.float64)
        std_dev = float(gray_arr.std())
        contrast_ok = std_dev >= 30.0

        laplacian_kernel = ImageFilter.Kernel(
            (3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=0)
        laplacian = np.array(
            image.convert("L").filter(laplacian_kernel), dtype=np.float64)
        if laplacian.shape[0] > 20 and laplacian.shape[1] > 20:
            interior = laplacian[2:-2, 2:-2]
        else:
            interior = laplacian
        lap_var = float(interior.var())
        if lap_var > 150:
            blur_level = "low"
        elif lap_var > 50:
            blur_level = "medium"
        else:
            blur_level = "high"

        needs_enhancement = (
            not resolution_ok
            or not contrast_ok
            or blur_level == "high"
        )

        return {
            "resolution_ok": resolution_ok,
            "contrast_ok": contrast_ok,
            "blur_level": blur_level,
            "laplacian_var": round(lap_var, 1),
            "std_dev": round(std_dev, 1),
            "needs_enhancement": needs_enhancement,
        }

    # ------------------------------------------------------------------
    # v5.11.5.12: Explicit preprocessing modes
    # ------------------------------------------------------------------

    def preprocess_safe_base(self, image: Image.Image) -> Image.Image:
        """v5.11.5.12: Safe baseline preprocessing — ZERO destructive enhancement.
        Only applies:
          - EXIF orientation correction
          - RGB conversion
          - Safe resize if max dimension > max_size (LANCZOS)
        Does NOT apply:
          - Contrast / Brightness / MedianFilter / Sharpen / autocontrast
          - Any text modification
        """
        if self.correct_exif:
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

        image = self._resize_if_needed(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def preprocess_mild(self, image: Image.Image) -> Image.Image:
        """v5.11.5.12: Mild enhancement — safe_base + autocontrast + gentle sharpen.
        No MedianFilter, no aggressive contrast/brightness tuning.
        """
        image = self.preprocess_safe_base(image)

        try:
            image = ImageOps.autocontrast(image, cutoff=2)
        except Exception:
            pass

        try:
            image = ImageEnhance.Sharpness(image).enhance(1.2)
        except Exception:
            pass

        return image

    def preprocess_legacy(self, image: Image.Image) -> Image.Image:
        """v5.11.5.12: Legacy full pipeline — for backward compatibility only.
        NOT recommended for production use.
        """
        image = self.preprocess_safe_base(image)

        if self.enhance_contrast:
            image = self._enhance_contrast(image)
        if self.denoise:
            image = self._denoise(image)
        if self.sharpen:
            image = self._sharpen(image)

        return image

    # ------------------------------------------------------------------
    # v5.11.5.9 legacy: kept for backward compatibility
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image, force_enhance: bool = False) -> Image.Image:
        """v5.11.5.9 legacy: Adaptive preprocessing — kept for backward compat.
        v5.11.5.12: Use preprocess_safe_base() or preprocess_mild() for new code."""
        logger.debug(f"预处理开始 (legacy): size={image.size}, mode={image.mode}")

        if self.correct_exif:
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

        image = self._resize_if_needed(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.baseline_only:
            logger.debug("baseline_only=True, 跳过所有增强")
            return image

        if self.adaptive_mode and not force_enhance:
            quality = self.assess_quality(image)
            if not quality["needs_enhancement"]:
                logger.debug(f"图像质量良好, 跳过增强 (lap_var={quality['laplacian_var']})")
                return image
            logger.debug(f"图像质量较差, 应用增强 (blur={quality['blur_level']})")

        if self.enhance_contrast:
            image = self._enhance_contrast(image)
        if self.denoise:
            image = self._denoise(image)
        if self.sharpen:
            image = self._sharpen(image)

        logger.debug(f"预处理完成: size={image.size}")
        return image

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_dim = max(w, h)
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.LANCZOS)
        return image

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        mean_brightness = np.array(gray).mean()
        if mean_brightness < 100:
            cf, bf = 1.5, 1.3
        elif mean_brightness > 180:
            cf, bf = 1.3, 0.9
        else:
            cf, bf = 1.2, 1.1
        image = ImageEnhance.Contrast(image).enhance(cf)
        image = ImageEnhance.Brightness(image).enhance(bf)
        return image

    def _denoise(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.MedianFilter(size=3))

    def _sharpen(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Sharpness(image).enhance(1.5)


class OCREngine:
    """OCR识别引擎 v3.1 — v5.11.5.12.3: hard gates, malformed URL candidate, trust boundary"""

    # v5.11.5.12: Class-level default attributes (for __new__-based test setup)
    url_postprocess_enabled: bool = True
    url_secondary_ocr_enabled: bool = True
    url_speculative_fix_enabled: bool = False
    url_candidate_detection_enabled: bool = True
    url_speculative_protocol_fix_enabled: bool = False
    hard_gate_min_confidence: float = 0.15
    hard_gate_max_uncertain_ratio: float = 0.95
    mid_gate_min_confidence: float = 0.35
    mid_gate_max_uncertain_ratio: float = 0.80

    # ------------------------------------------------------------------
    # v5.11.5.12: URL constants
    # ------------------------------------------------------------------
    _URL_TLDS = frozenset({
        "com", "cn", "xyz", "top", "cc", "tk", "cf", "ga", "ml", "gq",
        "work", "click", "link", "online", "site", "tech", "live", "cyou",
        "icu", "fun", "shop", "store", "edu", "org", "net", "gov", "io",
        "co", "info", "biz", "pw", "ws", "mobi", "name", "tv", "me",
        "app", "dev", "page", "run", "space", "website", "press", "host",
    })

    _URL_ALLOWLIST = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "-._~:/?#[]@!$&'()*+,;=%"
    )

    _URL_PROTO_RE = re.compile(
        r'(?:https?|ftp)\s*:\s*(?://|／／|\\\\|／|//)',
        re.IGNORECASE,
    )

    def __init__(self, engine: Optional[str] = None):
        self.engine_name = engine or OCR_CONFIG["engine"]
        self.config = OCR_CONFIG.get(self.engine_name, {})
        self.preprocessor = ImagePreprocessor()
        self.uncertain_threshold = OCR_CONFIG.get("uncertain_threshold", 0.5)
        # v5.11.5.12: URL processing config
        self.url_postprocess_enabled = OCR_CONFIG.get("url_postprocess_enabled", True)
        self.url_secondary_ocr_enabled = OCR_CONFIG.get("url_secondary_ocr_enabled", True)
        self.url_speculative_fix_enabled = OCR_CONFIG.get("url_speculative_fix_enabled", False)
        # v5.11.5.12: Hard gate thresholds
        pp = OCR_CONFIG.get("preprocessing", {})
        self.hard_gate_min_confidence = pp.get("hard_gate_min_confidence", 0.15)
        self.hard_gate_max_uncertain_ratio = pp.get("hard_gate_max_uncertain_ratio", 0.95)
        self.mid_gate_min_confidence = pp.get("mid_gate_min_confidence", 0.35)
        self.mid_gate_max_uncertain_ratio = pp.get("mid_gate_max_uncertain_ratio", 0.80)
        # v5.11.5.12: URL candidate detection
        self.url_candidate_detection_enabled = pp.get("url_candidate_detection_enabled", True)
        self.url_speculative_protocol_fix_enabled = pp.get(
            "url_speculative_protocol_fix_enabled", False)
        self._engine = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        logger.info(f"加载OCR引擎: {self.engine_name}")
        try:
            if self.engine_name == "easyocr":
                import easyocr
                gpu = self.config.get("gpu", True)
                model_dir = self.config.get("model_storage_directory", None)
                kwargs = {
                    "lang_list": self.config.get("languages", ["ch_sim", "en"]),
                    "gpu": gpu,
                    "verbose": False,
                }
                if model_dir:
                    kwargs["model_storage_directory"] = model_dir
                self._engine = easyocr.Reader(**kwargs)
            elif self.engine_name == "tesseract":
                import pytesseract
                if self.config.get("tesseract_cmd"):
                    pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
                self._engine = pytesseract
            elif self.engine_name == "paddleocr":
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    lang=self.config.get("lang", "ch"),
                    use_angle_cls=True,
                    use_gpu=self.config.get("use_gpu", True),
                    show_log=False,
                )
            else:
                raise ValueError(f"不支持的OCR引擎: {self.engine_name}")
            self._loaded = True
            logger.info(f"OCR引擎加载成功: {self.engine_name}")
        except Exception as e:
            logger.error(f"OCR引擎加载失败: {e}")
            if self.engine_name != "easyocr":
                logger.warning("降级到EasyOCR")
                self.engine_name = "easyocr"
                self.config = OCR_CONFIG.get("easyocr", {})
                self._load()
            else:
                raise

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def extract_text(self, image_input, preprocess: bool = True,
                     mode: Optional[str] = None,
                     adaptive: Optional[bool] = None) -> str:
        """提取纯文本（已后处理）。v5.11.5.12: mode parameter replaces adaptive."""
        result = self.extract_text_with_details(
            image_input, preprocess=preprocess, mode=mode, adaptive=adaptive)
        return result.get("corrected_text", result.get("full_text", ""))

    # ------------------------------------------------------------------
    # v5.11.5.12: Composite quality scoring
    # ------------------------------------------------------------------

    def _compute_quality(self, blocks: List[Dict]) -> Dict:
        """v5.11.5.12: Composite quality with HARD GATES that cannot be overridden.

        Hard gates (checked first, cannot be bypassed by composite weighted score):
          1. avg_confidence < hard_gate_min_confidence → insufficient_evidence
          2. uncertain_ratio >= hard_gate_max_uncertain_ratio → insufficient_evidence
          3. confident_block_count == 0 → insufficient_evidence
          4. avg_confidence < mid_gate_min_confidence → max low_confidence
          5. uncertain_ratio >= mid_gate_max_uncertain_ratio → max low_confidence

        Returns dict with score, status, hard_gate_applied, hard_gate_reason,
        confident_block_count, uncertain_block_count, malformed_url_candidate_count.
        """
        n = len(blocks)
        confident_blocks = [b for b in blocks if b.get("status") != "uncertain"]
        confident_block_count = len(confident_blocks)
        uncertain_block_count = n - confident_block_count

        # Count malformed URL candidates
        malformed_count = 0
        if self.url_candidate_detection_enabled:
            for b in blocks:
                candidate = self._is_url_candidate(b.get("text", ""))
                if candidate.get("is_candidate") and candidate.get("malformed"):
                    malformed_count += 1

        # Gate 0: No blocks → insufficient_evidence
        if n == 0:
            return {
                "score": 0.0, "status": "insufficient_evidence",
                "hard_gate_applied": True, "hard_gate_reason": "no_blocks",
                "confident_block_count": 0, "uncertain_block_count": 0,
                "malformed_url_candidate_count": 0,
                "components": {"confidence": 0.0, "certainty": 0.0,
                               "char_coverage": 0.0, "abnormal_char": 0.0,
                               "url_integrity": 1.0},
                "raw": {"avg_conf": 0.0, "uncertain_ratio": 1.0,
                        "total_chars": 0, "abnormal_ratio": 0.0,
                        "url_fragmentation": 0.0},
            }

        confs = [b["confidence"] for b in blocks]
        avg_conf = sum(confs) / n
        uncertain_count = sum(1 for b in blocks if b["status"] == "uncertain")
        uncertain_ratio = uncertain_count / n

        # Gate 1: avg_confidence < hard_gate_min_confidence → insufficient_evidence
        if avg_conf < self.hard_gate_min_confidence:
            return self._build_hard_gate_result(
                blocks, "insufficient_evidence",
                f"avg_confidence={avg_conf:.4f} < {self.hard_gate_min_confidence}")

        # Gate 2: uncertain_ratio >= hard_gate_max_uncertain_ratio → insufficient_evidence
        if uncertain_ratio >= self.hard_gate_max_uncertain_ratio:
            return self._build_hard_gate_result(
                blocks, "insufficient_evidence",
                f"uncertain_ratio={uncertain_ratio:.4f} >= {self.hard_gate_max_uncertain_ratio}")

        # Gate 3: No confident blocks → insufficient_evidence
        if confident_block_count == 0:
            return self._build_hard_gate_result(
                blocks, "insufficient_evidence",
                "confident_block_count=0 — no reliable OCR blocks")

        # Gate 4: avg_confidence < mid_gate_min_confidence → max low_confidence
        if avg_conf < self.mid_gate_min_confidence:
            return self._build_hard_gate_result(
                blocks, "low_confidence",
                f"avg_confidence={avg_conf:.4f} < {self.mid_gate_min_confidence}")

        # Gate 5: uncertain_ratio >= mid_gate_max_uncertain_ratio → max low_confidence
        if uncertain_ratio >= self.mid_gate_max_uncertain_ratio:
            return self._build_hard_gate_result(
                blocks, "low_confidence",
                f"uncertain_ratio={uncertain_ratio:.4f} >= {self.mid_gate_max_uncertain_ratio}")

        total_chars = sum(len(b.get("text", "")) for b in blocks)
        abnormal_count = sum(
            1 for b in blocks
            for c in b.get("text", "")
            if ord(c) > 127 and not ('一' <= c <= '鿿')
            and not ('　' <= c <= '〿')
            and not ('＀' <= c <= '￯')
            and c not in '，。！？；：“”‘’（）《》—…'
        )
        abnormal_ratio = abnormal_count / max(total_chars, 1)

        url_blocks = [b for b in blocks
                      if re.search(r'https?://|www\.|[a-z]+\.[a-z]{2,}',
                                   b.get("text", ""), re.I)]
        url_frag = 0.0
        if url_blocks:
            frag_count = 0
            for ub in url_blocks:
                txt = ub.get("text", "")
                if re.search(r'(?:\s+[a-z]{2,3}\b)', txt):
                    frag_count += 1
            url_frag = frag_count / len(url_blocks)

        conf_score = min(avg_conf / 0.70, 1.0)
        cert_score = 1.0 - uncertain_ratio
        char_score = min(total_chars / 20.0, 1.0)
        abn_score = max(0.0, 1.0 - abnormal_ratio * 3)
        url_score = 1.0 - url_frag

        score = (
            0.25 * conf_score +
            0.25 * cert_score +
            0.20 * char_score +
            0.15 * abn_score +
            0.15 * url_score
        )

        if score < 0.25:
            status = "insufficient_evidence"
        elif score < 0.45:
            status = "low_confidence"
        elif score < 0.65:
            status = "usable_with_warnings"
        else:
            status = "reliable"

        return {
            "score": round(score, 4),
            "status": status,
            "components": {
                "confidence": round(conf_score, 4),
                "certainty": round(cert_score, 4),
                "char_coverage": round(char_score, 4),
                "abnormal_char": round(abn_score, 4),
                "url_integrity": round(url_score, 4),
            },
            "raw": {
                "avg_conf": round(avg_conf, 4),
                "uncertain_ratio": round(uncertain_ratio, 4),
                "total_chars": total_chars,
                "abnormal_ratio": round(abnormal_ratio, 4),
                "url_fragmentation": round(url_frag, 4),
            },
        }

    def _build_hard_gate_result(self, blocks: List[Dict],
                                 forced_status: str, reason: str) -> Dict:
        """v5.11.5.12: Build quality result dict when a hard gate is triggered.
        The composite score is still computed for diagnostic/margin comparison,
        but the status is forced by the gate.
        """
        n = len(blocks)
        confs = [b["confidence"] for b in blocks]
        avg_conf = sum(confs) / n if n > 0 else 0.0
        uncertain_count = sum(1 for b in blocks if b["status"] == "uncertain")
        uncertain_ratio = uncertain_count / n if n > 0 else 1.0
        confident_block_count = n - uncertain_count
        uncertain_block_count = uncertain_count

        total_chars = sum(len(b.get("text", "")) for b in blocks)
        abnormal_count = sum(
            1 for b in blocks for c in b.get("text", "")
            if ord(c) > 127 and not ('一' <= c <= '鿿')
            and not ('　' <= c <= '〿')
            and not ('＀' <= c <= '￯')
            and c not in '，。！？；：“”‘’（）《》—…'
        )
        abnormal_ratio = abnormal_count / max(total_chars, 1)

        url_blocks = [b for b in blocks
                      if re.search(r'https?://|www\.|[a-z]+\.[a-z]{2,}',
                                   b.get("text", ""), re.I)]
        url_frag = 0.0
        if url_blocks:
            frag_count = sum(1 for ub in url_blocks
                           if re.search(r'(?:\s+[a-z]{2,3}\b)', ub.get("text", "")))
            url_frag = frag_count / len(url_blocks)

        malformed_count = 0
        if self.url_candidate_detection_enabled:
            for b in blocks:
                candidate = self._is_url_candidate(b.get("text", ""))
                if candidate.get("is_candidate") and candidate.get("malformed"):
                    malformed_count += 1

        conf_score = min(avg_conf / 0.70, 1.0)
        cert_score = 1.0 - uncertain_ratio
        char_score = min(total_chars / 20.0, 1.0)
        abn_score = max(0.0, 1.0 - abnormal_ratio * 3)
        url_score = 1.0 - url_frag
        score = 0.25 * conf_score + 0.25 * cert_score + 0.20 * char_score + 0.15 * abn_score + 0.15 * url_score

        return {
            "score": round(score, 4),
            "status": forced_status,
            "hard_gate_applied": True,
            "hard_gate_reason": reason,
            "confident_block_count": confident_block_count,
            "uncertain_block_count": uncertain_block_count,
            "malformed_url_candidate_count": malformed_count,
            "components": {
                "confidence": round(conf_score, 4),
                "certainty": round(cert_score, 4),
                "char_coverage": round(char_score, 4),
                "abnormal_char": round(abn_score, 4),
                "url_integrity": round(url_score, 4),
            },
            "raw": {
                "avg_conf": round(avg_conf, 4),
                "uncertain_ratio": round(uncertain_ratio, 4),
                "total_chars": total_chars,
                "abnormal_ratio": round(abnormal_ratio, 4),
                "url_fragmentation": round(url_frag, 4),
            },
        }

    # ------------------------------------------------------------------
    # v5.11.5.12: True baseline-first adaptive extraction
    # ------------------------------------------------------------------

    def extract_text_with_details(self, image_input, preprocess: bool = True,
                                   mode: Optional[str] = None,
                                   adaptive: Optional[bool] = None) -> Dict:
        """v5.11.5.12: True baseline-first adaptive OCR.

        Modes:
          - "safe_base": safe baseline only (EXIF+RGB+resize), single pass
          - "mild_enhanced": mild enhancement, single pass
          - "legacy_enhanced": legacy full pipeline, single pass
          - "none": no preprocessing, raw image
          - None / "adaptive": adaptive — safe_base first, mild fallback if needed

        Backward compat: if 'adaptive' parameter is passed (old API), it maps to
        adaptive mode. 'mode' parameter takes precedence.

        Returns comprehensive dict with audit fields.
        """
        start_time = time.time()
        image = self._load_image(image_input)

        # Determine effective mode
        effective_mode = mode
        if effective_mode is None:
            if adaptive is not None:
                # Backward compat: old adaptive parameter
                if adaptive and preprocess:
                    effective_mode = "adaptive"
                elif not preprocess:
                    effective_mode = "none"
                else:
                    effective_mode = self.preprocessor.production_mode
            else:
                if not preprocess:
                    effective_mode = "none"
                else:
                    effective_mode = self.preprocessor.production_mode

        # ---- Single-pass modes ----
        if effective_mode == "none":
            blocks = self._run_ocr_pass_raw(image)
            return self._build_result_v510(
                blocks, image, start_time,
                mode=PreprocessMode.NONE,
                attempted_modes=[PreprocessMode.NONE],
                selected_reason="preprocessing_disabled",
            )

        if effective_mode == "safe_base":
            blocks = self._run_ocr_pass_safe_base(image)
            return self._build_result_v510(
                blocks, image, start_time,
                mode=PreprocessMode.SAFE_BASE,
                attempted_modes=[PreprocessMode.SAFE_BASE],
                selected_reason="safe_base_requested",
            )

        if effective_mode == "mild_enhanced":
            blocks = self._run_ocr_pass_mild(image)
            return self._build_result_v510(
                blocks, image, start_time,
                mode=PreprocessMode.MILD_ENHANCED,
                attempted_modes=[PreprocessMode.MILD_ENHANCED],
                selected_reason="mild_enhanced_requested",
            )

        if effective_mode == "legacy_enhanced":
            blocks = self._run_ocr_pass_legacy(image)
            return self._build_result_v510(
                blocks, image, start_time,
                mode=PreprocessMode.LEGACY_ENHANCED,
                attempted_modes=[PreprocessMode.LEGACY_ENHANCED],
                selected_reason="legacy_enhanced_requested",
            )

        # ---- v5.11.5.9 backward compat: old "adaptive" with legacy preprocessor ----
        if adaptive is not None and preprocess:
            return self._extract_text_legacy_adaptive(image, start_time)

        # ---- v5.11.5.12: True adaptive (safe_base-first) ----
        # Pass 1: SAFE_BASE (true baseline — zero enhancement)
        baseline_blocks = self._run_ocr_pass_safe_base(image)
        baseline_quality = self._compute_quality(baseline_blocks)

        warnings = []
        url_uncertain_count = 0
        if baseline_quality["status"] in ("insufficient_evidence", "low_confidence"):
            warnings.append(f"safe_base quality: {baseline_quality['status']} "
                          f"(score={baseline_quality['score']:.3f})")

        # If baseline is already reliable or usable, return it (single pass)
        threshold = self.preprocessor.composite_quality_threshold
        if baseline_quality["score"] >= threshold:
            url_audits = self._audit_url_blocks(baseline_blocks, image)
            url_uncertain_count = sum(
                1 for a in url_audits
                if not a.get("deterministic_eligible", False)
            )
            return self._build_result_v510(
                baseline_blocks, image, start_time,
                mode=PreprocessMode.SAFE_BASE,
                attempted_modes=[PreprocessMode.SAFE_BASE],
                selected_reason=f"safe_base_quality_sufficient "
                              f"(score={baseline_quality['score']:.3f} >= {threshold})",
                baseline_quality=baseline_quality,
                selected_quality=baseline_quality,
                quality_status=baseline_quality["status"],
                warnings=warnings,
                url_audits=url_audits,
                url_uncertain_count=url_uncertain_count,
            )

        # Baseline insufficient — try mild enhancement if enabled
        if not self.preprocessor.mild_fallback_enabled:
            url_audits = self._audit_url_blocks(baseline_blocks, image)
            url_uncertain_count = sum(
                1 for a in url_audits
                if not a.get("deterministic_eligible", False)
            )
            warnings.append("mild_fallback disabled, returning safe_base despite low quality")
            return self._build_result_v510(
                baseline_blocks, image, start_time,
                mode=PreprocessMode.SAFE_BASE,
                attempted_modes=[PreprocessMode.SAFE_BASE],
                selected_reason="mild_fallback_disabled",
                baseline_quality=baseline_quality,
                selected_quality=baseline_quality,
                quality_status=baseline_quality["status"],
                warnings=warnings,
                url_audits=url_audits,
                url_uncertain_count=url_uncertain_count,
            )

        # Pass 2: MILD_ENHANCED
        logger.info(
            f"safe_base quality {baseline_quality['score']:.3f} < {threshold}, "
            f"attempting mild_enhanced"
        )
        mild_blocks = self._run_ocr_pass_mild(image)
        mild_quality = self._compute_quality(mild_blocks)

        attempted_modes = [PreprocessMode.SAFE_BASE, PreprocessMode.MILD_ENHANCED]
        candidate_scores = {
            PreprocessMode.SAFE_BASE: baseline_quality,
            PreprocessMode.MILD_ENHANCED: mild_quality,
        }

        # Mild must beat baseline by >=0.05 margin to be selected
        if mild_quality["score"] > baseline_quality["score"] + 0.05:
            selected_blocks = mild_blocks
            selected_mode = PreprocessMode.MILD_ENHANCED
            selected_reason = (
                f"mild_enhanced better: {mild_quality['score']:.3f} "
                f"> {baseline_quality['score']:.3f} + 0.05"
            )
            selected_quality = mild_quality
            quality_status = mild_quality["status"]
        else:
            selected_blocks = baseline_blocks
            selected_mode = PreprocessMode.SAFE_BASE
            selected_reason = (
                f"mild_enhanced insufficient margin: "
                f"{mild_quality['score']:.3f} <= {baseline_quality['score']:.3f} + 0.05"
            )
            selected_quality = baseline_quality
            quality_status = baseline_quality["status"]
            warnings.append("mild_enhanced did not significantly improve, keeping safe_base")

        url_audits = self._audit_url_blocks(selected_blocks, image)
        url_uncertain_count = sum(
            1 for a in url_audits
            if a.get("url_verification") in ("unverified", "conflict")
        )

        return self._build_result_v510(
            selected_blocks, image, start_time,
            mode=selected_mode,
            attempted_modes=attempted_modes,
            selected_reason=selected_reason,
            baseline_quality=baseline_quality,
            selected_quality=selected_quality,
            quality_status=quality_status,
            candidate_scores=candidate_scores,
            warnings=warnings,
            url_audits=url_audits,
            url_uncertain_count=url_uncertain_count,
        )

    def _extract_text_legacy_adaptive(self, image, start_time) -> Dict:
        """v5.11.5.9 backward compat: old adaptive flow using legacy preprocessor."""
        baseline_blocks = self._run_ocr_pass(image, preprocess=True, force_enhance=False)
        baseline_conf = (
            sum(b["confidence"] for b in baseline_blocks) / len(baseline_blocks)
            if baseline_blocks else 0.0
        )

        quality_threshold = self.preprocessor.config.get("quality_threshold", 0.5)
        if baseline_conf >= quality_threshold or not baseline_blocks:
            return self._build_result(
                baseline_blocks, image, start_time, preprocess_used="baseline"
            )

        enhanced_blocks = self._run_ocr_pass(image, preprocess=True, force_enhance=True)
        enhanced_conf = (
            sum(b["confidence"] for b in enhanced_blocks) / len(enhanced_blocks)
            if enhanced_blocks else 0.0
        )

        if enhanced_conf > baseline_conf:
            return self._build_result(
                enhanced_blocks, image, start_time, preprocess_used="enhanced"
            )
        else:
            return self._build_result(
                baseline_blocks, image, start_time, preprocess_used="baseline"
            )

    # ------------------------------------------------------------------
    # v5.11.5.12: Per-mode OCR pass methods
    # ------------------------------------------------------------------

    def _run_ocr_pass_safe_base(self, image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Run OCR on safe_base preprocessed image."""
        proc_image = self.preprocessor.preprocess_safe_base(image)
        return self._run_ocr_on_processed(proc_image)

    def _run_ocr_pass_mild(self, image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Run OCR on mild enhanced image."""
        proc_image = self.preprocessor.preprocess_mild(image)
        return self._run_ocr_on_processed(proc_image)

    def _run_ocr_pass_legacy(self, image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Run OCR on legacy enhanced image."""
        proc_image = self.preprocessor.preprocess_legacy(image)
        return self._run_ocr_on_processed(proc_image)

    def _run_ocr_pass_raw(self, image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Run OCR on raw image (no preprocessing at all)."""
        return self._run_ocr_on_processed(image)

    def _run_ocr_on_processed(self, proc_image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Common OCR execution on a preprocessed image.
        Returns list of block dicts with text, corrected_text, confidence, etc."""
        self._load()
        raw_results = self._do_ocr(proc_image)

        blocks = []
        for i, item in enumerate(raw_results):
            text = item.get("text", "").strip()
            if not text:
                continue
            confidence = item.get("confidence", 0.0)
            status = "uncertain" if confidence < self.uncertain_threshold else "certain"
            bbox = item.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]])
            position = self._describe_position(bbox, proc_image.size)
            corrected = self._postprocess_single(text)

            blocks.append({
                "block_id": i,
                "text": text,
                "corrected_text": corrected,
                "confidence": round(float(confidence), 4),
                "status": status,
                "bbox": bbox,
                "position": position,
                "reading_order": i,
            })

        blocks = self._sort_reading_order(blocks, proc_image.size)
        for i, b in enumerate(blocks):
            b["reading_order"] = i
        return blocks

    # ------------------------------------------------------------------
    # v5.11.5.9 backward compat: legacy OCR pass
    # ------------------------------------------------------------------

    def _run_ocr_pass(self, image, preprocess: bool, force_enhance: bool) -> List[Dict]:
        """v5.11.5.9 legacy: Run OCR pass with legacy preprocessor."""
        if preprocess:
            proc_image = self.preprocessor.preprocess(image, force_enhance=force_enhance)
        else:
            proc_image = image

        self._load()
        raw_results = self._do_ocr(proc_image)

        blocks = []
        for i, item in enumerate(raw_results):
            text = item.get("text", "").strip()
            if not text:
                continue
            confidence = item.get("confidence", 0.0)
            status = "uncertain" if confidence < self.uncertain_threshold else "certain"
            bbox = item.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]])
            position = self._describe_position(bbox, proc_image.size)
            corrected = self._postprocess_single(text)

            blocks.append({
                "block_id": i,
                "text": text,
                "corrected_text": corrected,
                "confidence": round(float(confidence), 4),
                "status": status,
                "bbox": bbox,
                "position": position,
                "reading_order": i,
            })

        blocks = self._sort_reading_order(blocks, proc_image.size)
        for i, b in enumerate(blocks):
            b["reading_order"] = i
        return blocks

    # ------------------------------------------------------------------
    # v5.11.5.12: Enhanced result builder with audit fields
    # ------------------------------------------------------------------

    def _build_result_v510(self, blocks, image, start_time, *,
                           mode: str,
                           attempted_modes: List[str],
                           selected_reason: str,
                           baseline_quality: Optional[Dict] = None,
                           selected_quality: Optional[Dict] = None,
                           quality_status: str = "usable_with_warnings",
                           candidate_scores: Optional[Dict] = None,
                           warnings: Optional[List[str]] = None,
                           url_audits: Optional[List[Dict]] = None,
                           url_uncertain_count: int = 0) -> Dict:
        """v5.11.5.12: Build comprehensive result dict with full audit trail."""
        processing_time = (time.time() - start_time) * 1000

        confidences = [b["confidence"] for b in blocks]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        uncertain_ids = [b["block_id"] for b in blocks if b["status"] == "uncertain"]
        uncertain_ratio = len(uncertain_ids) / len(blocks) if blocks else 0.0

        if selected_quality is None:
            selected_quality = self._compute_quality(blocks)
        if baseline_quality is None:
            baseline_quality = selected_quality

        # v5.11.5.12: Hard gate fields from quality
        hard_gate_applied = selected_quality.get("hard_gate_applied", False)
        hard_gate_reason = selected_quality.get("hard_gate_reason", "")
        confident_block_count = selected_quality.get("confident_block_count",
                                                     len([b for b in blocks if b.get("status") != "uncertain"]))
        uncertain_block_count_v511 = selected_quality.get("uncertain_block_count",
                                                          len(uncertain_ids))
        malformed_url_candidate_count = selected_quality.get("malformed_url_candidate_count", 0)

        # v5.11.5.12: Count deterministic-eligible URLs
        deterministic_url_eligible_count = sum(
            1 for a in (url_audits or [])
            if a.get("deterministic_eligible", False)
        )

        result = {
            "full_text": "\n".join(b["text"] for b in blocks),
            "corrected_text": "\n".join(b["corrected_text"] for b in blocks),
            "blocks": blocks,
            "engine": self.engine_name,
            "language": self.config.get("languages", self.config.get("lang", "unknown")),
            "processing_time_ms": round(processing_time, 1),
            "ocr_confidence_avg": round(float(avg_conf), 4),
            "uncertain_block_ids": uncertain_ids,

            # v5.11.5.9 backward compat
            "preprocess_used": mode,

            # v5.11.5.12: Preprocessing audit
            "preprocessing_mode": mode,
            "preprocessing_attempted_modes": attempted_modes,
            "preprocessing_selected_reason": selected_reason,

            # v5.11.5.12: Quality audit
            "baseline_confidence": (
                round(baseline_quality["raw"]["avg_conf"], 4)
                if baseline_quality else None
            ),
            "selected_confidence": round(float(avg_conf), 4),
            "baseline_uncertain_ratio": (
                round(baseline_quality["raw"]["uncertain_ratio"], 4)
                if baseline_quality else None
            ),
            "selected_uncertain_ratio": round(uncertain_ratio, 4),
            "candidate_quality_scores": candidate_scores,
            "ocr_quality_status": quality_status,
            "ocr_warnings": warnings or [],

            # v5.11.5.12: Hard gate audit
            "hard_gate_applied": hard_gate_applied,
            "hard_gate_reason": hard_gate_reason,
            "confident_block_count": confident_block_count,
            "uncertain_block_count": uncertain_block_count_v511,
            "malformed_url_candidate_count": malformed_url_candidate_count,

            # v5.11.5.12/5.11: URL audit
            "url_uncertain_count": url_uncertain_count,
            "url_audits": url_audits or [],
            "deterministic_url_eligible_count": deterministic_url_eligible_count,
        }

        logger.info(
            f"OCR完成 ({mode}): {len(blocks)} blocks, "
            f"{len(uncertain_ids)} uncertain, "
            f"quality={quality_status}, "
            f"avg_conf={avg_conf:.2%}, "
            f"elapsed={processing_time:.0f}ms"
        )
        return _to_native(result)

    # ------------------------------------------------------------------
    # v5.11.5.9 backward compat: old _build_result
    # ------------------------------------------------------------------

    def _build_result(self, blocks, image, start_time, preprocess_used: str = "standard") -> Dict:
        """v5.11.5.9 legacy: Build result dict. Kept for backward compat."""
        processing_time = (time.time() - start_time) * 1000

        confidences = [b["confidence"] for b in blocks]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        uncertain_ids = [b["block_id"] for b in blocks if b["status"] == "uncertain"]

        result = {
            "full_text": "\n".join(b["text"] for b in blocks),
            "corrected_text": "\n".join(b["corrected_text"] for b in blocks),
            "blocks": blocks,
            "engine": self.engine_name,
            "language": self.config.get("languages", self.config.get("lang", "unknown")),
            "processing_time_ms": round(processing_time, 1),
            "ocr_confidence_avg": round(float(avg_conf), 4),
            "uncertain_block_ids": uncertain_ids,
            "preprocess_used": preprocess_used,
        }

        logger.info(f"OCR完成 ({preprocess_used}): {len(blocks)} blocks, "
                     f"{len(uncertain_ids)} uncertain, "
                     f"avg_conf={avg_conf:.2%}, "
                     f"elapsed={processing_time:.0f}ms")
        return _to_native(result)

    # ------------------------------------------------------------------
    # v5.11.5.12: URL bbox secondary OCR recognition (P0-3)
    # ------------------------------------------------------------------

    def _secondary_url_ocr(self, original_image: Image.Image,
                           bbox: List[List[float]]) -> Optional[Dict]:
        """v5.11.5.12: Secondary OCR on URL region with allowlist.

        Crops the bbox from the original (unprocessed) image, adds padding,
        upscales with LANCZOS, and runs EasyOCR restricted to URL-safe characters.

        Returns dict with {text, confidence} or None if OCR fails.
        """
        try:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(original_image.width, int(max(xs))), \
                     min(original_image.height, int(max(ys)))

            if x2 <= x1 or y2 <= y1:
                return None

            pad_w = int((x2 - x1) * 0.10)
            pad_h = int((y2 - y1) * 0.10)
            x1_p = max(0, x1 - pad_w)
            y1_p = max(0, y1 - pad_h)
            x2_p = min(original_image.width, x2 + pad_w)
            y2_p = min(original_image.height, y2 + pad_h)

            crop = original_image.crop((x1_p, y1_p, x2_p, y2_p))
            if crop.width < 10 or crop.height < 10:
                return None

            new_w = crop.width * 2
            new_h = crop.height * 2
            if new_w > 2000 or new_h > 2000:
                scale = min(2000 / new_w, 2000 / new_h)
                new_w = int(new_w * scale)
                new_h = int(new_h * scale)
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

            self._load()
            if hasattr(self._engine, 'readtext'):
                raw = self._engine.readtext(
                    np.array(crop),
                    allowlist=self._URL_ALLOWLIST,
                )
                if raw:
                    best = max(raw, key=lambda r: r[2])
                    return {
                        "text": best[1].strip(),
                        "confidence": round(float(best[2]), 4),
                    }
            return None
        except Exception as e:
            logger.debug(f"Secondary URL OCR failed: {e}")
            return None

    def _is_url_like_block(self, text: str) -> bool:
        """Check if block text contains URL-like patterns."""
        if not text:
            return False
        return bool(
            re.search(r'https?://|www\.|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-z]{2,}', text, re.I)
        )

    def _is_url_candidate(self, text: str) -> Dict:
        """v5.11.5.12: Detect potential URL candidates including malformed ones.

        Distinguishes between:
          - Clean URL patterns (http://, www.domain.com) → is_candidate=True, malformed=False
          - Malformed patterns (https/domain, httpsJdomain, http;domain) → is_candidate=True, malformed=True
          - Non-URL text → is_candidate=False

        Returns dict with is_candidate, reason, raw_candidate_text, malformed, prefix_detected.
        """
        if not text or len(text.strip()) < 3:
            return {"is_candidate": False, "reason": "", "raw_candidate_text": text or "",
                    "malformed": False, "prefix_detected": None}

        stripped = text.strip()

        # Pattern 1: Damaged http/https prefix — colon-slash replaced/separated
        # Matches: https/domain, https:/domain, httpsJdomain, http;domain, https domain
        damaged_http = re.search(
            r'\b(https?)\s*[:;JjI|/\\](?:\s*/\s*)?\s*[a-zA-Z0-9]', stripped, re.I)
        if damaged_http:
            after = stripped[damaged_http.start():]
            prefix_len = len(damaged_http.group(0))
            # Check for domain-like content after the damaged prefix
            domain_like = re.search(
                r'[a-zA-Z0-9](?:[-a-zA-Z0-9\s.]*[a-zA-Z0-9])?', after[prefix_len:])
            if domain_like and len(domain_like.group()) >= 2:
                # It's malformed if the separator isn't ://
                prefix_text = damaged_http.group(0)
                is_malformed = not re.match(r'https?://', prefix_text)
                return {
                    "is_candidate": True,
                    "reason": f"damaged_http_prefix: {prefix_text[:12]}",
                    "raw_candidate_text": stripped,
                    "malformed": is_malformed,
                    "prefix_detected": damaged_http.group(1).lower(),
                }

        # Pattern 2: www with damaged separators (space, colon, semicolon — NOT dot)
        www_damaged = re.search(r'\bwww[\s:;]+[a-zA-Z0-9]', stripped, re.I)
        if www_damaged:
            return {
                "is_candidate": True,
                "reason": "www_prefix_damaged",
                "raw_candidate_text": stripped,
                "malformed": True,
                "prefix_detected": "www",
            }

        # Pattern 3: Clean standard URL (also caught by _is_url_like_block)
        clean_url = re.search(r'https?://[^\s]{3,}|www\.[a-zA-Z0-9][^\s]{1,}', stripped, re.I)
        if clean_url:
            return {
                "is_candidate": True,
                "reason": "standard_url_pattern",
                "raw_candidate_text": stripped,
                "malformed": False,
                "prefix_detected": (
                    "http" if re.match(r'https?://', clean_url.group(), re.I)
                    else "www"
                ),
            }

        # Pattern 4: Bare domain with known TLD (e.g., example.com)
        bare_domain = re.search(
            r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:[/?#][^\s]*)?$',
            stripped, re.I)
        if bare_domain and len(bare_domain.group()) >= 5:
            # Avoid false positives on version numbers (v5.11) and common English words
            domain_text = bare_domain.group()
            common_false = {'the.com', 'is.com', 'at.com', 'or.com', 'in.net',
                           'to.com', 'be.org', 'of.com', 'no.com', 'go.com'}
            if domain_text.lower() not in common_false:
                return {
                    "is_candidate": True,
                    "reason": "bare_domain",
                    "raw_candidate_text": stripped,
                    "malformed": False,
                    "prefix_detected": None,
                }

        return {"is_candidate": False, "reason": "", "raw_candidate_text": stripped,
                "malformed": False, "prefix_detected": None}

    def _audit_url_blocks(self, blocks: List[Dict],
                          original_image: Image.Image) -> List[Dict]:
        """v5.11.5.12: Audit URL-like AND URL-candidate blocks with secondary OCR.

        For each block:
        1. Check if clean URL (via _is_url_like_block) OR malformed candidate (via _is_url_candidate)
        2. Record raw OCR text and candidate status
        3. Attempt secondary OCR on original image bbox (clean URL blocks only)
        4. Malformed candidates are audited but NEVER verified
        5. Return full audit trail with trust-boundary fields
        """
        audits = []
        for block in blocks:
            text = block.get("text", "")
            is_clean_url = self._is_url_like_block(text)
            candidate = self._is_url_candidate(text) if self.url_candidate_detection_enabled else None
            is_url_candidate = candidate.get("is_candidate", False) if candidate else False
            is_malformed = candidate.get("malformed", False) if candidate else False

            # Skip blocks that are neither clean URLs nor malformed candidates
            if not is_clean_url and not (is_url_candidate and is_malformed):
                continue

            raw_text = text
            bbox = block.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]])

            # Conservative normalization: ONLY 1:1 char mapping, NO speculation
            normalized = self._postprocess_url_conservative(raw_text)

            # Only attempt secondary OCR for clean URL patterns (not malformed candidates)
            secondary_result = None
            secondary_attempted = False
            if is_clean_url and self.url_secondary_ocr_enabled:
                secondary_attempted = True
                secondary_result = self._secondary_url_ocr(original_image, bbox)

            candidates = {
                "raw": raw_text,
                "normalized": normalized,
            }
            if secondary_result:
                candidates["secondary"] = secondary_result["text"]

            selected_text = normalized
            selected_candidate = "normalized"
            selection_reason = "conservative_normalization_only"
            warnings_list = []
            url_verification = "unverified"
            deterministic_eligible = False

            if is_malformed:
                # Malformed candidates: audited but NEVER verified
                warnings_list.append(f"malformed_url_candidate: {candidate.get('reason', 'unknown')}")
                url_verification = "unverified"
                selection_reason = "malformed_candidate_no_fix"
                deterministic_eligible = False
            elif secondary_result and secondary_result["text"]:
                sec_text = secondary_result["text"]
                sec_conf = secondary_result["confidence"]

                strict_syntax = bool(re.match(
                    r'^https?://[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}'
                    r'(?:/[\w\-._~:/?#\[\]@!$&()*+,;=%]*)?$',
                    sec_text, re.I))

                # v5.11.5.12: Distinguish pixel_redecoded from heuristic fabrication
                raw_chars = set(raw_text)
                sec_added = set(sec_text) - raw_chars
                # Characters that could be pixel-redecoded (not heuristic fabrication):
                # . / : are common OCR errors that secondary OCR can genuinely recover
                pixel_redecodable = set('./:\\')
                sec_fabricated = sec_added - pixel_redecodable

                if strict_syntax and not sec_fabricated and sec_conf > block["confidence"] + 0.10:
                    selected_text = sec_text
                    selected_candidate = "secondary"
                    selection_reason = (
                        f"secondary_ocr_valid: strict_syntax=True, "
                        f"confidence={sec_conf:.4f} > {block['confidence']:.4f} + 0.10"
                    )
                    if sec_added:
                        warnings_list.append(
                            f"secondary_ocr_pixel_redecoded: {sec_added} — "
                            f"characters recovered by pixel re-recognition, not heuristic fix"
                        )
                    url_verification = "verified"
                    deterministic_eligible = True
                elif strict_syntax and sec_added:
                    # True fabrication — characters not in raw OCR AND not pixel-redecodable
                    if sec_fabricated:
                        warnings_list.append(
                            f"secondary_ocr_conflict_fabricated_chars: {sec_fabricated}")
                    if sec_added & pixel_redecodable:
                        warnings_list.append(
                            f"secondary_ocr_pixel_redecoded: {sec_added & pixel_redecodable}")
                    url_verification = "conflict"
                    deterministic_eligible = False
                else:
                    warnings_list.append(
                        f"secondary_ocr_insufficient: "
                        f"conf={sec_conf:.4f}, strict_syntax={strict_syntax}"
                    )
                    url_verification = "unverified"
                    deterministic_eligible = False
            else:
                if secondary_attempted:
                    warnings_list.append("secondary_ocr_failed_or_empty")

            # Strict URL syntax validation
            try:
                from urllib.parse import urlparse
                parsed = urlparse(selected_text)
                strict_syntax_valid = bool(parsed.scheme and parsed.netloc)
            except Exception:
                strict_syntax_valid = False

            # v5.11.5.12: Only 1:1 normalization rules recorded (no protocol slash fix)
            norm_rules = []
            if '：' in raw_text:
                norm_rules.append("fullwidth_colon_to_halfwidth")
            if '．' in raw_text:
                norm_rules.append("fullwidth_period_to_halfwidth")
            if '／' in raw_text:
                norm_rules.append("fullwidth_slash_to_halfwidth")
            if re.search(r'(https?)：//', raw_text, re.I):
                norm_rules.append("fullwidth_protocol_colon_fix")

            audits.append({
                "block_id": block.get("block_id", -1),
                "raw_ocr_text": raw_text,
                # v5.11.5.12: URL candidate detection fields
                "url_candidate_detected": is_url_candidate,
                "url_candidate_reason": candidate.get("reason", "") if candidate else "",
                "raw_candidate_text": raw_text,
                "malformed_candidate": is_malformed,
                # v5.11.5.12: Secondary OCR fields
                "secondary_ocr_attempted": secondary_attempted,
                "secondary_ocr_text": secondary_result["text"] if secondary_result else None,
                "secondary_ocr_confidence": (
                    secondary_result["confidence"] if secondary_result else None
                ),
                # v5.11.5.12: Selection + trust boundary
                "selected_text": selected_text,
                "selected_candidate": selected_candidate,
                "selection_reason": selection_reason,
                "normalization_rules": norm_rules,
                "strict_syntax_valid": strict_syntax_valid,
                "url_verification": url_verification,
                "deterministic_eligible": deterministic_eligible,
                "url_warnings": warnings_list,
            })

        return audits

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_extract(self, images, preprocess: bool = True,
                      mode: Optional[str] = None) -> List[Dict]:
        """批量处理 — v5.11.5.12: supports mode parameter."""
        results = []
        total = len(images)
        logger.info(f"批量OCR: {total} 张")
        for i, img in enumerate(images, 1):
            try:
                r = self.extract_text_with_details(img, preprocess=preprocess, mode=mode)
                r["batch_index"] = i
                r["status"] = "success"
                results.append(r)
            except Exception as e:
                logger.error(f"批量OCR第{i}张失败: {e}")
                results.append({"batch_index": i, "status": "error", "error": str(e)})
        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_image(self, image_input):
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"图片文件不存在: {path}")
            if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"不支持的格式: {path.suffix}")
            return Image.open(path)
        if isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input))
        raise ValueError(f"不支持的输入类型: {type(image_input)}")

    def _do_ocr(self, image: Image.Image) -> List[Dict]:
        """执行OCR — all results pass through _to_native() for JSON safety"""
        if self.engine_name == "easyocr":
            raw = self._engine.readtext(np.array(image))
            results = []
            for item in raw:
                results.append(_to_native({
                    "text": item[1],
                    "confidence": item[2],
                    "bbox": item[0],
                }))
            return results
        elif self.engine_name == "tesseract":
            import pytesseract
            data = self._engine.image_to_data(
                image, lang=self.config.get("languages", "chi_sim+eng"),
                output_type=pytesseract.Output.DICT,
            )
            results = []
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                if text:
                    conf_val = data["conf"][i]
                    conf = int(conf_val) / 100.0 if conf_val != "-1" else 0.5
                    results.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": [
                            [data["left"][i], data["top"][i]],
                            [data["left"][i] + data["width"][i], data["top"][i]],
                            [data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
                            [data["left"][i], data["top"][i] + data["height"][i]],
                        ],
                    })
            return results
        elif self.engine_name == "paddleocr":
            raw = self._engine.ocr(np.array(image), cls=True)
            results = []
            if raw and raw[0]:
                for line in raw[0]:
                    results.append(_to_native({
                        "text": line[1][0],
                        "confidence": line[1][1],
                        "bbox": line[0],
                    }))
            return results
        return []

    def _sort_reading_order(self, blocks: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        if len(blocks) <= 1:
            return blocks

        img_w, img_h = image_size

        x_centers = []
        for b in blocks:
            bbox = b.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]])
            xs = [p[0] for p in bbox]
            x_centers.append(sum(xs) / len(xs))

        if len(x_centers) >= 3:
            x_std = np.std(x_centers)
            if x_std > img_w * 0.25:
                sorted_x = sorted(x_centers)
                gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
                if gaps:
                    median_gap = np.median(gaps)
                    threshold = median_gap * 2
                    columns = []
                    current_col = [blocks[0]]
                    prev_x = x_centers[0]
                    for i in range(1, len(blocks)):
                        if x_centers[i] - prev_x > threshold:
                            columns.append(current_col)
                            current_col = []
                        current_col.append(blocks[i])
                        prev_x = x_centers[i]
                    columns.append(current_col)

                    result = []
                    for col in columns:
                        col_sorted = sorted(col, key=lambda b: (
                            sum(p[1] for p in b.get("bbox", [[0,0],[0,0],[0,0],[0,0]])) / 4
                        ))
                        result.extend(col_sorted)
                    return result

        return sorted(blocks, key=lambda b: (
            sum(p[1] for p in b.get("bbox", [[0,0],[0,0],[0,0],[0,0]])) / 4
        ))

    def _describe_position(self, bbox: List[List[float]], image_size: Tuple[int, int]) -> str:
        if not bbox:
            return "未知"
        img_w, img_h = image_size
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / len(xs) / img_w
        cy = sum(ys) / len(ys) / img_h
        h_pos = "左侧" if cx < 0.33 else ("右侧" if cx > 0.67 else "中间")
        v_pos = "顶部" if cy < 0.33 else ("底部" if cy > 0.67 else "中部")
        return f"{v_pos}{h_pos}"

    # ------------------------------------------------------------------
    # v5.11.5.12: Post-processing — conservative URL normalization only
    # ------------------------------------------------------------------

    def _postprocess_single(self, text: str) -> str:
        """单条文本后处理 — v5.11.5.12: conservative, no character fabrication."""
        if not text:
            return text

        fixes = {
            "http：": "http:",
            "https：": "https:",
            "http;//": "http://",
            "https;//": "https://",
        }
        for old, new in fixes.items():
            text = text.replace(old, new)

        if self.url_postprocess_enabled:
            text = self._postprocess_url_conservative(text)

        return text.strip()

    def _postprocess_url_conservative(self, text: str) -> str:
        """v5.11.5.12: Conservative URL post-processing — STRICT 1:1 normalization only.

        ALLOWED (1:1 char mapping, zero information gain):
          - Fullwidth ：→:  (U+FF1A→U+003A)
          - Fullwidth ．→.  (U+FF0E→U+002E)
          - Fullwidth ／→/  (U+FF0F→U+002F)
          - Fullwidth ＼→\\ (U+FF3C→U+005C)

        FORBIDDEN (character fabrication or speculation):
          - Protocol slash fix: https:/domain → https://domain (ADDS '/' not in raw OCR)
          - Dot insertion before known TLDs: sdu-verifyxyz → sdu-verify.xyz (ADDS '.')
          - Any character not present in raw OCR output
          - Deleting or replacing I/l/1 characters
          - Removing spaces within what might be a URL
        """
        if not text:
            return text

        # Pre-fix full-width colon after protocol name (1:1: ：→:)
        text = re.sub(r'(https?)：//', r'\1://', text, flags=re.IGNORECASE)

        # v5.11.5.12: Protocol slash fix REMOVED from production.
        # https:/domain → https://domain ADDS a '/' character not in raw OCR.
        # Only active when url_speculative_protocol_fix_enabled is explicitly True.
        if self.url_speculative_protocol_fix_enabled:
            text = re.sub(
                r'(https?):(/)([^\s/\\\.　＀-￯])',
                r'\1://\3',
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r'(https?):\s*/\s+([^\s])',
                r'\1://\2',
                text,
                flags=re.IGNORECASE,
            )

        # Rule 1 (was Rule 2): Full-width characters within URL spans (1:1 only)
        url_like = self._detect_url_like(text)
        if url_like:
            for start, end, _ in reversed(url_like):
                segment = text[start:end]
                fixed = segment
                fixed = fixed.replace('：', ':')
                fixed = fixed.replace('．', '.')
                fixed = fixed.replace('／', '/')
                fixed = fixed.replace('＼', '\\')
                fixed = fixed.replace('＠', '@')
                if fixed != segment:
                    text = text[:start] + fixed + text[end:]

        # Rule 2 (was Rule 3): Lowercase hostname
        def _lowercase_hostname(m):
            full = m.group(0)
            proto_match = re.match(r'(https?://)', full, re.IGNORECASE)
            if not proto_match:
                return full
            proto = proto_match.group(0)
            rest = full[len(proto):]
            slash_pos = rest.find('/')
            if slash_pos > 0:
                hostname = rest[:slash_pos]
                path = rest[slash_pos:]
            else:
                hostname = rest
                path = ''
            return proto + hostname.lower() + path

        text = re.sub(
            r'https?://[^\s<>"\'{}\[\]()（）一-鿿　-〿＀-￯]*',
            _lowercase_hostname,
            text,
            flags=re.IGNORECASE,
        )

        # v5.11.5.12: Rule 4 (dot restoration) REMOVED from production.
        # Only active when url_speculative_fix_enabled is explicitly True.
        if self.url_speculative_fix_enabled:
            for tld in sorted(self._URL_TLDS, key=len, reverse=True):
                dot_restore_re = re.compile(
                    r'(?<=[-a-zA-Z0-9])'
                    r'(?<!\.)'
                    r'(' + re.escape(tld) + r')'
                    r'(?=[\s\)\]\},;:!?.。，、：》）""''/\\　-]|$)',
                    re.IGNORECASE,
                )
                text = dot_restore_re.sub(r'.\1', text)

        return text

    # ------------------------------------------------------------------
    # v5.11.5.9 backward compat: old URL postprocessing
    # ------------------------------------------------------------------

    def _postprocess_url(self, text: str) -> str:
        """v5.11.5.9 legacy URL post-processing (kept for backward compat tests).
        v5.11.5.12: For production, use _postprocess_url_conservative() instead.
        """
        if not text:
            return text

        # Rule 0: Pre-fix full-width characters
        text = re.sub(r'(https?)：//', r'\1://', text, flags=re.IGNORECASE)
        for tld in sorted(self._URL_TLDS, key=len, reverse=True):
            text = re.sub(
                rf'．({re.escape(tld)})(?=[\s\)\]\}},;:!?.。，、：》）""''　-]|$)',
                rf'.\1', text)

        # Rule 1: Protocol fix
        text = re.sub(
            r'(https?):(/)([^\s/\\\.　＀-￯])',
            r'\1://\3', text, flags=re.IGNORECASE)
        text = re.sub(
            r'(https?):\s*/\s+([^\s])',
            r'\1://\2', text, flags=re.IGNORECASE)

        # Rule 2: Full-width within URL spans
        url_like = self._detect_url_like(text)
        if url_like:
            for start, end, _ in reversed(url_like):
                segment = text[start:end]
                fixed = segment
                fixed = fixed.replace('：', ':')
                fixed = fixed.replace('．', '.')
                fixed = fixed.replace('／', '/')
                fixed = fixed.replace('＼', '\\')
                fixed = fixed.replace('＠', '@')
                if fixed != segment:
                    text = text[:start] + fixed + text[end:]

        # Rule 3: Lowercase hostname
        def _lowercase_hostname(m):
            full = m.group(0)
            proto_match = re.match(r'(https?://)', full, re.IGNORECASE)
            if not proto_match:
                return full
            proto = proto_match.group(0)
            rest = full[len(proto):]
            slash_pos = rest.find('/')
            if slash_pos > 0:
                hostname = rest[:slash_pos]
                path = rest[slash_pos:]
            else:
                hostname = rest
                path = ''
            return proto + hostname.lower() + path

        text = re.sub(
            r'https?://[^\s<>"\'{}\[\]()（）一-鿿　-〿＀-￯]*',
            _lowercase_hostname, text, flags=re.IGNORECASE)

        # Rule 4: Dot restoration (v5.11.5.12: DEPRECATED, only in legacy path)
        for tld in sorted(self._URL_TLDS, key=len, reverse=True):
            dot_restore_re = re.compile(
                r'(?<=[-a-zA-Z0-9])'
                r'(?<!\.)'
                r'(' + re.escape(tld) + r')'
                r'(?=[\s\)\]\},;:!?.。，、：》）""''/\\　-]|$)',
                re.IGNORECASE,
            )
            text = dot_restore_re.sub(r'.\1', text)

        return text

    def _detect_url_like(self, text: str) -> List[Tuple[int, int, str]]:
        """v5.11.5.9: Detect URL-like substrings in text."""
        matches = []

        for m in re.finditer(
            r'(?:https?|ftp)\s*:\s*(?://|／／|\\\\|／|//)'
            r'[^\s<>"\'{}\[\]()（）一-鿿　-〿＀-￯]*',
            text, re.IGNORECASE,
        ):
            matches.append((m.start(), m.end(), m.group()))

        tld_alt = '|'.join(sorted(self._URL_TLDS, key=len, reverse=True))
        bare_domain_re = re.compile(
            r'(?<![@\w.\-])'
            r'([a-zA-Z0-9](?:[-a-zA-Z0-9]*[a-zA-Z0-9])?'
            r'(?:\.[a-zA-Z0-9](?:[-a-zA-Z0-9]*[a-zA-Z0-9])?)*'
            r'\.(?:' + tld_alt + r'))'
            r'(?:/[\w\-._~:/?#\[\]@!$&()*+,;=%]*)?'
            r'(?=[\s\)\]\},;:!?.。，、：》）""''　-]|$)',
            re.IGNORECASE,
        )
        seen_spans = set()
        for m in bare_domain_re.finditer(text):
            span = (m.start(), m.end())
            overlaps = any(
                s <= m.start() < e or s < m.end() <= e
                for s, e, _ in matches
            )
            if not overlaps and span not in seen_spans:
                seen_spans.add(span)
                matches.append((m.start(), m.end(), m.group()))

        return matches


# ============================================================================
# 便捷函数
# ============================================================================

def extract_text_from_image(image_path, engine: str = "easyocr") -> str:
    ocr = OCREngine(engine=engine)
    return ocr.extract_text(image_path)


def extract_text_from_bytes(image_bytes: bytes, engine: str = "easyocr") -> str:
    ocr = OCREngine(engine=engine)
    return ocr.extract_text(image_bytes)
