"""evaluate — EvaluationRunConfig, all-sample bootstrap, full CI, manifest"""
import argparse, hashlib, json, logging, math, random, sys, time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROMPT_HASH, PROMPT_HASH_SHORT, PROMPT_VERSION,
    PROMPT_BUNDLE_HASH, SCHEMA_REPAIR_PROMPT_VERSION, SCHEMA_REPAIR_PROMPT_HASH,
    SCHEMA_REPAIR_PROMPT_TEMPLATE,
    RUNTIME_REPORT_DIR, MANIFEST_DIR, APP_VERSION,
    API_RETRY_CONFIG, LLM_SCHEMA_RETRY_MAX,
)
from pipeline import AnalysisPipeline

logger = logging.getLogger("evaluate")

@dataclass
class EvaluationRunConfig:
    """CLI args → immutable run config, passed through to manifest/report"""
    split: str = "all"
    max_samples: Optional[int] = None
    seed: int = 42
    strategy: str = "weighted"
    use_llm: bool = False
    test_data: str = ""
    output: str = ""
    # Derived at runtime
    run_id: str = ""
    data_sha256: str = ""
    total_before_filter: int = 0
    total_after_filter: int = 0
    total_evaluated: int = 0
    n_phishing: int = 0
    n_legitimate: int = 0
    final_sample_ids: List[str] = field(default_factory=list)
    # v5.11.5.9: targeted evaluation by explicit sample IDs
    selection_mode: str = ""  # "" | "explicit_ids"
    requested_sample_ids: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    success_count: int = 0
    failure_count: int = 0

    def make_run_id(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # microseconds prevent collision
        self.run_id = f"run-{ts}-s{self.seed}-{self.strategy}-{'llm' if self.use_llm else 'rule'}"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-data", default=None)
    p.add_argument("--output", default=str(RUNTIME_REPORT_DIR))
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-id", action="append", default=None, dest="sample_ids",
                   help="Target specific sample ID(s) for evaluation. Repeatable. "
                        "Mutually exclusive with --max-samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["dev","blind","adversarial","all"], default="all")
    p.add_argument("--strategy", choices=["weighted","rule_only","hybrid"], default="weighted")
    p.add_argument("--allow-deterministic-fallback", action="store_true",
                   help="Allow pipeline to continue with rule-only when LLM config is incomplete")
    # v5.11.5.4: comparison report CLI
    p.add_argument("--compare-rule", default=None,
                   help="Path to rule-only report directory (for --compare mode)")
    p.add_argument("--compare-hybrid", default=None,
                   help="Path to hybrid/LLM report directory (for --compare mode)")
    p.add_argument("--compare-out", default=None,
                   help="Output path for comparison markdown (for --compare mode)")
    return p.parse_args()

def wilson_ci(p, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; m = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / d
    return (max(0, c-m), min(1, c+m))

def f1_bootstrap(y_true, y_pred, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap F1 using (y_true, y_pred) paired resampling. Same samples as point estimate."""
    if len(y_true) == 0: return (0.0, 0.0)
    idx = np.arange(len(y_true))
    rng = np.random.RandomState(seed)
    f1s = []
    for _ in range(n_boot):
        bi = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[bi], y_pred[bi]
        tp = int(np.sum((yt==True)&(yp==True)))
        fp = int(np.sum((yt==False)&(yp==True)))
        fn = int(np.sum((yt==True)&(yp==False)))
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f1s.append(2*p*r/(p+r) if (p+r)>0 else 0.0)
    return (float(np.percentile(f1s, alpha/2*100)), float(np.percentile(f1s, (1-alpha/2)*100)))

def fmt_pct(v): return f"{v:.2%}" if v is not None else "N/A"

class ModelEvaluator:
    def __init__(self, config: EvaluationRunConfig, allow_deterministic_fallback: bool = False):
        self.config = config
        self.pipeline = AnalysisPipeline(
            use_llm=config.use_llm,
            allow_deterministic_fallback=allow_deterministic_fallback)
        self.version = self.pipeline.version
        self.results: List[Dict] = []
        self.metrics: Dict = {}

    def load_data(self, path):
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(str(p))
        with open(p, "r", encoding="utf-8") as f: data = json.load(f)
        return data if isinstance(data, list) else data.get("samples", [])

    def stratified_sample(self, samples, n):
        random.seed(self.config.seed)
        phishing = [s for s in samples if s.get("is_phishing")]
        legit = [s for s in samples if not s.get("is_phishing")]
        random.shuffle(phishing); random.shuffle(legit)
        half = n // 2
        p_n = min(half, len(phishing)); l_n = min(n - p_n, len(legit))
        result = phishing[:p_n] + legit[:l_n]
        random.shuffle(result)
        return result

    def run(self, samples):
        if self.config.max_samples:
            samples = self.stratified_sample(samples, self.config.max_samples)
        total = len(samples)
        self.results = []
        failed = 0
        t0 = time.time()
        strategy = self.config.strategy

        p_count = sum(1 for s in samples if s.get("is_phishing"))
        # v5.11.5.1: detailed LLM status at start
        llm_requested = self.pipeline.llm_requested
        llm_configured = self.pipeline.llm_configured
        llm_active = self.pipeline.use_llm and self.pipeline.llm_client is not None
        print(f"\nEval: {total} samples (P:{p_count} L:{total-p_count}) | {strategy}")
        print(f"LLM requested={llm_requested} configured={llm_configured} active={llm_active}")

        for i, s in enumerate(samples, 1):
            sid = s.get("id", str(i)); tl = s.get("is_phishing", False)
            content = s.get("content", "")
            content_sha = hashlib.sha256(content.encode()).hexdigest()
            t1 = time.time()
            rec = {
                "sample_id": sid,
                "content_sha256": content_sha,
                "content_type": s.get("content_type", "email"),
                "source_channel": s.get("source_channel", "unknown"),
                "phishing_type": s.get("phishing_type", "normal"),
                "attack_pattern": s.get("attack_pattern", ""),
                "true_label": tl,
                "split": s.get("split", "unknown"),
                "strategy": strategy,
                "prediction_success": False,
                "response_status": "error",
            }
            try:
                resp = self.pipeline.analyze(
                    content=content, content_type=s.get("content_type", "email"),
                    source_channel=s.get("source_channel", ""), metadata=s.get("metadata"),
                    strategy=strategy)
                sr = resp.score_result
                elapsed = (time.time() - t1) * 1000
                rule_ids = []
                for ev in resp.verified_evidence:
                    rule_ids.extend(list(ev.rule_ids))
                # v5.11: enhanced auditability fields
                ve_summary = []
                for ev in resp.verified_evidence[:20]:
                    q = (ev.quote or "")[:80]
                    ve_summary.append({
                        "type": str(ev.type) if ev.type else "unknown",
                        "source": str(ev.source) if hasattr(ev, 'source') else "unknown",
                        "severity": str(ev.severity) if ev.severity else "low",
                        "rule_ids": list(ev.rule_ids) if hasattr(ev, 'rule_ids') else [],
                        "quote_hash": hashlib.sha256((ev.quote or "").encode()).hexdigest()[:12],
                        "quote_preview": q,
                    })
                llm_stats = self.pipeline.get_llm_stats()
                rec.update({
                    "prediction_success": True, "response_status": "success",
                    "pred_is_phishing": sr.is_phishing,
                    "pred_risk_score": sr.risk_score, "pred_risk_level": sr.risk_level,
                    "confidence": sr.confidence, "elapsed_ms": round(elapsed, 1),
                    "model_used": resp.model_used,
                    "verified_count": len(resp.verified_evidence),
                    "unverified_count": len(resp.unverified_evidence),
                    "rule_ids": list(set(rule_ids)),
                    "error_type": None,
                    # v5.11: enhanced auditability fields
                    "score_strategy": sr.strategy,
                    "score_breakdown": sr.score_breakdown,
                    "validated_feature_scores": self._get_validated_scores(resp),
                    "verified_evidence_summary": ve_summary,
                    "unverified_evidence_count": len(resp.unverified_evidence),
                    # v5.11.2: per-sample LLM status from real response (not use_llm)
                    "llm_status": resp.llm_sample_status,
                    "llm_model": llm_stats.get("llm_model", "unknown"),
                    "llm_latency_ms": round(resp.llm_total_latency_ms, 1),
                    "validation_error_count": len(resp.validation_errors),
                    "warning_types": [w.get("type","") for w in resp.warnings] if resp.warnings else [],
                    # v5.11.2: per-sample LLM counters from telemetry summation
                    "llm_request_count_for_sample": resp.llm_request_count_for_sample,
                    "llm_http_success_count_for_sample": resp.llm_http_success_count_for_sample,
                    "llm_http_error_count_for_sample": resp.llm_http_error_count_for_sample,
                    "llm_transport_retry_count_for_sample": resp.llm_transport_retry_count_for_sample,
                    "llm_schema_retry_count_for_sample": resp.llm_schema_retry_count_for_sample,
                    "llm_invalid_response_count_for_sample": resp.llm_invalid_response_count_for_sample,
                    "llm_json_mode_fallback_count_for_sample": resp.llm_json_mode_fallback_count_for_sample,
                    "llm_validation_error_types": resp.llm_validation_error_types,
                    "fallback_used": resp.fallback_used,
                    "fallback_reason": resp.fallback_reason,
                    # v5.11.4: null normalization audit
                    "llm_null_normalization_count_for_sample": resp.llm_null_normalization_count_for_sample,
                    "llm_normalized_field_paths": resp.llm_normalized_field_paths,
                    # v5.11.5: enum normalization + deterministic fallback audit
                    "llm_enum_normalization_count_for_sample": resp.llm_enum_normalization_count_for_sample,
                    "llm_enum_normalized_paths": resp.llm_enum_normalized_paths,
                    "llm_fallback_to_deterministic": resp.llm_fallback_to_deterministic,
                    # v5.11.5.9: conservative enum fallback + unknown enum security audit
                    "llm_unknown_enum_fallback_count_for_sample": resp.llm_unknown_enum_fallback_count_for_sample,
                    "llm_unknown_enum_paths": resp.llm_unknown_enum_paths,
                    "llm_unknown_enum_candidates": resp.llm_unknown_enum_candidates,
                    "llm_conservative_enum_fallback_used": resp.llm_conservative_enum_fallback_used,
                    # v5.11.5.9: add recovered_after_conservative_enum_fallback to valid statuses
                    "llm_evaluation_valid": resp.llm_sample_status in (
                        "usable_first_attempt", "recovered_after_validation_retry",
                        "recovered_after_conservative_enum_fallback"),
                    "invalid_reason": resp.llm_sample_status if resp.llm_sample_status not in (
                        "usable_first_attempt", "recovered_after_validation_retry",
                        "recovered_after_conservative_enum_fallback", "not_requested") else None,
                })
                ok = "OK" if sr.is_phishing == tl else "WRONG"
                # v5.11.5: mark terminal LLM failures on console line
                terminal_flag = ""
                if resp.llm_sample_status in ("terminal_validation_failed", "terminal_transport_failed"):
                    terminal_flag = " ⚠LLM_FAIL"
                elif resp.llm_sample_status == "not_configured":
                    terminal_flag = " ⚠LLM_NOCFG"
                print(f"[{i}/{total}] {ok} {sid} t={'P' if tl else 'N'} p={'P' if sr.is_phishing else 'N'} s={sr.risk_score}{terminal_flag}")
            except Exception as e:
                failed += 1
                rec.update({"error_type": type(e).__name__, "response_status": "error"})
                print(f"[{i}/{total}] FAIL {sid}: {type(e).__name__}")
            self.results.append(rec)

        self.config.success_count = total - failed
        self.config.failure_count = failed
        self.config.total_evaluated = total
        self.config.n_phishing = p_count
        self.config.n_legitimate = total - p_count
        self.config.final_sample_ids = [r["sample_id"] for r in self.results]
        self.config.end_time = datetime.now(timezone.utc).isoformat()
        print(f"Done: {time.time()-t0:.1f}s | fail={failed}/{total}")

        # v5.11.5.2: use config.success_count/failure_count (pipeline-level), not llm_client counters
        llm_stats = self.pipeline.get_llm_stats()
        print(f"Pipeline success: {self.config.success_count}/{total}")
        print(f"Pipeline failure: {self.config.failure_count}")
        print(f"LLM requested samples: {llm_stats.get('llm_sample_requested_count', 0)}")
        print(f"LLM usable samples: {llm_stats.get('llm_usable_response_count', 0)}")
        print(f"Terminal validation failures: {llm_stats.get('llm_terminal_validation_failure_count', 0)}")
        print(f"Terminal transport failures: {llm_stats.get('llm_terminal_transport_failure_count', 0)}")
        print(f"Deterministic fallback count: {llm_stats.get('llm_deterministic_fallback_count', 0)}")
        eval_valid = self._compute_llm_evaluation_valid(llm_stats, total)
        print(f"LLM evaluation valid: {eval_valid}")
        return self.results

    @staticmethod
    def _validate_count_int(value, name: str) -> bool:
        """v5.11.5.3: Validate that a count value is a non-negative integer.
        Rejects: bool (True/False are ints in Python), float, str, None, negative, missing."""
        if value is None:
            return False
        if isinstance(value, bool):
            return False  # bool is subclass of int in Python — reject explicitly
        if not isinstance(value, int):
            return False
        if value < 0:
            return False
        return True

    @staticmethod
    def _compute_llm_evaluation_valid(llm_stats: dict, total_evaluated: int = 0) -> bool:
        """v5.11.5.3: Determine if LLM evaluation results are valid.
        Priority chain:
        - not_requested → False
        - not_configured → False
        - not_active (no calls attempted) → False
        - terminal validation or transport failures > 0 → False
        - deterministic fallback > 0 → False
        - count mismatch (any of 7 invariants fail) → False
        - All checks pass → True

        Count closure invariants (v5.11.5.3 — all 7 must pass):
        1. sample_requested_count == total_evaluated (when total_evaluated > 0)
        2. usable + terminal_validation + terminal_transport == sample_requested
        3. request_attempt_count + cache_hit_count >= sample_requested_count
        4. request_attempt_count == http_success_count + http_error_count
        5. latency_sample_count == request_attempt_count
        6. usable_response_count <= http_success_count + cache_hit_count
        7. All counts are non-negative ints (bool rejected, None treated as missing)
        """
        # v5.11.5.3: Validate all count fields are proper ints (not bool, not None, not negative)
        _vi = ModelEvaluator._validate_count_int
        count_fields = [
            "llm_sample_requested_count", "llm_usable_response_count",
            "llm_terminal_validation_failure_count", "llm_terminal_transport_failure_count",
            "llm_deterministic_fallback_count", "llm_request_attempt_count",
            "llm_http_success_count", "llm_http_error_count",
            "llm_latency_sample_count", "llm_cache_hit_count",
        ]
        # Collect missing or invalid fields for count_mismatch
        has_invalid_counts = False
        for f in count_fields:
            if f not in llm_stats:
                has_invalid_counts = True
            elif not _vi(llm_stats.get(f), f):
                has_invalid_counts = True

        if not llm_stats.get("llm_requested", False):
            return False  # not applicable
        if not llm_stats.get("llm_configured", False):
            return False
        # v5.11.5.2: check llm_active — must have actually made calls
        if not llm_stats.get("llm_call_attempted", False):
            return False
        if llm_stats.get("llm_terminal_validation_failure_count", 0) > 0:
            return False
        if llm_stats.get("llm_terminal_transport_failure_count", 0) > 0:
            return False
        if llm_stats.get("llm_deterministic_fallback_count", 0) > 0:
            return False

        # v5.11.5.3: if any count fields are invalid types → count_mismatch
        if has_invalid_counts:
            return False

        # Must have actually made LLM calls
        if llm_stats.get("llm_sample_requested_count", 0) == 0:
            return False

        # Invariant 1: sample_requested must match total_evaluated (if provided)
        if total_evaluated > 0:
            req = llm_stats.get("llm_sample_requested_count", 0)
            if req != total_evaluated:
                return False

        # Invariant 2: usable + terminal_validation + terminal_transport == sample_requested
        usable = llm_stats.get("llm_usable_response_count", 0)
        term_val = llm_stats.get("llm_terminal_validation_failure_count", 0)
        term_tport = llm_stats.get("llm_terminal_transport_failure_count", 0)
        req = llm_stats.get("llm_sample_requested_count", 0)
        if usable + term_val + term_tport != req:
            return False

        # Invariant 3: request_attempt_count + cache_hit_count >= sample_requested_count
        attempts = llm_stats.get("llm_request_attempt_count", 0)
        cache_hits = llm_stats.get("llm_cache_hit_count", 0)
        if attempts + cache_hits < req:
            return False

        # Invariant 4: request_attempt_count == http_success_count + http_error_count
        successes = llm_stats.get("llm_http_success_count", 0)
        errors = llm_stats.get("llm_http_error_count", 0)
        if attempts != successes + errors:
            return False

        # Invariant 5: latency_sample_count == request_attempt_count
        latency_count = llm_stats.get("llm_latency_sample_count", 0)
        if latency_count != attempts:
            return False

        # Invariant 6: usable_response_count <= http_success_count + cache_hit_count
        if usable > successes + cache_hits:
            return False

        return True

    @staticmethod
    def _compute_llm_invalid_reason(llm_stats: dict, total_evaluated: int = 0) -> str:
        """v5.11.5.3: Generate invalid_reason string for manifest.
        Priority chain: not_requested > not_configured > not_active >
        terminal_llm_failure > deterministic_fallback_used > count_mismatch > ""
        """
        # v5.11.5.3: validate count fields for type safety
        _vi = ModelEvaluator._validate_count_int
        count_fields = [
            "llm_sample_requested_count", "llm_usable_response_count",
            "llm_terminal_validation_failure_count", "llm_terminal_transport_failure_count",
            "llm_deterministic_fallback_count", "llm_request_attempt_count",
            "llm_http_success_count", "llm_http_error_count",
            "llm_latency_sample_count", "llm_cache_hit_count",
        ]
        has_invalid_counts = False
        for f in count_fields:
            if f not in llm_stats:
                has_invalid_counts = True
            elif not _vi(llm_stats.get(f), f):
                has_invalid_counts = True

        if not llm_stats.get("llm_requested", False):
            return "not_requested"
        if not llm_stats.get("llm_configured", False):
            return "not_configured"
        if not llm_stats.get("llm_call_attempted", False):
            return "not_active"
        if (llm_stats.get("llm_terminal_validation_failure_count", 0) > 0 or
                llm_stats.get("llm_terminal_transport_failure_count", 0) > 0):
            return "terminal_llm_failure"
        if llm_stats.get("llm_deterministic_fallback_count", 0) > 0:
            return "deterministic_fallback_used"

        # v5.11.5.3: invalid count types → count_mismatch (before invariant checks)
        if has_invalid_counts:
            return "count_mismatch"

        # v5.11.5.3: full count closure checks
        if total_evaluated > 0:
            req = llm_stats.get("llm_sample_requested_count", 0)
            if req != total_evaluated:
                return "count_mismatch"
        usable = llm_stats.get("llm_usable_response_count", 0)
        term_val = llm_stats.get("llm_terminal_validation_failure_count", 0)
        term_tport = llm_stats.get("llm_terminal_transport_failure_count", 0)
        req = llm_stats.get("llm_sample_requested_count", 0)
        if usable + term_val + term_tport != req:
            return "count_mismatch"
        attempts = llm_stats.get("llm_request_attempt_count", 0)
        cache_hits = llm_stats.get("llm_cache_hit_count", 0)
        if attempts + cache_hits < req:
            return "count_mismatch"
        successes = llm_stats.get("llm_http_success_count", 0)
        errors = llm_stats.get("llm_http_error_count", 0)
        if attempts != successes + errors:
            return "count_mismatch"
        latency_count = llm_stats.get("llm_latency_sample_count", 0)
        if latency_count != attempts:
            return "count_mismatch"
        if usable > successes + cache_hits:
            return "count_mismatch"
        return ""

    @staticmethod
    def _get_validated_scores(resp):
        """Extract validated feature scores from pipeline response for auditability"""
        # The validated features are internal to the pipeline; extract from score_breakdown
        bd = resp.score_result.score_breakdown
        return {k: v for k, v in bd.items()
                if k in ("sender_credibility","link_safety","content_urgency",
                         "information_request","language_quality","attachment_risk")}

    def compute_metrics(self):
        if not self.results: return {}
        total = len(self.results)
        seed = self.config.seed

        # All-sample conservative: failures counted as wrong
        yt = np.array([r["true_label"] for r in self.results])
        yp = np.array([r.get("pred_is_phishing") if r.get("prediction_success") else (not r["true_label"]) for r in self.results])

        tp = int(np.sum((yt==True)&(yp==True))); tn = int(np.sum((yt==False)&(yp==False)))
        fp = int(np.sum((yt==False)&(yp==True))); fn = int(np.sum((yt==True)&(yp==False)))
        n_phish = int(np.sum(yt==True)); n_legit = int(np.sum(yt==False))
        has_both = n_phish > 0 and n_legit > 0

        acc = (tp+tn)/total if total>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 and has_both else None
        rec = tp/(tp+fn) if (tp+fn)>0 and has_both else None
        f1 = 2*prec*rec/(prec+rec) if prec and rec and (prec+rec)>0 else (0.0 if tp==0 and has_both else None)
        spec = tn/(tn+fp) if (tn+fp)>0 and has_both else None
        fpr_v = fp/(fp+tn) if (fp+tn)>0 and has_both else None
        fnr_v = fn/(fn+tp) if (fn+tp)>0 and has_both else None
        fail_rate = sum(1 for r in self.results if not r.get("prediction_success"))/total if total>0 else 0

        # F1 bootstrap on ALL-SAMPLE y_true/y_pred (same conservative array as point estimate)
        f1_ci = None
        if has_both and len(yt) >= 2:
            f1_ci = f1_bootstrap(yt, yp, seed=seed)

        self.metrics = {
            "total": total, "n_phishing": n_phish, "n_legitimate": n_legit,
            "has_both": has_both, "invalid": not has_both,
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            # Accuracy
            "accuracy": round(acc, 4),
            "accuracy_95ci": wilson_ci(acc, total),
            # Precision
            "precision": round(prec, 4) if prec is not None else None,
            "precision_95ci": wilson_ci(prec, tp+fp) if prec is not None and (tp+fp)>0 else None,
            # Recall
            "recall": round(rec, 4) if rec is not None else None,
            "recall_95ci": wilson_ci(rec, tp+fn) if rec is not None and (tp+fn)>0 else None,
            # F1
            "f1": round(f1, 4) if f1 is not None else None,
            "f1_bootstrap_95ci": (round(f1_ci[0],4), round(f1_ci[1],4)) if f1_ci else None,
            # Specificity
            "specificity": round(spec, 4) if spec is not None else None,
            "specificity_95ci": wilson_ci(spec, tn+fp) if spec is not None and (tn+fp)>0 else None,
            # FPR
            "fpr": round(fpr_v, 4) if fpr_v is not None else None,
            "fpr_95ci": wilson_ci(fpr_v, fp+tn) if fpr_v is not None and (fp+tn)>0 else None,
            # FNR
            "fnr": round(fnr_v, 4) if fnr_v is not None else None,
            "fnr_95ci": wilson_ci(fnr_v, fn+tp) if fnr_v is not None and (fn+tp)>0 else None,
            # Failure
            "failure_rate": round(fail_rate, 4),
            "failure_rate_95ci": wilson_ci(fail_rate, total),
            "failure_count": self.config.failure_count,
            "success_count": self.config.success_count,
        }
        return self.metrics

    def generate_report(self, out_dir):
        if not self.metrics: self.compute_metrics()
        m = self.metrics; cfg = self.config; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        cm = m["confusion"]
        inv = "\n**WARNING: INVALID EVALUATION**" if m.get("invalid") else ""

        # v5.11: LLM stats for report header
        llm_stats = self.pipeline.get_llm_stats()
        llm_status_line = self._llm_report_line(llm_stats)

        # Pre-compute all formatted strings (no Python expressions in markdown)
        acc_ci = f"[{m['accuracy_95ci'][0]:.2%},{m['accuracy_95ci'][1]:.2%}]"
        prec_ci = f"[{m['precision_95ci'][0]:.2%},{m['precision_95ci'][1]:.2%}]" if m.get('precision_95ci') else ""
        rec_ci = f"[{m['recall_95ci'][0]:.2%},{m['recall_95ci'][1]:.2%}]" if m.get('recall_95ci') else ""
        f1_ci = f"[{m['f1_bootstrap_95ci'][0]:.2%},{m['f1_bootstrap_95ci'][1]:.2%}]" if m.get('f1_bootstrap_95ci') else ""
        spec_ci = f"[{m['specificity_95ci'][0]:.2%},{m['specificity_95ci'][1]:.2%}]" if m.get('specificity_95ci') else ""
        fpr_ci = f"[{m['fpr_95ci'][0]:.2%},{m['fpr_95ci'][1]:.2%}]" if m.get('fpr_95ci') else ""
        fnr_ci = f"[{m['fnr_95ci'][0]:.2%},{m['fnr_95ci'][1]:.2%}]" if m.get('fnr_95ci') else ""
        fr_ci = f"[{m['failure_rate_95ci'][0]:.2%},{m['failure_rate_95ci'][1]:.2%}]" if m.get('failure_rate_95ci') else ""

        # v5.11: auto-generated error analysis from raw predictions
        error_table = self._build_error_table()
        # v5.11.1: LLM quality table for non-trivial samples
        llm_quality_table = self._build_llm_quality_table()

        rep = f"""# Phishing Detection Evaluation v{APP_VERSION}{inv}
**Run**: {cfg.run_id} | **Time**: {cfg.end_time}
**Split**: {cfg.split} | **Strategy**: {cfg.strategy} | **Seed**: {cfg.seed}
{llm_status_line}
**Prompt**: {PROMPT_VERSION} ({PROMPT_HASH_SHORT}...) | **Bundle**: {PROMPT_BUNDLE_HASH[:16]}... | **SchemaRepair**: {SCHEMA_REPAIR_PROMPT_VERSION} ({SCHEMA_REPAIR_PROMPT_HASH[:16]}...)

## Confusion
| | Pred:Phish | Pred:Normal |
|---|---|---|
| True:Phish | TP={cm['TP']} | FN={cm['FN']} |
| True:Normal | FP={cm['FP']} | TN={cm['TN']} |

## Metrics
| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | {fmt_pct(m['accuracy'])} | {acc_ci} |
| Precision | {fmt_pct(m['precision'])} | {prec_ci} |
| Recall | {fmt_pct(m['recall'])} | {rec_ci} |
| F1 | {fmt_pct(m['f1'])} | {f1_ci} |
| Specificity | {fmt_pct(m['specificity'])} | {spec_ci} |
| FPR | {fmt_pct(m['fpr'])} | {fpr_ci} |
| FNR | {fmt_pct(m['fnr'])} | {fnr_ci} |
| Failure Rate | {fmt_pct(m['failure_rate'])} | {fr_ci} |
| Success/Total | {m['success_count']}/{m['total']} | |
{error_table}
{llm_quality_table}
*Auto-generated v{APP_VERSION} run={cfg.run_id}*
"""
        rp = out / f"evaluation_report_{cfg.run_id}.md"
        rp.write_text(rep, encoding="utf-8")
        (out / "evaluation_report_latest.md").write_text(rep, encoding="utf-8")

        # Raw predictions
        pp = out / f"raw_predictions_{cfg.run_id}.json"
        pp.write_text(json.dumps({
            "run_id": cfg.run_id,
            "config": {
                "app_version": APP_VERSION, "pipeline_version": get_pipeline_version(),
                "scorer_version": get_scorer_version(),
                "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
                "prompt_bundle_hash": PROMPT_BUNDLE_HASH,
                "schema_repair_prompt_version": SCHEMA_REPAIR_PROMPT_VERSION,
                "schema_repair_prompt_hash": SCHEMA_REPAIR_PROMPT_HASH,
                "split": cfg.split, "max_samples": cfg.max_samples,
                "seed": cfg.seed, "strategy": cfg.strategy, "use_llm": cfg.use_llm,
                "data_sha256": cfg.data_sha256,
                "total_before_filter": cfg.total_before_filter,
                "total_after_filter": cfg.total_after_filter,
            },
            "llm_stats": llm_stats,
            "metrics": m, "predictions": self.results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Manifest with real LLM stats
        run_manifest, scorer_ver = build_manifest(cfg, m, pipeline=self.pipeline)
        mp1 = out / f"manifest_{cfg.run_id}.json"
        mp1.write_text(run_manifest, encoding="utf-8")
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        mp2 = MANIFEST_DIR / f"manifest_{cfg.run_id}.json"
        mp2.write_text(run_manifest, encoding="utf-8")

        print(f"Report: {rp}")
        print(f"Manifest: {mp1}")
        print(f"Manifest: {mp2}")
        return str(rp)

    @staticmethod
    def _llm_report_line(llm_stats: dict) -> str:
        """v5.11.5: generate LLM status line with terminal failure visibility"""
        if not llm_stats.get("llm_configured"):
            return "**LLM**: not_configured | **OCR**: not_run"
        v = llm_stats
        line = (f"**LLM**: {v.get('llm_vendor','?')}/{v.get('llm_model','?')} "
                f"(samples:{v.get('llm_sample_requested_count',0)} "
                f"attempts:{v.get('llm_request_attempt_count',0)} "
                f"usable:{v.get('llm_usable_response_count',0)} "
                f"tport_retry:{v.get('llm_transport_retry_count',0)} "
                f"schema_retry:{v.get('llm_schema_retry_count',0)} "
                f"json_fb:{v.get('llm_json_mode_fallback_count',0)})")
        # v5.11.5: show terminal failures clearly if any
        terminal_val = v.get('llm_terminal_validation_failure_count', 0)
        terminal_trans = v.get('llm_terminal_transport_failure_count', 0)
        det_fallback = v.get('llm_deterministic_fallback_count', 0)
        if terminal_val or terminal_trans:
            line += f" ⚠ TerminalFail: val={terminal_val} tport={terminal_trans}"
        if det_fallback:
            line += f" det_fallback={det_fallback}"
        line += f" | **OCR**: not_run"
        return line

    def _build_error_table(self) -> str:
        """v5.11: auto-generate FP/FN error analysis from raw predictions"""
        if not self.results: return ""
        fps = [r for r in self.results if r.get("prediction_success") and not r["true_label"] and r.get("pred_is_phishing")]
        fns = [r for r in self.results if r.get("prediction_success") and r["true_label"] and not r.get("pred_is_phishing")]

        lines = []
        if fps:
            lines.append("\n## False Positives (正常判为钓鱼)")
            lines.append("| Sample | Score | Strategy | Top Contributions |")
            lines.append("|---|---|---|---|")
            for r in fps:
                bd = r.get("score_breakdown", {})
                contribs = self._top_contributions(bd, 3)
                contrib_str = "; ".join(contribs) if contribs else "N/A"
                lines.append(f"| {r['sample_id']} | {r.get('pred_risk_score','?')} | {r.get('score_strategy','?')} | {contrib_str} |")
        if fns:
            lines.append("\n## False Negatives (钓鱼判为正常)")
            lines.append("| Sample | Score | Strategy | Top Contributions |")
            lines.append("|---|---|---|---|")
            for r in fns:
                bd = r.get("score_breakdown", {})
                contribs = self._top_contributions(bd, 3)
                contrib_str = "; ".join(contribs) if contribs else "N/A"
                lines.append(f"| {r['sample_id']} | {r.get('pred_risk_score','?')} | {r.get('score_strategy','?')} | {contrib_str} |")

        if not lines:
            lines.append("\n*No FP or FN in this run.*")
        return "\n".join(lines)

    def _build_llm_quality_table(self) -> str:
        """v5.11.5: list all non-trivial samples — recovered, terminal-failure, null-normalized,
        AND enum-normalized. Shows both normalization types and evaluation validity."""
        if not self.results:
            return ""
        interesting = [r for r in self.results if r.get("llm_status") not in (
            "not_requested", "not_configured", None)]
        if not interesting:
            return ""
        # Separate normalized samples (even if usable_first_attempt) from purely trivial ones
        null_normalized = [r for r in interesting if r.get("llm_null_normalization_count_for_sample", 0) > 0]
        enum_normalized = [r for r in interesting if r.get("llm_enum_normalization_count_for_sample", 0) > 0]
        non_trivial = [r for r in interesting if r.get("llm_status") not in ("usable_first_attempt",)]
        # Show all normalized + all non-trivial, deduplicated
        shown = {r["sample_id"]: r for r in (null_normalized + enum_normalized + non_trivial)}
        if not shown:
            return "\n*LLM quality: all requested samples returned usable on first attempt with no normalization.*"

        lines = ["\n## LLM Quality (non-trivial samples)"]
        lines.append("| Sample | Status | Requests | HTTP OK | HTTP Err | TportRtry | SchemaRtry | Invalid | JSON FB | NullNorm | EnumNorm | NormPaths | Error Types | Fallback |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(shown.values(), key=lambda x: x["sample_id"]):
            status = r.get("llm_status", "?")
            reqs = r.get("llm_request_count_for_sample", 0)
            http_ok = r.get("llm_http_success_count_for_sample", 0)
            http_err = r.get("llm_http_error_count_for_sample", 0)
            tport_r = r.get("llm_transport_retry_count_for_sample", 0)
            schema_r = r.get("llm_schema_retry_count_for_sample", 0)
            invalid = r.get("llm_invalid_response_count_for_sample", 0)
            json_fb = r.get("llm_json_mode_fallback_count_for_sample", 0)
            null_norm = r.get("llm_null_normalization_count_for_sample", 0)
            enum_norm = r.get("llm_enum_normalization_count_for_sample", 0)
            # Merge null + enum norm paths
            all_norm_paths = (r.get("llm_normalized_field_paths", []) or []) + (r.get("llm_enum_normalized_paths", []) or [])
            norm_paths = ", ".join(all_norm_paths[:3]) or "—"
            err_types = ", ".join(r.get("llm_validation_error_types", [])[:3]) or "—"
            fb = "yes" if r.get("fallback_used") else "no"
            lines.append(f"| {r['sample_id']} | {status} | {reqs} | {http_ok} | {http_err} | {tport_r} | {schema_r} | {invalid} | {json_fb} | {null_norm} | {enum_norm} | {norm_paths} | {err_types} | {fb} |")
        return "\n".join(lines)

    @staticmethod
    def _top_contributions(bd: dict, n: int = 3) -> list:
        """Extract top score contributions from breakdown dict"""
        contribs = []
        for k, v in bd.items():
            if not isinstance(v, dict): continue
            ws = v.get("weighted_score")
            if ws is not None and ws > 0:
                contribs.append((k, ws))
        contribs.sort(key=lambda x: -x[1])
        return [f"{k}={v:.1f}" for k, v in contribs[:n]]

    def print_summary(self):
        if not self.metrics: self.compute_metrics()
        m = self.metrics; cm = m["confusion"]
        print(f"\nCM: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} Acc={fmt_pct(m['accuracy'])}")
        if m.get("invalid"): print("WARNING: invalid evaluation")

# ---- helpers ----

def get_scorer_version():
    from risk_scorer import RiskScorer
    return RiskScorer.VERSION

def get_pipeline_version():
    try:
        from pipeline import AnalysisPipeline
        return AnalysisPipeline(use_llm=False).version
    except Exception:
        return APP_VERSION

def build_manifest(cfg: EvaluationRunConfig, metrics: Dict, pipeline=None):
    from risk_scorer import RiskScorer
    import platform
    # v5.11.1: get real LLM stats from live pipeline
    llm = {}
    if pipeline is not None:
        llm = pipeline.get_llm_stats()
    else:
        llm = {
            "llm_requested": cfg.use_llm,
            "llm_configured": False,
            "llm_vendor": "unknown",
            "llm_client_adapter": "none",
            "llm_api_type": "none",
            "llm_base_host": "",
            "llm_model": "not_configured",
            "llm_call_attempted": False,
            "llm_http_success_count": 0,
            "llm_usable_response_count": 0,
            "llm_failure_count": 0,
            "llm_validation_failure_count": 0,
            "llm_cache_hit_count": 0,
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
            "llm_json_mode_fallback_count": 0,
            "llm_http_error_count": 0,
            "llm_latency_sample_count": 0,
        }
        llm["prompt_bundle_hash"] = PROMPT_BUNDLE_HASH
        llm["schema_repair_prompt_version"] = SCHEMA_REPAIR_PROMPT_VERSION
        llm["schema_repair_prompt_hash"] = SCHEMA_REPAIR_PROMPT_HASH
        llm["max_http_attempts_per_sample"] = 0
        llm["llm_null_normalization_count"] = 0
        llm["llm_samples_with_null_normalization_count"] = 0
        # v5.11.5: new counters
        llm["llm_enum_normalization_count"] = 0
        llm["llm_samples_with_enum_normalization_count"] = 0
        llm["llm_pipeline_success_count"] = 0
        llm["llm_deterministic_fallback_count"] = 0
    m = {
        "run_id": cfg.run_id,
        "time": cfg.end_time,
        "app_version": APP_VERSION,
        "pipeline_version": get_pipeline_version(),
        "scorer_version": RiskScorer.VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "split": cfg.split,
        "max_samples": cfg.max_samples,
        "selection_mode": cfg.selection_mode,
        "requested_sample_ids": cfg.requested_sample_ids,
        "seed": cfg.seed,
        "strategy": cfg.strategy,
        "use_llm": cfg.use_llm,
        # v5.11: real LLM metadata from pipeline
        "llm_requested": llm.get("llm_requested", cfg.use_llm),
        "llm_configured": llm.get("llm_configured", False),
        "llm_vendor": llm.get("llm_vendor", "unknown"),
        "llm_client_adapter": llm.get("llm_client_adapter", "none"),
        "llm_api_type": llm.get("llm_api_type", "none"),
        "llm_base_host": llm.get("llm_base_host", ""),
        "llm_model": llm.get("llm_model", "not_configured"),
        "llm_call_attempted": llm.get("llm_call_attempted", False),
        "llm_http_success_count": llm.get("llm_http_success_count", 0),
        "llm_usable_response_count": llm.get("llm_usable_response_count", 0),
        "llm_failure_count": llm.get("llm_failure_count", 0),
        "llm_validation_failure_count": llm.get("llm_validation_failure_count", 0),
        "llm_cache_hit_count": llm.get("llm_cache_hit_count", 0),
        "llm_latency_p50_ms": llm.get("llm_latency_p50_ms", 0),
        "llm_latency_p95_ms": llm.get("llm_latency_p95_ms", 0),
        # v5.11.1: redefined LLM statistics
        "llm_sample_requested_count": llm.get("llm_sample_requested_count", 0),
        "llm_request_attempt_count": llm.get("llm_request_attempt_count", 0),
        "llm_transport_retry_count": llm.get("llm_transport_retry_count", 0),
        "llm_schema_retry_count": llm.get("llm_schema_retry_count", 0),
        "llm_invalid_response_count": llm.get("llm_invalid_response_count", 0),
        "llm_recovered_after_validation_count": llm.get("llm_recovered_after_validation_count", 0),
        "llm_terminal_validation_failure_count": llm.get("llm_terminal_validation_failure_count", 0),
        "llm_terminal_transport_failure_count": llm.get("llm_terminal_transport_failure_count", 0),
        "llm_json_mode_used": llm.get("llm_json_mode_used", False),
        "llm_json_mode_fallback_triggered": llm.get("llm_json_mode_fallback_triggered", False),
        "llm_json_mode_fallback_count": llm.get("llm_json_mode_fallback_count", 0),
        "llm_http_error_count": llm.get("llm_http_error_count", 0),
        # v5.11.3: latency sample count for invariant validation
        "llm_latency_sample_count": llm.get("llm_latency_sample_count", 0),
        # v5.11.4: null normalization audit
        "llm_null_normalization_count": llm.get("llm_null_normalization_count", 0),
        "llm_samples_with_null_normalization_count": llm.get("llm_samples_with_null_normalization_count", 0),
        # v5.11.5: enum normalization audit
        "llm_enum_normalization_count": llm.get("llm_enum_normalization_count", 0),
        "llm_samples_with_enum_normalization_count": llm.get("llm_samples_with_enum_normalization_count", 0),
        # v5.11.5.9: conservative enum fallback + unknown enum security audit
        "llm_unknown_enum_fallback_count": llm.get("llm_unknown_enum_fallback_count", 0),
        "llm_samples_with_unknown_enum_fallback_count": llm.get("llm_samples_with_unknown_enum_fallback_count", 0),
        "llm_unknown_enum_candidate_counts": llm.get("llm_unknown_enum_candidate_counts", {}),
        # v5.11.5: pipeline success vs LLM usable separation
        "llm_pipeline_success_count": llm.get("llm_pipeline_success_count", 0),
        "llm_deterministic_fallback_count": llm.get("llm_deterministic_fallback_count", 0),
        # v5.11.5.2: top-level evaluation validity fields with count closure
        "llm_evaluation_applicable": llm.get("llm_requested", cfg.use_llm),
        "llm_evaluation_valid": ModelEvaluator._compute_llm_evaluation_valid(llm, cfg.total_evaluated),
        "llm_invalid_reason": ModelEvaluator._compute_llm_invalid_reason(llm, cfg.total_evaluated),
        "pipeline_success_count": cfg.success_count,  # v5.11.5.2: from config, not llm_client
        "pipeline_failure_count": cfg.failure_count,
        "deterministic_fallback_count": llm.get("llm_deterministic_fallback_count", 0),
        "llm_active": (llm.get("llm_requested", False) and llm.get("llm_configured", False) and
                       llm.get("llm_call_attempted", False)),
        # v5.11.2: prompt bundle traceability
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "prompt_bundle_hash": PROMPT_BUNDLE_HASH,
        "schema_repair_prompt_version": SCHEMA_REPAIR_PROMPT_VERSION,
        "schema_repair_prompt_hash": SCHEMA_REPAIR_PROMPT_HASH,
        "max_http_attempts_per_sample": llm.get("max_http_attempts_per_sample",
            (API_RETRY_CONFIG["max_retries"] + 1) * (LLM_SCHEMA_RETRY_MAX + 1) if llm.get("llm_configured") else 0),
        "ocr_status": "not_run",
        "risk_threshold_high": 70,
        "risk_threshold_medium": 40,
        "data_file": cfg.test_data,
        "data_sha256": cfg.data_sha256,
        "total_before_filter": cfg.total_before_filter,
        "total_after_filter": cfg.total_after_filter,
        "total_evaluated": cfg.total_evaluated,
        "n_phishing": cfg.n_phishing,
        "n_legitimate": cfg.n_legitimate,
        "final_sample_ids": cfg.final_sample_ids,
        "success_count": cfg.success_count,
        "failure_count": cfg.failure_count,
        "python_version": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    return json.dumps(m, ensure_ascii=False, indent=2), RiskScorer.VERSION


def generate_llm_comparison(rule_dir: str, llm_dir: str, out_path: str):
    """v5.11.5.9: auto-generate llm_comparison.md from two manifest/raw pairs.

    Strict validation (v5.11.5.9 — tightened):
    - Both directories must contain manifest_*.json and raw_predictions_*.json
    - Rule manifest strategy must be "rule_only"
    - Rule manifest llm_invalid_reason must be "not_requested"
    - Hybrid manifest strategy must be "hybrid"
    - Hybrid manifest llm_evaluation_valid must be true
    - Both sides: split, seed, max_samples, total_evaluated, data_sha256, selection_mode must match
    - final_sample_ids must be identical in order and value
    - requested_sample_ids must be identical in order and value (v5.11.5.9)
    - raw predictions run_id must match respective manifest run_id
    - raw predictions sample_ids must match manifest final_sample_ids
    - metrics/confusion must be complete with valid numeric values
    - Any mismatch → specific error + non-zero exit + no comparison file
    - Real Delta calculated from raw numeric values (not parsed from formatted strings)
    """
    from risk_scorer import RiskScorer
    rule_manifest = _find_manifest(Path(rule_dir))
    llm_manifest = _find_manifest(Path(llm_dir))
    rule_raw = _find_raw(Path(rule_dir))
    llm_raw = _find_raw(Path(llm_dir))

    # v5.11.5.5: strict validation — fail early with clear errors
    errors = []
    if not rule_manifest:
        errors.append(f"Rule manifest not found in: {rule_dir}")
    if not llm_manifest:
        errors.append(f"Hybrid manifest not found in: {llm_dir}")
    if not rule_raw:
        errors.append(f"Rule raw predictions not found in: {rule_dir}")
    if not llm_raw:
        errors.append(f"Hybrid raw predictions not found in: {llm_dir}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    rm = json.loads(Path(rule_manifest).read_text(encoding="utf-8"))
    lm = json.loads(Path(llm_manifest).read_text(encoding="utf-8"))
    rp = json.loads(Path(rule_raw).read_text(encoding="utf-8"))
    lp = json.loads(Path(llm_raw).read_text(encoding="utf-8"))

    # ── v5.11.5.5: per-manifest strategy validation ──
    rule_strategy = rm.get("strategy", "")
    if rule_strategy != "rule_only":
        print(f"ERROR: Rule manifest strategy must be 'rule_only', got '{rule_strategy}'.", file=sys.stderr)
        sys.exit(1)

    rule_invalid_reason = rm.get("llm_invalid_reason", "")
    if rule_invalid_reason != "not_requested":
        print(f"ERROR: Rule manifest llm_invalid_reason must be 'not_requested', "
              f"got '{rule_invalid_reason}'.", file=sys.stderr)
        sys.exit(1)

    hybrid_strategy = lm.get("strategy", "")
    if hybrid_strategy != "hybrid":
        print(f"ERROR: Hybrid manifest strategy must be 'hybrid', got '{hybrid_strategy}'.", file=sys.stderr)
        sys.exit(1)

    # v5.11.5.5: hybrid manifest must have valid LLM evaluation
    llm_eval_valid = lm.get("llm_evaluation_valid", False)
    if not llm_eval_valid:
        reason = lm.get("llm_invalid_reason", "unknown")
        print(f"ERROR: Hybrid manifest llm_evaluation_valid is not true (reason: {reason}). "
              f"Cannot generate formal comparison report.", file=sys.stderr)
        sys.exit(1)

    # ── v5.11.5.5: cross-manifest field matching ──
    cross_checks = [
        ("split", "split"),
        ("seed", "seed"),
        ("max_samples", "max_samples"),
        ("total_evaluated", "total_evaluated"),
        ("data_sha256", "data_sha256"),
        ("selection_mode", "selection_mode"),
    ]
    for field, label in cross_checks:
        rv = rm.get(field)
        lv = lm.get(field)
        if rv != lv:
            print(f"ERROR: {label} mismatch — rule={rv!r}, hybrid={lv!r}.", file=sys.stderr)
            sys.exit(1)

    # final_sample_ids must be identical in order and value
    rule_ids = rm.get("final_sample_ids", [])
    hybrid_ids = lm.get("final_sample_ids", [])
    if rule_ids != hybrid_ids:
        print(f"ERROR: final_sample_ids differ between rule and hybrid manifests. "
              f"Rule has {len(rule_ids)} ids, hybrid has {len(hybrid_ids)} ids.", file=sys.stderr)
        sys.exit(1)

    # v5.11.5.9: requested_sample_ids must be identical in order and value
    rule_req_ids = rm.get("requested_sample_ids", [])
    hybrid_req_ids = lm.get("requested_sample_ids", [])
    if rule_req_ids != hybrid_req_ids:
        print(f"ERROR: requested_sample_ids differ between rule and hybrid manifests. "
              f"Rule has {len(rule_req_ids)} ids, hybrid has {len(hybrid_req_ids)} ids.", file=sys.stderr)
        sys.exit(1)

    # ── v5.11.5.5: raw predictions validation ──
    rule_raw_run_id = rp.get("run_id", "")
    hybrid_raw_run_id = lp.get("run_id", "")
    if rule_raw_run_id != rm.get("run_id", ""):
        print(f"ERROR: Rule raw predictions run_id ({rule_raw_run_id}) "
              f"does not match manifest run_id ({rm.get('run_id')}).", file=sys.stderr)
        sys.exit(1)
    if hybrid_raw_run_id != lm.get("run_id", ""):
        print(f"ERROR: Hybrid raw predictions run_id ({hybrid_raw_run_id}) "
              f"does not match manifest run_id ({lm.get('run_id')}).", file=sys.stderr)
        sys.exit(1)

    # raw predictions sample_ids must match manifest final_sample_ids
    rule_pred_ids = [p.get("sample_id", "") for p in rp.get("predictions", [])]
    hybrid_pred_ids = [p.get("sample_id", "") for p in lp.get("predictions", [])]
    if rule_pred_ids != rule_ids:
        print(f"ERROR: Rule raw predictions sample_ids do not match manifest final_sample_ids.", file=sys.stderr)
        sys.exit(1)
    if hybrid_pred_ids != hybrid_ids:
        print(f"ERROR: Hybrid raw predictions sample_ids do not match manifest final_sample_ids.", file=sys.stderr)
        sys.exit(1)

    # ── v5.11.5.5: metrics/confusion completeness ──
    for label, raw_data in [("Rule", rp), ("Hybrid", lp)]:
        m = raw_data.get("metrics", {})
        if not m:
            print(f"ERROR: {label} raw predictions has no metrics field.", file=sys.stderr)
            sys.exit(1)
        cm = m.get("confusion", {})
        if not cm:
            print(f"ERROR: {label} raw predictions has no confusion matrix.", file=sys.stderr)
            sys.exit(1)
        # All 4 confusion fields must be present as valid ints
        for field in ["TP", "TN", "FP", "FN"]:
            val = cm.get(field)
            if val is None or not isinstance(val, (int, float)) or (isinstance(val, bool)):
                print(f"ERROR: {label} confusion.{field} is missing or invalid (got {val!r}).", file=sys.stderr)
                sys.exit(1)
        # Core metrics must be present as valid floats
        for field in ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]:
            val = m.get(field)
            if val is None:
                print(f"ERROR: {label} metrics.{field} is missing (null/None).", file=sys.stderr)
                sys.exit(1)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                print(f"ERROR: {label} metrics.{field} is not a valid number (got {val!r}).", file=sys.stderr)
                sys.exit(1)

    # ── v5.11.5.5: Real Delta calculation from raw numeric values ──
    def _get_raw_metrics(raw_data):
        m = raw_data.get("metrics", {})
        cm = m.get("confusion", {})
        return {
            "TP": int(cm.get("TP", 0)), "TN": int(cm.get("TN", 0)),
            "FP": int(cm.get("FP", 0)), "FN": int(cm.get("FN", 0)),
            # Raw numeric values (0-1 range floats)
            "accuracy": float(m.get("accuracy", 0)),
            "precision": float(m.get("precision", 0)) if m.get("precision") is not None else None,
            "recall": float(m.get("recall", 0)) if m.get("recall") is not None else None,
            "f1": float(m.get("f1", 0)) if m.get("f1") is not None else None,
            "fpr": float(m.get("fpr", 0)) if m.get("fpr") is not None else None,
            "fnr": float(m.get("fnr", 0)) if m.get("fnr") is not None else None,
            "failure_count": int(m.get("failure_count", 0)),
            "success_count": int(m.get("success_count", 0)),
        }

    rule_raw_m = _get_raw_metrics(rp)
    hybrid_raw_m = _get_raw_metrics(lp)

    def _delta_pp(rule_val, hybrid_val):
        """Compute delta in percentage points: Hybrid - Rule.
        Returns formatted string like '+5.26 pp', '-2.00 pp', '0.00 pp'."""
        if rule_val is None or hybrid_val is None:
            return "—"
        delta = (hybrid_val - rule_val) * 100.0  # Convert to pp
        if abs(delta) < 0.005:
            return "0.00 pp"
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.2f} pp"

    def _delta_int(rule_val, hybrid_val):
        """Compute integer delta: Hybrid - Rule."""
        d = hybrid_val - rule_val
        if d == 0:
            return "0"
        sign = "+" if d > 0 else ""
        return f"{sign}{d}"

    # ── Build comparison report ──
    out_lines = [
        f"# LLM Comparison Report v{APP_VERSION}",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Run Identity",
        f"- **Rule Run ID**: {rm.get('run_id', '?')}",
        f"- **Hybrid Run ID**: {lm.get('run_id', '?')}",
        f"- **Split**: {rm.get('split', '?')}",
        f"- **Seed**: {rm.get('seed', '?')}",
        f"- **Samples**: {rm.get('total_evaluated', '?')}",
        f"- **Data SHA-256**: {rm.get('data_sha256', '?')[:16]}...",
        "",
        "## Model Info",
        f"- **Vendor**: {lm.get('llm_vendor', '?')}",
        f"- **Model**: {lm.get('llm_model', '?')}",
        f"- **Base Host**: {lm.get('llm_base_host', '?')}",
        f"- **Adapter**: {lm.get('llm_client_adapter', '?')}",
        f"- **API Type**: {lm.get('llm_api_type', '?')}",
        f"- **Prompt**: {lm.get('prompt_version', '?')} ({lm.get('prompt_hash', '?')[:16]}...)",
        f"- **Scorer**: v{RiskScorer.VERSION}",
        "",
        "## API Statistics (LLM Run)",
        f"- **Sample Requested**: {lm.get('llm_sample_requested_count', '?')}",
        f"- **HTTP Attempts**: {lm.get('llm_request_attempt_count', '?')}",
        f"- **HTTP Success**: {lm.get('llm_http_success_count', '?')}",
        f"- **Transport Retries**: {lm.get('llm_transport_retry_count', '?')}",
        f"- **Schema Retries**: {lm.get('llm_schema_retry_count', '?')}",
        f"- **Invalid Responses**: {lm.get('llm_invalid_response_count', '?')}",
        f"- **Usable Responses**: {lm.get('llm_usable_response_count', '?')}",
        f"- **Recovered**: {lm.get('llm_recovered_after_validation_count', '?')}",
        f"- **Terminal Validation Fail**: {lm.get('llm_terminal_validation_failure_count', '?')}",
        f"- **Terminal Transport Fail**: {lm.get('llm_terminal_transport_failure_count', '?')}",
        f"- **JSON Mode**: {'used' if lm.get('llm_json_mode_used') else 'not_used'}",
        f"- **JSON Mode Fallbacks**: {lm.get('llm_json_mode_fallback_count', 0)}",
        f"- **HTTP Errors**: {lm.get('llm_http_error_count', 0)}",
        f"- **P50 Latency**: {lm.get('llm_latency_p50_ms', '?')}ms",
        f"- **P95 Latency**: {lm.get('llm_latency_p95_ms', '?')}ms",
        "",
        "## Side-by-Side Metrics",
        f"| Metric | Rule-Only | DeepSeek Hybrid | Delta |",
        "|---|---|---|---|",
    ]

    metric_specs = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("FPR", "fpr"),
        ("FNR", "fnr"),
    ]
    for metric_name, key in metric_specs:
        rv = rule_raw_m[key]
        hv = hybrid_raw_m[key]
        rv_str = fmt_pct(rv) if rv is not None else "N/A"
        hv_str = fmt_pct(hv) if hv is not None else "N/A"
        delta_str = _delta_pp(rv, hv)
        out_lines.append(f"| {metric_name} | {rv_str} | {hv_str} | {delta_str} |")

    out_lines.extend([
        "",
        "## Confusion Matrix Comparison",
        f"### Rule-Only",
        f"TP={rule_raw_m['TP']} TN={rule_raw_m['TN']} FP={rule_raw_m['FP']} FN={rule_raw_m['FN']}",
        f"### DeepSeek Hybrid",
        f"TP={hybrid_raw_m['TP']} TN={hybrid_raw_m['TN']} FP={hybrid_raw_m['FP']} FN={hybrid_raw_m['FN']}",
        "",
        "## FP/FN Changes",
        f"| Direction | Rule-Only | Hybrid | Change |",
        "|---|---|---|---|",
        f"| FP | {rule_raw_m['FP']} | {hybrid_raw_m['FP']} | {_delta_int(rule_raw_m['FP'], hybrid_raw_m['FP'])} |",
        f"| FN | {rule_raw_m['FN']} | {hybrid_raw_m['FN']} | {_delta_int(rule_raw_m['FN'], hybrid_raw_m['FN'])} |",
        "",
        "## Capability Boundary",
        "- Data: 130 simulated samples (not production data)",
        "- OCR: not_run (models not downloaded)",
        "- Blind/Adversarial: not used for tuning",
        "- Main production strategy: rule_only (if hybrid ≤ rule-only)",
        "- DeepSeek: experimental auxiliary analysis",
        "",
        f"*Auto-generated v{APP_VERSION}*",
    ])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"LLM Comparison: {out}")

def _find_manifest(dir_path: Path) -> str:
    for f in sorted(dir_path.rglob("manifest_*.json"), reverse=True):
        return str(f)
    return ""

def _find_raw(dir_path: Path) -> str:
    for f in sorted(dir_path.rglob("raw_predictions_*.json"), reverse=True):
        return str(f)
    return ""

def _delta(a, b):
    d = b - a
    if d == 0: return "0"
    sign = "+" if d > 0 else ""
    return f"{sign}{d}"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # v5.11.5.4: comparison mode — generate llm_comparison.md
    if args.compare_rule or args.compare_hybrid or args.compare_out:
        if not (args.compare_rule and args.compare_hybrid and args.compare_out):
            print("ERROR: --compare-rule, --compare-hybrid, and --compare-out "
                  "must all be provided for comparison mode.", file=sys.stderr)
            sys.exit(1)
        generate_llm_comparison(args.compare_rule, args.compare_hybrid, args.compare_out)
        return

    if not args.test_data:
        print("ERROR: --test-data is required for evaluation mode.", file=sys.stderr)
        sys.exit(1)

    # v5.11.5.9: --sample-id validation
    if args.sample_ids and args.max_samples is not None:
        print("ERROR: --sample-id and --max-samples are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    data_path = str(Path(args.test_data).resolve())
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    # Build config
    cfg = EvaluationRunConfig(
        split=args.split, max_samples=args.max_samples, seed=args.seed,
        strategy=args.strategy, use_llm=not args.no_llm,
        test_data=data_path, output=args.output, data_sha256=data_hash,
        start_time=datetime.now(timezone.utc).isoformat(),
    )
    cfg.make_run_id()

    # Load and filter data
    evaluator = ModelEvaluator(cfg, allow_deterministic_fallback=args.allow_deterministic_fallback)
    samples_raw = evaluator.load_data(data_path)
    cfg.total_before_filter = len(samples_raw)

    if cfg.split != "all":
        split_samples = [s for s in samples_raw if s.get("split") == cfg.split]
        if not split_samples:
            print(f"ERROR: No samples with split='{cfg.split}'.")
            sys.exit(1)
        samples = split_samples
        print(f"Split '{cfg.split}': {len(samples)} samples")
    else:
        samples = samples_raw
    cfg.total_after_filter = len(samples)

    # v5.11.5.9: targeted evaluation by explicit sample IDs
    if args.sample_ids:
        cfg.selection_mode = "explicit_ids"
        cfg.requested_sample_ids = list(args.sample_ids)

        # Validate: no duplicates
        if len(cfg.requested_sample_ids) != len(set(cfg.requested_sample_ids)):
            print(f"ERROR: Duplicate sample IDs in --sample-id: {args.sample_ids}", file=sys.stderr)
            sys.exit(1)

        # Validate: all IDs exist and belong to the correct split
        id_to_sample = {s.get("id", ""): s for s in samples}
        missing = []
        wrong_split = []
        for sid in cfg.requested_sample_ids:
            if sid not in id_to_sample:
                missing.append(sid)
            elif id_to_sample[sid].get("split") != cfg.split:
                wrong_split.append(sid)

        if missing:
            print(f"ERROR: Sample ID(s) not found in data: {missing}", file=sys.stderr)
            sys.exit(1)
        if wrong_split:
            print(f"ERROR: Sample ID(s) not in split '{cfg.split}': {wrong_split}", file=sys.stderr)
            sys.exit(1)

        # Select samples in requested order
        samples = [id_to_sample[sid] for sid in cfg.requested_sample_ids]
        print(f"Explicit IDs: {len(samples)} samples: {[s.get('id') for s in samples]}")

    cfg.total_after_filter = len(samples)

    evaluator.run(samples)
    evaluator.compute_metrics()
    evaluator.print_summary()
    evaluator.generate_report(cfg.output)


if __name__ == "__main__":
    main()
