"""评测工具：对 evaluation_cases.yaml 运行全量评测，生成报告.

用法:
    python -m phishing_rule_detector.evaluation                          # 静默运行
    python -m phishing_rule_detector.evaluation --report                 # 文本报告
    python -m phishing_rule_detector.evaluation --json                   # JSON 到 stdout
    python -m phishing_rule_detector.evaluation                          \
      --dataset path/to/cases.yaml                                       \
      --json-out artifacts/metrics.json                                  \
      --report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from phishing_rule_detector.detector import detect

LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_BINARY_POSITIVE = {"medium", "high", "critical"}
_VALID_LABELS = {"normal", "phishing"}
_VALID_INPUT_TYPES = {"url", "html", "email", "sms", "text"}

_DEFAULT_DATASET = (
    Path(__file__).resolve().parent / "tests" / "fixtures" / "evaluation_cases.yaml"
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _configure_piped_utf8() -> None:
    """Use UTF-8 for redirected CLI output while preserving interactive consoles."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phishing_rule_detector.evaluation",
        description="对 evaluation_cases.yaml 运行全量评测",
    )
    p.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help="评测数据集 YAML 路径 (默认: tests/fixtures/evaluation_cases.yaml)",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="将完整评测指标写入指定 JSON 文件（自动创建父目录）",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="向 stdout 输出文本报告",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="向 stdout 输出 JSON",
    )
    return p


# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载评测案例."""
    p = Path(path) if path else _DEFAULT_DATASET
    if not p.exists():
        raise FileNotFoundError(f"数据集文件不存在: {p}")
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"数据集无法解析: {exc}") from exc
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("数据集缺少顶层 'cases' 字段")
    cases = data["cases"]
    _validate_cases(cases)
    return sorted(cases, key=lambda c: c.get("id", ""))


def _validate_cases(cases: Any) -> None:
    """Validate the evaluation schema before any metric is calculated."""
    if not isinstance(cases, list):
        raise ValueError("数据集 cases 必须是列表")

    seen_ids: set[str] = set()
    required_fields = ("id", "label", "input_type", "input_text")
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{prefix} 必须是对象")

        missing = [field for field in required_fields if field not in case]
        if missing:
            raise ValueError(f"{prefix} 缺少字段: {', '.join(missing)}")

        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{prefix}.id 必须是非空字符串")
        if case_id in seen_ids:
            raise ValueError(f"案例 id 重复: {case_id}")
        seen_ids.add(case_id)

        if case["label"] not in _VALID_LABELS:
            raise ValueError(
                f"{prefix}.label 必须是 normal 或 phishing，实际: {case['label']!r}"
            )
        if case["input_type"] not in _VALID_INPUT_TYPES:
            raise ValueError(
                f"{prefix}.input_type 无效: {case['input_type']!r}"
            )
        if not isinstance(case["input_text"], str) or not case["input_text"].strip():
            raise ValueError(f"{prefix}.input_text 必须是非空字符串")

        expected_level = case.get("expected_level")
        if expected_level is not None and expected_level not in LEVEL_ORDER:
            raise ValueError(
                f"{prefix}.expected_level 无效: {expected_level!r}"
            )

        allowed_levels = case.get("allowed_levels")
        if allowed_levels is not None:
            if not isinstance(allowed_levels, list) or not allowed_levels:
                raise ValueError(f"{prefix}.allowed_levels 必须是非空列表")
            invalid_levels = [level for level in allowed_levels if level not in LEVEL_ORDER]
            if invalid_levels:
                raise ValueError(
                    f"{prefix}.allowed_levels 包含无效等级: {invalid_levels}"
                )

        for field in ("required_rule_ids", "forbidden_rule_ids"):
            value = case.get(field, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{prefix}.{field} 必须是字符串列表")

        context = case.get("context", {})
        if not isinstance(context, dict):
            raise ValueError(f"{prefix}.context 必须是对象")
        base_url = case.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError(f"{prefix}.base_url 必须是字符串或 null")


def _display_dataset_path(path: str | Path | None) -> str:
    dataset_path = Path(path) if path else _DEFAULT_DATASET
    resolved = dataset_path.resolve()
    try:
        return resolved.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


# ──────────────────────────────────────────────────────────────
# 单例评测
# ──────────────────────────────────────────────────────────────


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    context = dict(case.get("context", {}))
    kwargs: dict[str, Any] = {
        "input_text": case["input_text"],
        "input_type": case["input_type"],
        "context": context,
    }
    if case.get("base_url"):
        kwargs["base_url"] = case["base_url"]
    return detect(**kwargs)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """评测单个案例 — 返回不含敏感原文的结果."""
    cid = case["id"]
    label = case.get("label", "")
    t0 = time.perf_counter()
    result = _run_case(case)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    passed = True
    failures: list[str] = []

    if not result.get("success"):
        return {
            "id": cid, "passed": False, "label": label,
            "level": "error", "score": 0, "evidence_ids": [],
            "failures": ["返回 success=False"],
            "duration_ms": duration_ms,
        }

    level = result["risk"]["level"]
    evidence_ids = sorted({e["rule_id"] for e in result["evidence"]})
    score = result["risk"]["raw_score"]

    # ── 等级断言 ──
    expected_level = case.get("expected_level", "")
    allowed_levels = case.get("allowed_levels", [])
    if expected_level:
        if LEVEL_ORDER.get(level, -1) < LEVEL_ORDER.get(expected_level, -1):
            passed = False
            failures.append(f"等级不足：期望>={expected_level}，实际{level} (score={score})")
    elif allowed_levels:
        if level not in allowed_levels:
            passed = False
            failures.append(f"等级不在允许范围：允许{allowed_levels}，实际{level}")

    # ── 必现规则 ──
    for rid in case.get("required_rule_ids", []):
        if rid not in evidence_ids:
            passed = False
            failures.append(f"缺少必现规则 {rid}")

    # ── 禁止规则 ──
    for rid in case.get("forbidden_rule_ids", []):
        if rid in evidence_ids:
            passed = False
            failures.append(f"出现禁止规则 {rid}")

    # ── normal 样本不得高于 low ──
    if label == "normal" and level != "low":
        passed = False
        failures.append(f"normal 样本等级过高: {level}")

    # ── phishing 样本必须有证据 ──
    if label == "phishing" and len(evidence_ids) == 0:
        passed = False
        failures.append("钓鱼样本无证据")

    return {
        "id": cid,
        "passed": passed,
        "label": label,
        "level": level,
        "score": score,
        "evidence_ids": evidence_ids,
        "failures": failures,
        "duration_ms": duration_ms,
    }


# ──────────────────────────────────────────────────────────────
# 批量评测 & 指标
# ──────────────────────────────────────────────────────────────


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """评测批量案例，返回二分类指标与独立的契约通过率。"""
    details = [evaluate_case(c) for c in cases]
    result = compute_metrics(details)
    result["details"] = details
    return result


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 6) if b else 0.0


def compute_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    """从案例详情计算完整的二分类和契约指标."""
    total = len(details)
    phishing_count = sum(1 for d in details if d["label"] == "phishing")
    normal_count = sum(1 for d in details if d["label"] == "normal")

    # ── 混淆矩阵 ──
    tp = fp = tn = fn = 0
    fn_ids: list[str] = []
    fp_ids: list[str] = []
    for d in details:
        pred_positive = d["level"] in _BINARY_POSITIVE
        if d["label"] == "phishing":
            if pred_positive:
                tp += 1
            else:
                fn += 1
                fn_ids.append(d["id"])
        else:  # normal
            if pred_positive:
                fp += 1
                fp_ids.append(d["id"])
            else:
                tn += 1

    confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)

    # ── 等级分布 ──
    level_dist: dict[str, int] = {}
    for d in details:
        lv = d.get("level", "error")
        level_dist[lv] = level_dist.get(lv, 0) + 1

    # ── 契约验证 ──
    contract_passed = sum(1 for d in details if d.get("passed", False))
    contract_failed = total - contract_passed
    contract_pass_rate = _safe_div(contract_passed, total)

    # ── 规则命中 ──
    rule_hits: dict[str, int] = {}
    for d in details:
        for rid in d.get("evidence_ids", []):
            rule_hits[rid] = rule_hits.get(rid, 0) + 1

    # 从检测结果收集 group 信息需要重新运行 —— 这里简单按规则名前缀推断
    # 实际上我们需要真实 group 数据。重新做一次轻量运行：
    # 但为避免重复运行，我们从已缓存的 evidence 中获取。
    # 由于 evaluate_case 只存了 rule_id，我们需重新运行一次完整检测来收集 group。
    # 更简洁的方式: 额外保存 group 信息。

    # ── 延迟 ──
    latencies = [d.get("duration_ms", 0) for d in details]
    latencies_sorted = sorted(latencies)
    latency_stats = {
        "total": sum(latencies),
        "average": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p50": _latency_percentile(latencies_sorted, 50),
        "p95": _latency_percentile(latencies_sorted, 95),
        "max": max(latencies) if latencies else 0,
    }

    # ── 按输入类型 ──
    # (此处需要原始 case 信息，由上层函数注入)

    # ── 质量门禁 ──
    gates = {
        "precision_min": 0.95,
        "recall_min": 0.75,
        "f1_min": 0.85,
        "fpr_max": 0.05,
        "contract_pass_rate_min": 1.0,
        "passed": (
            precision >= 0.95
            and recall >= 0.75
            and f1 >= 0.85
            and fpr <= 0.05
            and contract_pass_rate >= 1.0
        ),
    }

    return {
        "total": total,
        "phishing_count": phishing_count,
        "normal_count": normal_count,
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "level_distribution": level_dist,
        "contract_validation": {
            "total": total,
            "passed": contract_passed,
            "failed": contract_failed,
            "pass_rate": contract_pass_rate,
        },
        "false_negative_ids": fn_ids,
        "false_positive_ids": fp_ids,
        "latency_ms": latency_stats,
        "quality_gates": gates,
    }


def _latency_percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 2)


# ──────────────────────────────────────────────────────────────
# 顶层入口
# ──────────────────────────────────────────────────────────────


def run_full_evaluation(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """运行完整评测，返回所有指标（包含 by_input_type、rule_hit_counts 等）。

    这是推荐的新顶层入口。
    """
    cases = load_cases(path)

    # 运行评测
    details = [evaluate_case(c) for c in cases]

    # 基础指标
    base = compute_metrics(details)
    base["details"] = details

    # 规则和组命中（需要真实 group — 重新跑一次收集 group，用已缓存结果加速）
    rule_hits: dict[str, int] = {}
    group_hits: dict[str, int] = {}
    # 我们重新运行检测收集 group 信息
    from phishing_rule_detector.detector import detect as _detect

    for case in cases:
        ctx = dict(case.get("context", {}))
        kw: dict[str, Any] = {
            "input_text": case["input_text"],
            "input_type": case["input_type"],
            "context": ctx,
        }
        if case.get("base_url"):
            kw["base_url"] = case["base_url"]
        r = _detect(**kw)
        if r.get("success"):
            for e in r["evidence"]:
                rid = e["rule_id"]
                rule_hits[rid] = rule_hits.get(rid, 0) + 1
                group_hits[e["group"]] = group_hits.get(e["group"], 0) + 1

    base["rule_hit_counts"] = dict(
        sorted(rule_hits.items(), key=lambda kv: kv[1], reverse=True)
    )
    base["group_hit_counts"] = dict(
        sorted(group_hits.items(), key=lambda kv: kv[1], reverse=True)
    )

    # 按输入类型
    by_input: dict[str, dict[str, int]] = {}
    for case, detail in zip(cases, details):
        itype = case.get("input_type", "unknown")
        if itype not in by_input:
            by_input[itype] = {"total": 0, "phishing": 0, "normal": 0}
        by_input[itype]["total"] += 1
        if detail["label"] == "phishing":
            by_input[itype]["phishing"] += 1
        else:
            by_input[itype]["normal"] += 1
    base["by_input_type"] = by_input

    # 附加元信息
    base["rule_version"] = "3.0.0"
    base["dataset"] = _display_dataset_path(path)
    base["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    return base


def run_evaluation(path: str | Path | None = None) -> dict[str, Any]:
    """程序化兼容入口，返回与 CLI 相同的完整二分类指标."""
    return run_full_evaluation(path)


# ──────────────────────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────────────────────


def generate_report(result: dict[str, Any]) -> str:
    """生成可读文本报告（新格式）。"""
    lines = []
    sep = "=" * 64
    lines.append(sep)
    lines.append("  钓鱼规则引擎 — 评测报告")
    lines.append(sep)

    # 基础统计
    lines.append(f"  总案例数:         {result.get('total', 0)}")
    lines.append(f"  phishing:         {result.get('phishing_count', '?')}")
    lines.append(f"  normal:           {result.get('normal_count', '?')}")
    lines.append("")

    # 混淆矩阵
    cm = result.get("confusion_matrix", {})
    lines.append("  混淆矩阵 (二分类，medium+ 为正类):")
    lines.append(f"    TP={cm.get('tp', 0)}  FP={cm.get('fp', 0)}")
    lines.append(f"    FN={cm.get('fn', 0)}  TN={cm.get('tn', 0)}")
    lines.append("")

    # 核心指标
    lines.append("  二分类指标:")
    lines.append(f"    Accuracy:  {result.get('accuracy', 0):.4f}")
    lines.append(f"    Precision: {result.get('precision', 0):.4f}")
    lines.append(f"    Recall:    {result.get('recall', 0):.4f}")
    lines.append(f"    F1:        {result.get('f1', 0):.6f}")
    lines.append(f"    FPR:       {result.get('false_positive_rate', 0):.4f}")
    lines.append("")

    # 契约验证
    cv = result.get("contract_validation", {})
    lines.append("  案例契约验证:")
    lines.append(
        f"    {cv.get('passed', 0)}/{cv.get('total', 0)} 通过 "
        f"(pass_rate={cv.get('pass_rate', 0):.4f})"
    )
    lines.append("")

    # 等级分布
    ld = result.get("level_distribution", {})
    lines.append("  等级分布:")
    for lv in ["low", "medium", "high", "critical"]:
        if lv in ld:
            lines.append(f"    {lv}: {ld[lv]}")
    lines.append("")

    # 质量门禁
    qg = result.get("quality_gates", {})
    lines.append("  质量门禁:")
    for k in ["precision_min", "recall_min", "f1_min", "fpr_max", "contract_pass_rate_min"]:
        lines.append(f"    {k}: {qg.get(k, '?')}")
    lines.append(f"    passed: {qg.get('passed', False)}")
    lines.append("")

    # 延迟
    lat = result.get("latency_ms", {})
    lines.append("  延迟 (ms):")
    lines.append(f"    total={lat.get('total', 0)}  avg={lat.get('average', 0)}")
    lines.append(f"    p50={lat.get('p50', 0)}  p95={lat.get('p95', 0)}  max={lat.get('max', 0)}")

    lines.append(sep)
    return "\n".join(lines)


def export_json(result: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 移除 details 以保护隐私（不输出敏感输入文本）
    clean = {k: v for k, v in result.items() if k != "details"}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2, default=str)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    _configure_piped_utf8()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse 在未知参数时自动 exit(2)
        return e.code if isinstance(e.code, int) else 2

    # ── 加载数据集 ──
    try:
        result = run_full_evaluation(args.dataset)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"评测运行失败: {e}", file=sys.stderr)
        return 1

    # ── 输出 ──
    if args.json_out:
        try:
            export_json(result, args.json_out)
        except Exception as e:
            print(f"写入 JSON 失败: {e}", file=sys.stderr)
            return 1

    if args.json_stdout:
        clean = {k: v for k, v in result.items() if k != "details"}
        print(json.dumps(clean, ensure_ascii=False, indent=2, default=str))
    elif args.report:
        print(generate_report(result))
    else:
        # 静默模式
        cv = result.get("contract_validation", {})
        qg = result.get("quality_gates", {})
        if cv.get("failed", 0) > 0 or not qg.get("passed", True):
            lines = []
            lines.append(
                f"契约: {cv.get('passed', 0)}/{cv.get('total', 0)}"
            )
            cm = result.get("confusion_matrix", {})
            lines.append(
                f"混淆矩阵: TP={cm.get('tp', 0)} FP={cm.get('fp', 0)} "
                f"TN={cm.get('tn', 0)} FN={cm.get('fn', 0)}"
            )
            lines.append(
                f"Accuracy={result.get('accuracy', 0):.4f} "
                f"Precision={result.get('precision', 0):.4f} "
                f"Recall={result.get('recall', 0):.4f} "
                f"F1={result.get('f1', 0):.6f}"
            )
            lines.append(f"Quality gates passed: {qg.get('passed', False)}")
            print("\n".join(lines), file=sys.stderr)

    # ── 退出码 ──
    cv = result.get("contract_validation", {})
    qg = result.get("quality_gates", {})
    if cv.get("failed", 0) > 0:
        return 1
    if not qg.get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
