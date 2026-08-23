"""风险评分引擎：有效分计算、规则抑制、同组衰减、组上限、等级门控.

算法流程（遵循设计文档 §9-10）：
  1. effective_score = round(base_score × confidence × context_factor)
  2. 包含关系抑制（具体规则抑制泛化规则）
  3. 同组衰减（100% / 50% / 25%）
  4. 组分值上限
  5. raw_score = sum of all effective_scores, capped at 100
  6. 等级上限：only low → max low; only medium → max medium
  7. 强证据下限：qualified medium → at least medium; qualified high → at least high
  8. 两组独立 qualified high → critical_lock
  9. score = clamp(raw_score, level_min, level_max)
"""
from __future__ import annotations

from collections import defaultdict
from typing import cast

from phishing_rule_detector.config_loader import load_config
from phishing_rule_detector.models import EvidenceItem, RISK_LEVELS, RiskResult

LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def compute_effective_score(
    base_score: int, confidence: float, context_factor: float = 1.0
) -> int:
    """effective_score = round(base_score × confidence × context_factor)."""
    return round(base_score * confidence * context_factor)


def apply_suppression(
    evidence: list[EvidenceItem],
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """执行规则包含关系抑制，以 subject_id 限定作用域.

    规则：
      - 同一 subject_id 内：具体规则抑制泛化规则（默认行为）。
      - 两者都没有 subject_id：视为旧版 legacy 全局作用域（向后兼容）。
      - 不同 subject_id：不得互相抑制。
      - 一个有 subject_id、另一个没有：不得互相抑制。
      - 若配置中规则显式声明 scope: global，则突破 subject_id 限制，全局抑制。

    Returns:
        (active_evidence, suppressed_evidence)
    """
    config = load_config()
    suppression_map = config.rules.rule_suppression

    # 构建 (rule_id, subject_id) → bool 的抑制判定集合
    # 先找出所有被抑制的 (rule_id, subject_id) 组合
    suppressed_keys: set[tuple[str, str | None]] = set()

    for item in evidence:
        if item.rule_id in suppression_map:
            entry = suppression_map[item.rule_id]
            is_global = entry.scope == "global"

            for target_rule_id in entry.suppresses:
                # 找到所有 target_rule_id 的匹配 evidence
                for target_item in evidence:
                    if target_item.rule_id != target_rule_id:
                        continue

                    # 判定是否在同一作用域
                    if is_global:
                        # scope: global → 无条件抑制
                        suppressed_keys.add((target_item.rule_id, target_item.subject_id))
                    elif item.subject_id is None and target_item.subject_id is None:
                        # 两者都没有 subject_id → legacy 全局作用域
                        suppressed_keys.add((target_item.rule_id, None))
                    elif item.subject_id is not None and item.subject_id == target_item.subject_id:
                        # 同一 subject_id → 抑制
                        suppressed_keys.add((target_item.rule_id, target_item.subject_id))
                    # 否则：不同作用域，不抑制

    active = []
    suppressed = []
    for item in evidence:
        key = (item.rule_id, item.subject_id)
        if key in suppressed_keys:
            suppressed.append(item)
        else:
            active.append(item)

    return active, suppressed


def apply_group_attenuation(
    evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    """同组衰减：第一条 100%，第二条 50%，第三条及以后 25%.

    组内按 effective_score 降序排列后衰减.
    """
    config = load_config()
    att = config.scoring.group_attenuation  # [1.0, 0.5, 0.25]

    # 按证据组分组
    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence:
        groups[item.group].append(item)

    result = []
    for _group, items in groups.items():
        # 组内按有效分降序
        items.sort(key=lambda e: e.effective_score, reverse=True)
        for i, item in enumerate(items):
            factor = att[i] if i < len(att) else att[-1]
            new_score = round(item.effective_score * factor)
            # 返回新 EvidenceItem 但保留原始 effective_score 做衰减基准
            updated = item.model_copy(update={"effective_score": new_score})
            result.append(updated)

    return result


def apply_group_caps(
    evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    """应用组分值上限.

    每组总分不超过配置的 group_score_caps.
    """
    config = load_config()
    caps = config.scoring.group_score_caps

    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence:
        groups[item.group].append(item)

    # 只对未排序的做简单处理：超出的有效分按比例缩减
    result = []
    for group, items in groups.items():
        cap = caps.get(group)
        if cap is None:
            result.extend(items)
            continue

        total = sum(e.effective_score for e in items)
        if total <= cap:
            result.extend(items)
        else:
            # 按比例缩减到 cap 内
            scale = cap / total
            acc = 0
            sorted_items = sorted(
                items, key=lambda e: e.effective_score, reverse=True
            )
            for item in sorted_items:
                scaled = round(item.effective_score * scale)
                acc += scaled
                result.append(
                    item.model_copy(update={"effective_score": scaled})
                )
            # 如果因舍入导致溢出，从最后一条扣除
            if acc > cap:
                overflow = acc - cap
                last = result[-1]
                result[-1] = last.model_copy(
                    update={
                        "effective_score": max(0, last.effective_score - overflow)
                    }
                )

    return result


def compute_raw_score(evidence: list[EvidenceItem]) -> int:
    """计算所有证据有效分之和，上限 100."""
    config = load_config()
    total = sum(e.effective_score for e in evidence)
    return min(total, config.scoring.max_raw_score)


def _get_decision_confidence(item: EvidenceItem) -> float:
    """decision_confidence = confidence × context_factor."""
    return item.confidence * item.context_factor


def determine_level(
    evidence: list[EvidenceItem], raw_score: int
) -> RiskResult:
    """根据证据和 raw_score 确定最终风险等级.

    综合等级上限、强证据下限和 critical_lock 机制.

    §10.2 等级上限（审查 Fix 1）：
      - 所有 evidence severity ≤ low → max low
      - 无 qualified high → max medium（即使有 unqualified high severity）
      - 无 qualified medium → max low
    §10.3 强证据下限：
      - qualified medium → floor medium
      - qualified high (≥1组) → floor high
      - qualified high (≥2组, 独立) → critical_lock
    §10.4 confidence：
      - critical_lock → per-group max dc, then min of group maxes
      - floor=high → per-group max dc, then max of group maxes
      - floor=medium → max dc among qualifying medium evidence
      - floor=low (no qualifying) → max dc of all active evidence
    """
    config = load_config()
    thresholds = config.scoring.qualifying_thresholds
    level_ranges = config.scoring.level_ranges
    level_thresholds = config.scoring.level_thresholds

    # ── 0. 预扫描：收集合格证据信息 ──
    qualified_high_groups: set[str] = set()
    qualified_medium_evidence: list[EvidenceItem] = []
    all_decision_confidences: list[float] = []

    max_severity = "low"
    has_any_high_severity = False
    has_any_medium_severity = False

    for item in evidence:
        dc = _get_decision_confidence(item)
        all_decision_confidences.append(dc)

        sev = item.severity
        if sev == "high":
            has_any_high_severity = True
            if dc >= thresholds.high:
                qualified_high_groups.add(item.group)
        if sev == "medium":
            has_any_medium_severity = True
            if dc >= thresholds.medium:
                qualified_medium_evidence.append(item)
        if LEVEL_ORDER[sev] > LEVEL_ORDER[max_severity]:
            max_severity = sev

    # ── 1. 初始候选等级（基于 raw_score）──
    if raw_score <= level_thresholds.low_max:
        candidate_level = "low"
    elif raw_score <= level_thresholds.medium_max:
        candidate_level = "medium"
    elif raw_score <= level_thresholds.high_max:
        candidate_level = "high"
    else:
        candidate_level = "critical"

    # ── 2. 等级上限约束（审查 Fix 1）──
    # 只有 low → 最高 low
    if max_severity == "low":
        candidate_level = "low"
    # 有 medium 但没有 high → 最高 medium
    elif has_any_medium_severity and not has_any_high_severity:
        candidate_level = min(
            candidate_level, "medium", key=lambda x: LEVEL_ORDER[x]
        )
    # 有 high severity 但没有 qualified high → 最高 medium
    elif has_any_high_severity and len(qualified_high_groups) == 0:
        candidate_level = min(
            candidate_level, "medium", key=lambda x: LEVEL_ORDER[x]
        )

    # ── 3. 强证据等级下限 ──
    level_floor: RISK_LEVELS = "low"

    # 合格 medium → 至少 medium
    if len(qualified_medium_evidence) > 0:
        if LEVEL_ORDER["medium"] > LEVEL_ORDER[level_floor]:
            level_floor = "medium"

    # 合格 high → 至少 high
    if len(qualified_high_groups) >= 1:
        if LEVEL_ORDER["high"] > LEVEL_ORDER[level_floor]:
            level_floor = "high"

    # 两个独立高危组 → critical_lock
    critical_lock = False
    if len(qualified_high_groups) >= 2:
        critical_lock = True
        if LEVEL_ORDER["critical"] > LEVEL_ORDER[level_floor]:
            level_floor = "critical"

    # ── 4. 最终等级 ──
    final_level = cast(
        RISK_LEVELS,
        max(candidate_level, level_floor, key=lambda x: LEVEL_ORDER[x]),
    )

    # ── 5. 展示分 clamp ──
    level_range = getattr(level_ranges, final_level)
    configured_floor = getattr(
        config.scoring.level_floor_scores,
        final_level,
    )
    score = max(configured_floor, min(raw_score, level_range[1]))

    # ── 6. confidence（审查 Fix 8 + Fix 3）──
    if critical_lock:
        # critical: 每组取 max dc，然后取各组 max 的最小值
        # 避免同组弱证据拉低整体置信度
        high_group_max: dict[str, float] = {}
        for e in evidence:
            dc = _get_decision_confidence(e)
            if (
                e.severity == "high"
                and dc >= thresholds.high
                and e.group in qualified_high_groups
            ):
                if e.group not in high_group_max or dc > high_group_max[e.group]:
                    high_group_max[e.group] = dc
        group_maxes = list(high_group_max.values())
        risk_confidence = min(group_maxes) if group_maxes else 0.0
    elif level_floor == "high":
        # high floor: 每组取 max dc，然后取各组 max 的最大值
        # 使用与 critical_lock 一致的 per-group 聚合，避免同组多条证据影响
        per_group_max: dict[str, float] = {}
        for e in evidence:
            dc = _get_decision_confidence(e)
            if (
                e.severity == "high"
                and dc >= thresholds.high
                and e.group in qualified_high_groups
            ):
                if e.group not in per_group_max or dc > per_group_max[e.group]:
                    per_group_max[e.group] = dc
        group_maxes = list(per_group_max.values())
        risk_confidence = max(group_maxes) if group_maxes else 0.0
    elif level_floor == "medium":
        # medium floor: 取 qualified medium 中最高的 decision_confidence
        risk_confidence = (
            max(_get_decision_confidence(e) for e in qualified_medium_evidence)
            if qualified_medium_evidence
            else 0.0
        )
    else:
        # no qualifying evidence: 取所有有效证据中最高的 decision_confidence
        risk_confidence = (
            max(all_decision_confidences) if all_decision_confidences else 0.0
        )

    return RiskResult(
        score=score,
        raw_score=raw_score,
        level=final_level,
        level_floor=level_floor,
        confidence=round(risk_confidence, 4),
        critical_lock=critical_lock,
    )


def score_pipeline(
    evidence: list[EvidenceItem],
) -> tuple[RiskResult, list[EvidenceItem], list[EvidenceItem]]:
    """执行完整的评分管线.

    顺序：抑制 → 衰减 → 组上限 → raw_score → 等级判定.

    Returns:
        (risk_result, active_evidence, suppressed_evidence)
        active_evidence 是评分后的证据（已应用衰减和组上限）.
    """
    # 1. 包含关系抑制
    active, suppressed = apply_suppression(evidence)

    # 2. 同组衰减
    attenuated = apply_group_attenuation(active)

    # 3. 组分值上限
    capped = apply_group_caps(attenuated)

    # 4. raw_score
    raw_score = compute_raw_score(capped)

    # 5. 等级判定（基于原始 active 证据判断门控，使用衰减后的分值）
    result = determine_level(active, raw_score)

    return result, capped, suppressed
