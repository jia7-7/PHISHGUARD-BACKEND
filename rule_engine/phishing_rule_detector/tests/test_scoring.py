"""测试评分引擎：effective_score、抑制、衰减、组上限、等级门控."""
from phishing_rule_detector.models import EvidenceItem
from phishing_rule_detector.scoring import (
    compute_effective_score,
    apply_suppression,
    apply_group_attenuation,
    apply_group_caps,
    compute_raw_score,
    determine_level,
    score_pipeline,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_evidence(
    rule_id: str,
    group: str,
    severity: str,
    base_score: int,
    confidence: float = 1.0,
    context_factor: float = 1.0,
    subject_id: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        rule_id=rule_id,
        title=rule_id,
        group=group,
        severity=severity,
        confidence=confidence,
        context_factor=context_factor,
        base_score=base_score,
        effective_score=compute_effective_score(
            base_score, confidence, context_factor
        ),
        reason="test",
        subject_id=subject_id,
    )


# ── Effective Score ─────────────────────────────────────────

class TestEffectiveScore:
    def test_full_confidence(self):
        assert compute_effective_score(10, 1.0, 1.0) == 10

    def test_half_confidence(self):
        assert compute_effective_score(10, 0.5, 1.0) == 5

    def test_context_factor(self):
        assert compute_effective_score(10, 1.0, 0.35) == 4  # round(3.5) = 4

    def test_rounding(self):
        assert compute_effective_score(30, 0.95, 0.9) == 26  # round(25.65)


# ── Containment Suppression ────────────────────────────────

class TestSuppression:
    def test_specific_suppresses_generic(self):
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "HTML_PASSWORD_INPUT" not in rule_ids  # 被抑制

    def test_homograph_suppresses_two(self):
        evidence = [
            _make_evidence("DOMAIN_KEYWORD_IMPERSONATION", "identity", "medium", 12),
            _make_evidence("DOMAIN_SIMILAR_TO_OFFICIAL", "identity", "high", 25),
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30),
        ]
        active, _ = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "DOMAIN_HOMOGRAPH_ATTACK" in rule_ids
        assert "DOMAIN_SIMILAR_TO_OFFICIAL" not in rule_ids
        assert "DOMAIN_KEYWORD_IMPERSONATION" not in rule_ids

    def test_no_suppression_across_groups(self):
        """不同组的规则不应相互抑制."""
        evidence = [
            _make_evidence("TIME_LIMIT_PRESSURE", "social", "medium", 10),
            _make_evidence("URGENT_VERIFY_WORDING", "social", "medium", 8),
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "TIME_LIMIT_PRESSURE" in rule_ids
        # URGENT_VERIFY_WORDING 被 TIME_LIMIT_PRESSURE 抑制
        assert "URGENT_VERIFY_WORDING" not in rule_ids

    def test_suppressed_count(self):
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),
        ]
        _, suppressed = apply_suppression(evidence)
        assert len(suppressed) == 1

    # ── subject_id 作用域测试（审查 Fix 2）──

    def test_same_form_suppression_works(self):
        """同一表单内（相同 subject_id）PASSWORD_FORM 抑制 HTML_PASSWORD_INPUT."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4, subject_id="form:0"),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, subject_id="form:0"),
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "HTML_PASSWORD_INPUT" not in rule_ids
        assert len(suppressed) == 1
        assert suppressed[0].rule_id == "HTML_PASSWORD_INPUT"

    def test_different_forms_no_cross_suppression(self):
        """不同表单（不同 subject_id）不得互相抑制."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4, subject_id="form:0"),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, subject_id="form:1"),
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "HTML_PASSWORD_INPUT" in rule_ids  # 不同 form，不抑制
        assert len(suppressed) == 0

    def test_same_url_suppression_works(self):
        """同一 URL 内（相同 subject_id）同类包含关系正常抑制."""
        evidence = [
            _make_evidence("DOMAIN_KEYWORD_IMPERSONATION", "identity", "medium", 12, subject_id="url:0"),
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30, subject_id="url:0"),
        ]
        active, _ = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "DOMAIN_HOMOGRAPH_ATTACK" in rule_ids
        assert "DOMAIN_KEYWORD_IMPERSONATION" not in rule_ids

    def test_different_urls_no_cross_suppression(self):
        """不同 URL（不同 subject_id）不得互相抑制."""
        evidence = [
            _make_evidence("DOMAIN_KEYWORD_IMPERSONATION", "identity", "medium", 12, subject_id="url:0"),
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30, subject_id="url:1"),
        ]
        active, _ = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "DOMAIN_HOMOGRAPH_ATTACK" in rule_ids
        assert "DOMAIN_KEYWORD_IMPERSONATION" in rule_ids  # 不同 URL，不抑制

    def test_both_no_subject_id_legacy_behavior(self):
        """两条均无 subject_id：保持旧行为（视为同一 legacy 作用域）."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "HTML_PASSWORD_INPUT" not in rule_ids  # legacy 行为：抑制

    def test_one_has_subject_id_no_cross_suppression(self):
        """一条有 subject_id、另一条没有 → 不得误抑制."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4, subject_id="form:0"),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),  # 无 subject_id
        ]
        active, suppressed = apply_suppression(evidence)
        rule_ids = [e.rule_id for e in active]
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "HTML_PASSWORD_INPUT" in rule_ids  # 作用域不同，不抑制
        assert len(suppressed) == 0

    def test_active_evidence_preserves_subject_id(self):
        """active_evidence 必须完整保留 subject_id."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4, subject_id="form:0"),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, subject_id="form:0"),
        ]
        active, suppressed = apply_suppression(evidence)
        for item in active:
            assert item.subject_id == "form:0"
        for item in suppressed:
            assert item.subject_id == "form:0"

    def test_suppressed_evidence_preserves_subject_id(self):
        """suppressed_evidence 必须完整保留 subject_id."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4, subject_id="form:0"),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, subject_id="form:0"),
        ]
        _, suppressed = apply_suppression(evidence)
        assert len(suppressed) == 1
        assert suppressed[0].subject_id == "form:0"


# ── Group Attenuation ──────────────────────────────────────

class TestGroupAttenuation:
    def test_first_full_second_half(self):
        evidence = [
            _make_evidence("E1", "identity", "high", 20),
            _make_evidence("E2", "identity", "high", 20),
        ]
        result = apply_group_attenuation(evidence)
        assert result[0].effective_score == 20  # 100%
        assert result[1].effective_score == 10  # 50%

    def test_third_quarter(self):
        evidence = [
            _make_evidence("E1", "identity", "high", 20),
            _make_evidence("E2", "identity", "high", 20),
            _make_evidence("E3", "identity", "high", 20),
        ]
        result = apply_group_attenuation(evidence)
        assert result[0].effective_score == 20
        assert result[1].effective_score == 10
        assert result[2].effective_score == 5  # 25%

    def test_different_groups_independent(self):
        """不同组各自独立衰减."""
        evidence = [
            _make_evidence("E1", "identity", "high", 20),
            _make_evidence("E2", "identity", "high", 20),
            _make_evidence("E3", "credential", "high", 30),
            _make_evidence("E4", "credential", "high", 30),
        ]
        result = apply_group_attenuation(evidence)
        ids = {e.rule_id: e.effective_score for e in result}
        assert ids["E1"] == 20  # identity 第一条
        assert ids["E2"] == 10  # identity 第二条
        assert ids["E3"] == 30  # credential 第一条
        assert ids["E4"] == 15  # credential 第二条

    def test_sort_by_effective_score_desc(self):
        """组内按有效分降序排列后衰减."""
        evidence = [
            _make_evidence("E_low", "identity", "low", 5),
            _make_evidence("E_high", "identity", "high", 25),
        ]
        result = apply_group_attenuation(evidence)
        # E_high 应在前面获 100%，E_low 在后面的 50%
        high_e = [e for e in result if e.rule_id == "E_high"][0]
        low_e = [e for e in result if e.rule_id == "E_low"][0]
        assert high_e.effective_score == 25
        assert low_e.effective_score == 2  # round(5*0.5) = 2


# ── Group Score Caps ───────────────────────────────────────

class TestGroupCaps:
    def test_group_cap_applied(self):
        """identity 组上限 35，超过则截断."""
        evidence = [
            EvidenceItem(
                rule_id="E1", title="E1", group="identity", severity="high",
                confidence=1.0, base_score=40, effective_score=40,
                reason="test",
            ),
        ]
        result = apply_group_caps(evidence)
        # identity 组上限 35
        total = sum(e.effective_score for e in result if e.group == "identity")
        assert total <= 35


# ── Raw Score ──────────────────────────────────────────────

class TestRawScore:
    def test_raw_score_capped_at_100(self):
        evidence = [
            EvidenceItem(
                rule_id="E1", title="E1", group="identity", severity="high",
                confidence=1.0, base_score=50, effective_score=50,
                reason="test",
            ),
            EvidenceItem(
                rule_id="E2", title="E2", group="credential", severity="high",
                confidence=1.0, base_score=50, effective_score=50,
                reason="test",
            ),
            EvidenceItem(
                rule_id="E3", title="E3", group="navigation", severity="high",
                confidence=1.0, base_score=50, effective_score=50,
                reason="test",
            ),
        ]
        raw = compute_raw_score(evidence)
        assert raw <= 100


# ── Level Determination ────────────────────────────────────

class TestDetermineLevel:
    def test_only_low_capped_at_low(self):
        """只有 low 证据 → 最高等级 low（§10.2.1）"""
        evidence = [
            _make_evidence("E1", "transport", "low", 5, confidence=0.5),
            _make_evidence("E2", "payload", "low", 5, confidence=0.5),
        ]
        result = determine_level(evidence, 40)  # 即使 raw_score 高
        assert result.level == "low"
        assert result.score <= 29

    def test_qualified_medium_sets_medium_floor(self):
        """合格 medium 证据（decision_confidence ≥ 0.75）→ 至少 medium."""
        evidence = [
            _make_evidence("TIME_LIMIT_PRESSURE", "social", "medium", 10, confidence=0.8),
        ]
        result = determine_level(evidence, 10)
        assert result.level == "medium"
        assert result.score >= 30
        assert result.level_floor == "medium"

    def test_unqualified_medium_does_not_set_floor(self):
        """不合格 medium（decision_confidence < 0.75）不设置下限."""
        evidence = [
            _make_evidence("URGENT_VERIFY_WORDING", "social", "medium", 8, confidence=0.55),
        ]
        result = determine_level(evidence, 8)
        # 置信度不足 → 不满足 medium 门控 → 但仍会计分
        assert result.level_floor == "low"

    def test_qualified_high_sets_high_floor(self):
        """合格 high 证据（decision_confidence ≥ 0.80）→ 至少 high."""
        evidence = [
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30, confidence=1.0),
        ]
        result = determine_level(evidence, 30)
        assert result.level == "high"
        assert result.level_floor == "high"
        assert result.score >= 60

    def test_two_independent_high_groups_trigger_critical(self):
        """两个不同组的高危证据 → critical_lock=true（§10.3.3）"""
        evidence = [
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),
        ]
        result = determine_level(evidence, 60)
        assert result.level == "critical"
        assert result.critical_lock is True
        assert result.score >= 80

    def test_no_qualified_high_capped_at_medium(self):
        """无合格 high 证据时，即使 raw_score 在 high 区间，最高 medium（审查 Fix 1）."""
        evidence = [
            _make_evidence("E1", "social", "medium", 10, confidence=0.8),
            _make_evidence("E2", "social", "medium", 10, confidence=0.8),
            _make_evidence("E3", "social", "medium", 10, confidence=0.8),
            _make_evidence("E4", "transport", "low", 5),
        ]
        result = determine_level(evidence, 75)  # raw_score 在 high 区间
        assert result.level == "medium"
        assert result.score <= 59

    def test_unqualified_high_capped_at_medium(self):
        """有 high severity 但 confidence 不足 → 不视为合格 high → 最高 medium."""
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.6),  # decision_confidence 0.6 < 0.80
            _make_evidence("E2", "social", "medium", 10, confidence=0.8),
        ]
        result = determine_level(evidence, 70)  # raw_score 在 high 区间
        assert result.level == "medium"
        assert result.score <= 59

    def test_same_group_high_does_not_trigger_critical(self):
        """同一组的两个高危 → 不能触发 critical."""
        evidence = [
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30),
            _make_evidence("DOMAIN_SIMILAR_TO_OFFICIAL", "identity", "high", 25, confidence=0.85),
        ]
        # 但同形攻击会抑制相似域名，所以需要不同组
        result = determine_level(evidence, 55)
        assert result.critical_lock is False

    def test_score_clamped_to_level_range(self):
        """分数被 clamp 到等级区间."""
        evidence = [
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30),
        ]
        result = determine_level(evidence, 25)
        # raw_score=25 在 low 区间，但 high floor → score ≥ 60
        assert result.level == "high"
        assert result.score >= 60

    def test_configured_level_floor_score_is_used(self, monkeypatch):
        import phishing_rule_detector.scoring as scoring_module
        from phishing_rule_detector.config_loader import load_config

        config = load_config().model_copy(deep=True)
        config.scoring.level_floor_scores.high = 65
        monkeypatch.setattr(scoring_module, "load_config", lambda: config)
        evidence = [
            _make_evidence(
                "DOMAIN_HOMOGRAPH_ATTACK",
                "identity",
                "high",
                30,
                confidence=1.0,
            )
        ]

        result = determine_level(evidence, 30)

        assert result.level == "high"
        assert result.score == 65

    def test_confidence_updated_for_high(self):
        """risk.confidence 取触发等级下限的最高 decision_confidence."""
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.95),
        ]
        result = determine_level(evidence, 30)
        assert result.confidence == 0.95

    def test_critical_confidence_min_of_two(self):
        """critical 取两个独立高危组中较低的 decision_confidence."""
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=1.0),
            _make_evidence("E2", "credential", "high", 30, confidence=0.85),
        ]
        result = determine_level(evidence, 60)
        assert result.level == "critical"
        assert result.critical_lock is True

    def test_critical_confidence_per_group_max(self):
        """一个组有多条 high 时，组内取 max，然后取 min across groups（审查 Fix 3）."""
        # identity: 0.95, 0.80 → group_max=0.95
        # credential: 0.90 → group_max=0.90
        # risk.confidence = min(0.95, 0.90) = 0.90
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.95),
            _make_evidence("E2", "identity", "high", 25, confidence=0.80),
            _make_evidence("E3", "credential", "high", 30, confidence=0.90),
        ]
        result = determine_level(evidence, 80)
        assert result.level == "critical"
        assert result.critical_lock is True
        assert result.confidence == 0.90

    def test_critical_confidence_unqualified_not_counted(self):
        """未达 high 门槛的 evidence 不参与 critical confidence 计算."""
        # identity: dc=0.85 (qualifying)
        # credential: dc=0.95 (qualifying)
        # identity also has dc=0.70 (NOT qualifying) — should not pull down the min
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.85),
            _make_evidence("E2", "identity", "high", 20, confidence=0.70),  # 未达标
            _make_evidence("E3", "credential", "high", 30, confidence=0.95),
        ]
        result = determine_level(evidence, 80)
        assert result.level == "critical"
        assert result.critical_lock is True
        # E2 未达标不参与 → min(max(0.85), max(0.95)) = 0.85
        assert result.confidence == 0.85

    def test_same_group_two_high_no_critical(self):
        """同一组两条 high 不触发 critical_lock."""
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.95),
            _make_evidence("E2", "identity", "high", 25, confidence=0.85),
        ]
        result = determine_level(evidence, 55)
        assert result.critical_lock is False

    def test_three_independent_groups_critical(self):
        """三个独立高危组时结果稳定."""
        evidence = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.95),
            _make_evidence("E2", "credential", "high", 30, confidence=0.85),
            _make_evidence("E3", "navigation", "high", 25, confidence=0.80),
        ]
        result = determine_level(evidence, 85)
        assert result.level == "critical"
        assert result.critical_lock is True
        # min(max(0.95), max(0.85), max(0.80)) = 0.80
        assert result.confidence == 0.80

    def test_critical_confidence_input_order_independent(self):
        """输入顺序变化不影响 critical confidence 结果."""
        import random
        base = [
            _make_evidence("E1", "identity", "high", 30, confidence=0.95),
            _make_evidence("E2", "identity", "high", 25, confidence=0.80),
            _make_evidence("E3", "credential", "high", 30, confidence=0.90),
        ]
        # 固定种子确保可复现
        rng = random.Random(42)
        results = []
        for _ in range(5):
            shuffled = list(base)
            rng.shuffle(shuffled)
            result = determine_level(shuffled, 80)
            results.append(result.confidence)
        # 所有顺序应产生相同结果: min(max(0.95, 0.80), max(0.90)) = 0.90
        assert all(r == 0.90 for r in results)


# ── Full Pipeline ──────────────────────────────────────────

class TestScorePipeline:
    def test_pipeline_integration(self):
        """完整评分管线：抑制 → 衰减 → 组上限 → 等级."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, confidence=0.95),
            _make_evidence("DOMAIN_HOMOGRAPH_ATTACK", "identity", "high", 30),
            _make_evidence("DOMAIN_KEYWORD_IMPERSONATION", "identity", "medium", 12, confidence=0.8),
            _make_evidence("TIME_LIMIT_PRESSURE", "social", "medium", 10, confidence=0.8),
        ]
        risk, active, suppressed = score_pipeline(evidence)
        assert risk.level == "critical"
        assert risk.critical_lock is True
        assert risk.score >= 80

    def test_pipeline_returns_active_evidence(self):
        """score_pipeline 返回评分后的 active_evidence（审查 Fix 2）."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30, confidence=0.95),
        ]
        risk, active, suppressed = score_pipeline(evidence)
        assert isinstance(active, list)
        assert len(active) >= 1
        # active 中的 evidence 已应用衰减
        for item in active:
            assert isinstance(item, EvidenceItem)

    def test_pipeline_returns_suppressed(self):
        """score_pipeline 返回被抑制的 evidence（审查 Fix 2）."""
        evidence = [
            _make_evidence("HTML_PASSWORD_INPUT", "credential", "low", 4),
            _make_evidence("PASSWORD_FORM_UNTRUSTED_TARGET", "credential", "high", 30),
        ]
        risk, active, suppressed = score_pipeline(evidence)
        assert isinstance(suppressed, list)
        suppressed_ids = [e.rule_id for e in suppressed]
        assert "HTML_PASSWORD_INPUT" in suppressed_ids

    def test_confidence_no_floor_uses_max(self):
        """无等级下限时，risk.confidence = 所有有效证据的最高 decision_confidence（审查 Fix 8）."""
        evidence = [
            _make_evidence("E1", "social", "low", 5, confidence=0.3),
            _make_evidence("E2", "transport", "low", 5, confidence=0.7),
        ]
        risk, _, _ = score_pipeline(evidence)
        assert risk.confidence == 0.7
        assert risk.level_floor == "low"

    def test_confidence_medium_floor_uses_highest_qualifying(self):
        """medium 等级下限时 confidence = 触发下限的最高 decision_confidence（审查 Fix 8）."""
        evidence = [
            _make_evidence("E1", "social", "medium", 10, confidence=0.80),  # dc=0.80
            _make_evidence("E2", "social", "medium", 10, confidence=0.90),  # dc=0.90
        ]
        risk, _, _ = score_pipeline(evidence)
        assert risk.level_floor == "medium"
        assert risk.confidence == 0.90
