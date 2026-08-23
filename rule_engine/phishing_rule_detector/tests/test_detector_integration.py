"""集成测试：对 100 条评测案例运行完整检测管线.

Phase 2 验收：严格检查 required_rule_ids / forbidden_rule_ids / expected_level.
"""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from phishing_rule_detector.detector import detect


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_cases() -> list[dict]:
    """加载所有评测案例."""
    path = _FIXTURE_DIR / "evaluation_cases.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["cases"]


ALL_CASES = _load_cases()

LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _case_id(case: dict) -> str:
    desc = case.get("description", "")[:60]
    return f"{case['id']}: {desc}"


def _run_detect(case: dict) -> dict:
    """按 case 配置调用 detect."""
    context = dict(case.get("context", {}))
    kwargs: dict = {
        "input_text": case["input_text"],
        "input_type": case["input_type"],
        "context": context,
    }
    if case.get("base_url"):
        kwargs["base_url"] = case["base_url"]
    return detect(**kwargs)


# ── 参数化运行所有案例 ──


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_evaluation_case(case: dict):
    """每个评测案例应通过 required/forbidden/level 断言."""
    result = _run_detect(case)

    assert result["success"], f"[{case['id']}] 检测应成功"

    level = result["risk"]["level"]
    evidence_rule_ids = {e["rule_id"] for e in result["evidence"]}
    label = case["label"]

    # ── 等级断言 ──
    expected_level = case.get("expected_level", "")
    allowed_levels = case.get("allowed_levels", [])

    if expected_level:
        actual_rank = LEVEL_ORDER.get(level, -1)
        expected_rank = LEVEL_ORDER.get(expected_level, -1)
        assert actual_rank >= expected_rank, (
            f"[{case['id']}] 等级不足：期望 ≥{expected_level}，实际 {level} "
            f"(score={result['risk']['raw_score']}, "
            f"evidence={evidence_rule_ids})"
        )
    elif allowed_levels:
        assert level in allowed_levels, (
            f"[{case['id']}] 等级不在允许范围：允许 {allowed_levels}，实际 {level}"
        )

    # ── 必现规则断言 ──
    required = case.get("required_rule_ids", [])
    for rule_id in required:
        assert rule_id in evidence_rule_ids, (
            f"[{case['id']}] 缺少必现规则 {rule_id}，"
            f"实际证据: {evidence_rule_ids}"
        )

    # ── 禁止规则断言 ──
    forbidden = case.get("forbidden_rule_ids", [])
    for rule_id in forbidden:
        assert rule_id not in evidence_rule_ids, (
            f"[{case['id']}] 出现禁止规则 {rule_id}"
        )

    # ── 钓鱼样本必须产生证据 ──
    if label == "phishing":
        assert len(result["evidence"]) > 0, (
            f"[{case['id']}] 钓鱼样本应产生至少一条证据"
        )

    # ── 正常样本不应有 high/critical 等级 ──
    if label == "normal":
        assert level == "low", (
            f"[{case['id']}] normal 样本不应触发 medium/high/critical，"
            f"实际: level={level}, score={result['risk']['raw_score']}"
        )


# ── 统计汇总 ──


def test_evaluation_summary():
    """输出评测汇总统计（总是通过，仅打印信息）."""
    results = {"total": 0, "passed": 0, "failed": 0, "details": []}

    for case in ALL_CASES:
        result = _run_detect(case)
        assert result["success"], f"{case['id']}: 检测失败"

        level = result["risk"]["level"]
        evidence_ids = {e["rule_id"] for e in result["evidence"]}
        score = result["risk"]["raw_score"]
        label = case["label"]
        cid = case["id"]
        passed = True
        failures = []

        # 等级检查
        expected_level = case.get("expected_level", "")
        allowed_levels = case.get("allowed_levels", [])
        if expected_level:
            if LEVEL_ORDER.get(level, -1) < LEVEL_ORDER.get(expected_level, -1):
                passed = False
                failures.append(f"等级不足：期望≥{expected_level}，实际{level}")
        elif allowed_levels:
            if level not in allowed_levels:
                passed = False
                failures.append(f"等级不在允许范围：{allowed_levels}，实际{level}")

        # 必现规则
        for rid in case.get("required_rule_ids", []):
            if rid not in evidence_ids:
                passed = False
                failures.append(f"缺少必现规则 {rid}")

        # 禁止规则
        for rid in case.get("forbidden_rule_ids", []):
            if rid in evidence_ids:
                passed = False
                failures.append(f"出现禁止规则 {rid}")

        # 正常样本不应有高等级
        if label == "normal" and level != "low":
            passed = False
            failures.append(f"normal 样本等级过高: {level}")

        # 钓鱼样本必须有证据
        if label == "phishing" and len(result["evidence"]) == 0:
            passed = False
            failures.append("钓鱼样本无证据")

        results["total"] += 1
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({
                "id": cid,
                "label": label,
                "level": level,
                "score": score,
                "evidence": sorted(evidence_ids),
                "failures": failures,
            })

    # 输出统计
    print(f"\n{'='*60}")
    print("评测汇总")
    print(f"{'='*60}")
    accuracy = results["passed"] / results["total"] * 100 if results["total"] else 0
    print(f"通过: {results['passed']}/{results['total']} ({accuracy:.1f}%)")
    if results["failed"]:
        print(f"失败: {results['failed']} 条")
        for d in results["details"]:
            print(f"  [{d['id']}] level={d['level']} score={d['score']}")
            for f in d["failures"]:
                print(f"    - {f}")
            print(f"    evidence: {d['evidence']}")
    print(f"{'='*60}")

    # 质量门禁
    assert accuracy == 100, f"评测存在失败案例: {accuracy:.1f}%"


# ── 关键场景验收 ──


class TestAcceptanceScenarios:
    """验收场景 N01-N05, P01-P08."""

    def test_N01_normal_email_passes(self):
        """N01: 正常学校邮件应评为 low."""
        result = detect(
            input_text="关于 2025 年度国家奖学金评审工作的通知已发布，详见附件。",
            input_type="email",
            context={"sender": "xsc@sdu.edu.cn"},
        )
        assert result["success"]
        assert result["risk"]["level"] == "low"
        # 确保无身份冒充
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "EMAIL_SENDER_IMPERSONATION" not in rule_ids

    def test_N02_normal_url_passes(self):
        """N02: 学校官网 URL 应安全."""
        result = detect(
            input_text="https://sdu.edu.cn/admissions/2025/schedule",
            input_type="url",
        )
        assert result["success"]
        assert result["risk"]["level"] == "low"

    def test_N03_normal_html_passes(self):
        """N03: 学校官网 HTML 应安全."""
        result = detect(
            input_text="<html><body><h1>山东大学</h1><a href='https://sdu.edu.cn'>官网</a></body></html>",
            input_type="html",
        )
        assert result["success"]
        assert result["risk"]["level"] == "low"

    def test_N04_normal_text_passes(self):
        """N04: 日常文本应安全."""
        result = detect(
            input_text="今天的课程改到线上，请大家准时参加。",
            input_type="text",
        )
        assert result["success"]
        assert result["risk"]["level"] == "low"

    def test_N05_mixed_external_links_with_official_stays_low(self):
        """N05: 混合外链但以官方为主应低风险."""
        result = detect(
            input_text="请查看学校通知 https://sdu.edu.cn/news 和 https://github.com/sdu/repo",
            input_type="text",
        )
        assert result["success"]
        high_count = sum(1 for e in result["evidence"] if e["severity"] == "high")
        assert high_count == 0, f"不应有 high 严重度证据: {high_count} 条"

    def test_P01_phishing_email_detected(self):
        """P01: 假冒域名钓鱼邮件应检测到 DOMAIN_SIMILAR_TO_OFFICIAL."""
        result = detect(
            input_text="【紧急】您的账号出现异常，请立即点击 https://sdu-edu.cn/verify 验证",
            input_type="email",
            context={"sender": "security@sdu-edu.cn"},
        )
        assert result["risk"]["level"] in ("medium", "high", "critical")
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "DOMAIN_SIMILAR_TO_OFFICIAL" in rule_ids

    def test_P02_url_phishing_detected(self):
        """P02: 嵌套官方域名 URL 应检测到."""
        result = detect(
            input_text="https://sdu.edu.cn.evil.com/login",
            input_type="url",
        )
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "URL_NESTED_OFFICIAL_DOMAIN" in rule_ids

    def test_P03_html_form_phishing_detected(self):
        """P03: 伪造登录表单应检测到 critical（identity+credential 两组 qualified high → critical_lock）."""
        result = detect(
            input_text="<html><body><form action='https://evil.com/steal'><input type='password' name='pwd'><input name='student_id'></form></body></html>",
            input_type="html",
            base_url="https://sdu-edu.cn/login",
        )
        assert result["risk"]["level"] == "critical", (
            f"期望 critical，实际 {result['risk']['level']} "
            f"(score={result['risk']['raw_score']}, evidence={[e['rule_id'] for e in result['evidence']]})"
        )
        assert result["risk"]["critical_lock"] is True, "应触发 critical_lock"
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in rule_ids
        assert "DOMAIN_SIMILAR_TO_OFFICIAL" in rule_ids

    def test_P04_punycode_domain_detected(self):
        """P04: 可疑域名应触发 DOMAIN_SIMILAR_TO_OFFICIAL 检测."""
        result = detect(
            input_text="https://sdu-edu.cn/login",
            input_type="url",
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "DOMAIN_SIMILAR_TO_OFFICIAL" in rule_ids

    def test_P05_short_link_detected(self):
        """P05: 短链接应触发检测."""
        result = detect(
            input_text="请点击 https://bit.ly/sdu-verify 验证账号",
            input_type="url",
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "URL_SHORT_LINK" in rule_ids

    def test_P06_attachment_detected(self):
        """P06: 危险附件应触发检测."""
        result = detect(
            input_text="请下载附件查看",
            input_type="email",
            context={
                "sender": "fake@evil.com",
                "attachments": ["invoice.pdf.exe"],
            },
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        # 双重后缀应触发
        assert "ATTACHMENT_DOUBLE_EXTENSION" in rule_ids or "ATTACHMENT_EXECUTABLE" in rule_ids

    def test_P07_time_pressure_and_disable_threat_detected(self):
        """P07: 具体时间限制与账号停用威胁必须同时识别."""
        result = detect(
            input_text="请在12小时内完成验证，否则账号将被停用",
            input_type="sms",
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert result["risk"]["level"] in ("medium", "high", "critical")
        assert "TIME_LIMIT_PRESSURE" in rule_ids
        assert "ACCOUNT_DISABLE_THREAT" in rule_ids

    def test_qr_code_external_url_detected(self):
        """二维码指向外域作为独立验收场景保留."""
        result = detect(
            input_text="请扫描二维码登录",
            input_type="email",
            context={"sender": "admin@fake.com", "qr_urls": ["https://evil.com/login"]},
        )
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "QR_CREDENTIAL_URL" in rule_ids or "QR_CODE_EXTERNAL_URL" in rule_ids

    def test_P08_sender_impersonation_detected(self):
        """P08: 发件人冒充学校部门应检测到."""
        result = detect(
            input_text="您好，这里是山东大学信息技术中心，请点击链接验证您的账号",
            input_type="email",
            context={"sender": "fake@evil.com"},
        )
        assert result["success"]
        rule_ids = {e["rule_id"] for e in result["evidence"]}
        assert "EMAIL_SENDER_IMPERSONATION" in rule_ids


# ── debug 模式测试 ──


class TestDebugMode:
    def test_debug_returns_suppressed_evidence(self):
        """debug=true 时应返回 suppressed_evidence."""
        result = detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com", "debug": True},
        )
        assert result["success"]
        assert "suppressed_evidence" in result

    def test_debug_false_no_suppressed(self):
        """debug=false 时不应返回 suppressed_evidence."""
        result = detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com", "debug": False},
        )
        assert result["success"]
        assert result.get("suppressed_evidence") is None
