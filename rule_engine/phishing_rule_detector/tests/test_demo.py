"""演示模块测试."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import pytest


class TestDemoResults:
    """验证演示案例结果正确性."""

    def test_all_10_cases_pass(self):
        """10 个演示案例应全部 passed=true."""
        from phishing_rule_detector.demo import run_demo

        results = run_demo(silent=True)
        assert len(results) == 10, f"应有 10 个案例，实际 {len(results)}"

        for case in results:
            assert case["passed"], (
                f"[{case['id']}] 应通过: {case.get('failures', [])}"
            )

    def test_group_a_all_low(self):
        """Group A 所有案例应为 low."""
        from phishing_rule_detector.demo import run_demo

        results = run_demo(silent=True)
        for case in results:
            if case["id"].startswith("DEMO-A"):
                assert case["actual_level"] == "low", (
                    f"[{case['id']}] 期望 low，实际 {case['actual_level']}"
                )

    def test_group_b_medium(self):
        """B1/B2 应为 medium."""
        from phishing_rule_detector.demo import run_demo

        results = run_demo(silent=True)
        b_cases = {c["id"]: c for c in results if c["id"].startswith("DEMO-B")}
        assert b_cases["DEMO-B1"]["actual_level"] == "medium"
        assert b_cases["DEMO-B2"]["actual_level"] == "medium"
        # B2 应有 TIME_LIMIT_PRESSURE
        assert "TIME_LIMIT_PRESSURE" in b_cases["DEMO-B2"]["evidence_rules"]

    def test_group_c_high(self):
        """C1/C2/C3 应为 high."""
        from phishing_rule_detector.demo import run_demo

        results = run_demo(silent=True)
        c_cases = {c["id"]: c for c in results if c["id"].startswith("DEMO-C")}
        assert c_cases["DEMO-C1"]["actual_level"] == "high"
        assert c_cases["DEMO-C2"]["actual_level"] == "high"
        assert c_cases["DEMO-C3"]["actual_level"] == "high"
        # C2 应有 CREDENTIAL_REQUEST_TEXT
        assert "CREDENTIAL_REQUEST_TEXT" in c_cases["DEMO-C2"]["evidence_rules"]

    def test_d1_critical_with_lock(self):
        """D1 应为 critical 且 critical_lock=true."""
        from phishing_rule_detector.demo import run_demo

        results = run_demo(silent=True)
        d1 = next(c for c in results if c["id"] == "DEMO-D1")
        assert d1["actual_level"] == "critical"
        assert d1["critical_lock"] is True


class TestDemoCLI:
    """演示 CLI 测试."""

    def test_main_returns_zero_when_all_pass(self):
        """全部通过时 main() 返回 0."""
        from phishing_rule_detector.demo import main

        exit_code = main([])
        assert exit_code == 0

    def test_main_returns_one_when_failure(self):
        """有失败时 main() 返回 1."""
        # 构造一个必失败的案例 — 通过 monkeypatch detect
        import phishing_rule_detector.demo as demo_mod

        orig = demo_mod.DEMO_GROUPS
        try:
            # 添加一个期望 critical 但实际是 low 的案例
            demo_mod.DEMO_GROUPS = {
                "Test Fail": [{
                    "id": "FAIL-1",
                    "title": "will fail",
                    "description": "should fail",
                    "expected_level": "critical",
                    "expected_rule_ids": [],
                    "call": lambda: {
                        "success": True,
                        "risk": {"level": "low", "score": 5, "raw_score": 5,
                                  "confidence": 0.5, "critical_lock": False},
                        "evidence": [],
                        "warnings": [],
                        "trace_id": "test",
                        "duration_ms": 0,
                    },
                }],
            }
            exit_code = demo_mod.main([])
            assert exit_code == 1
        finally:
            demo_mod.DEMO_GROUPS = orig


class TestDemoJSON:
    """演示 JSON 导出测试."""

    def test_export_json_contains_expected_fields(self):
        """JSON 产物应包含 expected_level、actual_level、passed."""
        from phishing_rule_detector.demo import export_demo_json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "demo.json")
            export_demo_json(path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert "groups" in data
            assert "T" in data["generated_at"]
            datetime.fromisoformat(data["generated_at"])
            all_cases = []
            for g in data["groups"]:
                assert g["group"].startswith("Group ")
                all_cases.extend(g["cases"])
            assert len(all_cases) == 10
            for case in all_cases:
                assert "expected_level" in case
                assert "actual_level" in case
                assert "passed" in case
                assert case["passed"] is True, f"{case['id']} not passed"


class TestDemoSubprocess:
    """使用 subprocess 验证 CLI 行为."""

    def test_demo_cli_exit_zero(self):
        """python -m phishing_rule_detector.demo 返回 0."""
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.demo"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"

    def test_demo_json_export(self):
        """python -m phishing_rule_detector.demo --json 返回 0."""
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.demo", "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "演示结果已导出" in result.stdout

    def test_demo_cli_is_utf8_when_captured(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.demo"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        assert result.returncode == 0
        assert "钓鱼规则引擎" in result.stdout.decode("utf-8")

    @pytest.mark.parametrize(
        "command",
        [
            [sys.executable, "phishing_rule_detector/examples.py"],
            [sys.executable, "-m", "phishing_rule_detector.examples"],
        ],
    )
    def test_examples_cli_is_utf8_when_captured(self, command):
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        assert result.returncode == 0
        assert "钓鱼规则引擎" in result.stdout.decode("utf-8")
