"""评测工具测试."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────


class TestEvaluationLoadCases:
    """测试加载评测案例."""

    def test_load_cases_returns_list(self):
        from phishing_rule_detector.evaluation import load_cases

        cases = load_cases()
        assert isinstance(cases, list)
        assert len(cases) == 100

    def test_each_case_has_required_fields(self):
        from phishing_rule_detector.evaluation import load_cases

        cases = load_cases()
        for case in cases:
            for field in ["id", "label", "input_type", "input_text"]:
                assert field in case, f"案例缺少字段: {field}"

    def test_load_cases_missing_file_raises(self):
        from phishing_rule_detector.evaluation import load_cases

        with pytest.raises(FileNotFoundError):
            load_cases("definitely_missing_file.yaml")

    @staticmethod
    def _write_dataset(tmp_path, cases):
        path = tmp_path / "cases.yaml"
        path.write_text(
            yaml.safe_dump({"cases": cases}, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def test_invalid_label_rejected(self, tmp_path):
        from phishing_rule_detector.evaluation import load_cases

        path = self._write_dataset(tmp_path, [{
            "id": "X01", "label": "typo", "input_type": "text",
            "input_text": "hello", "expected_level": "low",
        }])
        with pytest.raises(ValueError, match="label"):
            load_cases(path)

    def test_invalid_expected_level_rejected(self, tmp_path):
        from phishing_rule_detector.evaluation import load_cases

        path = self._write_dataset(tmp_path, [{
            "id": "X01", "label": "normal", "input_type": "text",
            "input_text": "hello", "expected_level": "nonsense",
        }])
        with pytest.raises(ValueError, match="expected_level"):
            load_cases(path)

    def test_cases_must_be_list(self, tmp_path):
        from phishing_rule_detector.evaluation import load_cases

        path = tmp_path / "cases.yaml"
        path.write_text("cases:\n  id: X01\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cases.*列表"):
            load_cases(path)

    def test_duplicate_case_ids_rejected(self, tmp_path):
        from phishing_rule_detector.evaluation import load_cases

        case = {
            "id": "X01", "label": "normal", "input_type": "text",
            "input_text": "hello", "expected_level": "low",
        }
        path = self._write_dataset(tmp_path, [case, dict(case)])
        with pytest.raises(ValueError, match="重复"):
            load_cases(path)


# ──────────────────────────────────────────────
# 基础运行
# ──────────────────────────────────────────────


class TestEvaluationRun:
    def test_run_full_evaluation_returns_structured_result(self):
        from phishing_rule_detector.evaluation import run_full_evaluation

        result = run_full_evaluation()
        for key in [
            "total", "phishing_count", "normal_count",
            "confusion_matrix", "accuracy", "precision", "recall", "f1",
            "false_positive_rate", "level_distribution",
            "contract_validation", "quality_gates", "latency_ms",
            "dataset",
        ]:
            assert key in result, f"缺少 key: {key}"

    def test_legacy_run_evaluation_uses_binary_accuracy(self):
        from phishing_rule_detector.evaluation import run_evaluation

        result = run_evaluation()
        assert result["accuracy"] == 0.89
        assert result["contract_validation"]["pass_rate"] == 1.0

    def test_empty_cases(self):
        from phishing_rule_detector.evaluation import evaluate_cases

        result = evaluate_cases([])
        assert result["total"] == 0
        assert result["contract_validation"]["passed"] == 0
        assert result["contract_validation"]["failed"] == 0
        assert result["accuracy"] == 0.0


# ──────────────────────────────────────────────
# 混淆矩阵 & 指标
# ──────────────────────────────────────────────


class TestConfusionMatrix:
    def test_confusion_matrix_matches_baseline(self):
        """验证混淆矩阵 = TP=39, FP=0, TN=50, FN=11."""
        from phishing_rule_detector.evaluation import run_full_evaluation

        result = run_full_evaluation()
        cm = result["confusion_matrix"]
        assert cm["tp"] == 39, f"TP={cm['tp']}"
        assert cm["fp"] == 0, f"FP={cm['fp']}"
        assert cm["tn"] == 50, f"TN={cm['tn']}"
        assert cm["fn"] == 11, f"FN={cm['fn']}"

    def test_core_metrics_accuracy_precision_recall_f1_fpr(self):
        """验证五项核心指标值."""
        from phishing_rule_detector.evaluation import run_full_evaluation

        result = run_full_evaluation()
        assert abs(result["accuracy"] - 0.89) < 0.001
        assert abs(result["precision"] - 1.0) < 0.001
        assert abs(result["recall"] - 0.78) < 0.001
        assert abs(result["f1"] - 0.876404) < 0.001
        assert abs(result["false_positive_rate"] - 0.0) < 0.001

    def test_contract_pass_rate_is_separate_from_accuracy(self):
        """契约通过率与二分类准确率是不同的字段."""
        from phishing_rule_detector.evaluation import run_full_evaluation

        result = run_full_evaluation()
        cv = result["contract_validation"]
        assert cv["pass_rate"] == 1.0, f"contract_pass_rate={cv['pass_rate']}"
        # 明确区分: 这是两个不同的字段
        assert result["accuracy"] != 1.0 or cv["pass_rate"] == 1.0
        # accuracy 是二分类指标, contract pass_rate 是契约验证指标
        assert "accuracy" in result
        assert "contract_validation" in result
        assert "pass_rate" in result["contract_validation"]


# ──────────────────────────────────────────────
# 延迟计算
# ──────────────────────────────────────────────


class TestLatencyMetrics:
    def test_latency_percentile_correct(self):
        from phishing_rule_detector.evaluation import _latency_percentile

        vals = [0, 0, 0, 1, 2, 5, 10, 20, 50, 100]
        p50 = _latency_percentile(vals, 50)
        # 10 elements, idx = 0.5 * 9 = 4.5 → lo=4(val=2), hi=5(val=5), frac=0.5
        # → 2 + 0.5*(5-2) = 3.5
        assert p50 == 3.5, f"p50={p50}"

        p95 = _latency_percentile(vals, 95)
        # 10 elements, idx = 0.95 * 9 = 8.55 → lo=8(val=50), hi=9(val=100), frac=0.55
        # → 50 + 0.55*(100-50) = 77.5
        assert p95 == 77.5, f"p95={p95}"

        p_max = _latency_percentile(vals, 100)
        assert p_max == 100.0, f"max={p_max}"


# ──────────────────────────────────────────────
# 输出
# ──────────────────────────────────────────────


class TestEvaluationOutput:
    def test_generate_report_returns_string(self):
        from phishing_rule_detector.evaluation import generate_report, run_full_evaluation

        result = run_full_evaluation()
        report = generate_report(result)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "评测报告" in report

    def test_export_json_writes_file(self):
        from phishing_rule_detector.evaluation import export_json, run_full_evaluation

        result = run_full_evaluation()
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "metrics.json")
            export_json(result, outpath)
            assert os.path.exists(outpath)
            with open(outpath, encoding="utf-8") as f:
                data = json.load(f)
            assert data["total"] == result["total"]
            assert "details" not in data, "导出 JSON 不应包含 details (隐私保护)"


# ──────────────────────────────────────────────
# Subprocess CLI 测试
# ──────────────────────────────────────────────


class TestEvaluationCLISubprocess:
    """使用 subprocess 运行真实命令."""

    def test_cli_no_args_exit_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"

    def test_cli_report_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation", "--report"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0
        assert "评测报告" in result.stdout

    def test_cli_report_is_utf8_when_captured(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation", "--report"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        assert result.returncode == 0
        assert "钓鱼规则引擎" in result.stdout.decode("utf-8")

    def test_cli_json_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation", "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "confusion_matrix" in data
        assert data["confusion_matrix"]["tp"] == 39

    def test_cli_dataset_flag(self):
        dataset = str(
            Path(__file__).resolve().parent
            / "fixtures" / "evaluation_cases.yaml"
        )
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation",
             "--dataset", dataset],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0

    def test_cli_missing_dataset_exit_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation",
             "--dataset", "definitely-missing.yaml"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode != 0, f"退出码应为非零: {result.returncode}"

    def test_cli_missing_dataset_no_json_out(self):
        """缺失数据集时不应生成 --json-out 文件."""
        outpath = "artifacts/should_not_exist.json"
        # 确保文件不存在
        if os.path.exists(outpath):
            os.remove(outpath)
        subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation",
             "--dataset", "definitely-missing.yaml",
             "--json-out", outpath],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert not os.path.exists(outpath), (
            "缺失数据集时不应生成 JSON 文件"
        )

    def test_cli_json_out_generates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "metrics.json")
            dataset = str(
                Path(__file__).resolve().parent
                / "fixtures" / "evaluation_cases.yaml"
            )
            result = subprocess.run(
                [sys.executable, "-m", "phishing_rule_detector.evaluation",
                 "--dataset", dataset,
                 "--json-out", outpath],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            assert result.returncode == 0
            assert os.path.exists(outpath)
            with open(outpath, encoding="utf-8") as f:
                data = json.load(f)
            assert "confusion_matrix" in data

    def test_cli_unknown_arg_exit_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation",
             "--unknown-flag"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode != 0, f"未知参数应返回非零: {result.returncode}"

    def test_cli_json_does_not_contain_sensitive_input_text(self):
        """JSON --json 输出不应包含完整测试输入文本（隐私保护）."""
        result = subprocess.run(
            [sys.executable, "-m", "phishing_rule_detector.evaluation",
             "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        # --json 输出只含指标聚合，不应包含 details / 原始输入
        assert "details" not in data, (
            "--json 产物不应包含 details（含原始输入文本）"
        )
        # 验证没有泄露具体测试案例文本（检查几个已知的完整句子）
        text = json.dumps(data, ensure_ascii=False)
        sensitive_patterns = [
            "请输入您的账号密码",
            "token=abc123",
            "您的银行卡号",
        ]
        for pattern in sensitive_patterns:
            assert pattern not in text, f"JSON 泄露敏感输入: {pattern!r}"


# ──────────────────────────────────────────────
# 质量门禁
# ──────────────────────────────────────────────


class TestQualityGates:
    def test_quality_gates_passed_true(self):
        from phishing_rule_detector.evaluation import run_full_evaluation

        result = run_full_evaluation()
        qg = result["quality_gates"]
        assert qg["passed"] is True, f"quality gates failed: {qg}"

    def test_failed_gate_detected(self):
        """构造失败案例验证质量门禁能检测到失败."""
        from phishing_rule_detector.evaluation import compute_metrics

        # 构造 50 phishing (20 pred_positive) + 50 normal (10 pred_positive)
        details = []
        for i in range(50):
            if i < 20:
                details.append({"id": f"P{i}", "label": "phishing", "level": "high", "passed": True})
            else:
                details.append({"id": f"P{i}", "label": "phishing", "level": "low", "passed": True})
        for i in range(50):
            if i < 10:
                details.append({"id": f"N{i}", "label": "normal", "level": "high", "passed": True})
            else:
                details.append({"id": f"N{i}", "label": "normal", "level": "low", "passed": True})

        result = compute_metrics(details)
        cm = result["confusion_matrix"]
        assert cm["tp"] == 20
        assert cm["fp"] == 10  # 有 FP，precision < 0.95, recall drops
        qg = result["quality_gates"]
        assert qg["passed"] is False, "有 FP 时质量门禁应失败"


# ──────────────────────────────────────────────
# 离线 / 无网络
# ──────────────────────────────────────────────


class TestEvaluationOffline:
    def test_evaluation_no_network(self, monkeypatch):
        """验证完整 evaluation 不发起网络请求."""
        import urllib.request
        import socket

        call_count = [0]

        def _block(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("不应发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", _block)
        monkeypatch.setattr(socket.socket, "connect", _block)
        monkeypatch.setattr(socket, "create_connection", _block)

        from phishing_rule_detector.evaluation import run_full_evaluation

        run_full_evaluation()
        assert call_count[0] == 0, f"发起了 {call_count[0]} 次网络请求"


# ──────────────────────────────────────────────
# 程序化 API 兼容
# ──────────────────────────────────────────────


class TestProgrammaticAPI:
    def test_main_direct_call(self):
        from phishing_rule_detector.evaluation import main

        exit_code = main([])
        assert exit_code == 0

    def test_main_report_flag_direct(self):
        from phishing_rule_detector.evaluation import main

        exit_code = main(["--report"])
        assert exit_code == 0
