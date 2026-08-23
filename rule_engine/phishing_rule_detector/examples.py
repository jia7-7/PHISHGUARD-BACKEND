#!/usr/bin/env python3
"""钓鱼检测规则引擎 — 使用示例集.

两种运行方式:
    python phishing_rule_detector/examples.py
    python -m phishing_rule_detector.examples

每个示例展示 detect() 的一种典型用法。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 支持直接脚本运行: 将项目根目录加入 sys.path
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from phishing_rule_detector.detector import detect


def _configure_piped_utf8() -> None:
    """Use UTF-8 for redirected CLI output while preserving interactive consoles."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _sep(title: str) -> None:
    """打印分隔线."""
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def _show(result: dict) -> None:
    """简洁打印检测结果的关键信息."""
    if not result["success"]:
        print(f"  [FAIL] 检测失败: {result['error']['code']} — {result['error']['message']}")
        return

    risk = result["risk"]
    level = risk["level"].upper()
    icons = {"LOW": "[LOW]", "MEDIUM": "[MED]", "HIGH": "[HIGH]", "CRITICAL": "[CRIT]"}
    print(f"  等级: {icons.get(level, '[???]')} {level}")
    print(f"  分数: {risk['score']} (raw={risk['raw_score']})")
    print(f"  置信度: {risk['confidence']:.2%}")
    if risk["critical_lock"]:
        print("  [LOCK] critical_lock: True (两组独立高危证据)")
    summary = result["summary"]
    print(f"  证据: high×{summary['high_count']} medium×{summary['medium_count']} low×{summary['low_count']}")
    if result["evidence"]:
        print("  触发规则:")
        for e in result["evidence"]:
            icon = "[H]" if e["severity"] == "high" else "[M]" if e["severity"] == "medium" else "[L]"
            print(f"    {icon} [{e['rule_id']}] {e['title']}")
            print(f"       → {e['reason']}")
    if result["warnings"]:
        print(f"  警告: {', '.join(result['warnings'])}")
    print(f"  trace_id: {result['trace_id']} (耗时 {result['duration_ms']}ms)")


# ══════════════════════════════════════════════════════════════════════
# 示例 1: URL 检测
# ══════════════════════════════════════════════════════════════════════


def example_1_url_detection():
    """检测单个 URL：学校官网 vs 仿冒域名."""
    _sep("示例 1: URL 检测")

    # 1a: 学校官网 — 安全
    print("\n  [1a] 学校官网:")
    result = detect(
        input_text="https://sdu.edu.cn/admissions/2025/schedule",
        input_type="url",
    )
    _show(result)

    # 1b: 仿冒域名 — 高危
    print("\n  [1b] 仿冒域名:")
    result = detect(
        input_text="https://sdu-edu.cn/login",
        input_type="url",
    )
    _show(result)

    # 1c: 嵌套官方域名 — 高危
    print("\n  [1c] 嵌套官方域名的恶意 URL:")
    result = detect(
        input_text="https://sdu.edu.cn.evil.com/login",
        input_type="url",
    )
    _show(result)


# ══════════════════════════════════════════════════════════════════════
# 示例 2: 邮件检测
# ══════════════════════════════════════════════════════════════════════


def example_2_email_detection():
    """检测邮件内容：正常邮件 vs 钓鱼邮件."""
    _sep("示例 2: 邮件检测")

    # 2a: 正常学校通知 — 低风险
    print("\n  [2a] 学校官方选课通知:")
    result = detect(
        input_text="您好，本学期选课系统将于 9 月 15 日开放，请登录 https://sdu.edu.cn/course 查看课程安排。",
        input_type="email",
        context={"sender": "registrar@sdu.edu.cn"},
    )
    _show(result)

    # 2b: 仿冒发件人 — 高危
    print("\n  [2b] 仿冒学校发件人钓鱼邮件:")
    result = detect(
        input_text=(
            "您好，这里是山东大学信息技术中心，"
            "请点击链接验证您的账号 https://evil.com/verify"
        ),
        input_type="email",
        context={"sender": "fake@evil.com"},
    )
    _show(result)

    # 2c: 带危险附件
    print("\n  [2c] 带危险附件的邮件:")
    result = detect(
        input_text="请下载附件查看最新通知",
        input_type="email",
        context={
            "sender": "unknown@external.com",
            "attachments": ["通知.pdf.exe", "详情.doc"],
        },
    )
    _show(result)


# ══════════════════════════════════════════════════════════════════════
# 示例 3: HTML 表单检测
# ══════════════════════════════════════════════════════════════════════


def example_3_html_form_detection():
    """检测 HTML 页面：正常登录 vs 钓鱼表单."""
    _sep("示例 3: HTML 表单检测")

    # 3a: 正常同域表单 — 低风险
    print("\n  [3a] 正常同域登录表单:")
    result = detect(
        input_text=(
            "<html><body>"
            "<form action='https://normal-site.com/login'>"
            "<input type='password' name='pwd'>"
            "<input type='text' name='user'>"
            "</form>"
            "</body></html>"
        ),
        input_type="html",
        base_url="https://normal-site.com/login",
    )
    _show(result)

    # 3b: 仿冒域名钓鱼表单 — critical
    print("\n  [3b] 仿冒域名+密码+学号表单 (critical_lock):")
    result = detect(
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
    )
    _show(result)


# ══════════════════════════════════════════════════════════════════════
# 示例 4: 短信检测
# ══════════════════════════════════════════════════════════════════════


def example_4_sms_detection():
    """检测短信内容."""
    _sep("示例 4: 短信检测")

    # 4a: 钓鱼短信
    print("\n  [4a] 时间限制+账号停用钓鱼短信:")
    result = detect(
        input_text="【山东大学】请在12小时内完成验证，否则账号将被停用！点击 https://short.link/verify",
        input_type="sms",
    )
    _show(result)


# ══════════════════════════════════════════════════════════════════════
# 示例 5: Debug 模式
# ══════════════════════════════════════════════════════════════════════


def example_5_debug_mode():
    """Debug 模式：查看被抑制的证据."""
    _sep("示例 5: Debug 模式查看被抑制证据")

    result = detect(
        input_text="请立即验证您的账号 https://evil.com/login",
        input_type="email",
        context={"sender": "fake@evil.com", "debug": True},
    )
    _show(result)

    if result.get("suppressed_evidence"):
        print(f"\n  [SUPPRESSED] 被抑制的证据 ({len(result['suppressed_evidence'])} 条):")
        for e in result["suppressed_evidence"]:
            print(f"    [{e['rule_id']}] {e['title']}")


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════


def main():
    """运行所有示例."""
    print("=" * 64)
    print("  钓鱼规则引擎 — 使用示例集")
    print("  版本: 3.0.0")
    print("=" * 64)

    example_1_url_detection()
    example_2_email_detection()
    example_3_html_form_detection()
    example_4_sms_detection()
    example_5_debug_mode()

    print(f"\n{'=' * 64}")
    print("  所有示例执行完毕")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    _configure_piped_utf8()
    main()
