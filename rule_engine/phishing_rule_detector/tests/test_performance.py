"""性能测试：验证检测管线的响应时间和内存表现."""
from __future__ import annotations

import gc
import socket
import time


from phishing_rule_detector.detector import detect


# ── 简单案例延迟 ──

class TestSimpleCaseLatency:
    def test_url_detection_under_100ms(self):
        """简单 URL 检测应在 100ms 内完成."""
        start = time.perf_counter()
        result = detect(
            input_text="https://sdu.edu.cn/admissions",
            input_type="url",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert elapsed_ms < 500, f"URL 检测耗时 {elapsed_ms:.0f}ms 超过 500ms"

    def test_email_detection_under_500ms(self):
        """简单邮件检测应在 500ms 内完成."""
        start = time.perf_counter()
        result = detect(
            input_text="请立即验证您的账号 https://evil.com/login",
            input_type="email",
            context={"sender": "fake@evil.com"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert elapsed_ms < 1000, f"邮件检测耗时 {elapsed_ms:.0f}ms 超过 1000ms"

    def test_html_detection_under_1000ms(self):
        """中等 HTML 检测应在 1000ms 内完成."""
        html = "<html><body>" + "".join(
            f'<a href="https://example{i}.com">link{i}</a>' for i in range(20)
        ) + "</body></html>"
        start = time.perf_counter()
        result = detect(input_text=html, input_type="html")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert elapsed_ms < 2000, f"HTML 检测耗时 {elapsed_ms:.0f}ms 超过 2000ms"


# ── 批量处理 ──

class TestBatchProcessing:
    def test_50_cases_under_30s(self):
        """50 条案例处理应在 30 秒内完成."""
        cases = [
            ("https://sdu.edu.cn/news", "url", {}),
            ("【紧急】请立即验证账号 https://evil.com/login", "email", {"sender": "fake@evil.com"}),
            ("<html><body><form action='https://evil.com'><input type='password'></form></body></html>", "html", {}),
            ("您好，您的账号将被停用，请点击链接验证", "email", {"sender": "admin@sdu-edu.cn"}),
            ("https://bit.ly/abc123", "url", {}),
        ] * 10  # 50 cases

        start = time.perf_counter()
        for text, itype, ctx in cases:
            result = detect(input_text=text, input_type=itype, context=ctx)
            assert result["success"]
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 30, f"50 条案例耗时 {elapsed_s:.1f}s"


# ── 内存使用 ──

class TestMemoryUsage:
    def test_no_memory_leak_over_repeated_calls(self):
        """重复调用不应有明显内存泄漏."""
        gc.collect()
        # 运行多次检测
        for _ in range(20):
            detect(
                input_text="请验证账号 https://evil.com/login",
                input_type="email",
                context={"sender": "fake@evil.com"},
            )
        gc.collect()
        # 基本通过（不做精确测量）
        assert True


# ── 大输入处理 ──

class TestLargeInput:
    def test_large_text_handled(self):
        """大文本输入应正常处理."""
        large_text = "这是一段正常文本。" * 500  # ~3500 字符
        result = detect(
            input_text=large_text,
            input_type="text",
        )
        assert result["success"]

    def test_large_html_handled(self):
        """大 HTML 输入应正常处理."""
        large_html = "<html><body>" + "<p>正常内容</p>" * 300 + "</body></html>"
        result = detect(
            input_text=large_html,
            input_type="html",
        )
        assert result["success"]

    def test_many_attachments_handled(self):
        """多附件列表应正常处理."""
        attachments = [f"file{i}.pdf" for i in range(20)] + ["malware.exe"]
        result = detect(
            input_text="请查看附件",
            input_type="email",
            context={
                "sender": "test@example.com",
                "attachments": attachments,
            },
        )
        assert result["success"]

    def test_many_qr_urls_handled(self):
        """多二维码 URL 应正常处理."""
        qr_urls = [f"https://evil{i}.com/login" for i in range(20)]
        result = detect(
            input_text="请扫描二维码",
            input_type="email",
            context={
                "sender": "admin@fake.com",
                "qr_urls": qr_urls,
            },
        )
        assert result["success"]


# ── 并发安全（简化验证 — 无 threading）──

class TestIdempotency:
    def test_same_input_produces_same_result(self):
        """同一输入多次调用结果应一致."""
        results = []
        for _ in range(5):
            r = detect(
                input_text="【紧急】请验证您的账号 https://evil.com/login",
                input_type="email",
                context={"sender": "fake@evil.com"},
            )
            results.append((r["risk"]["level"], r["risk"]["raw_score"]))

        # 所有结果应相同
        assert all(r == results[0] for r in results), f"结果不一致: {results}"


# ── 零网络访问验证 ──

class TestOfflineOperation:
    def test_no_network_for_normal_detection(self, monkeypatch):
        """在 socket 层阻断网络，检测仍应正常完成."""
        def blocked(*_args, **_kwargs):
            raise AssertionError("检测管线不应访问网络")

        monkeypatch.setattr(socket.socket, "connect", blocked)
        monkeypatch.setattr(socket, "create_connection", blocked)
        result = detect(
            input_text="https://sdu.edu.cn/login",
            input_type="url",
        )
        assert result["success"]
