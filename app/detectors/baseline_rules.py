from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from app.config import Settings
from app.models import AnalysisPayload, DetectorResult, InputKind, Signal


_SHORTENERS = {
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "is.gd",
    "reurl.cc",
    "dwz.cn",
}

_URGENT_PATTERNS = (
    r"立即.{0,6}(验证|登录|处理|领取)",
    r"账号.{0,8}(停用|冻结|过期)",
    r"限时.{0,8}(领取|验证|提交)",
    r"逾期.{0,8}(失效|停用|取消)",
)

_CREDENTIAL_PATTERNS = (
    r"(输入|提交|填写).{0,8}(密码|验证码|银行卡|身份证)",
    r"(账号|密码).{0,4}(验证|更新|确认)",
)


class _HtmlFeatureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.password_inputs = 0
        self.form_actions: list[str] = []
        self.iframes = 0
        self.meta_refreshes = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        tag = tag.lower()
        if tag == "input" and values.get("type", "").lower() == "password":
            self.password_inputs += 1
        elif tag == "form" and values.get("action"):
            self.form_actions.append(values["action"])
        elif tag == "iframe":
            self.iframes += 1
        elif tag == "meta" and values.get("http-equiv", "").lower() == "refresh":
            self.meta_refreshes += 1


class BaselineRuleDetector:
    """Small offline baseline used until the security teammate supplies a module."""

    name = "baseline-rules"
    family = "rules"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def analyze(self, payload: AnalysisPayload) -> DetectorResult | None:
        signals: list[Signal] = []

        if payload.kind == InputKind.URL and payload.url:
            signals.extend(self._analyze_url(payload.url))
        elif payload.kind == InputKind.TEXT and payload.text:
            signals.extend(self._analyze_text(payload.text))
            signals.extend(self._extract_and_analyze_urls(payload.text))
        elif payload.kind == InputKind.HTML and payload.html:
            signals.extend(self._analyze_html(payload.html, payload.url))
            signals.extend(self._analyze_text(self._strip_markup(payload.html)))
        else:
            return None

        score = self._score(signals)
        confidence = 0.70 if signals else 0.45
        return DetectorResult(
            detector=self.name,
            family="rules",
            score=score,
            confidence=confidence,
            signals=signals,
            metadata={
                "version": "baseline-0.1",
                "network_access": False,
                "replace_with_teammate_module": True,
            },
        )

    def _analyze_url(self, url: str) -> list[Signal]:
        signals: list[Signal] = []
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")

        if parsed.scheme.lower() != "https":
            signals.append(self._signal("transport", "未使用HTTPS", 20, url))
        if "@" in parsed.netloc:
            signals.append(self._signal("url-obfuscation", "URL包含@符号", 55, parsed.netloc))
        if len(url) > 160:
            signals.append(self._signal("url-obfuscation", "URL异常冗长", 25, url[:200]))
        if hostname.startswith("xn--") or ".xn--" in hostname:
            signals.append(self._signal("domain", "域名使用Punycode", 45, hostname))
        if hostname in _SHORTENERS:
            signals.append(self._signal("short-link", "使用短链接隐藏目标地址", 35, hostname))

        try:
            ipaddress.ip_address(hostname.strip("[]"))
        except ValueError:
            pass
        else:
            signals.append(self._signal("domain", "使用IP地址代替域名", 55, hostname))

        if hostname and hostname.count(".") >= 4:
            signals.append(self._signal("domain", "域名层级过多", 25, hostname))

        for official in self.settings.official_domains:
            official = official.lower().rstrip(".")
            if "sdu" in hostname and hostname != official and not hostname.endswith(f".{official}"):
                signals.append(self._signal("impersonation", "疑似冒充学校域名", 80, hostname))
                break

        return signals

    def _extract_and_analyze_urls(self, text: str) -> list[Signal]:
        signals: list[Signal] = []
        urls = re.findall(r"https?://[^\s<>\"']+", text, flags=re.IGNORECASE)
        for url in urls[:10]:
            signals.extend(self._analyze_url(url.rstrip(".,;，。；)）")))
        return signals

    def _analyze_text(self, text: str) -> list[Signal]:
        signals: list[Signal] = []
        for pattern in _URGENT_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                signals.append(self._signal("social-engineering", "发现紧迫性诱导", 35, match.group(0)))
                break
        for pattern in _CREDENTIAL_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                signals.append(self._signal("credential-request", "疑似索取敏感凭据", 65, match.group(0)))
                break
        return signals

    def _analyze_html(self, html: str, source_url: str | None) -> list[Signal]:
        parser = _HtmlFeatureParser()
        parser.feed(html)
        signals: list[Signal] = []

        if parser.password_inputs:
            signals.append(self._signal("credential-form", "页面包含密码输入框", 55, "input[type=password]"))
        if parser.iframes:
            signals.append(self._signal("embedded-content", "页面包含嵌入式页面", 20, f"iframe x{parser.iframes}"))
        if parser.meta_refreshes:
            signals.append(self._signal("redirect", "页面包含自动跳转", 40, "meta refresh"))

        source_host = (urlsplit(source_url).hostname or "").lower() if source_url else ""
        for action in parser.form_actions[:10]:
            absolute_action = urljoin(source_url or "", action)
            action_host = (urlsplit(absolute_action).hostname or "").lower()
            if source_host and action_host and action_host != source_host:
                signals.append(self._signal("form-action", "表单提交到其他域名", 70, absolute_action[:200]))

        return signals

    @staticmethod
    def _strip_markup(html: str) -> str:
        return re.sub(r"<[^>]+>", " ", html)

    @staticmethod
    def _score(signals: list[Signal]) -> float:
        if not signals:
            return 5.0
        severities = sorted((signal.severity for signal in signals), reverse=True)
        return float(min(100, severities[0] + sum(value * 0.18 for value in severities[1:])))

    @staticmethod
    def _signal(category: str, title: str, severity: int, evidence: str) -> Signal:
        return Signal(
            category=category,
            title=title,
            description=f"基线规则检测到：{title}。该结果需要与AI判断和正式安全规则共同验证。",
            severity=severity,
            evidence=evidence,
        )

