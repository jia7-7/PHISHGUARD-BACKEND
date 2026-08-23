"""Regression tests for the final Phase 2 review findings."""
from __future__ import annotations

import json
import socket

import pytest
from pydantic import ValidationError

import phishing_rule_detector.normalizer as normalizer_module
from phishing_rule_detector.config_loader import RuleDefinition
from phishing_rule_detector.detector import detect
from phishing_rule_detector.rules.common import ExtractedURL, make_evidence


def _rule_ids(result: dict) -> set[str]:
    return {item["rule_id"] for item in result.get("evidence", [])}


def test_trusted_service_forbidden_password_is_high_risk():
    result = detect(
        input_text=(
            "<form action='https://www.wjx.cn/jq/123'>"
            "<input type='password' name='password'>"
            "</form>"
        ),
        input_type="html",
        base_url="https://www.wjx.cn/jq/123",
    )

    assert result["risk"]["level"] == "high"
    assert "TRUSTED_SERVICE_FORBIDDEN_FIELD" in _rule_ids(result)


def test_trusted_service_forbidden_token_is_high_risk():
    result = detect(
        input_text=(
            "<form action='https://www.wjx.cn/jq/123'>"
            "<input type='text' name='token'>"
            "</form>"
        ),
        input_type="html",
        base_url="https://www.wjx.cn/jq/123",
    )

    assert result["risk"]["level"] == "high"
    assert "TRUSTED_SERVICE_FORBIDDEN_FIELD" in _rule_ids(result)


def test_complete_result_redacts_userinfo_password():
    secret = "SuperSecret"
    result = detect(
        f"https://sdu.edu.cn:{secret}@evil.com/login?token=abc",
        "url",
    )

    assert secret not in json.dumps(result, ensure_ascii=False)
    assert "[REDACTED]" in json.dumps(result, ensure_ascii=False)


def test_complete_result_redacts_short_brand_userinfo_password():
    secret = "AnotherSecret"
    result = detect(f"https://sdu:{secret}@evil.com/login", "url")

    assert secret not in json.dumps(result, ensure_ascii=False)


def test_complete_result_redacts_fragment_secret_and_personal_data():
    result = detect(
        "scan",
        "text",
        context={
            "qr_urls": [
                "https://evil.example/login?phone=13800138000&email=alice@example.com"
                "&id=370102200001011234#token=FragmentSecret"
            ]
        },
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert "FragmentSecret" not in serialized
    assert "13800138000" not in serialized
    assert "alice@example.com" not in serialized
    assert "370102200001011234" not in serialized
    assert "138****8000" in serialized
    assert "a***e@example.com" in serialized
    assert "370*************34" in serialized


def test_original_p07_detects_time_limit_and_disable_threat():
    result = detect(
        "请在12小时内完成验证，否则账号将被停用",
        "sms",
    )

    assert result["risk"]["level"] in {"medium", "high", "critical"}
    assert {"TIME_LIMIT_PRESSURE", "ACCOUNT_DISABLE_THREAT"} <= _rule_ids(result)
    subjects = {
        item["subject_id"]
        for item in result["evidence"]
        if item["rule_id"] in {"TIME_LIMIT_PRESSURE", "ACCOUNT_DISABLE_THREAT"}
    }
    assert subjects == {"sentence:0"}


def test_wording_subject_id_tracks_matching_sentence():
    result = detect("这是普通通知。请立即验证您的账号。", "text")
    evidence = next(
        item for item in result["evidence"]
        if item["rule_id"] == "URGENT_VERIFY_WORDING"
    )

    assert evidence["subject_id"] == "sentence:1"


def test_malformed_port_returns_warning_and_keeps_later_url():
    result = detect(
        "http://evil.com:bad/login and http://8.8.8.8/login",
        "text",
    )

    assert any(warning.startswith("URL_PARSE_ERROR:url:") for warning in result["warnings"])
    assert "URL_IP_DOMAIN" in _rule_ids(result)


def test_external_link_dominance_requires_page_origin():
    result = detect(
        (
            '<a href="https://a.example/a">a</a>'
            '<a href="https://b.example/b">b</a>'
            '<a href="https://c.example/c">c</a>'
        ),
        "html",
    )

    assert "EXTERNAL_LINK_DOMINANCE" not in _rule_ids(result)
    assert result["risk"]["level"] == "low"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "x", "severity": "nonsense", "group": "identity", "base_score": 1, "confidence": 0.5},
        {"title": "x", "severity": "low", "group": "bad", "base_score": 1, "confidence": 0.5},
        {"title": "x", "severity": "low", "group": "identity", "base_score": -1, "confidence": 0.5},
        {"title": "x", "severity": "low", "group": "identity", "base_score": 1, "confidence": 1.1},
    ],
)
def test_rule_definition_rejects_invalid_metadata(kwargs):
    with pytest.raises(ValidationError):
        RuleDefinition(**kwargs)


def test_rule_definition_contains_complete_metadata():
    definition = RuleDefinition(
        title="Suspicious URL",
        severity="high",
        group="identity",
        base_score=25,
        confidence=0.9,
        tags=["domain", "spoofing"],
    )

    assert definition.title == "Suspicious URL"
    assert definition.tags == ["domain", "spoofing"]


def test_make_evidence_uses_configured_rule_metadata():
    evidence = make_evidence(
        rule_id="URL_IP_DOMAIN",
        title="wrong title",
        severity="low",
        group="payload",
        base_score=1,
        confidence=0.8,
        reason="test",
        tags=["wrong"],
    )

    assert evidence.title == "IP 地址登录域名"
    assert evidence.severity == "medium"
    assert evidence.group == "transport"
    assert evidence.base_score == 12
    assert evidence.tags == ["ip", "auth"]


def test_extracted_url_has_assignable_stable_subject_id():
    extracted = ExtractedURL(
        raw="https://example.com",
        normalized="https://example.com",
        scheme="https",
        hostname="example.com",
        port=None,
        path="",
        query="",
        fragment="",
        registered_domain="example.com",
        is_official=False,
        subject_id="url:7",
    )

    assert extracted.subject_id == "url:7"


def test_detection_remains_offline_when_socket_is_blocked(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    result = detect("https://sdu.edu.cn/login", "url")
    assert result["success"] is True


def test_html_parser_fallback_uses_stable_warning(monkeypatch):
    original = normalizer_module.BeautifulSoup

    def fail_lxml(markup, parser):
        if parser == "lxml":
            raise RuntimeError("lxml unavailable")
        return original(markup, parser)

    monkeypatch.setattr(normalizer_module, "BeautifulSoup", fail_lxml)
    result = detect("<p>hello</p>", "html")

    assert "HTML_PARSER_FALLBACK" in result["warnings"]


def test_html_parse_failure_uses_stable_warning(monkeypatch):
    def fail_all(*_args, **_kwargs):
        raise RuntimeError("all parsers unavailable")

    monkeypatch.setattr(normalizer_module, "BeautifulSoup", fail_all)
    result = detect("<p>hello</p>", "html")

    assert "HTML_PARSE_ERROR" in result["warnings"]
