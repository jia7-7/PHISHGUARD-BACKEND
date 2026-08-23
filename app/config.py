from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _get_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _default_rule_engine_path() -> str:
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / "rule_engine")


def _default_runtime_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / ".runtime")


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "PhishGuard API"
    app_version: str = "0.5.0"
    runtime_dir: str = ""
    max_text_chars: int = 100_000
    max_upload_bytes: int = 2 * 1024 * 1024
    medium_threshold: int = 30
    high_threshold: int = 70
    critical_threshold: int = 80
    ai_max_score_adjustment: int = 15
    official_domains: tuple[str, ...] = ("sdu.edu.cn",)
    rule_engine_path: str = ""
    ai_service_url: str = ""
    ai_service_token: str = ""
    ai_timeout_seconds: float = 45.0
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.runtime_dir:
            object.__setattr__(self, "runtime_dir", _default_runtime_dir())
        if not self.rule_engine_path:
            object.__setattr__(self, "rule_engine_path", _default_rule_engine_path())
        if self.max_text_chars <= 0 or self.max_upload_bytes <= 0:
            raise ValueError("Input limits must be positive")
        if not 0 <= self.medium_threshold < self.high_threshold < self.critical_threshold <= 100:
            raise ValueError(
                "Risk thresholds must satisfy 0 <= medium < high < critical <= 100"
            )
        if not 0 <= self.ai_max_score_adjustment <= 30:
            raise ValueError("AI score adjustment must be between 0 and 30")
        if self.ai_timeout_seconds <= 0:
            raise ValueError("AI timeout must be positive")


def load_settings() -> Settings:
    return Settings(
        runtime_dir=os.getenv("PHISHGUARD_RUNTIME_DIR", _default_runtime_dir()),
        max_text_chars=_get_int("PHISHGUARD_MAX_TEXT_CHARS", 100_000),
        max_upload_bytes=_get_int("PHISHGUARD_MAX_UPLOAD_BYTES", 2 * 1024 * 1024),
        medium_threshold=_get_int("PHISHGUARD_MEDIUM_THRESHOLD", 30),
        high_threshold=_get_int("PHISHGUARD_HIGH_THRESHOLD", 70),
        critical_threshold=_get_int("PHISHGUARD_CRITICAL_THRESHOLD", 80),
        ai_max_score_adjustment=_get_int("PHISHGUARD_AI_MAX_SCORE_ADJUSTMENT", 15),
        official_domains=_get_csv("PHISHGUARD_OFFICIAL_DOMAINS", ("sdu.edu.cn",)),
        rule_engine_path=os.getenv(
            "PHISHGUARD_RULE_ENGINE_PATH", _default_rule_engine_path()
        ).strip(),
        ai_service_url=os.getenv("PHISHGUARD_AI_SERVICE_URL", "").strip(),
        ai_service_token=os.getenv("PHISHGUARD_AI_SERVICE_TOKEN", "").strip(),
        ai_timeout_seconds=_get_float("PHISHGUARD_AI_TIMEOUT_SECONDS", 45.0),
        cors_origins=_get_csv("PHISHGUARD_CORS_ORIGINS"),
    )
