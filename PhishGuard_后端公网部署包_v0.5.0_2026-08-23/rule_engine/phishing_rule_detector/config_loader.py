"""YAML 配置加载与 Pydantic 校验."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from phishing_rule_detector.models import EVIDENCE_GROUPS, SEVERITY_LEVELS


_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Scoring Config ──────────────────────────────────────────


class StrictConfigModel(BaseModel):
    """配置模型默认拒绝拼写错误或未支持字段."""

    model_config = ConfigDict(extra="forbid")


class LevelThresholds(StrictConfigModel):
    low_max: int = 29
    medium_max: int = 59
    high_max: int = 79
    critical_min: int = 80

    @model_validator(mode="after")
    def validate_order(self) -> "LevelThresholds":
        if not (
            0 <= self.low_max < self.medium_max < self.high_max
            and self.critical_min == self.high_max + 1
            and self.critical_min <= 100
        ):
            raise ValueError("风险等级阈值必须递增且 critical_min 紧接 high_max")
        return self


class LevelRanges(StrictConfigModel):
    low: list[int] = [0, 29]
    medium: list[int] = [30, 59]
    high: list[int] = [60, 79]
    critical: list[int] = [80, 100]

    @field_validator("low", "medium", "high", "critical")
    @classmethod
    def validate_range(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or value[0] > value[1]:
            raise ValueError("等级区间必须是递增的 [min, max]")
        return value

    @model_validator(mode="after")
    def validate_contiguous(self) -> "LevelRanges":
        ranges = [self.low, self.medium, self.high, self.critical]
        if ranges[0][0] != 0 or ranges[-1][1] != 100:
            raise ValueError("等级区间必须覆盖 0 到 100")
        if any(left[1] + 1 != right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("等级区间必须连续且不得重叠")
        return self


class LevelFloorScores(StrictConfigModel):
    low: int = Field(default=0, ge=0, le=100)
    medium: int = Field(default=30, ge=0, le=100)
    high: int = Field(default=60, ge=0, le=100)
    critical: int = Field(default=80, ge=0, le=100)


class QualifyingThresholds(StrictConfigModel):
    medium: float = Field(default=0.75, ge=0.0, le=1.0)
    high: float = Field(default=0.80, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> "QualifyingThresholds":
        if self.medium > self.high:
            raise ValueError("medium 合格阈值不得高于 high")
        return self


class ScoringConfig(StrictConfigModel):
    schema_version: str
    rule_version: str
    updated_at: str
    level_thresholds: LevelThresholds = Field(
        default_factory=LevelThresholds
    )
    level_ranges: LevelRanges = Field(default_factory=LevelRanges)
    level_floor_scores: LevelFloorScores = Field(
        default_factory=LevelFloorScores
    )
    qualifying_thresholds: QualifyingThresholds = Field(
        default_factory=QualifyingThresholds
    )
    group_attenuation: list[float] = Field(
        default=[1.0, 0.5, 0.25], min_length=1
    )
    group_score_caps: dict[str, int] = Field(default_factory=dict)
    max_raw_score: int = Field(default=100, ge=1, le=100)

    @field_validator("group_attenuation")
    @classmethod
    def validate_attenuation(cls, value: list[float]) -> list[float]:
        if any(factor <= 0 or factor > 1 for factor in value):
            raise ValueError("同组衰减系数必须在 (0, 1] 范围内")
        if any(left < right for left, right in zip(value, value[1:])):
            raise ValueError("同组衰减系数必须单调不增")
        return value

    @field_validator("group_score_caps")
    @classmethod
    def validate_group_caps(cls, value: dict[str, int]) -> dict[str, int]:
        if any(cap <= 0 or cap > 100 for cap in value.values()):
            raise ValueError("证据组上限必须在 [1, 100] 范围内")
        return value

    @model_validator(mode="after")
    def validate_threshold_alignment(self) -> "ScoringConfig":
        thresholds = self.level_thresholds
        ranges = self.level_ranges
        expected = (
            ranges.low[1],
            ranges.medium[1],
            ranges.high[1],
            ranges.critical[0],
        )
        actual = (
            thresholds.low_max,
            thresholds.medium_max,
            thresholds.high_max,
            thresholds.critical_min,
        )
        if actual != expected:
            raise ValueError("等级阈值必须与等级区间边界一致")

        for level in ("low", "medium", "high", "critical"):
            floor_score = getattr(self.level_floor_scores, level)
            level_range = getattr(ranges, level)
            if not level_range[0] <= floor_score <= level_range[1]:
                raise ValueError(
                    f"{level} 展示分下限必须位于对应等级区间内"
                )
        return self


# ── Trusted Domains Config ──────────────────────────────────


class TrustedService(StrictConfigModel):
    domains: list[str]
    service: str
    allowed_fields: list[str] = Field(default_factory=list)
    forbidden_fields: list[str] = Field(default_factory=list)


class TrustedDomainsConfig(StrictConfigModel):
    schema_version: str
    updated_at: str
    official_domains: list[str] = Field(default_factory=list)
    trusted_services: list[TrustedService] = Field(default_factory=list)
    short_link_domains: list[str] = Field(default_factory=list)
    allowed_non_standard_ports: list[dict[str, Any]] = Field(
        default_factory=list
    )


# ── Rules Config ────────────────────────────────────────────


class SuppressionEntry(StrictConfigModel):
    suppresses: list[str]
    scope: Literal["subject", "global"] = "subject"  # 默认为 subject 级作用域


class RuleDefinition(StrictConfigModel):
    """单条规则的可配置元数据."""
    enabled: bool = True
    title: str = ""
    severity: SEVERITY_LEVELS = "low"
    group: EVIDENCE_GROUPS
    base_score: int = Field(default=0, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    dynamic_scoring: bool = False


class RulesConfig(StrictConfigModel):
    schema_version: str
    updated_at: str
    dangerous_extensions: list[str] = Field(default_factory=list)
    rule_suppression: dict[str, SuppressionEntry] = Field(
        default_factory=dict
    )
    rule_definitions: dict[str, RuleDefinition] = Field(
        default_factory=dict
    )
    brand_keywords: list[str] = Field(default_factory=list)
    login_keywords: list[str] = Field(default_factory=list)
    wording_patterns: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    sensitive_info_fields: list[str] = Field(default_factory=list)
    secret_fields: list[str] = Field(default_factory=list)
    max_subdomain_levels: int = 4

    @field_validator("rule_suppression", mode="before")
    @classmethod
    def _normalize_old_suppression_format(
        cls, v: dict[str, Any]
    ) -> dict[str, Any]:
        """将旧版列表格式转换为新版对象格式（审查 Fix 6）.

        旧格式: RULE_A: [RULE_B, RULE_C]
        新格式: RULE_A: {suppresses: [RULE_B, RULE_C], scope: subject}
        """
        if not isinstance(v, dict):
            return v
        result: dict[str, Any] = {}
        for rule_id, entry in v.items():
            if isinstance(entry, list):
                # 旧版列表格式 → 转换为 SuppressionEntry
                result[rule_id] = {
                    "suppresses": entry,
                    "scope": "subject",
                }
            else:
                result[rule_id] = entry
        return result


# ── App Config (聚合) ───────────────────────────────────────


class AppConfig(StrictConfigModel):
    scoring: ScoringConfig
    trusted: TrustedDomainsConfig
    rules: RulesConfig


# ── 单例加载 ────────────────────────────────────────────────

_config: AppConfig | None = None


def load_config(force_reload: bool = False) -> AppConfig:
    """加载并校验所有 YAML 配置，返回 AppConfig 实例."""
    global _config
    if _config is not None and not force_reload:
        return _config

    scoring_data = _load_yaml("scoring.yaml")
    trusted_data = _load_yaml("trusted_domains.yaml")
    rules_data = _load_yaml("rules.yaml")

    _config = AppConfig(
        scoring=ScoringConfig(**scoring_data),
        trusted=TrustedDomainsConfig(**trusted_data),
        rules=RulesConfig(**rules_data),
    )
    return _config
