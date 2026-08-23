from __future__ import annotations

import re
from collections import OrderedDict

from app.config import Settings
from app.models import DetectorResult, Evidence, RiskLevel, Signal


_LEVEL_RANK = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class FusionResult:
    def __init__(
        self,
        *,
        score: int,
        level: RiskLevel,
        confidence: float,
        summary: str,
        evidence: list[Evidence],
        recommendations: list[str],
    ) -> None:
        self.score = score
        self.level = level
        self.confidence = confidence
        self.summary = summary
        self.evidence = evidence
        self.recommendations = recommendations


class RiskFusionService:
    """Evidence-first fusion that preserves the formal rule engine's safety floor."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fuse(self, results: list[DetectorResult]) -> FusionResult:
        if not results:
            raise ValueError("At least one detector result is required")

        rule_results = [item for item in results if item.family == "rules"]
        ai_results = [item for item in results if item.family == "ai"]
        evidence = self._merge_evidence(results)

        if rule_results:
            primary_rule = max(rule_results, key=lambda item: item.score)
            raw_score = primary_rule.score
            floor = self._rule_floor(rule_results)
            critical_lock = any(
                bool(item.metadata.get("critical_lock")) for item in rule_results
            )
            raw_score, semantic_floor = self._apply_semantic_evidence(raw_score, ai_results)
            floor = self._higher_level(floor, semantic_floor)
        else:
            raw_score = max(item.score for item in ai_results)
            floor = self._semantic_floor(ai_results)
            critical_lock = False

        if critical_lock:
            floor = RiskLevel.CRITICAL
        raw_score = max(raw_score, self._minimum_score(floor))

        allow_critical = floor == RiskLevel.CRITICAL
        score = max(0, min(100, round(raw_score)))
        if score >= self.settings.critical_threshold and not allow_critical:
            score = self.settings.critical_threshold - 1
        level = self._higher_level(self._level(score, allow_critical), floor)

        confidence = max(
            item.confidence if item.family == "rules" else item.confidence * 0.85
            for item in results
        )
        return FusionResult(
            score=score,
            level=level,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            summary=self._summary(level),
            evidence=evidence,
            recommendations=self._recommendations(level),
        )

    def _apply_semantic_evidence(
        self,
        rule_score: float,
        ai_results: list[DetectorResult],
    ) -> tuple[float, RiskLevel]:
        score = rule_score
        for result in ai_results:
            if result.metadata.get("score_kind") != "llm_semantic_evidence":
                continue
            positive_signal = max(0.0, result.score - 50.0) / 50.0
            adjustment = (
                positive_signal
                * self.settings.ai_max_score_adjustment
                * result.confidence
            )
            score += adjustment
        return score, self._semantic_floor(ai_results)

    @staticmethod
    def _semantic_floor(ai_results: list[DetectorResult]) -> RiskLevel:
        floor = RiskLevel.LOW
        for result in ai_results:
            if result.metadata.get("score_kind") != "llm_semantic_evidence":
                continue
            count = int(result.metadata.get("semantic_evidence_count") or 0)
            if result.score >= 85 and count >= 3 and result.confidence >= 0.65:
                floor = RiskLevel.HIGH
            elif result.score >= 65 and count >= 2 and result.confidence >= 0.5:
                floor = RiskFusionService._higher_level(floor, RiskLevel.MEDIUM)
        return floor

    @staticmethod
    def _rule_floor(results: list[DetectorResult]) -> RiskLevel:
        floor = RiskLevel.LOW
        for result in results:
            for key in ("level_floor", "rule_level"):
                value = result.metadata.get(key)
                try:
                    candidate = RiskLevel(value)
                except (TypeError, ValueError):
                    continue
                floor = RiskFusionService._higher_level(floor, candidate)
        return floor

    def _minimum_score(self, level: RiskLevel) -> int:
        return {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: self.settings.medium_threshold,
            RiskLevel.HIGH: 60,
            RiskLevel.CRITICAL: self.settings.critical_threshold,
        }[level]

    def _level(self, score: int, allow_critical: bool) -> RiskLevel:
        if allow_critical and score >= self.settings.critical_threshold:
            return RiskLevel.CRITICAL
        if score >= self.settings.high_threshold:
            return RiskLevel.HIGH
        if score >= self.settings.medium_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @staticmethod
    def _higher_level(first: RiskLevel, second: RiskLevel) -> RiskLevel:
        return first if _LEVEL_RANK[first] >= _LEVEL_RANK[second] else second

    @staticmethod
    def _merge_evidence(results: list[DetectorResult]) -> list[Evidence]:
        merged: OrderedDict[str, Evidence] = OrderedDict()
        for result in results:
            for signal in result.signals:
                key = RiskFusionService._evidence_key(signal)
                existing = merged.get(key)
                if existing:
                    if result.detector not in existing.sources:
                        existing.sources.append(result.detector)
                    if signal.severity > existing.severity:
                        existing.severity = signal.severity
                    continue
                merged[key] = Evidence(
                    category=signal.category,
                    title=signal.title,
                    description=signal.description,
                    severity=signal.severity,
                    evidence=signal.evidence,
                    location=signal.location,
                    sources=[result.detector],
                )
        return sorted(merged.values(), key=lambda item: item.severity, reverse=True)

    @staticmethod
    def _evidence_key(signal: Signal) -> str:
        value = signal.evidence or signal.title
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        return f"{signal.category}:{normalized}"

    @staticmethod
    def _summary(level: RiskLevel) -> str:
        return {
            RiskLevel.LOW: "暂未发现明显钓鱼特征，但重要操作前仍应核对发送方。",
            RiskLevel.MEDIUM: "发现可疑特征，请通过官方渠道进行二次确认。",
            RiskLevel.HIGH: "发现高风险钓鱼特征，请停止操作并按建议处置。",
            RiskLevel.CRITICAL: "多组独立高危证据已锁定，请立即停止操作并上报。",
        }[level]

    @staticmethod
    def _recommendations(level: RiskLevel) -> list[str]:
        if level == RiskLevel.CRITICAL:
            return [
                "立即停止点击、下载、登录或付款操作。",
                "保留截图和原始消息，并联系学校网络安全人员。",
                "如已输入密码或验证码，立即修改密码并检查账号活动。",
            ]
        if level == RiskLevel.HIGH:
            return [
                "不要点击链接、下载附件或输入账号密码。",
                "通过山东大学官网或官方电话独立核实通知。",
                "如已输入密码，请立即修改并联系学校网络安全人员。",
            ]
        if level == RiskLevel.MEDIUM:
            return [
                "暂停操作，并核对发送方地址与链接真实域名。",
                "不要使用消息中的联系方式核实，应改用官方渠道。",
            ]
        return [
            "检测结果仅供辅助判断，重要操作前仍需核对发送方。",
            "访问校园服务时优先从学校官网进入。",
        ]
