"""测试配置加载与校验."""
import pytest
from phishing_rule_detector.config_loader import (
    ScoringConfig,
    load_config,
)


class TestConfigLoading:
    def test_load_config_succeeds(self):
        config = load_config()
        assert config is not None
        assert config.scoring.rule_version == "3.0.0"

    def test_official_domains_loaded(self):
        config = load_config()
        assert "sdu.edu.cn" in config.trusted.official_domains
        # 应只有 sdu.edu.cn 一个根域名
        assert len(config.trusted.official_domains) == 1

    def test_no_other_domains(self):
        """确认 trusted_domains.yaml 中只有 sdu.edu.cn."""
        config = load_config()
        assert config.trusted.official_domains == ["sdu.edu.cn"]

    def test_rule_suppression_loaded(self):
        config = load_config()
        supp = config.rules.rule_suppression
        assert "PASSWORD_FORM_UNTRUSTED_TARGET" in supp
        assert (
            "HTML_PASSWORD_INPUT"
            in supp["PASSWORD_FORM_UNTRUSTED_TARGET"].suppresses
        )
        # DOMAIN_HOMOGRAPH_ATTACK 抑制两条泛化规则
        assert (
            "DOMAIN_SIMILAR_TO_OFFICIAL"
            in supp["DOMAIN_HOMOGRAPH_ATTACK"].suppresses
        )
        assert (
            "DOMAIN_KEYWORD_IMPERSONATION"
            in supp["DOMAIN_HOMOGRAPH_ATTACK"].suppresses
        )

    def test_dangerous_extensions(self):
        config = load_config()
        assert ".exe" in config.rules.dangerous_extensions
        assert ".pdf" not in config.rules.dangerous_extensions
        assert ".scr" in config.rules.dangerous_extensions
        assert len(config.rules.dangerous_extensions) == 16

    def test_scoring_thresholds(self):
        config = load_config()
        assert config.scoring.level_thresholds.low_max == 29
        assert config.scoring.qualifying_thresholds.high == 0.80

    def test_trusted_services_only_wjx(self):
        config = load_config()
        assert len(config.trusted.trusted_services) == 1
        svc = config.trusted.trusted_services[0]
        assert svc.service == "survey"
        assert "wjx.cn" in svc.domains
        assert "wjx.com" in svc.domains

    def test_trusted_service_allowed_fields(self):
        config = load_config()
        svc = config.trusted.trusted_services[0]
        assert svc.allowed_fields == ["student_id", "phone", "email"]
        assert "password" not in svc.allowed_fields

    def test_trusted_service_forbidden_fields(self):
        config = load_config()
        svc = config.trusted.trusted_services[0]
        assert "password" in svc.forbidden_fields
        assert "sms_code" in svc.forbidden_fields
        assert "bank_card" in svc.forbidden_fields
        assert "token" in svc.forbidden_fields

    def test_short_link_domains(self):
        config = load_config()
        assert len(config.trusted.short_link_domains) == 6
        assert "bit.ly" in config.trusted.short_link_domains
        assert "t.cn" in config.trusted.short_link_domains

    def test_brand_keywords(self):
        config = load_config()
        assert "山东大学" in config.rules.brand_keywords
        assert "sdu" in config.rules.brand_keywords
        assert "统一认证" in config.rules.brand_keywords

    def test_wording_patterns_loaded(self):
        config = load_config()
        wp = config.rules.wording_patterns
        assert "urgent_verify" in wp
        assert "time_limit" in wp
        assert "account_disable" in wp
        assert "reward_bait" in wp
        assert "security_upgrade" in wp
        assert "credential_request" in wp
        # 验证每个模式有 keywords 和 confidence
        for name, pattern in wp.items():
            assert "keywords" in pattern, f"{name} 缺少 keywords"
            assert "confidence" in pattern, f"{name} 缺少 confidence"

    def test_secret_fields(self):
        config = load_config()
        assert "password" in config.rules.secret_fields
        assert "验证码" in config.rules.secret_fields
        assert "bank_card" in config.rules.secret_fields

    def test_group_score_caps(self):
        config = load_config()
        caps = config.scoring.group_score_caps
        assert caps["identity"] == 35
        assert caps["credential"] == 40
        assert caps["navigation"] == 25
        assert caps["social"] == 20
        assert caps["transport"] == 15
        assert caps["payload"] == 30

    def test_group_attenuation(self):
        config = load_config()
        att = config.scoring.group_attenuation
        assert att == [1.0, 0.5, 0.25]

    def test_reload_config(self):
        """force_reload 应该返回新实例."""
        c1 = load_config()
        c2 = load_config(force_reload=True)
        assert c1 is not c2  # force_reload 应创建新对象而非返回缓存

    def test_cache_config(self):
        """默认不 force_reload 应返回缓存实例."""
        c1 = load_config()
        c2 = load_config()
        assert c1 is c2  # 缓存命中，同一个对象

    def test_trust_policy_observes_force_reload(self):
        import phishing_rule_detector.trust_policy as trust_policy

        first = load_config(force_reload=True)
        assert trust_policy._get_config() is first

        second = load_config(force_reload=True)
        assert trust_policy._get_config() is second

    @pytest.mark.parametrize(
        "overrides",
        [
            {"group_attenuation": []},
            {
                "level_ranges": {
                    "low": [29, 0],
                    "medium": [30, 59],
                    "high": [60, 79],
                    "critical": [80, 100],
                }
            },
            {"max_raw_score": 0},
            {"unknown_setting": True},
        ],
    )
    def test_invalid_scoring_config_rejected(self, overrides):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScoringConfig(
                schema_version="3.0.0",
                rule_version="3.0.0",
                updated_at="2026-08-11",
                **overrides,
            )

    def test_level_floor_score_must_stay_inside_level_range(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScoringConfig(
                schema_version="3.0.0",
                rule_version="3.0.0",
                updated_at="2026-08-11",
                level_floor_scores={
                    "low": 0,
                    "medium": 30,
                    "high": 80,
                    "critical": 80,
                },
            )

    # ── Fix 5: scope 严格校验 ──

    def test_suppression_scope_default_subject(self):
        """scope 默认为 subject."""
        from phishing_rule_detector.config_loader import SuppressionEntry
        entry = SuppressionEntry(suppresses=["RULE_A"])
        assert entry.scope == "subject"

    def test_suppression_scope_global(self):
        """scope 可为 global."""
        from phishing_rule_detector.config_loader import SuppressionEntry
        entry = SuppressionEntry(suppresses=["RULE_A"], scope="global")
        assert entry.scope == "global"

    def test_suppression_scope_invalid_rejected(self):
        """无效 scope 值（如 globla）应被 Pydantic 拒绝."""
        import pytest
        from pydantic import ValidationError
        from phishing_rule_detector.config_loader import SuppressionEntry
        with pytest.raises(ValidationError):
            SuppressionEntry(suppresses=["RULE_A"], scope="globla")

    def test_suppression_scope_empty_string_rejected(self):
        """空字符串 scope 应被拒绝."""
        import pytest
        from pydantic import ValidationError
        from phishing_rule_detector.config_loader import SuppressionEntry
        with pytest.raises(ValidationError):
            SuppressionEntry(suppresses=["RULE_A"], scope="")

    # ── Fix 6: 旧版列表格式向后兼容 ──

    def test_old_list_format_suppression_accepted(self):
        """旧版列表格式应被接受并转换为新版 SuppressionEntry."""
        import yaml
        from phishing_rule_detector.config_loader import RulesConfig
        yaml_data = yaml.safe_load("""
schema_version: "2.1.0"
updated_at: "2026-08-11"
rule_suppression:
  RULE_A:
    - RULE_B
    - RULE_C
""")
        config = RulesConfig(**yaml_data)
        entry = config.rule_suppression["RULE_A"]
        assert entry.suppresses == ["RULE_B", "RULE_C"]
        assert entry.scope == "subject"

    def test_old_and_new_format_mixed(self):
        """混合旧版列表和新版对象格式应均可正常工作."""
        import yaml
        from phishing_rule_detector.config_loader import RulesConfig
        yaml_data = yaml.safe_load("""
schema_version: "2.1.0"
updated_at: "2026-08-11"
rule_suppression:
  RULE_A:
    - RULE_B
  RULE_C:
    suppresses:
      - RULE_D
    scope: global
""")
        config = RulesConfig(**yaml_data)
        assert config.rule_suppression["RULE_A"].suppresses == ["RULE_B"]
        assert config.rule_suppression["RULE_A"].scope == "subject"
        assert config.rule_suppression["RULE_C"].suppresses == ["RULE_D"]
        assert config.rule_suppression["RULE_C"].scope == "global"
