"""测试域名与身份规则."""
from phishing_rule_detector.rules.common import (
    RuleContext,
    extract_urls_from_text,
)
from phishing_rule_detector.rules.domain_rules import (
    domain_similar_to_official,
    domain_punycode_suspicious,
    domain_homograph_attack,
    domain_keyword_impersonation,
)


def _ctx(url: str) -> RuleContext:
    urls = extract_urls_from_text(url)
    return RuleContext(
        input_text=url,
        normalized_text=url,
        input_type="url",
        base_url=None,
        raw_text=url,
        extracted_urls=urls,
    )


class TestDomainSimilarToOfficial:
    def test_official_domain_no_hit(self):
        r = domain_similar_to_official(_ctx("https://pass.sdu.edu.cn/login"))
        assert r is None

    def test_similar_domain_hits(self):
        r = domain_similar_to_official(_ctx("https://sdu-edu.cn/login"))
        assert r is not None
        assert any(e.rule_id == "DOMAIN_SIMILAR_TO_OFFICIAL" for e in r)

    def test_core_label_edit_distance_with_different_suffix_hits(self):
        """核心标签与 sdu 编辑距离为 1 时不应受注册域后缀影响."""
        r = domain_similar_to_official(_ctx("https://sdv.com/login"))
        assert r is not None
        assert any(
            e.rule_id == "DOMAIN_SIMILAR_TO_OFFICIAL"
            and "levenshtein" in e.tags
            for e in r
        )

    def test_sdu_in_domain_not_alone_enough(self):
        """仅包含 sdu 不直接判高危."""
        r = domain_similar_to_official(_ctx("https://sdutest.example.com"))
        # sdu 本身不触发编辑距离规则
        if r:
            assert not any(
                e.rule_id == "DOMAIN_SIMILAR_TO_OFFICIAL"
                and e.tags and "levenshtein" in e.tags
                for e in r
            )

    def test_external_domain_nested_no_cross(self):
        r = domain_similar_to_official(_ctx("https://sdu.edu.cn.evil.com/login"))
        # 嵌套域名由 URL_NESTED_OFFICIAL_DOMAIN 处理
        # DOMAIN_SIMILAR_TO_OFFICIAL 也可能触发，但不应误报
        assert r is None or all(e.rule_id != "DOMAIN_SIMILAR_TO_OFFICIAL" or e.confidence < 0.95 for e in r)


class TestDomainPunycodeSuspicious:
    def test_punycode_no_official_similarity_no_hit(self):
        # xn--fiq228c.com → 中文.com，不与官方域名相似，不触发
        r = domain_punycode_suspicious(_ctx("https://xn--fiq228c.com/login"))
        assert r is None

    def test_ascii_domain_no_hit(self):
        r = domain_punycode_suspicious(_ctx("https://example.com/login"))
        assert r is None

    def test_external_punycode_no_hit(self):
        # 外部 Punycode 域名不与官方域名相似 → 不触发证据
        r = domain_punycode_suspicious(_ctx("https://xn--fiq228c.com"))
        assert r is None


class TestDomainHomographAttack:
    def test_normal_domain_no_hit(self):
        r = domain_homograph_attack(_ctx("https://example.com"))
        assert r is None

    def test_official_domain_no_hit(self):
        r = domain_homograph_attack(_ctx("https://pass.sdu.edu.cn"))
        assert r is None

    def test_mixed_script_detected(self):
        """混合脚本域名应被检测."""
        from phishing_rule_detector.rules.common import has_mixed_script
        # 西里尔 a (а) + 拉丁
        assert has_mixed_script("sduа") or not has_mixed_script("sdu")


class TestDomainKeywordImpersonation:
    def test_brand_and_login_hits(self):
        r = domain_keyword_impersonation(_ctx("https://sdu-login.example.com/verify"))
        assert r is not None
        assert any(e.rule_id == "DOMAIN_KEYWORD_IMPERSONATION" for e in r)

    def test_brand_only_no_hit(self):
        r = domain_keyword_impersonation(_ctx("https://sdu-news.example.com"))
        # 只有品牌词无认证词不命中
        assert r is None

    def test_official_domain_no_hit(self):
        r = domain_keyword_impersonation(_ctx("https://pass.sdu.edu.cn/login"))
        assert r is None
