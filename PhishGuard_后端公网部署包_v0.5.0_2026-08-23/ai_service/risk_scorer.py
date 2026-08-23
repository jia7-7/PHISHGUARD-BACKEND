"""Risk scorer — structured rule_ids, strict type boundary"""
import hashlib, logging, re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

EVIDENCE_SOURCE_LLM="llm"; EVIDENCE_SOURCE_DET_URL="deterministic_url"
EVIDENCE_SOURCE_DET_HDR="deterministic_header"; EVIDENCE_SOURCE_KW="keyword"; EVIDENCE_SOURCE_HTML="html_parser"
SCORABLE_SOURCES={EVIDENCE_SOURCE_LLM,EVIDENCE_SOURCE_DET_URL,EVIDENCE_SOURCE_DET_HDR,EVIDENCE_SOURCE_KW,EVIDENCE_SOURCE_HTML}
RULE_ONLY_SOURCES={EVIDENCE_SOURCE_DET_URL,EVIDENCE_SOURCE_DET_HDR,EVIDENCE_SOURCE_KW,EVIDENCE_SOURCE_HTML}

def canonical_evidence_key(ev:Dict)->str:
    q=(ev.get("quote","")or"").strip(); q=re.sub(r'\s+','',q)
    for f,h in [('：',':'),('（','('),('）',')'),('，',','),('。','.'),('！','!'),('？','?'),('；',';'),('＠','@'),('．','.')]: q=q.replace(f,h)
    return hashlib.sha256(f"{q.lower()}|{ev.get('type','')}".encode()).hexdigest()

@dataclass(frozen=True)
class ScoreResult:
    score:int; level:str; level_label:str; level_color:str; is_phishing:bool; confidence:float; score_breakdown:Dict; strategy:str
    # v5.11.5.1: promotion tracking
    promotion_applied: bool = False
    promotion_rule: str = ""
    pre_promotion_score: int = 0
    promotion_score: int = 0
    promotion_evidence_types: tuple = ()
    # v5.11.5.2: deterministic anchor tracking
    promotion_anchor_source: str = ""
    promotion_anchor_rule_ids: tuple = ()
    def to_dict(self)->Dict: return {"risk_score":self.score,"risk_level":self.level,"risk_level_label":self.level_label,"level_color":self.level_color,"is_phishing":self.is_phishing,"confidence":self.confidence,"score_breakdown":self.score_breakdown,"strategy":self.strategy,"promotion_applied":self.promotion_applied,"promotion_rule":self.promotion_rule,"pre_promotion_score":self.pre_promotion_score,"promotion_score":self.promotion_score,"promotion_evidence_types":list(self.promotion_evidence_types),"promotion_anchor_source":self.promotion_anchor_source,"promotion_anchor_rule_ids":list(self.promotion_anchor_rule_ids)}

class RiskScorer:
    VERSION="5.11.5.12.3"
    RULE_SCORES={
        "url_sdu_impersonation_and_suspicious_tld":85,"url_sdu_impersonation":60,
        "url_userinfo_deception":80,"url_valid_ip_host":40,"url_punycode_impersonation":70,
        "url_shortener_only":20,"url_ddns_service":35,"url_suspicious_tld_only":30,
        "auth_multiple_failures":45,"from_replyto_mismatch":30,"sdu_domain_impersonation":60,
    }
    # Email combination rules: when multiple rule_ids are present, use combined score
    EMAIL_COMBOS = {
        ("sdu_domain_impersonation","auth_multiple_failures","from_replyto_mismatch"):85,
        ("sdu_domain_impersonation","auth_multiple_failures"):80,
        ("sdu_domain_impersonation","from_replyto_mismatch"):75,
        ("auth_multiple_failures","from_replyto_mismatch"):55,
    }
    # v5.11.5: semantic composite patterns from verified evidence types
    # Extended to capture dangerous multi-indicator combinations for improved recall
    COMPOSITE_PATTERNS = [
        # (type_set, bonus, label)
        # ═══ Two-indicator patterns (baseline: 20-40) ═══
        ({"urgency","info_request"}, 35, "urgency_with_info_request"),
        ({"urgency","suspicious_link"}, 40, "urgency_with_suspicious_link"),
        ({"sender_anomaly","info_request"}, 30, "sender_anomaly_with_info_request"),
        ({"domain_anomaly","suspicious_link"}, 35, "domain_anomaly_with_suspicious_link"),
        ({"domain_anomaly","sender_anomaly"}, 25, "domain_and_sender_anomaly"),
        ({"sender_anomaly","urgency"}, 20, "sender_anomaly_with_urgency"),
        # v5.11.5: new two-indicator patterns
        ({"impersonation","info_request"}, 35, "impersonation_with_info_request"),
        ({"impersonation","suspicious_link"}, 35, "impersonation_with_suspicious_link"),
        ({"credential_request","urgency"}, 40, "credential_request_with_urgency"),
        ({"payment_request","urgency"}, 40, "payment_request_with_urgency"),
        ({"secrecy","payment_request"}, 35, "secrecy_with_payment_request"),
        ({"reward_lure","payment_request"}, 30, "reward_lure_with_payment_request"),
        ({"secrecy","info_request"}, 30, "secrecy_with_info_request"),
        # ═══ Three-indicator patterns (bonus: 45-50) ═══
        # Classic phishing triad
        ({"domain_anomaly","suspicious_link","urgency"}, 50, "phishing_triad_domain_link_urgency"),
        ({"sender_anomaly","suspicious_link","urgency"}, 50, "phishing_triad_sender_link_urgency"),
        ({"domain_anomaly","sender_anomaly","suspicious_link"}, 45, "triple_anomaly_link"),
        # Urgent credential/payment phishing
        ({"urgency","info_request","suspicious_link"}, 50, "urgent_info_link"),
        ({"credential_request","urgency","suspicious_link"}, 50, "credential_urgency_link"),
        ({"payment_request","urgency","suspicious_link"}, 50, "payment_urgency_link"),
        # Impersonation chains
        ({"impersonation","credential_request","urgency"}, 50, "impersonation_credential_urgency"),
        ({"impersonation","payment_request","urgency"}, 50, "impersonation_payment_urgency"),
        # ═══ Four-indicator patterns (bonus: 50) ═══
        ({"domain_anomaly","sender_anomaly","suspicious_link","urgency"}, 50, "quad_anomaly"),
    ]
    def __init__(self):
        from config import RISK_SCORING_CONFIG
        self.config=RISK_SCORING_CONFIG; self.dimensions=self.config["dimensions"]; self.levels=self.config["risk_levels"]
    def score(self, validated=None, evidence=None, strategy="weighted",**kwargs)->ScoreResult:
        evidence=evidence or []
        from models import ValidatedFeatureSet as VFS
        if validated is not None and not isinstance(validated,VFS):
            raise TypeError(f"RiskScorer.score() requires ValidatedFeatureSet or None, got {type(validated).__name__}")
        if kwargs: raise TypeError(f"Unexpected args: {list(kwargs)}")
        usable=[e for e in evidence if e.get("source") in (RULE_ONLY_SOURCES if strategy=="rule_only" else SCORABLE_SOURCES)]
        usable=self._dedup(usable)
        if strategy=="weighted": score,bd,conf,promo=self._weighted(validated,usable)
        elif strategy=="rule_only": score,bd,conf=self._rule_only(usable); promo=(False,"",0,0,(),"",[])
        elif strategy=="hybrid": score,bd,conf,promo=self._hybrid(validated,usable)
        else: raise ValueError(f"Unknown: {strategy}")
        score=max(0,min(100,int(round(score))))
        level,label=self._to_level(score); info=self.levels.get(level,{})
        return ScoreResult(score=score,level=level,level_label=label,level_color=info.get("color","#000"),is_phishing=(level=="high"),confidence=round(min(conf,1.0),4),score_breakdown=bd,strategy=strategy,promotion_applied=promo[0],promotion_rule=promo[1],pre_promotion_score=promo[2],promotion_score=promo[3],promotion_evidence_types=promo[4],promotion_anchor_source=promo[5],promotion_anchor_rule_ids=tuple(promo[6]) if promo[6] else ())
    def _weighted(self,validated,evidence):
        """v5.11: coverage-aware scoring — no renormalization, composite evidence bonus, coverage cap"""
        dims=self._read_v(validated); bd={}; total=0.0; tw=0.0; sc=0; unk=0
        for dk,dc in self.dimensions.items():
            w=dc["weight"]; info=dims.get(dk,{"score":None,"status":"unknown"})
            if info["status"]=="unknown": bd[dk]={"raw_score":None,"weight":w,"weighted_score":None,"status":"unknown"}; unk+=1; continue
            s=info.get("score",0); ws=s*w; total+=ws; tw+=w; sc+=1
            bd[dk]={"raw_score":s,"weight":w,"weighted_score":round(ws,2),"status":"scored"}
        # v5.11: composite evidence bonus from verified evidence type combinations
        composite_bonus, composite_rules = self._composite_evidence_bonus(evidence)
        # v5.11: coverage-aware — no renormalization
        # With full coverage (tw>=0.50): use raw weighted sum
        # With partial coverage (0.25<=tw<0.50): moderate boost for partial information
        # With minimal coverage (tw<0.25): no boost, apply coverage cap
        if tw >= 0.50:
            weighted_final = total  # full information
        elif tw >= 0.25:
            weighted_final = total * 1.5  # moderate boost for 2-dimension coverage
        else:
            weighted_final = total  # minimal information, no boost
        # Add composite bonus
        final = weighted_final + composite_bonus
        # v5.11: coverage cap — <2 semantic dimensions + no composite rules → max 69 (medium)
        if sc < 2 and not composite_rules:
            final = min(final, 69)
        # No dimensions scored at all
        if tw == 0:
            final = min(60, len(evidence) * 12) if evidence else 5
        # Pack breakdown
        bd["coverage"] = round(tw, 2)
        bd["scored_dimensions"] = sc
        bd["weighted_sum"] = round(total, 1)
        bd["composite_bonus"] = round(composite_bonus, 1)
        bd["composite_rules"] = composite_rules
        # v5.11.5.1: promotion fields (weighted strategy doesn't apply promotion)
        bd["promotion_applied"] = False
        bd["promotion_rule"] = ""
        bd["pre_promotion_score"] = round(final, 1)
        bd["promotion_anchor_source"] = ""
        bd["promotion_anchor_rule_ids"] = []
        return final, bd, self._conf(sc + len(composite_rules), unk), (False, "", int(round(final)), 0, (), "", [])
    def _composite_evidence_bonus(self, evidence):
        """v5.11.5.1: check verified evidence for dangerous type combinations.
        Uses max-bonus strategy (not sum) to prevent overlapping patterns from
        stacking infinitely. Only verified evidence participates."""
        types = set()
        for ev in evidence:
            if ev.get("verification") in ("matched", "deterministic"):
                t = ev.get("type", "")
                if t: types.add(t)
        best_bonus = 0; best_rules = []
        for pattern_set, pattern_bonus, pattern_label in self.COMPOSITE_PATTERNS:
            if pattern_set.issubset(types):
                if pattern_bonus > best_bonus:
                    best_bonus = pattern_bonus
                    best_rules = [pattern_label]
        return best_bonus, best_rules
    def _rule_only(self,evidence):
        det=[e for e in evidence if e.get("source") in RULE_ONLY_SOURCES]
        if not det: return 5,{"rule_score":5,"rule_ids":[],"total":0,"all_rids":[]},0.3
        best=0; counted=set(); all_rids=set()
        for e in det:
            for rid in e.get("rule_ids",[]):
                if rid in self.RULE_SCORES: best=max(best,self.RULE_SCORES[rid]); counted.add(rid); all_rids.add(rid)
            if e.get("source")==EVIDENCE_SOURCE_KW:
                best=max(best,{"high":10,"medium":5,"low":2}.get(e.get("severity","low"),0))
        # Check email combination rules
        for combo, combo_score in self.EMAIL_COMBOS.items():
            if all(r in all_rids for r in combo):
                best=max(best,combo_score)
                counted.update(combo)
        if best==0: best=max(5,sum(1 for e in det if e.get("severity")=="high")*25)
        score=min(100,best)
        return score,{"rule_score":score,"rule_ids":list(counted),"total":len(det),"all_rids":list(all_rids)},min(0.6,0.2+len(det)*0.06)
    # v5.11.5.2: HIGH_RISK_PROMOTION_RULES — strict gates to break the 69-point cap.
    # Only verified evidence (verification="matched" or "deterministic") counts.
    #
    # link_based_phishing: requires deterministic URL anchor (verification=="deterministic",
    # source=="deterministic_url", valid url_* rule_id). Pure LLM-matched domain_anomaly
    # must NOT trigger promotion.
    #
    # social_engineering_phishing: requires either secrecy OR (urgency + deterministic
    # sender anomaly from email headers). Impersonation+payment+urgency alone must NOT
    # promote. Normal tuition/fee/utility notices must NOT promote.
    HIGH_RISK_PROMOTION_RULES = {
        "link_based_phishing": {
            "required_types": {"suspicious_link", "domain_anomaly"},
            "pressure_types": {"urgency", "secrecy"},
            "demand_types": {"info_request", "credential_request", "payment_request"},
            "promotion_score": 75,
            "description": "Link-based phishing: deterministic URL anchor + suspicious link + pressure + demand"
        },
        "social_engineering_phishing": {
            "required_types": {"impersonation"},
            "pressure_types": {"urgency", "secrecy"},
            "demand_types": {"payment_request", "credential_request"},
            "promotion_score": 72,
            "description": "Social engineering: impersonation + payment/credential + (secrecy|urgency+sender_anomaly) + ≥3 matched evidence"
        },
    }

    def _check_promotion(self, evidence):
        """v5.11.5.3: Check if verified evidence qualifies for promotion past the 69-point cap.
        Returns (promotion_applied, rule_name, promotion_score, evidence_types_tuple,
                 anchor_source, anchor_rule_ids).

        Link-based phishing MUST have a deterministic URL anchor:
        - verification=="deterministic" AND source=="deterministic_url" AND
          at least one rule_id that is BOTH:
            (a) starts with "url_", AND
            (b) is present in RiskScorer.RULE_SCORES (strict whitelist).
        Fake/unknown url_* rules (e.g. "url_fake_rule") must NOT trigger promotion.
        Pure LLM-matched domain_anomaly (verification="matched", source="llm") MUST NOT trigger.

        Social-engineering phishing MUST have:
        - Either secrecy present in verified types, OR
        - (urgency present) AND (deterministic sender_anomaly from email headers:
          verification=="deterministic", source=="deterministic_header").
        Impersonation+payment+urgency alone → NO promotion.
        """
        # Collect distinct verified evidence types and per-evidence metadata
        verified_types = set()
        for ev in evidence:
            ver = ev.get("verification", "")
            if ver in ("matched", "deterministic"):
                t = ev.get("type", "")
                if t:
                    verified_types.add(t)

        # Guard: need at least 3 distinct types
        if len(verified_types) < 3:
            return False, "", 0, (), "", []

        # Guard: urgency + info_request alone never promotes (no other strong indicator)
        strong_types = verified_types - {"urgency", "info_request", "other", "language_issue"}
        if not strong_types:
            return False, "", 0, (), "", []

        # v5.11.5.3: Helper — validate a single rule_id against the strict whitelist.
        # Must BOTH start with "url_" AND be present in RULE_SCORES.
        # Unknown/fake url_* rules (url_fake_rule, url_unknown_rule, etc.) are rejected.
        # Non-url rules (e.g. sdu_domain_impersonation) are also rejected as URL anchors.
        # Non-string, None, and wrong-type rule_ids are safely skipped.
        def _is_valid_url_rule(rid):
            """Check if a rule_id is a valid URL rule for promotion anchor.
            Must be a string, start with 'url_', AND be in RULE_SCORES."""
            if not isinstance(rid, str):
                return False
            if not rid.startswith("url_"):
                return False
            return rid in self.RULE_SCORES

        # ═══ Link-based phishing check ═══
        link_cfg = self.HIGH_RISK_PROMOTION_RULES["link_based_phishing"]
        link_req = link_cfg["required_types"]
        link_pressure = link_cfg["pressure_types"]
        link_demand = link_cfg["demand_types"]

        if link_req.issubset(verified_types) and (verified_types & link_pressure) and (verified_types & link_demand):
            # v5.11.5.3: MUST have deterministic URL anchor — a domain_anomaly evidence
            # with verification=="deterministic", source=="deterministic_url",
            # and at least one valid URL rule (passes _is_valid_url_rule whitelist).
            det_anchor_src = ""
            det_anchor_rids = []
            for ev in evidence:
                ver = ev.get("verification", "")
                src = ev.get("source", "")
                rids = ev.get("rule_ids", [])
                etype = ev.get("type", "")
                if (ver == "deterministic" and src == "deterministic_url"
                        and etype == "domain_anomaly"):
                    # v5.11.5.3: strict whitelist via RULE_SCORES membership,
                    # not just prefix match. url_fake_rule → rejected.
                    valid = [r for r in rids if _is_valid_url_rule(r)]
                    if valid:
                        det_anchor_src = src
                        det_anchor_rids = valid
                        break

            if det_anchor_src:
                # Has real deterministic URL anchor → promote!
                promo_types = tuple(sorted(
                    link_req | (verified_types & link_pressure) | (verified_types & link_demand)))
                return (True, "link_based_phishing", link_cfg["promotion_score"],
                        promo_types, det_anchor_src, det_anchor_rids)

        # ═══ Social-engineering phishing check ═══
        soc_cfg = self.HIGH_RISK_PROMOTION_RULES["social_engineering_phishing"]
        soc_req = soc_cfg["required_types"]
        soc_pressure = soc_cfg["pressure_types"]
        soc_demand = soc_cfg["demand_types"]

        if soc_req.issubset(verified_types) and (verified_types & soc_pressure) and (verified_types & soc_demand):
            # v5.11.5.2: Must have either secrecy OR (urgency + deterministic sender anomaly)
            has_secrecy = "secrecy" in verified_types
            has_urgency = "urgency" in verified_types

            # Check for deterministic sender anomaly from email headers
            has_det_sender_anomaly = False
            for ev in evidence:
                ver = ev.get("verification", "")
                src = ev.get("source", "")
                etype = ev.get("type", "")
                if (ver == "deterministic" and src == "deterministic_header"
                        and etype == "sender_anomaly"):
                    has_det_sender_anomaly = True
                    break

            # Social engineering requires: secrecy OR (urgency + deterministic sender anomaly)
            if not (has_secrecy or (has_urgency and has_det_sender_anomaly)):
                return False, "", 0, (), "", []

            # Must have at least 3 verified evidence with distinct quotes
            distinct_quotes = set()
            for ev in evidence:
                ver = ev.get("verification", "")
                q = (ev.get("quote", "") or "").strip()
                if ver in ("matched", "deterministic") and q:
                    distinct_quotes.add(q)
            if len(distinct_quotes) < 3:
                return False, "", 0, (), "", []

            # All checks passed — promote!
            promo_types = tuple(sorted(
                soc_req | (verified_types & soc_pressure) | (verified_types & soc_demand)))
            return (True, "social_engineering_phishing", soc_cfg["promotion_score"],
                    promo_types, "deterministic_header" if has_det_sender_anomaly else "", [])

        return False, "", 0, (), "", []

    def _hybrid(self,validated,evidence):
        """v5.11.5.1: calibrated hybrid with promotion system.
        Default cap at 69 for ordinary LLM semantic evidence.
        Only HIGH_RISK_PROMOTION_RULES can break through to ≥70."""
        ws,wb,wc = self._weighted(validated,evidence)[:3]
        rs,rb,rc = self._rule_only(evidence)
        has_composite = bool(wb.get("composite_rules"))
        coverage = wb.get("coverage", 0)
        scored_dims = wb.get("scored_dimensions", 0)

        # Step 1: Base hybrid — rule tier + LLM adjustment, default cap at 69
        if rs >= 70:
            # High risk tier — LLM adds confirmatory signal (capped)
            base_hybrid = min(100, rs + ws * 0.15)
        elif rs >= 40:
            # Medium risk tier — LLM can elevate within bounds
            elevation = ws * 0.6 if has_composite else ws * 0.4
            base_hybrid = min(84, rs + elevation)
        else:
            # Low risk tier — LLM needs composite evidence to reach medium
            if has_composite and coverage >= 0.25:
                base_hybrid = min(69, rs + ws * 0.5)
            else:
                base_hybrid = min(49, rs + ws * 0.3)

        # Apply default cap at 69 (only promotion can break through)
        pre_promotion = int(round(min(base_hybrid, 69) if base_hybrid < 70 else base_hybrid))

        # Step 2: Check promotion (v5.11.5.2: returns 6-tuple with anchor fields)
        promoted, rule_name, promo_score, promo_types, anchor_src, anchor_rids = self._check_promotion(evidence)
        if promoted:
            hybrid = max(pre_promotion, promo_score)
        else:
            hybrid = pre_promotion

        return hybrid, {
            "llm_weighted": round(ws, 1), "rule_score": rs,
            "hybrid_score": round(hybrid, 1),
            "has_rule_evidence": rs > 5,
            "composite_rules": wb.get("composite_rules", []),
            "coverage": coverage,
            "scored_dimensions": scored_dims,
            # v5.11.5.1: promotion tracking
            "promotion_applied": promoted,
            "promotion_rule": rule_name,
            "pre_promotion_score": pre_promotion,
            "promotion_score": promo_score if promoted else 0,
            "promotion_evidence_types": list(promo_types),
            # v5.11.5.2: deterministic anchor tracking
            "promotion_anchor_source": anchor_src,
            "promotion_anchor_rule_ids": anchor_rids,
        }, max(wc, rc), (promoted, rule_name, pre_promotion, promo_score if promoted else 0, promo_types, anchor_src, anchor_rids)
    def _read_v(self,validated):
        dims={}
        if validated is None:
            for k in self.dimensions: dims[k]={"score":None,"status":"unknown"}
            return dims
        for field,dk in [("sender_credibility_score","sender_credibility"),("link_safety_score","link_safety"),("content_urgency_score","content_urgency"),("information_request_score","information_request"),("language_quality_score","language_quality"),("attachment_risk_score","attachment_risk")]:
            val=getattr(validated,field,None)
            if val is not None: dims[dk]={"score":val,"status":"scored"}
            else: dims[dk]={"score":None,"status":"unknown"}
        return dims
    def _to_level(self,s):
        if s>=self.levels["high"]["min_score"]: return "high",self.levels["high"]["label"]
        if s>=self.levels["medium"]["min_score"]: return "medium",self.levels["medium"]["label"]
        return "low",self.levels["low"]["label"]
    def _dedup(self,ev): seen=set(); r=[]; [r.append(e) for e in ev if not (canonical_evidence_key(e) in seen or seen.add(canonical_evidence_key(e)))]; return r
    def _rule_bonus(self,ev):
        b=0.0; cfg=self.config["rule_evidence_bonus"]
        for e in ev:
            if e.get("source")==EVIDENCE_SOURCE_LLM: continue
            b+=cfg["high"] if e.get("severity")=="high" else (cfg["medium"] if e.get("severity")=="medium" else 0)
        return min(b,cfg["max_total_bonus"]),[]
    def _conf(self,sc,unk): total=sc+unk; return 0.3 if total==0 else min(0.95,(sc/total)*0.85)
