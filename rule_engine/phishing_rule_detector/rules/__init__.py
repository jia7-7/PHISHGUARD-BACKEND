"""规则模块 — 导入即注册所有规则."""
from phishing_rule_detector.rules import url_rules       # noqa: F401
from phishing_rule_detector.rules import domain_rules    # noqa: F401
from phishing_rule_detector.rules import html_rules      # noqa: F401
from phishing_rule_detector.rules import text_rules      # noqa: F401
from phishing_rule_detector.rules import email_rules     # noqa: F401
from phishing_rule_detector.rules import image_rules     # noqa: F401
