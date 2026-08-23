"""官方域名判断、可信第三方策略与 tldextract 离线解析."""
from __future__ import annotations

import tldextract

from phishing_rule_detector.config_loader import (
    AppConfig,
    TrustedService,
    load_config,
)

# ── tldextract 单例（离线模式）──

_tld_extract: tldextract.TLDExtract | None = None


def _get_tldextract() -> tldextract.TLDExtract:
    """获取 tldextract 实例，使用 suffix_list_urls=() 强制离线."""
    global _tld_extract
    if _tld_extract is None:
        _tld_extract = tldextract.TLDExtract(suffix_list_urls=())
    return _tld_extract


def get_registered_domain(hostname: str) -> str:
    """提取注册域名（基于公共后缀列表）."""
    ext = _get_tldextract()
    result = ext(hostname)
    return result.registered_domain or hostname


# ── 域名规范化 ──


def _normalize_hostname(hostname: str) -> str:
    """小写化并移除末尾点."""
    return hostname.lower().strip().rstrip(".")


# ── 官方域名判断 ──


def is_official_domain(hostname: str) -> bool:
    """按域名标签边界判断 hostname 是否为官方域名或其子域名.

    判断规则:
      host == trusted_domain
      或
      host.endswith("." + trusted_domain)

    保证 sdu.edu.cn.evil.com 和 fakesdu.edu.cn 被正确拒绝。
    """
    host = _normalize_hostname(hostname)
    if not host:
        return False

    config = _get_config()
    for domain in config.trusted.official_domains:
        domain = _normalize_hostname(domain)
        if host == domain:
            return True
        if host.endswith("." + domain):
            return True

    return False


def is_official_subdomain(hostname: str) -> bool:
    """判断 hostname 是否为官方域名的子域名（非根域名本身）."""
    host = _normalize_hostname(hostname)
    if not host:
        return False

    config = _get_config()
    for domain in config.trusted.official_domains:
        domain = _normalize_hostname(domain)
        if host.endswith("." + domain):
            return True

    return False


# ── 可信第三方 ──


def match_trusted_service(hostname: str) -> TrustedService | None:
    """匹配可信第三方服务.

    按注册域名后缀匹配：host == service_domain 或 host.endswith("." + domain).
    """
    host = _normalize_hostname(hostname)
    if not host:
        return None

    config = _get_config()
    for svc in config.trusted.trusted_services:
        for domain in svc.domains:
            domain = _normalize_hostname(domain)
            if host == domain or host.endswith("." + domain):
                return svc

    return None


def check_field_allowed(
    service: TrustedService | None, field_name: str
) -> bool:
    """检查字段是否在可信服务的允许列表中."""
    if service is None:
        return False
    return field_name.lower() in [
        f.lower() for f in service.allowed_fields
    ]


def check_field_forbidden(
    service: TrustedService | None, field_name: str
) -> bool:
    """检查字段是否在可信服务的禁止列表中."""
    if service is None:
        return False
    return field_name.lower() in [
        f.lower() for f in service.forbidden_fields
    ]


# ── 短链接检测 ──


def is_short_link_domain(hostname: str) -> bool:
    """判断 hostname 是否为已知短链接服务域名."""
    host = _normalize_hostname(hostname)
    if not host:
        return False

    config = _get_config()
    for sl in config.trusted.short_link_domains:
        sl = _normalize_hostname(sl)
        if host == sl or host.endswith("." + sl):
            return True

    return False


# ── 内部辅助 ──

def _get_config() -> AppConfig:
    """返回配置加载器的当前快照，避免维护第二份过期缓存."""
    return load_config()
