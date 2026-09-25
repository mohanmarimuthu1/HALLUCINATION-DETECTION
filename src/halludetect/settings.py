"""Application settings, loaded from environment variables / .env.

Every provider credential is a SecretStr so it never renders in plain text
in logs, tracebacks, or repr() output (Phase 8.1 secrets review depends on
this holding for every key added here).
"""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    openrouter_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    custom_provider_base_url: str | None = None
    custom_provider_api_key: SecretStr | None = None

    # Web-search evidence source (docs/contract.md, Phase 0.2 decision: Tavily).
    # Absence disables the "web" evidence_source; it must never crash a request.
    tavily_api_key: SecretStr | None = None

    # Per-key auth for POST /v1/verify (Phase 5.2). Comma-separated list of
    # valid client keys - distinct from the provider keys above, which
    # authenticate *this service* to an LLM/search backend, not a caller to
    # this service. A static list is deliberate scope for v1: a full
    # user/key management system is not part of this project. No keys
    # configured means every request is rejected with 401, never silently
    # allowed through.
    client_api_keys: SecretStr | None = None

    # Token-bucket rate limiting per client key (Phase 5.2), in-memory and
    # per-process - correct for a single instance; a multi-instance
    # deployment needs a shared store instead (out of scope until Phase 6
    # introduces the first shared-cache dependency this project takes on).
    rate_limit_capacity: float = 60.0
    rate_limit_refill_per_s: float = 1.0

    # Result cache (Phase 6.1): identical requests within cache_ttl_s get
    # the same AnalysisResult back without repeating any LLM calls.
    # File-backed (diskcache), not in-memory, specifically so it survives a
    # process restart - unlike the rate limiter/health tracker above, a
    # cache that resets on every restart wouldn't be doing its job.
    cache_enabled: bool = True
    cache_dir: str = ".cache/halludetect"
    cache_ttl_s: float = 3600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
