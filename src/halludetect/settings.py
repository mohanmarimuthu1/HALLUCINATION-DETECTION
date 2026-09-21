"""Application settings, loaded from environment variables / .env.

Every provider credential is a SecretStr so it never renders in plain text
in logs, tracebacks, or repr() output (Phase 8.1 secrets review depends on
this holding for every key added here).
"""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    custom_provider_base_url: str | None = None
    custom_provider_api_key: SecretStr | None = None

    # Web-search evidence source (docs/contract.md, Phase 0.2 decision: Tavily).
    # Absence disables the "web" evidence_source; it must never crash a request.
    tavily_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
