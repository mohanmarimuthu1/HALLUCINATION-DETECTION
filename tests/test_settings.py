"""Settings tests - offline, no network, no live keys.

Regression coverage for a real bug found while live-testing Phase 5.2:
`.env.example` tells users to leave optional numeric settings blank, but
without `env_ignore_empty`, pydantic-settings treats `RATE_LIMIT_CAPACITY=`
(present, empty) as a literal empty string rather than "unset" - which
fails validation for a non-Optional `float` field and crashes `Settings()`
construction entirely, taking down every request that calls
`get_settings()`. Reproduced directly against a real `.env` file, not
assumed.
"""
from halludetect.settings import Settings


def _write_env(tmp_path, contents: str):
    env_file = tmp_path / ".env"
    env_file.write_text(contents, encoding="utf-8")
    return env_file


def test_blank_numeric_env_vars_fall_back_to_defaults(tmp_path):
    env_file = _write_env(tmp_path, "RATE_LIMIT_CAPACITY=\nRATE_LIMIT_REFILL_PER_S=\n")
    settings = Settings(_env_file=env_file)
    assert settings.rate_limit_capacity == 60.0
    assert settings.rate_limit_refill_per_s == 1.0


def test_blank_secret_env_vars_are_none_not_empty_string(tmp_path):
    env_file = _write_env(tmp_path, "OPENROUTER_API_KEY=\nCLIENT_API_KEYS=\n")
    settings = Settings(_env_file=env_file)
    assert settings.openrouter_api_key is None
    assert settings.client_api_keys is None


def test_numeric_env_vars_with_real_values_are_parsed(tmp_path):
    env_file = _write_env(tmp_path, "RATE_LIMIT_CAPACITY=10\nRATE_LIMIT_REFILL_PER_S=0.5\n")
    settings = Settings(_env_file=env_file)
    assert settings.rate_limit_capacity == 10.0
    assert settings.rate_limit_refill_per_s == 0.5


def test_no_env_file_uses_defaults(tmp_path):
    settings = Settings(_env_file=tmp_path / "does-not-exist.env")
    assert settings.rate_limit_capacity == 60.0
    assert settings.client_api_keys is None
