"""Router chain tests (Phase 2.3) - offline, FakeProvider stands in for
every model. Covers pinned -> free pool -> user key -> explicit fail, and
that a failing model never causes a crash, only a fail-over."""
import pytest

from halludetect.llm.exceptions import LLMResponseError, LLMTimeoutError
from halludetect.llm.fake import FakeProvider, fake_response
from halludetect.llm.health import HealthTracker
from halludetect.llm.openrouter import FreeModelCatalog
from halludetect.llm.router import Router


def make_catalog(models: list[str]) -> FreeModelCatalog:
    catalog = FreeModelCatalog(api_key="key")
    catalog._models = models
    catalog._fetched_at = 0.0
    return catalog


def test_pinned_model_success_skips_free_pool():
    providers = {"pinned/model": FakeProvider([fake_response("pinned answer")])}
    router = Router(
        catalog=make_catalog(["free/a"]),
        provider_factory=lambda model: providers[model],
        pinned_model="pinned/model",
    )
    result = router.complete("hi")
    assert result.text == "pinned answer"
    assert "free/a" not in providers


def test_pinned_failure_falls_over_to_free_pool():
    providers = {
        "pinned/model": FakeProvider([LLMTimeoutError("timed out")]),
        "free/a": FakeProvider([fake_response("free pool answer")]),
    }
    router = Router(
        catalog=make_catalog(["free/a"]),
        provider_factory=lambda model: providers[model],
        pinned_model="pinned/model",
    )
    result = router.complete("hi")
    assert result.text == "free pool answer"


def test_free_pool_exhausted_falls_over_to_user_provider():
    providers = {
        "free/a": FakeProvider([LLMTimeoutError("down")]),
        "free/b": FakeProvider([LLMTimeoutError("down")]),
    }
    user_provider = FakeProvider([fake_response("user key answer")])
    router = Router(
        catalog=make_catalog(["free/a", "free/b"]),
        provider_factory=lambda model: providers[model],
        user_provider=user_provider,
    )
    result = router.complete("hi")
    assert result.text == "user key answer"


def test_everything_failing_raises_explicit_error_not_a_guess():
    providers = {"free/a": FakeProvider([LLMTimeoutError("down")])}
    router = Router(
        catalog=make_catalog(["free/a"]),
        provider_factory=lambda model: providers[model],
    )
    with pytest.raises(LLMResponseError):
        router.complete("hi")


def test_health_updated_on_failure_and_success():
    providers = {
        "free/a": FakeProvider([LLMTimeoutError("down")]),
        "free/b": FakeProvider([fake_response("ok")]),
    }
    health = HealthTracker()
    router = Router(
        catalog=make_catalog(["free/a", "free/b"]),
        provider_factory=lambda model: providers[model],
        health=health,
    )
    router.complete("hi")
    assert health.get("free/a").consecutive_failures == 1
    assert health.get("free/b").successes == 1


def test_respects_max_free_models_tried_cap():
    providers = {f"free/{i}": FakeProvider([LLMTimeoutError("down")]) for i in range(10)}
    router = Router(
        catalog=make_catalog(list(providers.keys())),
        provider_factory=lambda model: providers[model],
        max_free_models_tried=3,
    )
    with pytest.raises(LLMResponseError):
        router.complete("hi")
    attempted = [model for model, p in providers.items() if p.call_count > 0]
    assert len(attempted) == 3
