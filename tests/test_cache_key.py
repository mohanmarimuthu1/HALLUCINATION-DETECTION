"""Cache key computation tests (Phase 6.1) - offline, pure functions."""
from halludetect.cache.key import compute_cache_key


def _key(**overrides) -> str:
    defaults = dict(
        answer="The Eiffel Tower is in Paris.",
        question=None,
        evidence=["The Eiffel Tower is in Paris, France."],
        evidence_source="none",
        model_provider="openrouter",
        allow_free_pool=True,
        pinned_model=None,
        user_api_key=None,
    )
    defaults.update(overrides)
    return compute_cache_key(**defaults)


def test_identical_inputs_produce_identical_keys():
    assert _key() == _key()


def test_different_answer_changes_the_key():
    assert _key(answer="a different answer") != _key()


def test_different_evidence_changes_the_key():
    assert _key(evidence=["different evidence"]) != _key()


def test_different_model_provider_changes_the_key():
    assert _key(model_provider="gemini") != _key()


def test_different_pinned_model_changes_the_key():
    assert _key(pinned_model="some/model") != _key()


def test_different_user_api_key_changes_the_key():
    """Two callers with different bring-your-own keys must never share a
    cached answer, even for an otherwise identical request.
    """
    assert _key(user_api_key="caller-a-key") != _key(user_api_key="caller-b-key")


def test_evidence_order_matters():
    assert _key(evidence=["a", "b"]) != _key(evidence=["b", "a"])
