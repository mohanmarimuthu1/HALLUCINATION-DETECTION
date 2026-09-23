import pytest

from halludetect.evidence.base import EvidenceSource
from halludetect.evidence.custom import CustomEvidenceSource


def test_custom_evidence_satisfies_evidence_source_protocol() -> None:
    assert isinstance(CustomEvidenceSource(lambda q: []), EvidenceSource)


def test_custom_evidence_passes_query_to_the_retriever() -> None:
    captured = {}

    def retriever(query: str) -> list[str]:
        captured["query"] = query
        return []

    source = CustomEvidenceSource(retriever)
    source.fetch("what is the capital of france")

    assert captured["query"] == "what is the capital of france"


def test_custom_evidence_short_strings_become_one_chunk_each() -> None:
    source = CustomEvidenceSource(lambda q: ["first fact.", "second fact."])

    chunks = source.fetch("q")

    assert [c.chunk_id for c in chunks] == ["custom-0", "custom-1"]
    assert [c.text for c in chunks] == ["first fact.", "second fact."]
    assert all(c.source == "custom" for c in chunks)


def test_custom_evidence_skips_blank_strings() -> None:
    source = CustomEvidenceSource(lambda q: ["", "   ", "real fact."])

    chunks = source.fetch("q")

    assert [c.chunk_id for c in chunks] == ["custom-2"]
    assert chunks[0].text == "real fact."


def test_custom_evidence_splits_long_result_into_bounded_chunks() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(10)]
    long_text = " ".join(sentences)
    source = CustomEvidenceSource(lambda q: [long_text], max_chunk_chars=60)

    chunks = source.fetch("q")

    assert len(chunks) > 1
    assert all(c.chunk_id.startswith("custom-0-") for c in chunks)
    assert all(c.source == "custom" for c in chunks)
    for c in chunks:
        assert len(c.text) <= 60


def test_custom_evidence_propagates_retriever_exceptions() -> None:
    def failing_retriever(query: str) -> list[str]:
        raise RuntimeError("caller's retriever is broken")

    source = CustomEvidenceSource(failing_retriever)

    with pytest.raises(RuntimeError, match="caller's retriever is broken"):
        source.fetch("q")


def test_custom_evidence_empty_retriever_result_returns_no_evidence() -> None:
    source = CustomEvidenceSource(lambda q: [])

    assert source.fetch("q") == []
