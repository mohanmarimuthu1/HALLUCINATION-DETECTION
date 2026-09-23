from halludetect.evidence.base import Evidence, EvidenceSource
from halludetect.evidence.direct import DirectEvidence


def test_direct_evidence_satisfies_evidence_source_protocol() -> None:
    assert isinstance(DirectEvidence(["x"]), EvidenceSource)


def test_direct_evidence_short_strings_become_one_chunk_each() -> None:
    source = DirectEvidence(["first fact.", "second fact."])

    chunks = source.fetch("irrelevant query")

    assert chunks == [
        Evidence(chunk_id="direct-0", text="first fact.", source="direct"),
        Evidence(chunk_id="direct-1", text="second fact.", source="direct"),
    ]


def test_direct_evidence_ignores_query() -> None:
    source = DirectEvidence(["fact."])

    assert source.fetch("q1") == source.fetch("q2")


def test_direct_evidence_skips_blank_strings() -> None:
    source = DirectEvidence(["", "   ", "real fact."])

    chunks = source.fetch("q")

    assert [c.chunk_id for c in chunks] == ["direct-2"]
    assert chunks[0].text == "real fact."


def test_direct_evidence_splits_long_paragraph_into_bounded_chunks() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(10)]
    long_text = " ".join(sentences)
    source = DirectEvidence([long_text], max_chunk_chars=60)

    chunks = source.fetch("q")

    assert len(chunks) > 1
    assert all(c.chunk_id.startswith("direct-0-") for c in chunks)
    assert all(c.source == "direct" for c in chunks)
    for c in chunks:
        assert len(c.text) <= 60
    rejoined = " ".join(c.text for c in chunks)
    for s in sentences:
        assert s in rejoined


def test_direct_evidence_never_drops_an_oversized_single_sentence() -> None:
    oversized_sentence = "x" * 500 + "."
    source = DirectEvidence([oversized_sentence], max_chunk_chars=60)

    chunks = source.fetch("q")

    assert len(chunks) == 1
    assert chunks[0].text == oversized_sentence


def test_direct_evidence_splits_on_paragraph_boundaries_first() -> None:
    para_a = "A" * 40
    para_b = "B" * 40
    source = DirectEvidence([f"{para_a}\n\n{para_b}"], max_chunk_chars=50)

    chunks = source.fetch("q")

    assert [c.text for c in chunks] == [para_a, para_b]


def test_direct_evidence_multiple_inputs_independently_chunked() -> None:
    short = "short."
    long_input = " ".join(f"Sentence {i} of the long input." for i in range(10))
    source = DirectEvidence([short, long_input], max_chunk_chars=60)

    chunks = source.fetch("q")

    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "direct-0"
    assert all(cid.startswith("direct-1-") for cid in ids[1:])
