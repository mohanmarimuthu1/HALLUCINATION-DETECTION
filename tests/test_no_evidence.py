from halludetect.evidence.base import EvidenceSource
from halludetect.evidence.none import NoEvidenceSource


def test_no_evidence_source_satisfies_evidence_source_protocol() -> None:
    assert isinstance(NoEvidenceSource(), EvidenceSource)


def test_no_evidence_source_always_returns_empty_regardless_of_query() -> None:
    source = NoEvidenceSource()

    assert source.fetch("anything") == []
    assert source.fetch("") == []
