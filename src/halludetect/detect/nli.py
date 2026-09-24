"""Optional NLI cross-encoder second signal (Phase 4.5).

Disabled by default. The pipeline (4.1-4.4) is fully functional without
this - quote-grounding is the primary signal. This module only defines
the plug-in point: an `NLIScorer` Protocol pipeline.py can call if one is
configured, and a concrete loader whose ML dependency is imported lazily,
so nothing that imports this module - or runs with NLI disabled, the
default - needs sentence-transformers/torch installed.

Wiring a *specific* cross-encoder checkpoint is deferred: label ordering
in NLI cross-encoder output varies by model and there is no universal
convention, so `CrossEncoderNLIScorer` takes `label_order` explicitly
rather than guessing at one - guessing wrong here would silently invert
entailment/contradiction, exactly the kind of unverified assumption this
project exists to avoid.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from halludetect.detect.schemas import Label


@runtime_checkable
class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> tuple[Label, float]:
        """Returns (label, confidence) for whether `premise` entails,
        contradicts, or says nothing about `hypothesis`. Uses the same
        `Label` enum as quote-grounded verification so a caller can
        combine both signals without a separate vocabulary.
        """
        ...


class CrossEncoderNLIScorer:
    """Lazily loads a sentence-transformers CrossEncoder on construction,
    not on import - `settings.nli_model` being unset must never require
    this dependency to be installed at all.
    """

    def __init__(self, model_name: str, label_order: tuple[Label, Label, Label]):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "settings.nli_model is configured but sentence-transformers "
                "is not installed - add it to the environment to use the "
                "NLI signal, or leave nli_model unset to skip it"
            ) from exc

        self._model = CrossEncoder(model_name)
        self._label_order = label_order
        self.model_name = model_name

    def score(self, premise: str, hypothesis: str) -> tuple[Label, float]:
        raw_scores = self._model.predict([(premise, hypothesis)])[0]
        best_index = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        return self._label_order[best_index], float(raw_scores[best_index])
