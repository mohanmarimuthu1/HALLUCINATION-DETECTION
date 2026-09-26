"""metrics.py tests (Phase 7.3) - each metric checked against a small,
hand-computed array so the numbers are verified, not just "runs without
crashing."
"""
import math

from halludetect.eval import metrics


def test_average_precision_perfect_ranking():
    # All positives score higher than all negatives -> AP == 1.0.
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]
    assert metrics.average_precision(scores, labels) == 1.0


def test_average_precision_worst_ranking():
    # All negatives score higher than all positives.
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [False, False, True, True]
    ap = metrics.average_precision(scores, labels)
    assert ap is not None
    assert ap < 0.6


def test_average_precision_none_when_no_positives():
    assert metrics.average_precision([0.1, 0.2], [False, False]) is None


def test_prevalence():
    assert metrics.prevalence([True, True, False, False]) == 0.5
    assert metrics.prevalence([]) == 0.0


def test_brier_score_perfect_predictions_is_zero():
    assert metrics.brier_score([1.0, 0.0], [True, False]) == 0.0


def test_brier_score_hand_computed():
    # (0.8 - 1)^2 + (0.3 - 0)^2 = 0.04 + 0.09 = 0.13; mean over 2 = 0.065.
    score = metrics.brier_score([0.8, 0.3], [True, False])
    assert math.isclose(score, 0.065, rel_tol=1e-9)


def test_ece_perfect_calibration_within_a_single_bin():
    # Avg predicted (1.0) exactly matches the observed positive rate (1.0).
    scores = [1.0, 1.0]
    labels = [True, True]
    ece, table = metrics.expected_calibration_error(scores, labels, n_bins=10)
    assert math.isclose(ece, 0.0, abs_tol=1e-9)
    populated = [b for b in table if b["count"] > 0]
    assert len(populated) == 1
    assert populated[0]["count"] == 2


def test_ece_miscalibrated_bucket():
    # Predicted 0.9 but observed rate 0.0 (never actually hallucinated) -> ECE == 0.9.
    scores = [0.9, 0.9]
    labels = [False, False]
    ece, _ = metrics.expected_calibration_error(scores, labels, n_bins=10)
    assert math.isclose(ece, 0.9, rel_tol=1e-9)


def test_precision_recall_hand_computed():
    predicted = [True, True, False, False]
    actual = [True, False, True, False]
    # TP=1 (idx0), FP=1 (idx1), FN=1 (idx2).
    precision, recall = metrics.precision_recall(predicted, actual)
    assert precision == 0.5
    assert recall == 0.5


def test_precision_recall_none_when_undefined():
    assert metrics.precision_recall([False, False], [False, False]) == (None, None)


def test_percentile_median_and_p95():
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert metrics.percentile(values, 0.5) == 5.5
    assert metrics.percentile([], 0.5) == 0.0


def test_percentile_single_value():
    assert metrics.percentile([42.0], 0.95) == 42.0
