"""Metric, thresholding and error-analysis tests with hand-computed ground truth."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gamesense.config import CONFIG, GENRES
from gamesense.evaluation.error_analysis import (
    build_error_frame,
    classify_outcome,
    confusion_pairs,
    per_class_error_summary,
    select_examples,
)
from gamesense.evaluation.evaluator import evaluate_predictions
from gamesense.evaluation.metrics import (
    aggregate_seeds,
    binarize,
    confusion_counts,
    multilabel_metrics,
)
from gamesense.evaluation.thresholding import (
    search_global_threshold,
    search_per_class_thresholds,
    threshold_sweep,
)

CLASSES: tuple[str, str, str] = ("A", "B", "C")
TOLERANCE = 1e-9

# --------------------------------------------------------------------------- #
# The worked 4-sample x 3-class example used throughout this module.
#
# truth predictions at threshold 0.5 row0: A .
#
# class A: TP=2 FP=0 FN=0 TN=2 -> P=1 R=1 F1=1 class B: TP=1 FP=1 FN=1 TN=1 -> P=1/2 R=1/2
# F1=1/2 class C: TP=1 FP=0 FN=1 TN=2 -> P=1 R=1/2 F1=2/3
# --------------------------------------------------------------------------- #
Y_TRUE = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
Y_PROB = np.array(
    [
        [0.90, 0.60, 0.20],
        [0.10, 0.80, 0.30],
        [0.70, 0.40, 0.10],
        [0.20, 0.30, 0.95],
    ]
)


# --------------------------------------------------------------------------- #
# multilabel_metrics -- hand-computed arithmetic
# --------------------------------------------------------------------------- #
def test_micro_and_macro_averages_match_hand_computation() -> None:
    metrics = multilabel_metrics(Y_TRUE, Y_PROB, threshold=0.5, classes=CLASSES)

    # Micro pools every (sample, class) decision: TP=2+1+1=4, FP=0+1+0=1, FN=0+1+1=2 micro-P = 4
    # / (4 + 1) = 0.8 micro-R = 4 / (4 + 2) = 2/3 micro-F1 = 2*4 / (2*4 + 1 + 2) = 8/11
    assert metrics["micro_precision"] == pytest.approx(0.8, abs=TOLERANCE)
    assert metrics["micro_recall"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert metrics["micro_f1"] == pytest.approx(8 / 11, abs=TOLERANCE)

    # Macro averages the per-class values unweighted: macro-P = (1 + 1/2 + 1 ) / 3 = 5/6 macro-R
    # = (1 + 1/2 + 1/2) / 3 = 2/3 macro-F1 = (1 + 1/2 + 2/3) / 3 = 13/18
    assert metrics["macro_precision"] == pytest.approx(5 / 6, abs=TOLERANCE)
    assert metrics["macro_recall"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert metrics["macro_f1"] == pytest.approx(13 / 18, abs=TOLERANCE)

    per_class = metrics["per_class"]
    assert per_class["A"]["f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert per_class["B"]["f1"] == pytest.approx(0.5, abs=TOLERANCE)
    assert per_class["C"]["f1"] == pytest.approx(2 / 3, abs=TOLERANCE)


def test_hamming_loss_is_the_fraction_of_wrong_label_decisions() -> None:
    # 4 samples x 3 classes = 12 decisions.  Wrong ones: (row0, B) spurious,
    # (row0, C) missed, (row2, B) missed  ->  3 / 12 = 0.25
    metrics = multilabel_metrics(Y_TRUE, Y_PROB, threshold=0.5, classes=CLASSES)
    assert metrics["hamming_loss"] == pytest.approx(3 / 12, abs=TOLERANCE)


def test_subset_accuracy_is_the_exact_match_fraction() -> None:
    # Exact label-set matches: row1 ({B}) and row3 ({C})  ->  2 / 4 = 0.5
    metrics = multilabel_metrics(Y_TRUE, Y_PROB, threshold=0.5, classes=CLASSES)
    assert metrics["subset_accuracy"] == pytest.approx(0.5, abs=TOLERANCE)


def test_perfect_predictions_saturate_every_metric() -> None:
    probabilities = np.where(Y_TRUE == 1, 0.95, 0.05)
    metrics = multilabel_metrics(Y_TRUE, probabilities, threshold=0.5, classes=CLASSES)
    assert metrics["micro_f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["macro_f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["hamming_loss"] == pytest.approx(0.0, abs=TOLERANCE)
    assert metrics["subset_accuracy"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["mAP"] == pytest.approx(1.0, abs=TOLERANCE)


def test_all_zero_predictions_score_zero_without_crashing_or_warning() -> None:
    """The "predict nothing" degenerate case must be 0, not ``NaN``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = multilabel_metrics(
            Y_TRUE, np.zeros_like(Y_PROB), threshold=0.5, classes=CLASSES
        )

    undefined = [w for w in caught if "UndefinedMetric" in type(w.message).__name__]
    assert undefined == [], f"zero_division must be handled explicitly, got {undefined}"

    for key in ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "macro_precision"):
        assert metrics[key] == pytest.approx(0.0, abs=TOLERANCE)
        assert np.isfinite(metrics[key])
    assert metrics["subset_accuracy"] == pytest.approx(0.0, abs=TOLERANCE)
    assert metrics["label_cardinality_pred"] == pytest.approx(0.0, abs=TOLERANCE)


def test_macro_falls_below_micro_when_a_rare_genre_is_missed() -> None:
    """Reporting both averages is only worth it if they can disagree."""
    n = 10
    truth = np.zeros((n, 3), dtype=int)
    truth[:6, 0] = 1  # A: 6 positives
    truth[:5, 1] = 1  # B: 5 positives
    truth[0, 2] = 1  # C: 1 positive
    probabilities = np.zeros((n, 3))
    probabilities[:, 0] = np.where(truth[:, 0] == 1, 0.9, 0.1)
    probabilities[:, 1] = np.where(truth[:, 1] == 1, 0.9, 0.1)
    probabilities[:, 2] = 0.1  # C is never predicted

    metrics = multilabel_metrics(truth, probabilities, threshold=0.5, classes=CLASSES)
    # macro-F1 = (1 + 1 + 0) / 3 = 2/3
    # micro-F1 = 2*11 / (2*11 + 0 + 1) = 22/23
    assert metrics["macro_f1"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert metrics["micro_f1"] == pytest.approx(22 / 23, abs=TOLERANCE)
    assert metrics["macro_f1"] < metrics["micro_f1"]


def test_average_precision_matches_the_hand_computed_ranking() -> None:
    """Class ``A`` has a known ranking; ``B`` and ``C`` are ranked perfectly."""
    truth = np.array(
        [[1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
    )
    scores = np.array(
        [
            [0.90, 0.90, 0.10],
            [0.80, 0.80, 0.10],
            [0.70, 0.10, 0.10],
            [0.40, 0.10, 0.10],
            [0.30, 0.05, 0.90],
            [0.20, 0.05, 0.80],
        ]
    )
    metrics = multilabel_metrics(truth, scores, threshold=0.5, classes=CLASSES)
    per_class = metrics["per_class"]
    assert per_class["A"]["average_precision"] == pytest.approx(29 / 36, abs=TOLERANCE)
    assert per_class["B"]["average_precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert per_class["C"]["average_precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics["mAP"] == pytest.approx((29 / 36 + 1.0 + 1.0) / 3, abs=TOLERANCE)


def test_map_excludes_classes_without_positives_instead_of_scoring_them_zero() -> None:
    truth = Y_TRUE.copy()
    truth[:, 2] = 0  # class C has no positives at all in this split
    metrics = multilabel_metrics(truth, Y_PROB, threshold=0.5, classes=CLASSES)

    assert np.isfinite(metrics["mAP"]), "an absent class must not poison mAP"
    assert metrics["mAP"] > 0.0
    assert metrics["per_class"]["C"]["support"] == 0
    assert np.isnan(metrics["per_class"]["C"]["average_precision"])


def test_multilabel_metrics_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        multilabel_metrics(Y_TRUE, Y_PROB[:2], classes=CLASSES)
    with pytest.raises(ValueError):
        multilabel_metrics(Y_TRUE, Y_PROB, classes=GENRES)


# --------------------------------------------------------------------------- #
# binarize
# --------------------------------------------------------------------------- #
def test_binarize_with_a_scalar_threshold() -> None:
    predictions = binarize(Y_PROB, 0.5)
    expected = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.array_equal(predictions, expected)


def test_binarize_with_a_per_class_threshold_vector() -> None:
    # Thresholds A=0.8, B=0.35, C=0.25 applied column-wise.
    predictions = binarize(Y_PROB, [0.8, 0.35, 0.25])
    expected = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(predictions, expected)


def test_binarize_treats_a_probability_on_the_threshold_as_positive() -> None:
    """``>=``, not ``>``: the whole threshold grid assumes inclusive boundaries."""
    probabilities = np.array([[0.5, 0.499999, 0.500001]])
    assert np.array_equal(binarize(probabilities, 0.5), np.array([[1, 0, 1]]))
    assert np.array_equal(binarize(probabilities, [0.5, 0.5, 0.5]), np.array([[1, 0, 1]]))


def test_binarize_rejects_a_wrong_length_threshold_vector() -> None:
    with pytest.raises(ValueError, match="per-class threshold"):
        binarize(Y_PROB, [0.5, 0.5])


# --------------------------------------------------------------------------- #
# Threshold search
# --------------------------------------------------------------------------- #
def _shifted_probabilities(labels: np.ndarray, positive: float, negative: float) -> np.ndarray:
    """Probabilities separable at exactly one point of the configured grid."""
    return np.where(labels == 1, positive, negative)


@pytest.fixture
def separable_labels() -> np.ndarray:
    """21 samples x 3 classes with every class both present and absent."""
    return np.array([[1 if (row + col) % 3 == 0 else 0 for col in range(3)] for row in range(21)])


def test_threshold_sweep_covers_the_whole_grid(separable_labels: np.ndarray) -> None:
    grid = CONFIG.evaluation.threshold_grid
    probabilities = _shifted_probabilities(separable_labels, 0.32, 0.27)
    sweep = threshold_sweep(separable_labels, probabilities, grid=grid, classes=CLASSES)

    assert len(sweep) == len(grid)
    assert sweep["threshold"].tolist() == [float(t) for t in grid]
    for column in ("micro_f1", "macro_f1", "hamming_loss", "label_cardinality_pred"):
        assert column in sweep.columns
        assert sweep[column].notna().all()


def test_global_search_returns_the_argmax_of_the_requested_metric(
    separable_labels: np.ndarray,
) -> None:
    grid = CONFIG.evaluation.threshold_grid
    probabilities = _shifted_probabilities(separable_labels, 0.32, 0.27)
    selected, info = search_global_threshold(
        separable_labels, probabilities, grid=grid, metric="micro_f1", classes=CLASSES
    )

    sweep = threshold_sweep(separable_labels, probabilities, grid=grid, classes=CLASSES)
    best = float(sweep["micro_f1"].max())
    maximisers = sweep.loc[np.isclose(sweep["micro_f1"], best), "threshold"].tolist()
    assert selected in maximisers
    assert info["best_score"] == pytest.approx(best)
    assert info["strategy"] == "global"
    assert len(info["scores"]) == len(grid)


def test_global_search_beats_a_provably_suboptimal_half(
    separable_labels: np.ndarray,
) -> None:
    """Negatives at 0.27 and positives at 0.32 make 0.50 the worst possible choice."""
    grid = CONFIG.evaluation.threshold_grid
    probabilities = _shifted_probabilities(separable_labels, 0.32, 0.27)
    selected, info = search_global_threshold(
        separable_labels, probabilities, grid=grid, metric="micro_f1", classes=CLASSES
    )
    at_half = multilabel_metrics(
        separable_labels, probabilities, threshold=0.5, classes=CLASSES
    )["micro_f1"]

    assert selected == pytest.approx(0.30)
    assert at_half == pytest.approx(0.0, abs=TOLERANCE)
    assert info["best_score"] > at_half
    assert info["best_score"] == pytest.approx(1.0, abs=TOLERANCE)


def test_per_class_search_never_loses_to_a_single_global_half() -> None:
    """One threshold per genre must be at least as good as 0.5 everywhere."""
    n = 60
    truth = np.zeros((n, 3), dtype=int)
    truth[:30, 0] = 1
    truth[:20, 1] = 1
    # class C deliberately has zero positives -> must fall back to the default.
    probabilities = np.zeros((n, 3))
    probabilities[:, 0] = np.where(truth[:, 0] == 1, 0.70, 0.20)
    probabilities[:, 1] = np.where(truth[:, 1] == 1, 0.45, 0.10)
    probabilities[:, 2] = 0.05

    thresholds, info = search_per_class_thresholds(truth, probabilities, classes=CLASSES)

    assert thresholds.shape == (len(CLASSES),)
    assert info["strategy"] == "per_class"
    assert info["per_class"]["C"]["support"] == 0
    assert info["per_class"]["C"]["f1"] is None
    assert thresholds[2] == pytest.approx(CONFIG.evaluation.default_threshold)

    per_class_macro = multilabel_metrics(
        truth, probabilities, threshold=thresholds, classes=CLASSES
    )["macro_f1"]
    global_macro = multilabel_metrics(
        truth, probabilities, threshold=0.5, classes=CLASSES
    )["macro_f1"]
    assert per_class_macro >= global_macro
    assert per_class_macro > global_macro, "class B should only be recoverable per-class"


# --------------------------------------------------------------------------- #
# aggregate_seeds
# --------------------------------------------------------------------------- #
def test_aggregate_seeds_uses_the_sample_standard_deviation() -> None:
    runs = [{"micro_f1": 0.60}, {"micro_f1": 0.70}, {"micro_f1": 0.80}]
    # mean = 0.70; ddof=1 -> sqrt(((0.1)^2 + 0 + (0.1)^2) / 2) = 0.10
    aggregated = aggregate_seeds(runs, keys=("micro_f1",))["micro_f1"]
    assert aggregated["mean"] == pytest.approx(0.70)
    assert aggregated["std"] == pytest.approx(0.10)
    assert aggregated["min"] == pytest.approx(0.60)
    assert aggregated["max"] == pytest.approx(0.80)
    assert aggregated["n"] == 3


def test_aggregate_seeds_reports_zero_spread_for_a_single_run() -> None:
    aggregated = aggregate_seeds([{"macro_f1": 0.42}], keys=("macro_f1",))["macro_f1"]
    assert aggregated == {"mean": 0.42, "std": 0.0, "min": 0.42, "max": 0.42, "n": 1}


def test_aggregate_seeds_skips_missing_and_non_finite_values() -> None:
    runs = [{"micro_f1": 0.5}, {"micro_f1": float("nan")}, {"macro_f1": 0.9}]
    aggregated = aggregate_seeds(runs, keys=("micro_f1", "macro_f1", "mAP"))
    assert aggregated["micro_f1"]["n"] == 1
    assert aggregated["macro_f1"]["n"] == 1
    assert "mAP" not in aggregated


# --------------------------------------------------------------------------- #
# Confusion counts / per-class error summary
# --------------------------------------------------------------------------- #
def test_confusion_counts_match_the_hand_counted_table() -> None:
    counts = confusion_counts(Y_TRUE, Y_PROB, threshold=0.5, classes=CLASSES)
    assert counts["A"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert counts["B"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert counts["C"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    for genre, entry in counts.items():
        assert sum(entry.values()) == Y_TRUE.shape[0], genre


def test_per_class_error_summary_agrees_with_the_confusion_counts() -> None:
    table = per_class_error_summary(
        labels=Y_TRUE, probabilities=Y_PROB, threshold=0.5, classes=CLASSES
    ).set_index("genre")
    counts = confusion_counts(Y_TRUE, Y_PROB, threshold=0.5, classes=CLASSES)

    for genre in CLASSES:
        row = table.loc[genre]
        assert (int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])) == (
            counts[genre]["tp"],
            counts[genre]["fp"],
            counts[genre]["fn"],
            counts[genre]["tn"],
        )
        assert int(row["tp"] + row["fp"] + row["fn"] + row["tn"]) == Y_TRUE.shape[0]
        assert int(row["support"]) == counts[genre]["tp"] + counts[genre]["fn"]

    assert table.loc["A", "f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert table.loc["B", "precision"] == pytest.approx(0.5, abs=TOLERANCE)
    assert table.loc["C", "recall"] == pytest.approx(0.5, abs=TOLERANCE)
    assert table.loc["C", "false_negative_rate"] == pytest.approx(0.5, abs=TOLERANCE)


# --------------------------------------------------------------------------- #
# Outcome classification and the qualitative error frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("true_row", "pred_row", "expected"),
    [
        ([1, 0, 1], [1, 0, 1], "exact"),
        ([1, 0, 1], [1, 0, 0], "partial"),
        ([1, 0, 1], [1, 1, 1], "partial"),
        ([1, 0, 0], [0, 1, 0], "wrong"),
        ([1, 0, 1], [0, 0, 0], "empty_prediction"),
        ([0, 0, 0], [0, 0, 0], "empty_prediction"),
    ],
)
def test_classify_outcome_on_hand_written_rows(
    true_row: list[int], pred_row: list[int], expected: str
) -> None:
    assert classify_outcome(np.array(true_row), np.array(pred_row)) == expected


def test_build_error_frame_columns_and_hand_checked_jaccard() -> None:
    frame = build_error_frame(
        labels=Y_TRUE,
        probabilities=Y_PROB,
        sample_ids=["s0", "s1", "s2", "s3"],
        threshold=0.5,
        classes=CLASSES,
    )

    expected_columns = {
        "sample_id", "n_true", "n_pred", "true_genres", "pred_genres",
        "tp", "fp", "fn", "missed_genres", "spurious_genres",
        "jaccard", "outcome", "max_prob",
    }
    assert expected_columns <= set(frame.columns)
    assert {f"p_{genre}" for genre in CLASSES} <= set(frame.columns)
    assert len(frame) == Y_TRUE.shape[0]

    row = frame.set_index("sample_id").loc["s0"]
    # row0: true {A, C}, predicted {A, B}  ->  |{A}| / |{A, B, C}| = 1/3
    assert row["true_genres"] == "A|C"
    assert row["pred_genres"] == "A|B"
    assert row["jaccard"] == pytest.approx(1 / 3, abs=TOLERANCE)
    assert row["outcome"] == "partial"
    assert row["missed_genres"] == "C"
    assert row["spurious_genres"] == "B"
    assert row["max_prob"] == pytest.approx(0.90)

    outcomes = frame["outcome"].value_counts().to_dict()
    assert outcomes == {"partial": 2, "exact": 2}
    assert int(frame["tp"].sum()) == 4  # matches the micro TP total
    assert int(frame["fp"].sum()) == 1
    assert int(frame["fn"].sum()) == 2


def test_select_examples_is_capped_per_outcome_and_invents_nothing() -> None:
    rng = np.random.default_rng(7)
    n = 40
    truth = (rng.random((n, 3)) < 0.4).astype(int)
    probabilities = rng.random((n, 3))
    sample_ids = [f"s{index}" for index in range(n)]
    frame = build_error_frame(
        labels=truth,
        probabilities=probabilities,
        sample_ids=sample_ids,
        threshold=0.5,
        classes=CLASSES,
    )

    n_per_outcome = 2
    picked = select_examples(frame, n_per_outcome=n_per_outcome)
    assert set(picked["sample_id"]) <= set(sample_ids)
    assert len(picked) == len(set(picked["sample_id"])), "no row may be duplicated"
    for outcome, count in picked["outcome"].value_counts().items():
        assert count <= n_per_outcome, outcome
        assert outcome in set(frame["outcome"])
    assert len(picked) <= n_per_outcome * frame["outcome"].nunique()

    empty = select_examples(frame.head(0), n_per_outcome=n_per_outcome)
    assert len(empty) == 0


# --------------------------------------------------------------------------- #
# confusion_pairs
# --------------------------------------------------------------------------- #
def test_confusion_pairs_is_square_and_its_diagonal_is_per_class_recall() -> None:
    matrix = confusion_pairs(
        labels=Y_TRUE, probabilities=Y_PROB, threshold=0.5, classes=CLASSES
    )
    assert matrix.shape == (len(CLASSES), len(CLASSES))
    assert list(matrix.index) == list(CLASSES)
    assert list(matrix.columns) == list(CLASSES)

    summary = per_class_error_summary(
        labels=Y_TRUE, probabilities=Y_PROB, threshold=0.5, classes=CLASSES
    ).set_index("genre")
    for genre in CLASSES:
        assert matrix.at[genre, genre] == pytest.approx(
            summary.at[genre, "recall"], abs=TOLERANCE
        )
    # Off-diagonal: every sample truly labelled A also got a spurious B.
    assert matrix.at["A", "B"] == pytest.approx(1.0, abs=TOLERANCE)


# --------------------------------------------------------------------------- #
# THE LEAKAGE-CRITICAL TEST
# --------------------------------------------------------------------------- #
def test_threshold_is_selected_on_validation_and_never_on_test(
    separable_labels: np.ndarray,
) -> None:
    """Prove the reported threshold cannot have been tuned on the test split."""
    val_probabilities = _shifted_probabilities(separable_labels, 0.32, 0.27)
    test_probabilities = _shifted_probabilities(separable_labels, 0.62, 0.57)

    result = evaluate_predictions(
        model_kind="multimodal",
        seed=CONFIG.training.seed,
        val_labels=separable_labels,
        val_probabilities=val_probabilities,
        test_labels=separable_labels,
        test_probabilities=test_probabilities,
        classes=CLASSES,
        config=CONFIG,
    )

    assert result.threshold_selected == pytest.approx(0.30)
    assert result.threshold_default == pytest.approx(CONFIG.evaluation.default_threshold)
    assert result.val_metrics_selected["micro_f1"] == pytest.approx(1.0, abs=TOLERANCE)

    # The reported test metrics are computed at the validation-selected threshold.
    expected_test = multilabel_metrics(
        separable_labels, test_probabilities, threshold=0.30, classes=CLASSES
    )
    assert result.test_metrics_selected["micro_f1"] == pytest.approx(
        expected_test["micro_f1"], abs=TOLERANCE
    )
    assert result.test_metrics_selected["macro_f1"] == pytest.approx(
        expected_test["macro_f1"], abs=TOLERANCE
    )

    # And the test-optimal threshold would have scored strictly better, which is
    # only possible because it was never allowed to influence the choice.
    test_optimal = multilabel_metrics(
        separable_labels, test_probabilities, threshold=0.60, classes=CLASSES
    )
    assert test_optimal["micro_f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert result.test_metrics_selected["micro_f1"] < test_optimal["micro_f1"]

    assert result.headline is result.test_metrics_selected
    assert len(result.threshold_sweep_val) == len(CONFIG.evaluation.threshold_grid)
    assert tuple(result.classes) == CLASSES
