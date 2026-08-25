"""Tests for the single inference entry point used by the notebook and the app."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from gamesense.config import CONFIG, GENRES, MODEL_DISPLAY_NAMES, MODEL_KINDS, Paths
from gamesense.inference.predictor import (
    GameSensePredictor,
    GradCAMResult,
    MissingCheckpointError,
    Prediction,
)
from gamesense.models import save_checkpoint
from gamesense.utils import save_json

SEED = 20260821
IMAGE_SIDE = CONFIG.image.image_size
#: A description long enough to clear the minimum-word warning threshold.
LONG_DESCRIPTION = (
    "Command a fleet of armoured hovercraft across a shattered desert continent, "
    "capturing supply depots, upgrading your engines and racing rival warlords to "
    "the last working refinery before the storm season closes every route."
)
SHORT_DESCRIPTION = "A tiny arcade blurb."


def _build_models() -> dict[str, Any]:
    """Instantiate one random-weight model per system, or skip when offline."""
    from gamesense.models import (
        ImageOnlyClassifier,
        MultimodalClassifier,
        TextOnlyClassifier,
    )

    torch.manual_seed(SEED)
    try:
        return {
            "image": ImageOnlyClassifier(pretrained=False),
            "text": TextOnlyClassifier(pretrained=False),
            "multimodal": MultimodalClassifier(pretrained=False),
        }
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"DistilBERT configuration unavailable offline ({type(exc).__name__}: {exc})")
        raise


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained_project(tmp_path_factory: pytest.TempPathFactory):
    """A temporary project root holding one checkpoint per system."""
    root = tmp_path_factory.mktemp("gamesense_trained")
    config = replace(CONFIG, paths=Paths(root=root))
    config.paths.ensure()
    for kind, model in _build_models().items():
        save_checkpoint(
            model,
            config.checkpoint_path(kind),
            metadata={"model_init_kwargs": {"pretrained": False}},
        )
    return config


@pytest.fixture(scope="module")
def predictor(trained_project) -> GameSensePredictor:
    return GameSensePredictor(config=trained_project, device="cpu")


@pytest.fixture(scope="module")
def screenshot():
    """A small synthetic RGB screenshot as a PIL image."""
    from PIL import Image

    pixels = np.random.default_rng(SEED).integers(0, 255, size=(90, 160, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.fixture(scope="module")
def screenshot_path(tmp_path_factory: pytest.TempPathFactory, screenshot) -> Path:
    path = tmp_path_factory.mktemp("shots") / "shot.jpg"
    screenshot.save(path, format="JPEG", quality=80)
    return path


# --------------------------------------------------------------------------- #
# Nothing trained yet
# --------------------------------------------------------------------------- #
def test_predictor_reports_every_model_as_unavailable_before_training(
    tmp_config, screenshot
) -> None:
    untrained = GameSensePredictor(config=tmp_config, device="cpu")

    availability = untrained.available_models()
    assert set(availability) == set(MODEL_KINDS)
    assert availability == {kind: False for kind in MODEL_KINDS}
    assert untrained.any_available() is False

    with pytest.raises(MissingCheckpointError) as excinfo:
        untrained.load_model("image")
    message = str(excinfo.value)
    assert "scripts/train_image.py" in message, message
    assert str(untrained.checkpoint_path("image")) in message

    with pytest.raises(MissingCheckpointError):
        untrained.predict(mode="image", image=screenshot)


def test_predict_all_returns_nothing_when_no_model_is_trained(
    tmp_config, screenshot
) -> None:
    untrained = GameSensePredictor(config=tmp_config, device="cpu")
    assert untrained.predict_all(image=screenshot, text=LONG_DESCRIPTION) == {}
    with pytest.raises(MissingCheckpointError):
        untrained.predict_all(image=screenshot, skip_missing=False)


# --------------------------------------------------------------------------- #
# Availability once checkpoints exist
# --------------------------------------------------------------------------- #
def test_all_three_systems_are_available_after_saving_checkpoints(
    predictor: GameSensePredictor,
) -> None:
    assert predictor.available_models() == {kind: True for kind in MODEL_KINDS}
    assert predictor.any_available() is True


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _assert_valid_prediction(prediction: Prediction, kind: str) -> None:
    assert isinstance(prediction, Prediction)
    assert prediction.model_kind == kind
    assert len(prediction.probabilities) == len(GENRES)
    assert set(prediction.probabilities) == set(GENRES)
    assert all(0.0 <= value <= 1.0 for value in prediction.probabilities.values())
    assert set(prediction.predicted_genres) <= set(GENRES)
    assert prediction.display_name == MODEL_DISPLAY_NAMES[kind]


def test_image_only_prediction(predictor: GameSensePredictor, screenshot) -> None:
    prediction = predictor.predict(mode="image", image=screenshot)
    _assert_valid_prediction(prediction, "image")
    assert prediction.used_image is True
    assert prediction.used_text is False
    assert prediction.warnings == []


def test_text_only_prediction(predictor: GameSensePredictor) -> None:
    prediction = predictor.predict(mode="text", text=LONG_DESCRIPTION)
    _assert_valid_prediction(prediction, "text")
    assert prediction.used_image is False
    assert prediction.used_text is True
    assert prediction.warnings == []


def test_multimodal_prediction(predictor: GameSensePredictor, screenshot) -> None:
    prediction = predictor.predict(
        mode="multimodal", image=screenshot, text=LONG_DESCRIPTION
    )
    _assert_valid_prediction(prediction, "multimodal")
    assert prediction.used_image is True
    assert prediction.used_text is True


def test_missing_inputs_and_unknown_modes_raise_value_error(
    predictor: GameSensePredictor, screenshot
) -> None:
    with pytest.raises(ValueError, match="screenshot"):
        predictor.predict(mode="image")
    with pytest.raises(ValueError, match="description"):
        predictor.predict(mode="text")
    with pytest.raises(ValueError, match="description"):
        predictor.predict(mode="multimodal", image=screenshot, text="   ")
    with pytest.raises(ValueError, match="unknown prediction mode"):
        predictor.predict(mode="audio", image=screenshot)  # type: ignore[arg-type]


def test_short_description_warns_instead_of_failing(predictor: GameSensePredictor) -> None:
    """Out-of-distribution input is a caveat to show the user, not an exception."""
    prediction = predictor.predict(mode="text", text=SHORT_DESCRIPTION)
    _assert_valid_prediction(prediction, "text")
    assert prediction.warnings, "a too-short description must be flagged"
    assert str(CONFIG.dataset.min_description_words) in prediction.warnings[0]


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def test_prepare_image_accepts_every_input_flavour(
    predictor: GameSensePredictor, screenshot, screenshot_path: Path
) -> None:
    expected = (1, 3, IMAGE_SIDE, IMAGE_SIDE)
    rng = np.random.default_rng(SEED)
    candidates: list[Any] = [
        screenshot,
        screenshot_path,
        str(screenshot_path),
        rng.integers(0, 255, size=(72, 96, 3), dtype=np.uint8),
        rng.random((72, 96, 3)).astype(np.float32),
        torch.rand(3, IMAGE_SIDE, IMAGE_SIDE),
        torch.rand(1, 3, IMAGE_SIDE, IMAGE_SIDE),
    ]
    for candidate in candidates:
        tensor = predictor.prepare_image(candidate)
        assert tuple(tensor.shape) == expected, type(candidate).__name__
        assert tensor.dtype == torch.float32


def test_prepare_text_respects_the_configured_maximum_length(
    predictor: GameSensePredictor,
) -> None:
    encoded = predictor.prepare_text(LONG_DESCRIPTION)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape == encoded["attention_mask"].shape
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["input_ids"].shape[1] <= CONFIG.text.max_length

    # A deliberately over-long description must be truncated, not rejected.
    overlong = predictor.prepare_text(" ".join(["strategy"] * 5_000))
    assert overlong["input_ids"].shape[1] == CONFIG.text.max_length
    assert overlong["attention_mask"].shape == overlong["input_ids"].shape


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def test_threshold_falls_back_to_the_default_without_a_metrics_file(
    predictor: GameSensePredictor, trained_project
) -> None:
    path = trained_project.metrics_path("multimodal", predictor.seed)
    assert not path.is_file()
    assert predictor.threshold_for("multimodal") == pytest.approx(
        CONFIG.evaluation.default_threshold
    )


def test_stored_threshold_is_read_back_and_changes_the_decision(
    predictor: GameSensePredictor, trained_project, screenshot
) -> None:
    """The served decision rule must be the one the report's metrics were computed with."""
    path = trained_project.metrics_path("image", predictor.seed)
    try:
        save_json({"threshold_selected": 0.0}, path)
        assert predictor.threshold_for("image") == pytest.approx(0.0)
        permissive = predictor.predict(mode="image", image=screenshot)
        assert permissive.threshold == pytest.approx(0.0)

        save_json({"threshold_selected": 1.0}, path)
        assert predictor.threshold_for("image") == pytest.approx(1.0)
        strict = predictor.predict(mode="image", image=screenshot)
        assert strict.threshold == pytest.approx(1.0)

        # A sigmoid is strictly inside (0, 1), so the two extremes must disagree.
        assert set(permissive.predicted_genres) == set(GENRES)
        assert strict.predicted_genres == []
        assert permissive.probabilities == strict.probabilities
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Comparison mode
# --------------------------------------------------------------------------- #
def test_predict_all_skips_models_whose_inputs_are_missing(
    predictor: GameSensePredictor, screenshot
) -> None:
    image_only = predictor.predict_all(image=screenshot)
    assert set(image_only) == {"image"}

    text_only = predictor.predict_all(text=LONG_DESCRIPTION)
    assert set(text_only) == {"text"}

    both = predictor.predict_all(image=screenshot, text=LONG_DESCRIPTION)
    assert set(both) == set(MODEL_KINDS)


def test_comparison_frame_has_one_row_per_genre_and_one_column_per_model(
    predictor: GameSensePredictor, screenshot
) -> None:
    predictions = predictor.predict_all(image=screenshot, text=LONG_DESCRIPTION)
    frame = predictor.comparison_frame(predictions)

    assert len(frame) == len(GENRES)
    assert set(frame["genre"]) == set(GENRES)
    assert len(frame.columns) == 1 + len(predictions)
    for kind in predictions:
        assert MODEL_DISPLAY_NAMES[kind] in frame.columns

    empty = predictor.comparison_frame({})
    assert list(empty.columns) == ["genre"]
    assert len(empty) == 0


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_explain_returns_matching_heatmap_base_and_overlay(
    predictor: GameSensePredictor, screenshot
) -> None:
    result = predictor.explain(image=screenshot, mode="image")

    assert isinstance(result, GradCAMResult)
    assert result.genre in GENRES
    assert result.class_index == GENRES.index(result.genre)
    assert 0.0 <= result.probability <= 1.0
    assert result.model_kind == "image"
    assert result.base_image.shape == (IMAGE_SIDE, IMAGE_SIDE, 3)
    assert result.overlay.shape == result.base_image.shape
    assert result.heatmap.shape == result.base_image.shape[:2]
    assert 0.0 <= float(result.heatmap.min()) and float(result.heatmap.max()) <= 1.0


def test_explain_rejects_models_without_a_vision_branch(
    predictor: GameSensePredictor, screenshot
) -> None:
    with pytest.raises(ValueError, match="vision branch"):
        predictor.explain(image=screenshot, mode="text")


def test_explaining_the_fused_model_requires_the_description(
    predictor: GameSensePredictor, screenshot
) -> None:
    """The fused logits depend on both branches, so the text is not optional."""
    with pytest.raises(ValueError, match="description"):
        predictor.explain(image=screenshot, mode="multimodal")


def test_explain_accepts_an_explicit_genre(
    predictor: GameSensePredictor, screenshot
) -> None:
    result = predictor.explain(image=screenshot, mode="image", genre="Strategy")
    assert result.genre == "Strategy"
    assert result.class_index == GENRES.index("Strategy")


# --------------------------------------------------------------------------- #
# Model information panel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_info_survives_minimal_metadata(
    predictor: GameSensePredictor, kind: str
) -> None:
    info = predictor.model_info(kind)

    assert info["model_kind"] == kind
    assert info["display_name"] == MODEL_DISPLAY_NAMES[kind]
    architecture = info["architecture"]
    assert architecture["num_classes"] == len(GENRES)
    assert architecture["parameters"]["total"] > 0
    assert info["threshold"] == pytest.approx(CONFIG.evaluation.default_threshold)
    # These checkpoints carry no training history; the panel must still render.
    assert info["best_epoch"] is None
    assert info["best_val_score"] is None
    assert info["seed"] == predictor.seed
    assert info["device"] == "cpu"


# --------------------------------------------------------------------------- #
# Prediction value object
# --------------------------------------------------------------------------- #
def test_prediction_ranking_and_frame(predictor: GameSensePredictor, screenshot) -> None:
    prediction = predictor.predict(mode="image", image=screenshot)

    ranked = prediction.ranked()
    assert len(ranked) == len(GENRES)
    probabilities = [value for _, value in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert [genre for genre, _ in prediction.ranked(top_k=3)] == [
        genre for genre, _ in ranked[:3]
    ]

    frame = prediction.to_frame()
    assert len(frame) == len(GENRES)
    assert set(frame["genre"]) == set(GENRES)
    assert set(frame.columns) == {"genre", "probability", "predicted", "threshold"}
    assert set(frame.loc[frame["predicted"], "genre"]) == set(prediction.predicted_genres)

    payload = prediction.as_dict()
    assert payload["model_kind"] == "image"
    assert payload["display_name"] == MODEL_DISPLAY_NAMES["image"]
    assert set(payload["probabilities"]) == set(GENRES)
