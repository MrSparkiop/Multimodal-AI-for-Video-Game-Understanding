"""Single inference entry point shared by the notebook and the Streamlit app."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..config import CONFIG, GENRES, MODEL_DISPLAY_NAMES, MODEL_KINDS, GameSenseConfig
from ..data.dataset import build_image_transforms
from ..data.preprocessing import clean_description
from ..utils import get_logger, load_json, resolve_device

__all__ = [
    "PredictionMode",
    "MissingCheckpointError",
    "Prediction",
    "GradCAMResult",
    "GameSensePredictor",
]

LOGGER = get_logger("gamesense.inference.predictor")

PredictionMode = Literal["image", "text", "multimodal"]


class MissingCheckpointError(FileNotFoundError):
    """Raised when a requested model has not been trained yet."""


@dataclass
class Prediction:
    """One model's output for one game."""

    model_kind: str
    threshold: float | list[float]
    probabilities: dict[str, float]
    predicted_genres: list[str]
    used_image: bool
    used_text: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY_NAMES.get(self.model_kind, self.model_kind)

    def ranked(self, *, top_k: int | None = None) -> list[tuple[str, float]]:
        """Genres sorted by probability, highest first."""
        items = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return items[:top_k] if top_k else items

    def to_frame(self) -> Any:
        """Tabular view: one row per genre, with the threshold that applied to it."""
        import pandas as pd

        if np.isscalar(self.threshold):
            per_genre = dict.fromkeys(self.probabilities, float(self.threshold))
        else:
            values = np.asarray(self.threshold, dtype=float).ravel()
            per_genre = {
                genre: float(values[index])
                for index, genre in enumerate(self.probabilities)
                if index < values.size
            }
        rows = [
            {
                "genre": genre,
                "probability": probability,
                "predicted": genre in self.predicted_genres,
                "threshold": per_genre.get(genre),
            }
            for genre, probability in self.ranked()
        ]
        return pd.DataFrame(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "display_name": self.display_name,
            "threshold": self.threshold,
            "probabilities": self.probabilities,
            "predicted_genres": self.predicted_genres,
            "used_image": self.used_image,
            "used_text": self.used_text,
            "warnings": self.warnings,
        }


@dataclass
class GradCAMResult:
    """A Grad-CAM explanation ready for display."""

    genre: str
    class_index: int
    probability: float
    heatmap: np.ndarray
    base_image: np.ndarray
    overlay: np.ndarray
    model_kind: str


class GameSensePredictor:
    """Lazy-loading inference facade over the three trained systems."""

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        device: str | torch.device | None = None,
        seed: int | None = None,
        classes: Sequence[str] = GENRES,
    ) -> None:
        self.config = config
        self.device = resolve_device(device if device is not None else config.device)
        self.seed = config.training.seed if seed is None else int(seed)
        self.classes = tuple(classes)
        self._models: dict[str, Any] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self._tokenizer: Any | None = None
        self._transform = build_image_transforms(train=False, image=config.image)

    # -- availability ------------------------------------------------------- #
    def checkpoint_path(self, model_kind: str) -> Path:
        return self.config.checkpoint_path(model_kind, self.seed)

    def available_models(self) -> dict[str, bool]:
        """Which of the three systems have a checkpoint on disk."""
        return {kind: self.checkpoint_path(kind).is_file() for kind in MODEL_KINDS}

    def any_available(self) -> bool:
        return any(self.available_models().values())

    # -- loading ------------------------------------------------------------ #
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from ..data.loader import get_tokenizer

            self._tokenizer = get_tokenizer(self.config.text.model_name)
        return self._tokenizer

    def load_model(self, model_kind: str) -> Any:
        """Load (and memoise) one trained model."""
        if model_kind in self._models:
            return self._models[model_kind]
        path = self.checkpoint_path(model_kind)
        if not path.is_file():
            raise MissingCheckpointError(
                f"No trained '{model_kind}' model at {path}.\n"
                f"Train it with:  python scripts/train_{model_kind}.py"
            )
        from ..models import load_model_from_checkpoint

        model, payload = load_model_from_checkpoint(
            path, kind=model_kind, config=self.config, device=self.device
        )
        model.eval()
        self._models[model_kind] = model
        self._payloads[model_kind] = payload
        LOGGER.info("Loaded %s model from %s", model_kind, path.name)
        return model

    def model_info(self, model_kind: str) -> dict[str, Any]:
        """Architecture + training metadata for the "Model Information" panel."""
        model = self.load_model(model_kind)
        payload = self._payloads.get(model_kind, {})
        metadata = payload.get("metadata", {})
        history_meta = metadata.get("history_meta", {})
        return {
            "model_kind": model_kind,
            "display_name": MODEL_DISPLAY_NAMES.get(model_kind, model_kind),
            "checkpoint": str(self.checkpoint_path(model_kind)),
            "architecture": model.describe(),
            "best_epoch": metadata.get("best_epoch"),
            "best_val_score": metadata.get("best_val_score"),
            "monitor": metadata.get("monitor"),
            "seed": history_meta.get("seed", self.seed),
            "epochs_run": history_meta.get("epochs_run"),
            "criterion": history_meta.get("criterion"),
            "threshold": self.threshold_for(model_kind),
            "device": str(self.device),
        }

    # -- thresholds --------------------------------------------------------- #
    def threshold_source(self, model_kind: str) -> tuple[float | list[float], str]:
        """Return ``(threshold, provenance)``: provenance is ``"validation"`` or ``"default"``."""
        path = self.config.metrics_path(model_kind, self.seed)
        if path.is_file():
            try:
                payload = load_json(path)
                selected = payload.get("threshold_selected")
                if selected is not None:
                    return selected, "validation"
            except (ValueError, OSError):  # pragma: no cover - corrupt file
                LOGGER.warning("Could not read threshold from %s", path)
        return float(self.config.evaluation.default_threshold), "default"

    def threshold_for(self, model_kind: str) -> float | list[float]:
        """Return the validation-selected threshold, falling back to the default."""
        return self.threshold_source(model_kind)[0]

    def skip_reasons(
        self, *, image: Any | None = None, text: str | None = None
    ) -> dict[str, str]:
        """Explain, per model, why it cannot be run on the given inputs.

        Returns only the models that cannot run, mapped to a human-readable reason.
        """
        available = self.available_models()
        has_image = image is not None
        has_text = bool(text and text.strip())
        reasons: dict[str, str] = {}
        for kind in MODEL_KINDS:
            if not available.get(kind):
                reasons[kind] = "no checkpoint yet - this model has not been trained"
            elif kind in ("image", "multimodal") and not has_image:
                reasons[kind] = "no screenshot supplied"
            elif kind in ("text", "multimodal") and not has_text:
                reasons[kind] = "no game description supplied"
        return reasons

    # -- preprocessing ------------------------------------------------------ #
    def prepare_image(self, image: Any) -> torch.Tensor:
        """Turn a PIL image / path / array into a ``(1, 3, H, W)`` model input."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                pil = handle.convert("RGB")
        elif isinstance(image, np.ndarray):
            array = image
            if array.dtype != np.uint8:
                array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(array).convert("RGB")
        elif torch.is_tensor(image):
            if image.dim() == 4:
                return image.to(self.device)
            return image.unsqueeze(0).to(self.device)
        else:
            pil = image.convert("RGB")
        return self._transform(pil).unsqueeze(0).to(self.device)

    def prepare_text(self, text: str) -> dict[str, torch.Tensor]:
        """Clean and tokenise a free-text description."""
        cleaned = clean_description(text)
        encoded = self.tokenizer()(
            [cleaned],
            padding=True,
            truncation=True,
            max_length=self.config.text.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    # -- prediction --------------------------------------------------------- #
    def predict(
        self,
        *,
        mode: PredictionMode = "multimodal",
        image: Any | None = None,
        text: str | None = None,
    ) -> Prediction:
        """Predict genre probabilities with one of the three systems.

        Raises ValueError if the inputs *mode* needs are missing, MissingCheckpointError if it
        is untrained.
        """
        if mode not in MODEL_KINDS:
            raise ValueError(f"unknown prediction mode {mode!r}; expected one of {MODEL_KINDS}")
        needs_image = mode in ("image", "multimodal")
        needs_text = mode in ("text", "multimodal")
        if needs_image and image is None:
            raise ValueError(f"mode '{mode}' requires a screenshot")
        if needs_text and not (text and text.strip()):
            raise ValueError(f"mode '{mode}' requires a game description")

        model = self.load_model(mode)
        inputs: dict[str, Any] = {}
        warnings: list[str] = []
        if needs_image:
            inputs["image"] = self.prepare_image(image)
        if needs_text:
            cleaned = clean_description(text or "")
            if len(cleaned.split()) < self.config.dataset.min_description_words:
                warnings.append(
                    f"The description has only {len(cleaned.split())} words; the model was "
                    f"trained on descriptions of at least "
                    f"{self.config.dataset.min_description_words} words, so the prediction "
                    "may be unreliable."
                )
            inputs.update(self.prepare_text(text or ""))

        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**inputs))[0].detach().cpu().numpy()

        threshold = self.threshold_for(mode)
        vector = (
            np.full(len(self.classes), float(threshold))
            if np.isscalar(threshold)
            else np.asarray(threshold, dtype=np.float64)
        )
        predicted = [
            genre
            for index, genre in enumerate(self.classes)
            if probabilities[index] >= vector[index]
        ]
        return Prediction(
            model_kind=mode,
            threshold=threshold,
            probabilities={
                genre: float(probabilities[index]) for index, genre in enumerate(self.classes)
            },
            predicted_genres=predicted,
            used_image=needs_image,
            used_text=needs_text,
            warnings=warnings,
        )

    def predict_all(
        self, *, image: Any | None = None, text: str | None = None, skip_missing: bool = True
    ) -> dict[str, Prediction]:
        """Run every model whose inputs are available (comparison mode)."""
        results: dict[str, Prediction] = {}
        for kind in MODEL_KINDS:
            if kind in ("image", "multimodal") and image is None:
                continue
            if kind in ("text", "multimodal") and not (text and text.strip()):
                continue
            try:
                results[kind] = self.predict(mode=kind, image=image, text=text)
            except MissingCheckpointError:
                if not skip_missing:
                    raise
                LOGGER.info("Skipping %s in comparison mode (no checkpoint)", kind)
        return results

    def comparison_frame(
        self, predictions: dict[str, Prediction], *, sort_by: str | None = None
    ) -> Any:
        """Genres as rows, models as columns -- the model-comparison table."""
        import pandas as pd

        if not predictions:
            return pd.DataFrame(columns=["genre"])
        data = {"genre": list(self.classes)}
        for kind, prediction in predictions.items():
            data[MODEL_DISPLAY_NAMES.get(kind, kind)] = [
                prediction.probabilities[genre] for genre in self.classes
            ]
        frame = pd.DataFrame(data)
        numeric = [column for column in frame.columns if column != "genre"]

        preferred = sort_by or ("multimodal" if "multimodal" in predictions else None)
        preferred_column = MODEL_DISPLAY_NAMES.get(preferred, preferred) if preferred else None
        if preferred_column in numeric:
            order = frame[preferred_column]
        else:
            order = frame[numeric].mean(axis=1)
        return frame.assign(_order=order).sort_values("_order", ascending=False).drop(
            columns="_order"
        ).reset_index(drop=True)

    # -- explainability ----------------------------------------------------- #
    def explain(
        self,
        *,
        image: Any,
        text: str | None = None,
        mode: PredictionMode = "image",
        genre: str | int | None = None,
        alpha: float = 0.45,
    ) -> GradCAMResult:
        """Produce a Grad-CAM explanation for the vision branch."""
        if mode not in ("image", "multimodal"):
            raise ValueError("Grad-CAM is only defined for models with a vision branch")
        from ..explainability.gradcam import GradCAM, denormalize_image, overlay_heatmap

        model = self.load_model(mode)
        image_tensor = self.prepare_image(image)
        extras: dict[str, Any] = {}
        if mode == "multimodal":
            if not (text and text.strip()):
                raise ValueError("explaining the multimodal model requires the description")
            extras.update(self.prepare_text(text))

        with GradCAM(model, config=self.config, classes=self.classes) as explainer:
            heatmap, class_index, probability = explainer(
                image_tensor, class_index=genre, model_inputs=extras
            )
        base = denormalize_image(image_tensor, config=self.config)
        return GradCAMResult(
            genre=self.classes[class_index],
            class_index=class_index,
            probability=probability,
            heatmap=heatmap,
            base_image=base,
            overlay=overlay_heatmap(base, heatmap, alpha=alpha, config=self.config),
            model_kind=mode,
        )
