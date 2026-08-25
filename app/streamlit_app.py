"""Streamlit front-end for GameSense."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path bootstrap: makes ``streamlit run app/streamlit_app.py`` work from a fresh clone without
# ``pip install -e .``.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import io  # noqa: E402  - after the bootstrap above
from typing import Any, Final  # noqa: E402

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from gamesense import __author__, __version__  # noqa: E402
from gamesense.config import CONFIG, GENRES, MODEL_DISPLAY_NAMES, MODEL_KINDS  # noqa: E402
from gamesense.inference import (  # noqa: E402
    GameSensePredictor,
    GradCAMResult,
    MissingCheckpointError,
    Prediction,
)
from gamesense.utils import set_seed  # noqa: E402

# --------------------------------------------------------------------------- #
# Presentation constants (never model or metric values -- those come from
# CONFIG and from the predictor).
# --------------------------------------------------------------------------- #
PAGE_TITLE: Final[str] = "GameSense"
PAGE_ICON: Final[str] = "🎮"
SUBTITLE: Final[str] = "Multimodal AI for Video Game Understanding"

#: Sidebar label -> predictor mode.  "Multimodal" is first, hence the default.
MODE_OPTIONS: Final[dict[str, str]] = {
    "Multimodal": "multimodal",
    "Image Only": "image",
    "Text Only": "text",
}
MODE_HELP: Final[dict[str, str]] = {
    "multimodal": "Late fusion: screenshot embedding + description embedding -> MLP.",
    "image": "ResNet18 over the screenshot only.",
    "text": "DistilBERT over the store description only.",
}
#: Modes whose forward pass consumes an image / a description.
IMAGE_MODES: Final[tuple[str, ...]] = ("image", "multimodal")
TEXT_MODES: Final[tuple[str, ...]] = ("text", "multimodal")
#: Models with a vision branch, i.e. the ones Grad-CAM is defined for.
GRADCAM_MODES: Final[tuple[str, ...]] = ("image", "multimodal")

UPLOAD_TYPES: Final[tuple[str, ...]] = ("jpg", "jpeg", "png")
TEXT_AREA_HEIGHT: Final[int] = 160
EXAMPLE_PLACEHOLDER: Final[str] = (
    "Explore an open fantasy world filled with quests, enemies, weapons and magical creatures."
)
#: Hand-written illustrative prompts -- not rows of the dataset, no ground truth.
EXAMPLE_DESCRIPTIONS: Final[tuple[tuple[str, str], ...]] = (
    (
        "Open-world quest",
        "A vast open world where you gather a party, craft weapons and armour, level up "
        "your character and take on story quests, side quests and monster hunts across "
        "forests, ruined castles and snowy mountain passes.",
    ),
    (
        "Arcade racing",
        "Take the wheel of tuned street cars and race rivals through neon city circuits "
        "and coastal highways, drifting through tight corners, unlocking new vehicles and "
        "climbing the online leaderboards in split-second time trials.",
    ),
    (
        "Management sim",
        "Design and manage a growing city: lay out roads and power grids, balance the "
        "budget, keep citizens happy, respond to natural disasters and watch your "
        "simulated population react to every planning decision you make.",
    ),
)

#: Commands that produce the artefacts this page serves.
SETUP_COMMANDS: Final[tuple[str, ...]] = (
    "python scripts/prepare_data.py",
    "python scripts/train_image.py",
    "python scripts/train_text.py",
    "python scripts/train_multimodal.py",
)

MAIN_COLUMN_RATIO: Final[tuple[int, int]] = (5, 6)
GENRE_ROW_RATIO: Final[tuple[int, int, int]] = (3, 7, 2)
CHART_HEIGHT: Final[int] = 320
HEATMAP_COLORMAP: Final[str] = "jet"
PERCENT: Final[float] = 100.0
UINT8_MAX: Final[float] = 255.0

# Session-state keys (kept as constants so widgets and callbacks cannot drift).
_STATE_MODE: Final[str] = "gs_mode"
_STATE_DESCRIPTION: Final[str] = "gs_description"
_STATE_UPLOAD: Final[str] = "gs_upload"
_STATE_RESULT: Final[str] = "gs_result"
_STATE_COMPARE: Final[str] = "gs_compare"
_STATE_CAM_GENRE: Final[str] = "gs_cam_genre"
_STATE_CAM_MODEL: Final[str] = "gs_cam_model"


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_predictor(device: str, seed: int) -> GameSensePredictor:
    """Build (once per session) the predictor that serves every inference."""
    set_seed(seed)
    return GameSensePredictor(device=device, seed=seed)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def format_threshold(threshold: float | list[float] | None) -> str:
    """Render a scalar or per-class decision threshold for display."""
    if threshold is None:
        return "unknown"
    if np.isscalar(threshold):
        return f"{float(threshold):.2f}"
    values = [float(value) for value in threshold]  # type: ignore[union-attr]
    if not values:
        return "unknown"
    return f"per-class ({min(values):.2f} - {max(values):.2f})"


def threshold_vector(threshold: float | list[float], size: int) -> list[float]:
    """Expand a scalar threshold to one value per genre."""
    if np.isscalar(threshold):
        return [float(threshold)] * size
    return [float(value) for value in threshold]  # type: ignore[union-attr]


def threshold_note(predictor: GameSensePredictor, model_kind: str) -> str:
    """Explain where the served threshold came from."""
    _, provenance = predictor.threshold_source(model_kind)
    path = predictor.config.metrics_path(model_kind, predictor.seed)
    if provenance == "validation":
        return f"selected on the validation split, read back from {path.name}"
    return (
        f"project default (CONFIG.evaluation.default_threshold = "
        f"{CONFIG.evaluation.default_threshold:.2f}); no {path.name} yet"
    )


def _number(value: Any) -> str:
    """Format an integer count, or say so when it was not recorded."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    return "not recorded"


def _score(value: Any) -> str:
    """Format a floating-point validation score without inventing one."""
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "not recorded"


def as_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a float image in ``[0, 1]`` (or ``[0, 255]``) to ``uint8`` RGB."""
    data = np.asarray(array, dtype=np.float32)
    if data.size and float(data.max()) > 1.5:  # already in 0-255
        data = data / UINT8_MAX
    return (np.clip(data, 0.0, 1.0) * UINT8_MAX).astype(np.uint8)


def colorize_heatmap(heatmap: np.ndarray, *, colormap: str = HEATMAP_COLORMAP) -> np.ndarray:
    """Map a ``(H, W)`` heat map in ``[0, 1]`` to an RGB ``uint8`` image."""
    coloured = mpl.colormaps[colormap](np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0))
    return (coloured[..., :3] * UINT8_MAX).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Static page chrome
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """Title, subtitle and the research-question expander."""
    st.title(PAGE_TITLE)
    st.markdown(f"##### {SUBTITLE}")
    with st.expander("About this project", expanded=False):
        st.markdown(
            "**Research question.** Does combining a gameplay *screenshot* with the "
            "game's *store description* improve multi-label genre prediction over using "
            "either modality alone? Three systems are trained on identical splits, the "
            "identical label space and identical metrics: an **image-only** classifier "
            "(ImageNet-pretrained ResNet18 over the screenshot), a **text-only** "
            "classifier (pretrained DistilBERT over the description) and a "
            "**multimodal** classifier that concatenates both embeddings and feeds them "
            "to a small fusion MLP.\n\n"
            f"The {len(GENRES)} genres ({', '.join(GENRES)}) are **not mutually "
            "exclusive**: a game can be Action *and* Adventure *and* RPG at once. The "
            "task is therefore multi-label, the output layer uses one independent "
            "**sigmoid** per genre rather than a softmax, and a genre is predicted when "
            "its probability passes a decision threshold that was tuned on the "
            "validation split. Probabilities across genres do not sum to 100%."
        )


def render_untrained_banner(predictor: GameSensePredictor) -> None:
    """Explain exactly how to produce the checkpoints this page needs."""
    st.warning(
        "**No trained models found yet.** This page serves checkpoints written by the "
        "training scripts, and none exist so far, so analysis is disabled. Run the "
        "following from the project root (each training script writes one checkpoint):",
        icon=":material/build:",
    )
    st.code("\n".join(SETUP_COMMANDS), language="bash")
    expected = ", ".join(f"`{predictor.checkpoint_path(kind).name}`" for kind in MODEL_KINDS)
    st.caption(
        f"Expected files in `{CONFIG.paths.checkpoints.name}/` for seed "
        f"{predictor.seed}: {expected}. Training even one of them re-enables the "
        "corresponding mode; the page never invents predictions in the meantime."
    )


def render_footer(predictor: GameSensePredictor) -> None:
    """One-line reproducibility note."""
    st.divider()
    st.caption(
        f"GameSense v{__version__} by **{__author__}** - every probability on this page is "
        f"computed live by `gamesense.inference.GameSensePredictor` from the checkpoints of "
        f"seed **{predictor.seed}** on device `{predictor.device}`, with the decision "
        "thresholds read back from `results/metrics/`. Nothing is hard-coded, so the "
        "page always agrees with the reported experiments."
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(predictor: GameSensePredictor) -> str:
    """Draw the sidebar and return the selected predictor mode."""
    with st.sidebar:
        st.header("Settings")
        label = st.radio(
            "Prediction mode",
            list(MODE_OPTIONS),
            index=0,
            key=_STATE_MODE,
            help="Which of the three trained systems should analyse the inputs.",
        )
        mode = MODE_OPTIONS.get(label or next(iter(MODE_OPTIONS)), "multimodal")
        st.caption(MODE_HELP[mode])
        st.divider()
        render_availability(predictor)
        st.divider()
        st.markdown("**Runtime**")
        st.markdown(
            f"- Device: `{predictor.device}`\n"
            f"- Checkpoint seed: `{predictor.seed}`\n"
            f"- Label space: {len(GENRES)} genres, multi-label (sigmoid)"
        )
        render_model_information(predictor)
    return mode


def render_availability(predictor: GameSensePredictor) -> None:
    """Green check / grey dash per model, straight from the filesystem."""
    st.markdown("**Model availability**")
    availability = predictor.available_models()
    for kind in MODEL_KINDS:
        name = MODEL_DISPLAY_NAMES.get(kind, kind)
        if availability.get(kind, False):
            st.markdown(f":green[**✔**] {name}")
        else:
            st.markdown(f":gray[**–** {name} (not trained)]")


def render_model_information(predictor: GameSensePredictor) -> None:
    """Architecture and training metadata for every trained model."""
    availability = predictor.available_models()
    trained = [kind for kind in MODEL_KINDS if availability.get(kind, False)]
    with st.expander("Model Information", expanded=False):
        if not trained:
            st.caption(
                "Architecture, parameter counts, best epoch, monitored metric, decision "
                "threshold and loss appear here once a checkpoint exists."
            )
            return
        for position, kind in enumerate(trained):
            try:
                with st.spinner(f"Reading the {kind} checkpoint..."):
                    info = predictor.model_info(kind)
            except Exception as error:  # noqa: BLE001 - a bad file must not break the page
                st.warning(f"Could not read the {kind} checkpoint: {error}")
                continue
            if position:
                st.divider()
            st.markdown("\n".join(model_info_lines(info)))


def model_info_lines(info: dict[str, Any]) -> list[str]:
    """Turn :meth:`GameSensePredictor.model_info` into markdown bullet lines."""
    architecture = info.get("architecture") or {}
    parameters = architecture.get("parameters") or {}
    lines = [f"**{info.get('display_name', info.get('model_kind', 'model'))}**", ""]
    lines.append(f"- Class: `{architecture.get('class', 'unknown')}`")
    if architecture.get("backbone"):
        lines.append(f"- Vision backbone: `{architecture['backbone']}`")
    if architecture.get("text_model"):
        lines.append(f"- Language model: `{architecture['text_model']}`")
    if architecture.get("fusion"):
        lines.append(f"- Fusion: {architecture['fusion']}")
    lines.append(
        f"- Parameters: {_number(parameters.get('total'))} total, "
        f"{_number(parameters.get('trainable'))} trainable, "
        f"{_number(parameters.get('frozen'))} frozen"
    )
    lines.append(f"- Best epoch: {_number(info.get('best_epoch'))}")
    lines.append(
        f"- Monitored: `{info.get('monitor') or CONFIG.training.monitor_metric}` = "
        f"{_score(info.get('best_val_score'))} (validation)"
    )
    lines.append(f"- Epochs run: {_number(info.get('epochs_run'))}, seed {info.get('seed')}")
    lines.append(f"- Decision threshold: {format_threshold(info.get('threshold'))}")
    lines.append(f"- Criterion: `{info.get('criterion') or 'not recorded'}`")
    lines.append(f"- Checkpoint: `{Path(str(info.get('checkpoint', ''))).name}`")
    return lines


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def read_uploaded_image(upload: Any) -> tuple[Image.Image | None, str | None]:
    """Decode an uploaded file into an RGB image."""
    try:
        payload = upload.getvalue()
        if not payload:
            return None, "That file is empty. Please upload a JPG or PNG screenshot."
        with Image.open(io.BytesIO(payload)) as handle:
            handle.load()
            return handle.convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return None, (
            f"That file could not be read as an image ({error}). Please upload an "
            "uncorrupted JPG or PNG screenshot."
        )


def render_image_input() -> Image.Image | None:
    """Screenshot uploader plus preview; ``None`` when nothing usable was given."""
    upload = st.file_uploader(
        "Game screenshot",
        type=list(UPLOAD_TYPES),
        key=_STATE_UPLOAD,
        help="A single gameplay screenshot (JPG or PNG). Used by the image and "
        "multimodal models.",
    )
    if upload is None:
        st.caption("No screenshot yet - required by the image and multimodal models.")
        return None
    image, error = read_uploaded_image(upload)
    if error is not None or image is None:
        st.error(error or "The uploaded image could not be decoded.")
        return None
    st.image(
        image,
        caption=f"{upload.name} - {image.width}x{image.height} px "
        f"(resized to {CONFIG.image.image_size}px for the model)",
        width="stretch",
    )
    return image


def _use_example(description: str) -> None:
    """Button callback: fill the description box before the next rerun."""
    st.session_state[_STATE_DESCRIPTION] = description


def render_example_buttons() -> None:
    """Offer a few illustrative descriptions so the page is usable without data."""
    st.caption(
        "Or try an illustrative example (hand-written prompts, not dataset rows; "
        "no screenshot is supplied, so they exercise the text-only model):"
    )
    columns = st.columns(len(EXAMPLE_DESCRIPTIONS))
    for column, (label, description) in zip(columns, EXAMPLE_DESCRIPTIONS, strict=True):
        column.button(
            label,
            key=f"gs_example_{label}",
            on_click=_use_example,
            args=(description,),
            width="stretch",
            help="Fills the description box with an example.",
        )


def render_text_input() -> str:
    """Description text area with a live word count and a soft length warning."""
    st.session_state.setdefault(_STATE_DESCRIPTION, "")
    text = st.text_area(
        "Game description",
        key=_STATE_DESCRIPTION,
        height=TEXT_AREA_HEIGHT,
        placeholder=EXAMPLE_PLACEHOLDER,
        help="The store description. Used by the text and multimodal models.",
    )
    text = text or ""
    words = len(text.split())
    minimum = CONFIG.dataset.min_description_words
    if not text.strip():
        st.caption(f"0 words - the text models need at least {minimum} words to be reliable.")
    elif words < minimum:
        st.warning(
            f"{words} words: below the {minimum}-word minimum used when building the "
            "training set, so a text-based prediction may be unreliable.",
            icon=":material/warning:",
        )
    else:
        st.caption(f"{words} words (truncated to {CONFIG.text.max_length} tokens by the model).")
    render_example_buttons()
    return text


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def missing_input_message(mode: str, image: Image.Image | None, text: str) -> str | None:
    """Return why *mode* cannot run on these inputs, or ``None`` when it can."""
    if mode in IMAGE_MODES and image is None:
        return (
            f"**{MODEL_DISPLAY_NAMES.get(mode, mode)}** needs a screenshot. Upload a JPG "
            "or PNG above, or switch the sidebar to *Text Only*."
        )
    if mode in TEXT_MODES and not text.strip():
        return (
            f"**{MODEL_DISPLAY_NAMES.get(mode, mode)}** needs a game description. Type "
            "one above (or press an example button), or switch the sidebar to "
            "*Image Only*."
        )
    return None


def run_analysis(
    predictor: GameSensePredictor, mode: str, image: Image.Image | None, text: str
) -> None:
    """Run one prediction and store it in the session, or explain why it cannot run."""
    st.session_state.pop(_STATE_RESULT, None)
    availability = predictor.available_models()
    if not availability.get(mode, False):
        trained = [MODEL_DISPLAY_NAMES.get(k, k) for k, ok in availability.items() if ok]
        alternative = (
            f" Already trained and selectable in the sidebar: {', '.join(trained)}."
            if trained
            else ""
        )
        st.error(
            f"The **{MODEL_DISPLAY_NAMES.get(mode, mode)}** model has no checkpoint yet. "
            f"Train it with `python scripts/train_{mode}.py`.{alternative}"
        )
        return
    message = missing_input_message(mode, image, text)
    if message is not None:
        st.warning(message, icon=":material/info:")
        return
    try:
        with st.spinner(f"Running {MODEL_DISPLAY_NAMES.get(mode, mode)}..."):
            prediction = predictor.predict(mode=mode, image=image, text=text or None)
    except MissingCheckpointError as error:
        st.error(str(error))
        return
    except (ValueError, RuntimeError, OSError) as error:
        st.error(f"The model could not process these inputs: {error}")
        return
    except Exception as error:  # noqa: BLE001 - the page must never show a traceback
        st.error(f"Unexpected inference error ({type(error).__name__}): {error}")
        return
    st.session_state[_STATE_RESULT] = {
        "mode": mode,
        "prediction": prediction,
        "image": image,
        "text": text,
    }


def render_input_column(predictor: GameSensePredictor, mode: str) -> None:
    """Inputs plus the primary action button."""
    st.subheader("Game inputs")
    image = render_image_input()
    text = render_text_input()
    trained = predictor.any_available()
    clicked = st.button(
        "Analyze Game",
        type="primary",
        width="stretch",
        disabled=not trained,
        help="Train a model first." if not trained else "Run the selected model.",
    )
    if clicked:
        run_analysis(predictor, mode, image, text)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def render_probability_rows(prediction: Prediction) -> None:
    """One row per genre: name, probability bar, percentage."""
    limits = threshold_vector(prediction.threshold, len(GENRES))
    index_of = {genre: index for index, genre in enumerate(GENRES)}
    for genre, probability in prediction.ranked():
        above = genre in prediction.predicted_genres
        limit = limits[index_of.get(genre, 0)]
        name_col, bar_col, value_col = st.columns(GENRE_ROW_RATIO, vertical_alignment="center")
        name_col.markdown(f"**:green[{genre}]**" if above else f":gray[{genre}]")
        bar_col.progress(float(np.clip(probability, 0.0, 1.0)))
        percentage = f"{probability * PERCENT:.1f}%"
        value_col.markdown(
            f"**:green[{percentage}]**" if above else f":gray[{percentage}]",
            help=f"threshold {limit:.2f}",
        )


def render_predictions(predictor: GameSensePredictor, prediction: Prediction) -> None:
    """The "Predicted Game Genres" block, including threshold provenance."""
    st.subheader("Predicted Game Genres")
    st.caption(
        f"{prediction.display_name} - decision threshold "
        f"**{format_threshold(prediction.threshold)}** "
        f"({threshold_note(predictor, prediction.model_kind)}). Genres are independent "
        "sigmoids, so the probabilities do not sum to 100%."
    )
    for warning in prediction.warnings:
        st.warning(warning, icon=":material/warning:")
    render_probability_rows(prediction)
    if prediction.predicted_genres:
        st.success(
            "**Predicted label set:** " + ", ".join(prediction.predicted_genres),
            icon=":material/label:",
        )
    else:
        st.info(
            "No genre reached the decision threshold, so the predicted label set is "
            "empty. That is a valid multi-label outcome; a longer description or a more "
            "representative screenshot usually raises the scores.",
            icon=":material/info:",
        )


def render_probability_chart(prediction: Prediction) -> None:
    """Horizontal bar chart of all genres, highest probability first."""
    frame = pd.DataFrame(prediction.ranked(), columns=["genre", "probability"])
    st.bar_chart(
        frame,
        x="genre",
        y="probability",
        horizontal=True,
        sort=False,
        height=CHART_HEIGHT,
    )


def render_result_column(
    predictor: GameSensePredictor, result: dict[str, Any] | None, mode: str
) -> None:
    """Right-hand column: either the guidance text or the prediction."""
    if result is None:
        st.subheader("Predicted Game Genres")
        st.info(
            "Provide the inputs on the left and press **Analyze Game** - the "
            "per-genre probabilities, the predicted label set and the model comparison "
            "will appear here.",
            icon=":material/insights:",
        )
        return
    prediction: Prediction = result["prediction"]
    if result.get("mode") != mode:
        st.caption(
            f"Showing the previous analysis by *{prediction.display_name}*. Press "
            "**Analyze Game** to re-run with the mode now selected in the sidebar."
        )
    render_predictions(predictor, prediction)
    render_probability_chart(prediction)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def percent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy *frame* with every probability column formatted as a percentage."""
    display = frame.copy()
    for column in display.columns:
        if column == "genre":
            continue
        display[column] = display[column].map(lambda value: f"{float(value) * PERCENT:.1f}%")
    return display


def skip_notes(
    predictor: GameSensePredictor,
    image: Image.Image | None,
    text: str,
    produced: dict[str, Prediction],
) -> list[str]:
    """Say explicitly which models were left out of the comparison, and why."""
    availability = predictor.available_models()
    notes: list[str] = []
    for kind in MODEL_KINDS:
        if kind in produced:
            continue
        reasons: list[str] = []
        if not availability.get(kind, False):
            reasons.append(f"no checkpoint yet (`python scripts/train_{kind}.py`)")
        if kind in IMAGE_MODES and image is None:
            reasons.append("no screenshot supplied")
        if kind in TEXT_MODES and not text.strip():
            reasons.append("no description supplied")
        if not reasons:
            reasons.append("it produced no output")
        notes.append(f"- **{MODEL_DISPLAY_NAMES.get(kind, kind)}**: " + "; ".join(reasons))
    return notes


def render_comparison(
    predictor: GameSensePredictor, image: Image.Image | None, text: str
) -> None:
    """Run every model whose inputs are available and tabulate the differences."""
    st.subheader("Model Comparison")
    enabled = st.toggle(
        "Compare all trained systems on these inputs",
        key=_STATE_COMPARE,
        help="Runs each model that has a checkpoint and the inputs it needs.",
    )
    if not enabled:
        st.caption(
            "Turn this on to see the image-only, text-only and multimodal probabilities "
            "side by side - the direct evidence for the research question."
        )
        return
    try:
        with st.spinner("Running every available model..."):
            predictions = predictor.predict_all(image=image, text=text or None)
    except Exception as error:  # noqa: BLE001 - comparison must not break the page
        st.error(f"The comparison could not be completed ({type(error).__name__}): {error}")
        return
    if predictions:
        frame = predictor.comparison_frame(predictions)
        st.dataframe(percent_frame(frame), hide_index=True, width="stretch")
        long = frame.melt(id_vars="genre", var_name="model", value_name="probability")
        st.bar_chart(
            long, x="genre", y="probability", color="model", stack=False, height=CHART_HEIGHT
        )
    else:
        st.info("No model could be run on these inputs.", icon=":material/info:")
    notes = skip_notes(predictor, image, text, predictions)
    if notes:
        st.info("**Not compared:**\n" + "\n".join(notes), icon=":material/filter_alt:")


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def render_gradcam_panels(result: GradCAMResult) -> None:
    """Original / heat map / overlay side by side, with an honest caption."""
    original, heatmap, overlay = st.columns(3)
    original.image(
        as_uint8(result.base_image),
        caption="Screenshot as the model sees it",
        width="stretch",
    )
    heatmap.image(
        colorize_heatmap(result.heatmap),
        caption=f"Grad-CAM heat map: {result.genre}",
        width="stretch",
    )
    overlay.image(as_uint8(result.overlay), caption="Heat map over screenshot", width="stretch")
    st.caption(
        f"Warm / red regions are the pixels whose features contributed most to the "
        f"**{result.genre}** logit of the "
        f"{MODEL_DISPLAY_NAMES.get(result.model_kind, result.model_kind)} model "
        f"(probability {result.probability * PERCENT:.1f}%); blue regions contributed "
        "little. Grad-CAM visualises a correlation between activations and that one "
        "score - it is not a causal explanation, and it says nothing about the text "
        "branch of the multimodal model."
    )


def render_gradcam(
    predictor: GameSensePredictor,
    image: Image.Image | None,
    text: str,
    prediction: Prediction,
) -> None:
    """Grad-CAM section, shown only when a vision model and a screenshot exist."""
    availability = predictor.available_models()
    options = [kind for kind in GRADCAM_MODES if availability.get(kind, False)]
    if not options:
        return
    if image is None:
        st.caption(
            "Grad-CAM explains the vision branch - upload a screenshot and analyse "
            "again to see which pixels drove a genre."
        )
        return
    st.subheader("Explainability (Grad-CAM)")
    ranked = prediction.ranked()
    default_genre = ranked[0][0] if ranked else GENRES[0]
    genre_col, model_col = st.columns(2)
    genre = genre_col.selectbox(
        "Genre to explain", GENRES, index=GENRES.index(default_genre), key=_STATE_CAM_GENRE
    )
    labels = {MODEL_DISPLAY_NAMES.get(kind, kind): kind for kind in options}
    label = model_col.selectbox("Model to explain", list(labels), key=_STATE_CAM_MODEL)
    mode = labels.get(label or next(iter(labels)), options[0])
    if mode == "multimodal" and not text.strip():
        st.info(
            "Explaining the multimodal model needs the description as well, because its "
            "logits depend on both branches. Add a description or explain the "
            "image-only model instead.",
            icon=":material/info:",
        )
        return
    try:
        with st.spinner("Computing Grad-CAM..."):
            result = predictor.explain(
                image=image, text=text or None, mode=mode, genre=genre or default_genre
            )
    except Exception as error:  # noqa: BLE001 - keep the rest of the page alive
        st.warning(
            f"Grad-CAM could not be computed ({type(error).__name__}: {error}). The "
            "predictions above are unaffected.",
            icon=":material/warning:",
        )
        return
    render_gradcam_panels(result)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose the page."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    try:
        predictor = get_predictor(CONFIG.device, CONFIG.training.seed)
    except Exception as error:  # noqa: BLE001 - show a message, not a traceback
        st.error(
            f"GameSense could not initialise its predictor ({type(error).__name__}): "
            f"{error}. Check that the project dependencies are installed "
            "(`pip install -r requirements.txt`)."
        )
        return
    render_header()
    mode = render_sidebar(predictor)
    if not predictor.any_available():
        render_untrained_banner(predictor)
    left, right = st.columns(MAIN_COLUMN_RATIO, gap="large")
    with left:
        render_input_column(predictor, mode)
    result = st.session_state.get(_STATE_RESULT)
    with right:
        render_result_column(predictor, result, mode)
    if result is not None:
        st.divider()
        render_comparison(predictor, result.get("image"), result.get("text") or "")
        st.divider()
        render_gradcam(
            predictor,
            result.get("image"),
            result.get("text") or "",
            result["prediction"],
        )
    render_footer(predictor)


main()
