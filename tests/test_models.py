"""Architectural contract tests for the three GameSense systems."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from gamesense.config import CONFIG, GENRES
from gamesense.explainability.gradcam import (
    GradCAM,
    denormalize_image,
    gradcam_heatmap,
    overlay_heatmap,
)
from gamesense.models import (
    BiLSTMTextClassifier,
    ImageEncoder,
    ImageOnlyClassifier,
    MLPHead,
    MultimodalClassifier,
    TextEncoder,
    TextOnlyClassifier,
    build_model,
    load_model_from_checkpoint,
    save_checkpoint,
)
from gamesense.models.text_model import masked_mean_pool

BATCH = 2
IMAGE_SIDE = CONFIG.image.image_size
# : Deliberately small images for the optimisation loops: a ResNet accepts any : side length,
# and 64 px keeps 20 forward/backward passes to a couple of : seconds while exercising exactly
SMALL_SIDE = 64
SEQ_LEN = 12
SEED = 20260821


def _skip_without_hf(exc: Exception) -> None:
    """Skip the calling test when the local Hugging Face cache is unusable."""
    pytest.skip(f"DistilBERT configuration unavailable offline ({type(exc).__name__}: {exc})")


def _text_classifier(**kwargs: Any) -> TextOnlyClassifier:
    """Build a random-weight DistilBERT classifier, or skip when offline."""
    try:
        return TextOnlyClassifier(pretrained=False, **kwargs)
    except Exception as exc:  # pragma: no cover - offline machine
        _skip_without_hf(exc)
        raise


def _text_encoder(**kwargs: Any) -> TextEncoder:
    """Build a random-weight DistilBERT encoder, or skip when offline."""
    try:
        return TextEncoder(pretrained=False, **kwargs)
    except Exception as exc:  # pragma: no cover - offline machine
        _skip_without_hf(exc)
        raise


def _multimodal(**kwargs: Any) -> MultimodalClassifier:
    """Build a random-weight fusion model, or skip when offline."""
    try:
        return MultimodalClassifier(pretrained=False, **kwargs)
    except Exception as exc:  # pragma: no cover - offline machine
        _skip_without_hf(exc)
        raise


# --------------------------------------------------------------------------- #
# Shared, read-only fixtures (module scope: building DistilBERT is the slow part)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def image_model() -> ImageOnlyClassifier:
    torch.manual_seed(SEED)
    return ImageOnlyClassifier(pretrained=False)


@pytest.fixture(scope="module")
def text_model() -> TextOnlyClassifier:
    torch.manual_seed(SEED)
    return _text_classifier()


@pytest.fixture(scope="module")
def fusion_model() -> MultimodalClassifier:
    torch.manual_seed(SEED)
    return _multimodal()


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(BATCH, 3, IMAGE_SIDE, IMAGE_SIDE)


@pytest.fixture(scope="module")
def tokens() -> dict[str, torch.Tensor]:
    """A tokenised batch with real padding, so masking is actually exercised."""
    torch.manual_seed(SEED)
    input_ids = torch.randint(1, 1000, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    attention_mask[:, SEQ_LEN // 2 :] = 0  # second half is padding
    input_ids = input_ids * attention_mask
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# --------------------------------------------------------------------------- #
# MLPHead
# --------------------------------------------------------------------------- #
def test_mlp_head_output_shape() -> None:
    head = MLPHead(32, len(GENRES), hidden_dims=(16,))
    assert head(torch.randn(5, 32)).shape == (5, len(GENRES))


def test_mlp_head_without_hidden_layers_is_linear() -> None:
    """``hidden_dims=()`` must degrade to a single linear projection."""
    head = MLPHead(32, 4, hidden_dims=(), dropout=0.0)
    linears = [layer for layer in head.net if isinstance(layer, nn.Linear)]
    assert len(linears) == 1
    assert not any(isinstance(layer, nn.LayerNorm) for layer in head.net)
    assert head(torch.randn(3, 32)).shape == (3, 4)


@pytest.mark.parametrize(("in_features", "num_classes"), [(0, 8), (-4, 8), (16, 0), (16, -1)])
def test_mlp_head_rejects_non_positive_dimensions(in_features: int, num_classes: int) -> None:
    with pytest.raises(ValueError):
        MLPHead(in_features, num_classes)


def test_mlp_head_is_deterministic_in_eval_and_stochastic_in_train() -> None:
    head = MLPHead(64, len(GENRES), hidden_dims=(32,), dropout=0.5)
    features = torch.randn(8, 64)

    head.eval()
    assert torch.equal(head(features), head(features))

    head.train()
    # Dropout is random, so at least one of a handful of pairs must differ.
    assert any(not torch.equal(head(features), head(features)) for _ in range(5))


# --------------------------------------------------------------------------- #
# MODEL A -- image only
# --------------------------------------------------------------------------- #
def test_image_model_returns_logits_not_probabilities(
    image_model: ImageOnlyClassifier, images: torch.Tensor
) -> None:
    image_model.eval()
    logits = image_model(images)
    assert logits.shape == (BATCH, len(GENRES))
    # The head must NOT squash its output: probabilities are produced explicitly.
    assert not bool(((logits >= 0.0) & (logits <= 1.0)).all())

    probabilities = image_model.predict_proba(image=images)
    assert probabilities.shape == (BATCH, len(GENRES))
    assert float(probabilities.min()) >= 0.0
    assert float(probabilities.max()) <= 1.0
    assert torch.allclose(probabilities, torch.sigmoid(logits), atol=1e-6)


def test_image_model_forward_from_cached_features(image_model: ImageOnlyClassifier) -> None:
    features = torch.randn(BATCH, CONFIG.image.embedding_dim)
    image_model.eval()
    assert image_model.forward_from_features(image_features=features).shape == (
        BATCH,
        len(GENRES),
    )


def test_image_encoder_embedding_and_feature_map(images: torch.Tensor) -> None:
    encoder = ImageEncoder(pretrained=False, freeze=True).eval()
    embeddings = encoder(images)
    assert embeddings.shape == (BATCH, CONFIG.image.embedding_dim)

    maps = encoder.feature_map(images)
    assert maps.dim() == 4
    # 224 px through a ResNet's five stride-2 stages -> a 7x7 grid.
    assert maps.shape == (BATCH, CONFIG.image.embedding_dim, 7, 7)


def test_frozen_image_encoder_has_no_trainable_parameters_and_stays_in_eval(
    image_model: ImageOnlyClassifier,
) -> None:
    trainable = [p for p in image_model.encoder.parameters() if p.requires_grad]
    assert trainable == []

    image_model.train()
    try:
        # A frozen BatchNorm left in train mode would keep updating its running
        # statistics, so the "frozen" features would silently drift.
        assert image_model.encoder.training is False
        assert image_model.head.training is True
    finally:
        image_model.eval()


def test_unfreeze_last_stages_opens_only_layer4() -> None:
    encoder = ImageEncoder(pretrained=False, freeze=True)
    opened = encoder.unfreeze_last_stages(1)
    assert opened == ["layer4"]

    assert all(p.requires_grad for p in encoder.layer4.parameters())
    for earlier in ("layer3", "layer2", "layer1", "stem"):
        assert not any(p.requires_grad for p in getattr(encoder, earlier).parameters())


# --------------------------------------------------------------------------- #
# MODEL B -- text only
# --------------------------------------------------------------------------- #
def test_text_model_forward_shapes(
    text_model: TextOnlyClassifier, tokens: dict[str, torch.Tensor]
) -> None:
    text_model.eval()
    logits = text_model(tokens["input_ids"], tokens["attention_mask"])
    assert logits.shape == (BATCH, len(GENRES))

    features = torch.randn(BATCH, CONFIG.text.embedding_dim)
    assert text_model.forward_from_features(text_features=features).shape == (
        BATCH,
        len(GENRES),
    )


def test_masked_mean_pool_ignores_padding() -> None:
    """Hand-computed: the pooled vector is the mean over unmasked rows only."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    mask = torch.tensor([[1, 1, 0, 0]])
    # mean over rows 0 and 1 only -> [(1+3)/2, (2+4)/2] = [2, 3]
    pooled = masked_mean_pool(hidden, mask)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]), atol=1e-6)

    # Padding must not shift the result even when the padded states are huge.
    hidden_noisy = hidden.clone()
    hidden_noisy[0, 2:] = 1_000.0
    assert torch.allclose(masked_mean_pool(hidden_noisy, mask), pooled, atol=1e-6)


def test_cls_and_mean_pooling_differ(tokens: dict[str, torch.Tensor]) -> None:
    """Same weights, different pooling -> different sentence embedding."""
    encoder = _text_encoder(freeze=True).eval()
    encoder.pooling = "mean"
    mean_embedding = encoder(tokens["input_ids"], tokens["attention_mask"])
    encoder.pooling = "cls"
    cls_embedding = encoder(tokens["input_ids"], tokens["attention_mask"])
    assert mean_embedding.shape == cls_embedding.shape
    assert not torch.allclose(mean_embedding, cls_embedding, atol=1e-4)


def test_frozen_text_encoder_has_no_trainable_parameters(
    text_model: TextOnlyClassifier,
) -> None:
    assert [p for p in text_model.encoder.parameters() if p.requires_grad] == []
    text_model.train()
    try:
        assert text_model.encoder.training is False
    finally:
        text_model.eval()


# --------------------------------------------------------------------------- #
# BiLSTM baseline
# --------------------------------------------------------------------------- #
def test_bilstm_baseline_is_fully_trainable(tokens: dict[str, torch.Tensor]) -> None:
    model = BiLSTMTextClassifier()
    model.eval()
    logits = model(tokens["input_ids"], tokens["attention_mask"])
    assert logits.shape == (BATCH, len(GENRES))
    # Nothing is pretrained here, so nothing may be frozen.
    assert all(p.requires_grad for p in model.parameters())
    assert model.frozen_pretrained_prefixes() == ()


def test_bilstm_baseline_has_no_feature_cache_path() -> None:
    model = BiLSTMTextClassifier()
    with pytest.raises(NotImplementedError):
        model.forward_from_features(text_features=torch.randn(BATCH, 256))


# --------------------------------------------------------------------------- #
# MODEL C -- multimodal fusion
# --------------------------------------------------------------------------- #
def test_multimodal_forward_from_raw_inputs_and_cached_features(
    fusion_model: MultimodalClassifier, images: torch.Tensor, tokens: dict[str, torch.Tensor]
) -> None:
    fusion_model.eval()
    raw = fusion_model(images, tokens["input_ids"], tokens["attention_mask"])
    assert raw.shape == (BATCH, len(GENRES))

    cached = fusion_model.forward_from_features(
        image_features=torch.randn(BATCH, fusion_model.image_dim),
        text_features=torch.randn(BATCH, fusion_model.text_dim),
    )
    assert cached.shape == (BATCH, len(GENRES))


def test_fused_dimension_is_the_sum_of_both_branches(
    fusion_model: MultimodalClassifier,
) -> None:
    assert fusion_model.image_dim == CONFIG.image.embedding_dim
    assert fusion_model.text_dim == CONFIG.text.embedding_dim
    assert fusion_model.fused_dim == fusion_model.image_dim + fusion_model.text_dim
    assert fusion_model.fused_dim == 1280


def test_fuse_rejects_misaligned_batches(fusion_model: MultimodalClassifier) -> None:
    """A screenshot must always be fused with *its own* game's description."""
    with pytest.raises(ValueError, match="same games"):
        fusion_model.fuse(
            torch.randn(3, fusion_model.image_dim),
            torch.randn(4, fusion_model.text_dim),
        )


def test_l2_normalisation_makes_both_halves_unit_norm() -> None:
    model = _multimodal(normalize_embeddings=True, dropout=0.0).eval()
    image_features = torch.randn(BATCH, model.image_dim) * 25.0
    text_features = torch.randn(BATCH, model.text_dim) * 0.01
    fused = model.fuse(image_features, text_features)

    assert fused.shape == (BATCH, model.fused_dim)
    image_half = fused[:, : model.image_dim].norm(dim=1)
    text_half = fused[:, model.image_dim :].norm(dim=1)
    assert torch.allclose(image_half, torch.ones(BATCH), atol=1e-5)
    assert torch.allclose(text_half, torch.ones(BATCH), atol=1e-5)


def test_modality_dropout_only_perturbs_training_mode() -> None:
    """``dropout=0.0`` isolates modality dropout as the only stochastic part."""
    torch.manual_seed(SEED)
    model = _multimodal(dropout=0.0, modality_dropout=0.5)
    image_features = torch.randn(16, model.image_dim)
    text_features = torch.randn(16, model.text_dim)

    model.eval()
    first = model.forward_from_features(
        image_features=image_features, text_features=text_features
    )
    second = model.forward_from_features(
        image_features=image_features, text_features=text_features
    )
    assert torch.equal(first, second)

    model.train()
    assert any(
        not torch.equal(
            model.forward_from_features(
                image_features=image_features, text_features=text_features
            ),
            model.forward_from_features(
                image_features=image_features, text_features=text_features
            ),
        )
        for _ in range(5)
    )


def test_multimodal_rejects_invalid_modality_dropout(
    fusion_model: MultimodalClassifier,
) -> None:
    # Re-uses the already-built encoders so the guard is tested without paying
    # for a second DistilBERT instantiation.
    with pytest.raises(ValueError, match="modality_dropout"):
        MultimodalClassifier(
            image_encoder=fusion_model.image_encoder,
            text_encoder=fusion_model.text_encoder,
            modality_dropout=1.0,
        )


def test_gradcam_target_layer_is_the_vision_branch(
    fusion_model: MultimodalClassifier, image_model: ImageOnlyClassifier
) -> None:
    assert fusion_model.gradcam_target_layer is fusion_model.image_encoder.layer4
    assert image_model.gradcam_target_layer is image_model.encoder.layer4


# --------------------------------------------------------------------------- #
# A real optimisation step
# --------------------------------------------------------------------------- #
def test_multimodal_takes_a_real_gradient_step_and_can_overfit_a_tiny_batch() -> None:
    """End-to-end sanity check that the fusion model is actually trainable."""
    torch.manual_seed(SEED)
    model = _multimodal(dropout=0.0)
    batch = 4
    inputs = {
        "image": torch.randn(batch, 3, SMALL_SIDE, SMALL_SIDE),
        "input_ids": torch.randint(1, 1000, (batch, 8)),
        "attention_mask": torch.ones(batch, 8, dtype=torch.long),
    }
    targets = torch.zeros(batch, len(GENRES))
    targets[torch.arange(batch), torch.arange(batch) % len(GENRES)] = 1.0

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "the fusion head must be trainable"
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG.training.head_lr * 10)

    model.train()
    losses: list[float] = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(**inputs), targets)
        loss.backward()
        if step == 0:
            assert torch.isfinite(loss)
            head_grads = [
                p.grad for p in model.fusion_head.parameters() if p.grad is not None
            ]
            assert head_grads, "the fusion head received no gradients at all"
            assert any(float(g.abs().sum()) > 0.0 for g in head_grads)
            for encoder in (model.image_encoder, model.text_encoder):
                for parameter in encoder.parameters():
                    assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------------- #
# Checkpoint round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "frozen_prefixes"),
    [
        ("image", ("encoder.",)),
        ("text", ("encoder.",)),
        ("multimodal", ("image_encoder.", "text_encoder.")),
    ],
)
def test_exported_checkpoint_omits_frozen_pretrained_tensors(
    kind: str, frozen_prefixes: tuple[str, ...], tmp_path: Path
) -> None:
    """Frozen encoders are reproducible from published weights, so they are not saved."""
    model = build_model(kind, pretrained=False) if kind != "text" else _text_classifier()
    exported = model.exportable_state_dict()
    assert exported, "something must still be saved (the trained head)"
    for key in exported:
        assert not any(key.startswith(prefix) for prefix in frozen_prefixes), key
    assert set(model.frozen_pretrained_prefixes()) == set(frozen_prefixes)

    path = save_checkpoint(model, tmp_path / f"{kind}_model.pt")
    assert path.is_file()


def test_checkpoint_round_trip_reproduces_identical_logits(tmp_path: Path) -> None:
    """A reloaded checkpoint must score a game exactly like the trained model did."""
    torch.manual_seed(SEED)
    model = ImageOnlyClassifier(pretrained=False).eval()
    init_kwargs = {"pretrained": False}
    features = torch.randn(3, CONFIG.image.embedding_dim)
    inputs = torch.randn(2, 3, SMALL_SIDE, SMALL_SIDE)

    # (a) default export: only the trained head travels, so the head must match.
    slim = save_checkpoint(
        model, tmp_path / "image_model.pt", metadata={"model_init_kwargs": init_kwargs}
    )
    reloaded, payload = load_model_from_checkpoint(slim, kind="image")
    assert payload["modality"] == "image"
    assert payload["classes"] == list(GENRES)
    assert payload["metadata"]["model_init_kwargs"] == init_kwargs
    assert torch.equal(
        model.forward_from_features(image_features=features),
        reloaded.eval().forward_from_features(image_features=features),
    )

    # (b) full export: every tensor travels, so even the raw-image path matches.
    full = save_checkpoint(
        model,
        tmp_path / "image_model_full.pt",
        metadata={"model_init_kwargs": init_kwargs},
        full=True,
    )
    restored, _ = load_model_from_checkpoint(full, kind="image")
    assert torch.equal(model(inputs), restored.eval()(inputs))


def test_describe_reports_sensible_parameter_counts(
    image_model: ImageOnlyClassifier,
) -> None:
    info = image_model.describe()
    counts = info["parameters"]
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert 0 < counts["trainable"] < counts["total"]
    assert info["modality"] == "image"
    assert info["num_classes"] == len(GENRES)
    assert info["backbone"] == CONFIG.image.backbone
    assert 0.0 < info["trainable_fraction"] < 1.0
    assert info["frozen_pretrained"] == ["encoder."]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("aliases", "expected"),
    [
        (("image", "image_only", "vision"), ImageOnlyClassifier),
        (("text", "text_only", "distilbert"), TextOnlyClassifier),
        (("text_bilstm", "bilstm", "lstm"), BiLSTMTextClassifier),
        (("multimodal", "fusion", "both"), MultimodalClassifier),
    ],
)
def test_build_model_resolves_every_alias(
    aliases: tuple[str, ...], expected: type
) -> None:
    for alias in aliases:
        # The BiLSTM baseline has nothing pretrained, hence no `pretrained` flag.
        kwargs = {} if expected is BiLSTMTextClassifier else {"pretrained": False}
        try:
            model = build_model(alias, **kwargs)
        except Exception as exc:  # pragma: no cover - offline machine
            if expected in (TextOnlyClassifier, MultimodalClassifier):
                _skip_without_hf(exc)
            raise
        assert isinstance(model, expected)
        assert model.num_classes == len(GENRES)


def test_build_model_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("audio")


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gradcam_setup(image_model: ImageOnlyClassifier) -> tuple[ImageOnlyClassifier, torch.Tensor]:
    torch.manual_seed(SEED)
    return image_model, torch.randn(1, 3, IMAGE_SIDE, IMAGE_SIDE)


def test_gradcam_heatmap_shape_and_range(
    gradcam_setup: tuple[ImageOnlyClassifier, torch.Tensor],
) -> None:
    model, image = gradcam_setup
    heatmap, class_index, probability = gradcam_heatmap(model, image)

    assert heatmap.shape == tuple(image.shape[-2:])
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert float(np.abs(heatmap).sum()) > 0.0, "an all-zero heat map explains nothing"
    assert 0 <= class_index < len(GENRES)
    assert 0.0 <= probability <= 1.0


def test_gradcam_accepts_genre_by_name_and_by_index(
    gradcam_setup: tuple[ImageOnlyClassifier, torch.Tensor],
) -> None:
    model, image = gradcam_setup
    genre = "Racing"
    by_name, index_from_name, prob_from_name = gradcam_heatmap(
        model, image, class_index=genre
    )
    by_index, index_from_index, prob_from_index = gradcam_heatmap(
        model, image, class_index=GENRES.index(genre)
    )
    assert index_from_name == index_from_index == GENRES.index(genre)
    assert prob_from_name == pytest.approx(prob_from_index)
    assert np.allclose(by_name, by_index, atol=1e-6)


def test_gradcam_rejects_unknown_genre(
    gradcam_setup: tuple[ImageOnlyClassifier, torch.Tensor],
) -> None:
    model, image = gradcam_setup
    with pytest.raises(ValueError, match="unknown genre"):
        gradcam_heatmap(model, image, class_index="Roguelike")


def test_gradcam_restores_frozen_flags_and_removes_hooks(
    gradcam_setup: tuple[ImageOnlyClassifier, torch.Tensor],
) -> None:
    """Explaining a prediction must not leave the model in a different state."""
    model, image = gradcam_setup
    layer = model.gradcam_target_layer

    with GradCAM(model) as explainer:
        assert len(layer._forward_hooks) == 1
        explainer(image, class_index=0)
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) == 0

    # Grad-CAM needs a differentiable path, so it temporarily unfreezes the
    # backbone; leaving it unfrozen would silently start training it.
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert model.training is False


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def test_denormalize_image_returns_hwc_in_unit_range(images: torch.Tensor) -> None:
    restored = denormalize_image(images[:1])
    assert restored.shape == (IMAGE_SIDE, IMAGE_SIDE, 3)
    assert 0.0 <= float(restored.min()) and float(restored.max()) <= 1.0

    # A 3-D tensor is accepted too (the app passes single images around).
    assert denormalize_image(images[0]).shape == (IMAGE_SIDE, IMAGE_SIDE, 3)


def test_overlay_heatmap_accepts_tensors_arrays_and_mismatched_maps(
    images: torch.Tensor,
) -> None:
    heatmap = np.random.default_rng(SEED).random((IMAGE_SIDE, IMAGE_SIDE)).astype(np.float32)

    from_tensor = overlay_heatmap(images[:1], heatmap)
    assert from_tensor.shape == (IMAGE_SIDE, IMAGE_SIDE, 3)
    assert 0.0 <= float(from_tensor.min()) and float(from_tensor.max()) <= 1.0

    base = denormalize_image(images[:1])
    from_array = overlay_heatmap(base, heatmap)
    assert from_array.shape == base.shape
    assert np.allclose(from_tensor, from_array, atol=1e-5)

    # A raw 7x7 Grad-CAM map must be resized to the image, not rejected.
    coarse = np.random.default_rng(SEED).random((7, 7)).astype(np.float32)
    assert overlay_heatmap(base, coarse).shape == base.shape


def test_overlay_heatmap_accepts_uint8_style_images() -> None:
    base = (np.random.default_rng(SEED).random((32, 48, 3)) * 255.0).astype(np.float32)
    heatmap = np.zeros((32, 48), dtype=np.float32)
    blended = overlay_heatmap(base, heatmap, alpha=0.0)
    assert blended.shape == (32, 48, 3)
    # alpha=0 keeps the (rescaled) base image untouched.
    assert np.allclose(blended, base / 255.0, atol=1e-6)


def test_image_encoder_rejects_non_4d_input() -> None:
    encoder = ImageEncoder(pretrained=False, freeze=True).eval()
    with pytest.raises(ValueError, match="4-D"):
        encoder(torch.randn(3, IMAGE_SIDE, IMAGE_SIDE))


def test_image_encoder_rejects_unsupported_backbone() -> None:
    config = replace(CONFIG, image=replace(CONFIG.image, backbone="vgg11"))
    with pytest.raises(ValueError, match="unsupported backbone"):
        ImageEncoder(config=config, pretrained=False)


# --------------------------------------------------------------------------- #
# Regression: BiLSTM embedding table must cover the whole WordPiece vocabulary
# --------------------------------------------------------------------------- #
def test_bilstm_embedding_covers_the_tokenizer_vocabulary(tokenizer) -> None:
    """The BiLSTM must accept any id the project tokenizer can emit."""
    from gamesense.models.text_model import BiLSTMTextClassifier

    model = BiLSTMTextClassifier()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    assert model.encoder.vocab_size >= vocab_size

    # The largest id the tokenizer can produce must be embeddable.
    extreme = torch.tensor([[0, 1, vocab_size - 1]], dtype=torch.long)
    mask = torch.ones_like(extreme)
    with torch.no_grad():
        logits = model(input_ids=extreme, attention_mask=mask)
    assert logits.shape == (1, len(GENRES))
    assert torch.isfinite(logits).all()


def test_bilstm_accepts_real_tokenized_text(tokenizer) -> None:
    """End-to-end: real descriptions through the real tokenizer, no IndexError."""
    from gamesense.models.text_model import BiLSTMTextClassifier

    texts = [
        "Race tuned street cars through neon city circuits and coastal highways.",
        "Command divisions across a historically detailed turn-based campaign map.",
        "A cheerful match-3 puzzler with hundreds of increasingly tricky levels.",
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    model = BiLSTMTextClassifier()
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )
    assert logits.shape == (len(texts), len(GENRES))
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# Regression: the small "backbone" learning rate is for PRETRAINED weights only
# --------------------------------------------------------------------------- #
# The small rate exists to nudge pretrained tensors gently.
HEAD_LR = 1e-3
BACKBONE_LR = 2e-5


def _lr_of(groups, predicate) -> set[float]:
    """Learning rates of the groups whose name satisfies *predicate*."""
    return {group["lr"] for group in groups if predicate(group["name"])}


def test_frozen_models_put_every_trainable_tensor_on_the_head_rate() -> None:
    """With a frozen encoder only the head trains, so nothing may use 2e-5."""
    from gamesense.models.model_utils import parameter_groups

    for model in (
        ImageOnlyClassifier(pretrained=False),
        TextOnlyClassifier(pretrained=False),
        MultimodalClassifier(pretrained=False),
    ):
        groups = parameter_groups(
            model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, weight_decay=1e-2
        )
        assert groups, f"{type(model).__name__} produced no parameter groups"
        assert all(group["lr"] == HEAD_LR for group in groups), (
            f"{type(model).__name__} put a frozen-encoder-free model on the backbone rate: "
            f"{[(g['name'], g['lr']) for g in groups]}"
        )


def test_from_scratch_bilstm_trains_at_the_head_rate() -> None:
    """A model with no pretrained tensors must train entirely at head_lr."""
    from gamesense.models.model_utils import parameter_groups
    from gamesense.models.text_model import BiLSTMTextClassifier

    model = BiLSTMTextClassifier()
    assert model.pretrained_children == ()
    groups = parameter_groups(
        model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, weight_decay=1e-2
    )
    assert all(group["lr"] == HEAD_LR for group in groups)

    # The encoder is the bulk of the model, so this is not a vacuous check.
    covered = sum(p.numel() for group in groups for p in group["params"])
    encoder = sum(p.numel() for p in model.encoder.parameters())
    assert covered > encoder, "the encoder must be inside a trainable group"


def test_partial_fine_tuning_keeps_pretrained_layers_on_the_small_rate() -> None:
    """Unfrozen ResNet stages get backbone_lr; the head still gets head_lr."""
    from gamesense.models.model_utils import parameter_groups

    model = ImageOnlyClassifier(pretrained=False, freeze_backbone=True, unfreeze_stages=1)
    groups = parameter_groups(
        model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR, weight_decay=1e-2
    )
    assert _lr_of(groups, lambda n: n.startswith("pretrained")) == {BACKBONE_LR}
    assert _lr_of(groups, lambda n: n.startswith("head")) == {HEAD_LR}

    pretrained_params = sum(
        p.numel() for g in groups if g["name"].startswith("pretrained") for p in g["params"]
    )
    assert pretrained_params > 1_000_000, "layer4 should be a large group"


def test_bias_and_norm_parameters_are_excluded_from_weight_decay() -> None:
    """AdamW convention: no decay on biases or normalisation parameters."""
    from gamesense.models.model_utils import parameter_groups

    groups = parameter_groups(
        ImageOnlyClassifier(pretrained=False),
        head_lr=HEAD_LR,
        backbone_lr=BACKBONE_LR,
        weight_decay=1e-2,
    )
    by_name = {group["name"]: group for group in groups}
    assert any(name.endswith("nodecay") for name in by_name)
    for name, group in by_name.items():
        expected = 0.0 if name.endswith("nodecay") else 1e-2
        assert group["weight_decay"] == expected
