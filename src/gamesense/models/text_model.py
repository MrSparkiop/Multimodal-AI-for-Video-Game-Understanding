"""MODEL B -- text-only multi-label genre classifier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..config import CONFIG, GENRES, GameSenseConfig
from ..utils import get_logger
from .model_utils import GameSenseModel, MLPHead, freeze_module, unfreeze_module

__all__ = [
    "masked_mean_pool",
    "TextEncoder",
    "TextOnlyClassifier",
    "BiLSTMTextEncoder",
    "BiLSTMTextClassifier",
    "build_text_model",
]

LOGGER = get_logger("gamesense.models.text")


def _tokenizer_vocab_size(config: GameSenseConfig = CONFIG) -> int:
    """Vocabulary size of the project tokenizer, with a configured fallback."""
    try:
        from ..data.loader import get_tokenizer

        tokenizer = get_tokenizer(config.text.model_name)
        size = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
        return max(size, config.text.lstm_vocab_size)
    except Exception:  # offline / tokenizer unavailable
        LOGGER.warning(
            "Could not read the tokenizer vocabulary; falling back to "
            "TextConfig.lstm_vocab_size=%d",
            config.text.lstm_vocab_size,
        )
        return config.text.lstm_vocab_size


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, *, eps: float = 1e-9
) -> torch.Tensor:
    """Average token states, ignoring padding positions.

    Padding positions contribute nothing, so the embedding does not depend on batch padding.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(eps)
    return summed / counts


class TextEncoder(nn.Module):
    """Pretrained Transformer encoder producing a fixed-size sentence embedding."""

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        freeze: bool | None = None,
        unfreeze_layers: int | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = config.text.model_name
        self.pooling = config.text.pooling
        if pretrained:
            self.transformer = AutoModel.from_pretrained(self.model_name)
        else:  # used by fast unit tests -- random weights, same shapes
            self.transformer = AutoModel.from_config(AutoConfig.from_pretrained(self.model_name))
        self.embedding_dim = int(self.transformer.config.hidden_size)
        self.pretrained = bool(pretrained)

        do_freeze = config.model.freeze_text_encoder if freeze is None else freeze
        layers = config.model.unfreeze_text_layers if unfreeze_layers is None else unfreeze_layers
        if do_freeze:
            freeze_module(self.transformer)
        if layers:
            self.unfreeze_last_layers(layers)
        LOGGER.info(
            "TextEncoder(%s, pooling=%s, frozen=%s, unfrozen_layers=%d)",
            self.model_name,
            self.pooling,
            do_freeze,
            layers or 0,
        )

    def _transformer_layers(self) -> nn.ModuleList | None:
        """Locate the list of transformer blocks across architectures."""
        for path in (
            ("transformer", "layer"),  # DistilBERT
            ("encoder", "layer"),  # BERT / RoBERTa
            ("layers",),
        ):
            module: Any = self.transformer
            for attribute in path:
                module = getattr(module, attribute, None)
                if module is None:
                    break
            if isinstance(module, nn.ModuleList):
                return module
        return None

    def unfreeze_last_layers(self, n_layers: int) -> int:
        """Re-enable gradients for the last *n_layers* transformer blocks."""
        layers = self._transformer_layers()
        if layers is None:  # pragma: no cover - unusual architecture
            LOGGER.warning("Could not locate transformer layers for %s", self.model_name)
            return 0
        selected = list(layers)[-max(0, int(n_layers)) :] if n_layers else []
        for block in selected:
            unfreeze_module(block)
        return len(selected)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a tokenised batch into ``(batch, embedding_dim)``."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        return masked_mean_pool(hidden, attention_mask)


class TextOnlyClassifier(GameSenseModel):
    """MODEL B: DistilBERT sentence embedding + multi-label MLP head."""

    modality = "text"
    pretrained_children = ("encoder",)
    _frozen_children = ("encoder",)

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        num_classes: int = len(GENRES),
        freeze_encoder: bool | None = None,
        unfreeze_layers: int | None = None,
        dropout: float | None = None,
        hidden_dims: Sequence[int] | None = None,
        pretrained: bool = True,
        encoder: TextEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = encoder or TextEncoder(
            config=config,
            freeze=freeze_encoder,
            unfreeze_layers=unfreeze_layers,
            pretrained=pretrained,
        )
        self.head = MLPHead(
            self.encoder.embedding_dim,
            num_classes,
            hidden_dims=(
                hidden_dims
                if hidden_dims is not None
                else ((config.model.head_hidden_dim,) if config.model.head_hidden_dim else ())
            ),
            dropout=config.model.dropout if dropout is None else dropout,
        )
        self.embedding_dim = self.encoder.embedding_dim

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        text_features: torch.Tensor | None = None,
        **_ignored: Any,
    ) -> torch.Tensor:
        """Return ``(batch, num_classes)`` logits from tokens or cached features."""
        if text_features is None:
            if input_ids is None or attention_mask is None:
                raise ValueError("provide 'input_ids' + 'attention_mask' or 'text_features'")
            text_features = self.encode(input_ids, attention_mask)
        return self.head(text_features)

    def forward_from_features(
        self, *, text_features: torch.Tensor, **_ignored: Any
    ) -> torch.Tensor:
        return self.head(text_features)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Text embedding, with gradients only if the encoder is unfrozen."""
        trainable = any(p.requires_grad for p in self.encoder.parameters())
        if trainable:
            return self.encoder(input_ids, attention_mask)
        with torch.no_grad():
            return self.encoder(input_ids, attention_mask)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "text_model": self.encoder.model_name,
                "pooling": self.encoder.pooling,
                "embedding_dim": self.embedding_dim,
                "max_length": self.config.text.max_length,
            }
        )
        return info


# --------------------------------------------------------------------------- #
# Optional recurrent baseline
# --------------------------------------------------------------------------- #
class BiLSTMTextEncoder(nn.Module):
    """Bidirectional LSTM sentence encoder trained from scratch."""

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        vocab_size: int | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        text = config.text
        # The embedding table must cover the tokenizer's whole vocabulary: the BiLSTM consumes
        # the *same* WordPiece ids as DistilBERT, and those run up to 30,521.
        resolved = vocab_size if vocab_size is not None else _tokenizer_vocab_size(config)
        self.vocab_size = int(resolved)
        self.embedding = nn.Embedding(
            self.vocab_size, text.lstm_embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=text.lstm_embedding_dim,
            hidden_size=text.lstm_hidden_dim,
            num_layers=text.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.embedding_dim = 2 * text.lstm_hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode tokens into ``(batch, 2 * hidden_dim)`` via masked mean pooling."""
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        return masked_mean_pool(outputs, attention_mask)


class BiLSTMTextClassifier(GameSenseModel):
    """Optional MODEL B baseline: BiLSTM encoder + multi-label MLP head."""

    modality = "text_bilstm"
    pretrained_children = ()

    def __init__(
        self,
        *,
        config: GameSenseConfig = CONFIG,
        num_classes: int = len(GENRES),
        vocab_size: int | None = None,
        dropout: float | None = None,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = BiLSTMTextEncoder(config=config, vocab_size=vocab_size)
        self.head = MLPHead(
            self.encoder.embedding_dim,
            num_classes,
            hidden_dims=(
                hidden_dims
                if hidden_dims is not None
                else ((config.model.head_hidden_dim,) if config.model.head_hidden_dim else ())
            ),
            dropout=config.model.dropout if dropout is None else dropout,
        )
        self.embedding_dim = self.encoder.embedding_dim

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_ignored: Any,
    ) -> torch.Tensor:
        if input_ids is None or attention_mask is None:
            raise ValueError("BiLSTM baseline requires 'input_ids' and 'attention_mask'")
        return self.head(self.encoder(input_ids, attention_mask))

    def forward_from_features(self, **_ignored: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "the BiLSTM baseline is trained end to end; there is no frozen feature cache"
        )


def build_text_model(
    *,
    config: GameSenseConfig = CONFIG,
    num_classes: int = len(GENRES),
    architecture: str = "distilbert",
    **kwargs: Any,
) -> GameSenseModel:
    """Build a text model: ``"distilbert"`` (default) or ``"bilstm"``."""
    if architecture in ("distilbert", "transformer", "text"):
        return TextOnlyClassifier(config=config, num_classes=num_classes, **kwargs)
    if architecture in ("bilstm", "lstm"):
        return BiLSTMTextClassifier(config=config, num_classes=num_classes, **kwargs)
    raise ValueError(f"unknown text architecture {architecture!r}")
