"""Train MODEL B -- the text-only DistilBERT genre classifier."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _train_common import add_common_arguments, run_training

from gamesense.config import CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the text-only (DistilBERT or BiLSTM) multi-label genre classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    language = parser.add_argument_group("language")
    language.add_argument(
        "--architecture",
        choices=("distilbert", "bilstm"),
        default="distilbert",
        help="pretrained Transformer (default) or the from-scratch recurrent baseline",
    )
    language.add_argument(
        "--unfreeze-layers",
        type=int,
        default=CONFIG.model.unfreeze_text_layers,
        help="number of trailing transformer blocks to fine-tune (0 = frozen). "
             "Any value > 0 disables the feature cache.",
    )
    language.add_argument("--head-hidden", type=int, default=CONFIG.model.head_hidden_dim,
                          help="hidden width of the classification head (0 = linear head)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = (args.head_hidden,) if args.head_hidden else ()

    if args.architecture == "bilstm":
        # The BiLSTM is trained end to end (its embedding table is learned from
        # scratch), so there is no frozen-encoder cache to reuse.
        return run_training(
            "text_bilstm",
            args,
            model_kwargs={"dropout": args.dropout, "hidden_dims": hidden},
            force_end_to_end=True,
        )

    # Freeze the whole encoder, then re-open the last N transformer blocks.
    model_kwargs = {
        "freeze_encoder": True,
        "unfreeze_layers": args.unfreeze_layers,
        "dropout": args.dropout,
        "hidden_dims": hidden,
    }
    return run_training(
        "text",
        args,
        model_kwargs=model_kwargs,
        force_end_to_end=args.unfreeze_layers > 0,
    )


if __name__ == "__main__":
    sys.exit(main())
