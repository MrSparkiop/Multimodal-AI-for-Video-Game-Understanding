"""Train MODEL C -- the multimodal (ResNet18 + DistilBERT) genre classifier."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _train_common import add_common_arguments, run_training

from gamesense.config import CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multimodal (late-fusion) multi-label genre classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    fusion = parser.add_argument_group("fusion")
    fusion.add_argument(
        "--fusion-hidden",
        type=int,
        nargs="*",
        default=list(CONFIG.model.fusion_hidden_dims),
        help="hidden widths of the fusion MLP",
    )
    fusion.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="concatenate raw embeddings instead of L2-normalised ones (ablation)",
    )
    fusion.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="probability of zeroing one branch per sample during training "
             "(robustness to a missing modality at serving time)",
    )
    fusion.add_argument("--unfreeze-stages", type=int, default=CONFIG.model.unfreeze_image_stages,
                        help="trailing ResNet stages to fine-tune (0 = frozen)")
    fusion.add_argument("--unfreeze-layers", type=int, default=CONFIG.model.unfreeze_text_layers,
                        help="trailing transformer blocks to fine-tune (0 = frozen)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Freeze both encoders, then re-open the last N ResNet stages / transformer
    # blocks -- not "unfreeze everything", which is a different experiment.
    model_kwargs = {
        "freeze_image_backbone": True,
        "freeze_text_encoder": True,
        "unfreeze_image_stages": args.unfreeze_stages,
        "unfreeze_text_layers": args.unfreeze_layers,
        "dropout": args.dropout,
        "fusion_hidden_dims": tuple(args.fusion_hidden),
        "normalize_embeddings": not args.no_normalize_embeddings,
        "modality_dropout": args.modality_dropout,
    }
    return run_training(
        "multimodal",
        args,
        model_kwargs=model_kwargs,
        force_end_to_end=args.unfreeze_stages > 0 or args.unfreeze_layers > 0,
    )


if __name__ == "__main__":
    sys.exit(main())
