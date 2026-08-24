"""Train MODEL A -- the image-only ResNet18 genre classifier."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _train_common import add_common_arguments, run_training

from gamesense.config import CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the image-only (ResNet18) multi-label genre classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    vision = parser.add_argument_group("vision")
    vision.add_argument(
        "--unfreeze-stages",
        type=int,
        default=CONFIG.model.unfreeze_image_stages,
        help="number of trailing ResNet stages to fine-tune (0 = frozen feature extraction; "
             "1 = layer4; 2 = layer4+layer3). Any value > 0 disables the feature cache.",
    )
    vision.add_argument(
        "--random-init",
        action="store_true",
        help="train the backbone from random weights instead of ImageNet weights "
             "(ablation showing the value of transfer learning)",
    )
    vision.add_argument("--head-hidden", type=int, default=CONFIG.model.head_hidden_dim,
                        help="hidden width of the classification head (0 = linear head)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The recipe is always "freeze the whole backbone, then re-open the last N stages".
    model_kwargs = {
        "freeze_backbone": True,
        "unfreeze_stages": args.unfreeze_stages,
        "pretrained": not args.random_init,
        "dropout": args.dropout,
        "hidden_dims": (args.head_hidden,) if args.head_hidden else (),
    }
    return run_training(
        "image",
        args,
        model_kwargs=model_kwargs,
        force_end_to_end=args.unfreeze_stages > 0 or args.random_init,
    )


if __name__ == "__main__":
    sys.exit(main())
