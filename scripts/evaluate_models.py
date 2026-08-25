"""Compare the three GameSense systems and produce every reported artefact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from gamesense.config import CONFIG, GENRES, MODEL_DISPLAY_NAMES, MODEL_KINDS
from gamesense.data.loader import load_bundle
from gamesense.evaluation.error_analysis import (
    build_error_frame,
    compare_model_errors,
    confusion_pairs,
    enrich_with_image_stats,
    grouped_error_rates,
    per_class_error_summary,
    select_examples,
)
from gamesense.evaluation.evaluator import (
    aggregate_over_seeds,
    build_comparison_table,
    collect_results,
    evaluate_model,
    load_predictions,
)
from gamesense.evaluation.metrics import METRIC_KEYS
from gamesense.utils import get_logger, load_json, save_json

LOGGER = get_logger("gamesense.evaluate")

DESCRIPTION_LENGTH_BINS = [0, 20, 30, 40, 50, 10_000]
DESCRIPTION_LENGTH_LABELS = ["<20", "20-29", "30-39", "40-49", "50+"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the image-only, text-only and multimodal systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_KINDS),
                        help="artefact names to compare (extra ablation tags are allowed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIG.training.seeds),
                        help="seeds to look for on disk")
    parser.add_argument("--primary-seed", type=int, default=CONFIG.training.seed,
                        help="seed used for per-sample error analysis and Grad-CAM")
    parser.add_argument("--reevaluate", action="store_true",
                        help="re-run inference from the checkpoints before reporting")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--gradcam-examples", type=int, default=6,
                        help="number of Grad-CAM panels to render (0 disables)")
    parser.add_argument("--image-stat-limit", type=int, default=1500,
                        help="how many screenshots to open when computing brightness/contrast")
    parser.add_argument("--device", default=CONFIG.device)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def write_table(frame: pd.DataFrame, path: Path, *, floatfmt: str = "%.4f") -> None:
    """Write a DataFrame as CSV plus a markdown twin for the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=floatfmt)
    markdown = path.with_suffix(".md")
    try:
        markdown.write_text(frame.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    except ImportError:  # tabulate is optional
        markdown.write_text(frame.to_string(index=False), encoding="utf-8")
    LOGGER.info("Wrote %s (%d rows)", path.name, len(frame))


def build_per_class_table(collected: dict[str, dict[int, dict[str, Any]]], *, seed: int) -> pd.DataFrame:
    """Per-genre metrics for every model at the validation-selected threshold."""
    rows: list[dict[str, Any]] = []
    for kind, per_seed in collected.items():
        payload = per_seed.get(seed) or next(iter(per_seed.values()), None)
        if payload is None:
            continue
        per_class = payload.get("test_metrics_selected", {}).get("per_class", {})
        for genre in GENRES:
            entry = per_class.get(genre)
            if entry is None:
                continue
            rows.append(
                {
                    "model": MODEL_DISPLAY_NAMES.get(kind, kind),
                    "model_kind": kind,
                    "genre": genre,
                    "support": entry.get("support"),
                    "precision": entry.get("precision"),
                    "recall": entry.get("recall"),
                    "f1": entry.get("f1"),
                    "average_precision": entry.get("average_precision"),
                    "predicted_positive": entry.get("predicted_positive"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def build_error_frames(
    collected: dict[str, dict[int, dict[str, Any]]],
    *,
    bundle: Any,
    seed: int,
    image_stat_limit: int,
) -> dict[str, pd.DataFrame]:
    """Per-model per-sample error frames for the test split."""
    frames: dict[str, pd.DataFrame] = {}
    samples = pd.concat(list(bundle.splits.values()), ignore_index=True)
    for kind, per_seed in collected.items():
        prediction_path = CONFIG.predictions_path(kind, "test", seed)
        if not prediction_path.is_file():
            LOGGER.info("No test predictions for %s seed %d - skipping error analysis", kind, seed)
            continue
        payload = per_seed.get(seed) or next(iter(per_seed.values()), {})
        threshold = payload.get("threshold_selected", CONFIG.evaluation.default_threshold)
        predictions = load_predictions(prediction_path)
        frame = build_error_frame(
            labels=predictions["labels"],
            probabilities=predictions["probabilities"],
            sample_ids=predictions["sample_ids"],
            samples=samples,
            games=bundle.games,
            threshold=np.asarray(threshold) if isinstance(threshold, list) else float(threshold),
            classes=bundle.classes,
        )
        if image_stat_limit:
            frame = enrich_with_image_stats(frame, config=CONFIG, limit=image_stat_limit)
        frames[kind] = frame
        LOGGER.info(
            "%s: %d test samples | exact %.1f%% | partial %.1f%% | wrong %.1f%% | empty %.1f%%",
            kind, len(frame),
            100 * (frame["outcome"] == "exact").mean(),
            100 * (frame["outcome"] == "partial").mean(),
            100 * (frame["outcome"] == "wrong").mean(),
            100 * (frame["outcome"] == "empty_prediction").mean(),
        )
    return frames


def error_analysis_tables(
    frames: dict[str, pd.DataFrame],
    collected: dict[str, dict[int, dict[str, Any]]],
    *,
    bundle: Any,
    seed: int,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Write every error-analysis table and return a digest for the summary."""
    digest: dict[str, Any] = {"outcome_rates": {}, "grouped": {}}

    for kind, frame in frames.items():
        digest["outcome_rates"][kind] = {
            outcome: float((frame["outcome"] == outcome).mean())
            for outcome in ("exact", "partial", "wrong", "empty_prediction")
        }
        prediction_path = CONFIG.predictions_path(kind, "test", seed)
        payload = collected.get(kind, {}).get(seed, {})
        threshold = payload.get("threshold_selected", CONFIG.evaluation.default_threshold)
        predictions = load_predictions(prediction_path)
        threshold_value = (
            np.asarray(threshold) if isinstance(threshold, list) else float(threshold)
        )

        write_table(
            per_class_error_summary(
                labels=predictions["labels"],
                probabilities=predictions["probabilities"],
                threshold=threshold_value,
                classes=bundle.classes,
            ),
            metrics_dir / f"error_per_class_{kind}.csv",
        )
        confusion = confusion_pairs(
            labels=predictions["labels"],
            probabilities=predictions["probabilities"],
            threshold=threshold_value,
            classes=bundle.classes,
        )
        confusion.to_csv(metrics_dir / f"error_confusion_{kind}.csv")

        grouped: dict[str, pd.DataFrame] = {
            "by_n_true_labels": grouped_error_rates(frame, by="n_true"),
        }
        if "description_word_count" in frame.columns:
            grouped["by_description_length"] = grouped_error_rates(
                frame, by="description_word_count",
                bins=DESCRIPTION_LENGTH_BINS, labels=DESCRIPTION_LENGTH_LABELS,
            )
        if "brightness" in frame.columns and frame["brightness"].notna().any():
            grouped["by_brightness"] = grouped_error_rates(frame, by="brightness", bins=4)
        for name, table in grouped.items():
            write_table(table, metrics_dir / f"error_{name}_{kind}.csv")
        digest["grouped"][kind] = {
            name: table.to_dict("records") for name, table in grouped.items()
        }

        examples = select_examples(frame, n_per_outcome=3, seed=seed)
        columns = [
            c for c in (
                "sample_id", "app_id", "name", "outcome", "true_genres", "pred_genres",
                "missed_genres", "spurious_genres", "jaccard", "max_prob",
                "description_word_count", "brightness", "image_path", "description",
            ) if c in examples.columns
        ]
        write_table(examples[columns], metrics_dir / f"error_examples_{kind}.csv")

    if len(frames) > 1:
        agreement = compare_model_errors(frames)
        write_table(agreement, metrics_dir / "error_model_agreement.csv")
        if {"jaccard_multimodal", "jaccard_image", "jaccard_text"} <= set(agreement.columns):
            means = {
                name: float(agreement[f"jaccard_{name}"].mean())
                for name in ("image", "text", "multimodal")
            }
            # Two different questions, and they have different answers: * against the best
            # SINGLE unimodal system -- the deployable baseline, and the honest reading of RQ3;
            best_single = "text" if means["text"] >= means["image"] else "image"
            single = agreement[f"jaccard_{best_single}"]
            oracle = agreement[["jaccard_image", "jaccard_text"]].max(axis=1)
            digest["multimodal_vs_unimodal"] = {
                "n_samples": int(len(agreement)),
                "best_single_unimodal": best_single,
                "vs_best_single_model": {
                    "fixed_by_fusion": int((agreement["jaccard_multimodal"] > single).sum()),
                    "broken_by_fusion": int((agreement["jaccard_multimodal"] < single).sum()),
                    "unchanged": int((agreement["jaccard_multimodal"] == single).sum()),
                },
                "vs_per_sample_oracle": {
                    "fixed_by_fusion": int((agreement["jaccard_multimodal"] > oracle).sum()),
                    "broken_by_fusion": int((agreement["jaccard_multimodal"] < oracle).sum()),
                    "unchanged": int((agreement["jaccard_multimodal"] == oracle).sum()),
                    "mean_jaccard_oracle": float(oracle.mean()),
                },
                "mean_jaccard_image": means["image"],
                "mean_jaccard_text": means["text"],
                "mean_jaccard_multimodal": means["multimodal"],
            }
    return digest


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def generate_figures(
    *,
    collected: dict[str, dict[int, dict[str, Any]]],
    comparison: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    bundle: Any,
    seed: int,
    n_gradcam: int,
    device: str,
) -> list[str]:
    """Render every report figure.  Missing inputs are skipped, never faked."""
    import matplotlib

    matplotlib.use("Agg")
    from gamesense.visualization import plots

    figures_dir = CONFIG.paths.figures
    written: list[str] = []

    def emit(name: str, builder: Any) -> None:
        """Render one figure, tolerating a missing input and never leaking figures."""
        import matplotlib.pyplot as plt

        path = figures_dir / name
        try:
            builder(path)
        except Exception as exc:  # a missing input must not abort the whole report
            LOGGER.warning("Figure %s skipped (%s: %s)", name, type(exc).__name__, exc)
            return
        finally:
            # Dozens of figures are produced in one run; leaving them open would
            # exhaust memory and trip matplotlib's max-open warning.
            plt.close("all")
        if path.is_file():
            written.append(path.name)

    # -- model comparison -------------------------------------------------- #
    if not comparison.empty:
        emit(
            "comparison_metrics.png",
            lambda path: plots.plot_metric_comparison(
                comparison, metrics=("micro_f1", "macro_f1", "mAP"), save_path=path
            ),
        )
        emit(
            "comparison_precision_recall.png",
            lambda path: plots.plot_precision_recall_scatter(
                {
                    kind: (per_seed.get(seed) or next(iter(per_seed.values())))[
                        "test_metrics_selected"
                    ]
                    for kind, per_seed in collected.items()
                },
                save_path=path,
            ),
        )
        emit(
            "comparison_per_class_f1.png",
            lambda path: plots.plot_per_class_f1(
                {
                    kind: (per_seed.get(seed) or next(iter(per_seed.values())))[
                        "test_metrics_selected"
                    ].get("per_class", {})
                    for kind, per_seed in collected.items()
                },
                classes=bundle.classes,
                save_path=path,
            ),
        )

    # -- training curves --------------------------------------------------- #
    histories: dict[str, dict[str, Any]] = {}
    for kind in collected:
        path = CONFIG.history_path(kind, seed)
        if path.is_file():
            histories[kind] = load_json(path)
            emit(
                f"training_curves_{kind}.png",
                lambda p, h=histories[kind], k=kind: plots.plot_training_curves(
                    h, title=MODEL_DISPLAY_NAMES.get(k, k), save_path=p
                ),
            )
    if len(histories) > 1:
        emit(
            "training_comparison.png",
            lambda path: plots.plot_history_comparison(
                histories, metric="val_micro_f1", save_path=path
            ),
        )

    # -- threshold analysis ------------------------------------------------ #
    for kind, per_seed in collected.items():
        payload = per_seed.get(seed) or next(iter(per_seed.values()))
        sweep = payload.get("threshold_sweep_val")
        if not sweep:
            continue
        best = payload.get("threshold_selected")
        emit(
            f"threshold_analysis_{kind}.png",
            lambda path, s=sweep, b=best, k=kind: plots.plot_threshold_analysis(
                pd.DataFrame(s),
                best_threshold=b if isinstance(b, (int, float)) else None,
                title=MODEL_DISPLAY_NAMES.get(k, k),
                save_path=path,
            ),
        )

    # -- precision-recall curves + confusion ------------------------------- #
    for kind in collected:
        prediction_path = CONFIG.predictions_path(kind, "test", seed)
        if not prediction_path.is_file():
            continue
        predictions = load_predictions(prediction_path)
        emit(
            f"pr_curves_{kind}.png",
            lambda path, p=predictions, k=kind: plots.plot_pr_curves(
                p["labels"], p["probabilities"], classes=bundle.classes,
                title=MODEL_DISPLAY_NAMES.get(k, k), save_path=path,
            ),
        )
    for kind, frame in frames.items():
        payload = collected.get(kind, {}).get(seed, {})
        threshold = payload.get("threshold_selected", CONFIG.evaluation.default_threshold)
        prediction_path = CONFIG.predictions_path(kind, "test", seed)
        if not prediction_path.is_file():
            continue
        predictions = load_predictions(prediction_path)
        matrix = confusion_pairs(
            labels=predictions["labels"],
            probabilities=predictions["probabilities"],
            threshold=np.asarray(threshold) if isinstance(threshold, list) else float(threshold),
            classes=bundle.classes,
        )
        emit(
            f"confusion_{kind}.png",
            lambda path, m=matrix, k=kind: plots.plot_confusion_heatmap(
                m, title=f"Genre confusion - {MODEL_DISPLAY_NAMES.get(k, k)}", save_path=path
            ),
        )

    # -- qualitative examples + grouped error rates ------------------------ #
    for kind, frame in frames.items():
        examples = select_examples(frame, n_per_outcome=2, seed=seed)
        emit(
            f"prediction_examples_{kind}.png",
            lambda path, e=examples: plots.plot_prediction_examples(
                e, root=CONFIG.paths.root, n=min(6, len(e)), classes=bundle.classes, save_path=path
            ),
        )
        emit(
            f"error_by_n_labels_{kind}.png",
            lambda path, f=frame: plots.plot_error_rate_by_group(
                grouped_error_rates(f, by="n_true"), x="n_true", save_path=path
            ),
        )
        if "description_word_count" in frame.columns:
            emit(
                f"error_by_description_length_{kind}.png",
                lambda path, f=frame: plots.plot_error_rate_by_group(
                    grouped_error_rates(
                        f, by="description_word_count",
                        bins=DESCRIPTION_LENGTH_BINS, labels=DESCRIPTION_LENGTH_LABELS,
                    ),
                    x="description_word_count",
                    save_path=path,
                ),
            )

    # -- Grad-CAM ---------------------------------------------------------- #
    if n_gradcam and CONFIG.checkpoint_path("image", seed).is_file():
        try:
            written.extend(
                _gradcam_figures(
                    bundle=bundle, frames=frames, seed=seed, n=n_gradcam, device=device
                )
            )
        except Exception as exc:
            LOGGER.warning("Grad-CAM figures skipped (%s: %s)", type(exc).__name__, exc)

    LOGGER.info("Generated %d figures in %s", len(written), figures_dir)
    return written


def _gradcam_figures(
    *, bundle: Any, frames: dict[str, pd.DataFrame], seed: int, n: int, device: str
) -> list[str]:
    """Render Grad-CAM panels for a mix of correct and incorrect predictions."""
    from gamesense.inference.predictor import GameSensePredictor
    from gamesense.visualization import plots

    frame = frames.get("image")
    if frame is None or frame.empty or "image_path" not in frame.columns:
        return []
    predictor = GameSensePredictor(config=CONFIG, device=device, seed=seed)
    chosen = select_examples(frame, n_per_outcome=max(1, n // 3), seed=seed)
    results = []
    for row in chosen.head(n).to_dict("records"):
        image_path = CONFIG.paths.root / str(row["image_path"])
        if not image_path.is_file():
            continue
        genre = (row.get("true_genres") or "").split("|")[0] or None
        try:
            explanation = predictor.explain(image=image_path, mode="image", genre=genre or None)
        except Exception as exc:
            LOGGER.warning("Grad-CAM failed for %s (%s)", row.get("sample_id"), exc)
            continue
        explanation.title = (  # type: ignore[attr-defined]
            f"{row.get('name', row['sample_id'])}\ntrue: {row.get('true_genres')} | "
            f"pred: {row.get('pred_genres')}"
        )
        results.append(explanation)
    if not results:
        return []
    path = CONFIG.paths.figures / "gradcam_examples.png"
    plots.plot_gradcam_grid(results, save_path=path)
    return [path.name] if path.is_file() else []


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    CONFIG.paths.ensure()
    metrics_dir = CONFIG.paths.metrics
    bundle = load_bundle(CONFIG)

    if args.reevaluate:
        for kind in args.models:
            for seed in args.seeds:
                if not CONFIG.checkpoint_path(kind, seed).is_file():
                    continue
                LOGGER.info("Re-evaluating %s seed %d from checkpoint", kind, seed)
                evaluate_model(kind, seed=seed, config=CONFIG, bundle=bundle, device=args.device)

    collected = collect_results(model_kinds=args.models, seeds=args.seeds, config=CONFIG)
    if not collected:
        LOGGER.error(
            "No metrics found in %s. Train the models first:\n"
            "  python scripts/train_image.py\n  python scripts/train_text.py\n"
            "  python scripts/train_multimodal.py",
            metrics_dir,
        )
        return 1
    LOGGER.info(
        "Found results for: %s",
        {kind: sorted(per_seed) for kind, per_seed in collected.items()},
    )

    # -- headline tables --------------------------------------------------- #
    comparison = build_comparison_table(collected, split="test", threshold="selected")
    write_table(comparison, metrics_dir / "comparison_test.csv")
    write_table(
        build_comparison_table(collected, split="test", threshold="default"),
        metrics_dir / "comparison_test_threshold050.csv",
    )
    write_table(
        build_comparison_table(collected, split="val", threshold="selected"),
        metrics_dir / "comparison_val.csv",
    )
    per_class = build_per_class_table(collected, seed=args.primary_seed)
    write_table(per_class, metrics_dir / "per_class_f1.csv")

    # -- error analysis ---------------------------------------------------- #
    frames = build_error_frames(
        collected, bundle=bundle, seed=args.primary_seed, image_stat_limit=args.image_stat_limit
    )
    error_digest = error_analysis_tables(
        frames, collected, bundle=bundle, seed=args.primary_seed, metrics_dir=metrics_dir
    )

    # -- figures ----------------------------------------------------------- #
    figures: list[str] = []
    if not args.no_figures:
        figures = generate_figures(
            collected=collected,
            comparison=comparison,
            frames=frames,
            bundle=bundle,
            seed=args.primary_seed,
            n_gradcam=args.gradcam_examples,
            device=args.device,
        )

    # -- machine readable digest ------------------------------------------- #
    aggregated = aggregate_over_seeds(collected, split="test", threshold="selected")
    summary = {
        "generated_from": {
            "models": sorted(collected),
            "seeds_found": {kind: sorted(per_seed) for kind, per_seed in collected.items()},
            "primary_seed": args.primary_seed,
        },
        "data": bundle.summary(),
        "metric_keys": list(METRIC_KEYS),
        "comparison_test_selected": comparison.to_dict("records"),
        "aggregated_over_seeds": aggregated,
        "thresholds_selected": {
            kind: (per_seed.get(args.primary_seed) or next(iter(per_seed.values())))[
                "threshold_selected"
            ]
            for kind, per_seed in collected.items()
        },
        "per_class": per_class.to_dict("records"),
        "error_analysis": error_digest,
        "figures": figures,
    }
    save_json(summary, metrics_dir / "results_summary.json")

    # -- console report ---------------------------------------------------- #
    LOGGER.info("=" * 78)
    LOGGER.info("MODEL COMPARISON (test split, validation-selected threshold)")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        LOGGER.info("\n%s", comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    LOGGER.info("-" * 78)
    LOGGER.info("Wrote %s and %d figures", (metrics_dir / "results_summary.json").name, len(figures))
    LOGGER.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
