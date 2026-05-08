"""
Generate matplotlib figures for the metasurface_aug_test experiment.

Reads artifacts written by:
- experiments.metasurface_aug_test.train_reconstruct  -> train_history.json
- experiments.metasurface_aug_test.evaluate           -> summary.json, examples.json

Writes PNGs into a dedicated subfolder (default: experiments/metasurface_aug_test/figures).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(train_history: dict, out_path: Path, show: bool) -> None:
    hist = train_history.get("history", {})
    train = hist.get("train_loss", [])
    val = hist.get("val_loss", [])

    fig = plt.figure(figsize=(6, 4))
    plt.plot(train, label="train")
    plt.plot(val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Masked reconstruction training")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_per_mask_bars(summary: dict, out_path: Path, show: bool) -> None:
    per_mask = summary.get("per_mask_type_masked_l1", {})
    baseline = summary.get("baseline_zero_fill_per_mask_type_masked_l1", {})

    mask_types = list(per_mask.keys())
    if not mask_types:
        raise ValueError("No mask types found in summary.json (per_mask_type_masked_l1 is empty).")

    score = np.array([per_mask[t] for t in mask_types], dtype=np.float32)
    base = np.array([baseline.get(t, np.nan) for t in mask_types], dtype=np.float32)

    x = np.arange(len(mask_types))
    w = 0.38

    fig = plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, base, width=w, label="baseline (zero fill)")
    plt.bar(x + w / 2, score, width=w, label="model")
    plt.xticks(x, mask_types)
    plt.ylabel("masked L1 (lower is better)")
    plt.title("Per mask-type difficulty")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_example_panel(example: dict, out_path: Path, show: bool) -> None:
    masked = np.array(example["masked_input"], dtype=np.float32)
    pred = np.array(example["prediction"], dtype=np.float32)
    target = np.array(example["target"], dtype=np.float32)
    visible = np.array(example["visible_mask"], dtype=np.float32)
    err = np.abs(pred - target)

    mats = [visible, masked, pred, target, err]
    titles = ["visible_mask (1=seen)", "masked_input", "prediction", "target", "|pred-target|"]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=True)
    last_im = None
    for ax, mat, title in zip(axes, mats, titles):
        last_im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("channel (6)")
    axes[0].set_ylabel("wavelength index (20)")
    fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)

    sample_index = example.get("sample_index", "?")
    mask_type = example.get("mask_type", "?")
    fig.suptitle(f"Example sample_index={sample_index} mask_type={mask_type}", y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate matplotlib figures for metasurface_aug_test.")
    p.add_argument(
        "--results_dir",
        type=str,
        default="experiments/metasurface_aug_test/results",
        help="Directory containing train_history.json, summary.json, examples.json.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="experiments/metasurface_aug_test/figures",
        help="Directory to write PNG figures into.",
    )
    p.add_argument(
        "--example_idx",
        type=int,
        default=0,
        help="Which entry in examples.json to visualize.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display figures interactively (in addition to saving).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    train_history_path = results_dir / "train_history.json"
    summary_path = results_dir / "summary.json"
    examples_path = results_dir / "examples.json"

    if not train_history_path.exists():
        raise FileNotFoundError(f"Missing {train_history_path}. Run train_reconstruct first.")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run evaluate first.")
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing {examples_path}. Run evaluate first.")

    train_history = _read_json(train_history_path)
    summary = _read_json(summary_path)
    examples = _read_json(examples_path)
    if not isinstance(examples, list) or len(examples) == 0:
        raise ValueError("examples.json is empty; re-run evaluate to generate sample examples.")
    if args.example_idx < 0 or args.example_idx >= len(examples):
        raise IndexError(f"example_idx {args.example_idx} out of range (0..{len(examples)-1}).")

    plot_loss_curves(train_history, out_dir / "loss_curve.png", show=args.show)
    plot_per_mask_bars(summary, out_dir / "per_mask_bars.png", show=args.show)

    ex = examples[args.example_idx]
    sample_index = ex.get("sample_index", args.example_idx)
    mask_type = ex.get("mask_type", "unknown")
    plot_example_panel(ex, out_dir / f"example_{sample_index}_{mask_type}_panel.png", show=args.show)

    print(f"Wrote figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

