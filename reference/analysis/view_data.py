"""
Exploratory viewer for the reference dataset.

Features:
- Loads a small `FieldDataset` using the same hyperparameters as `train_base_model.py`
- Inspects a few samples:
  - Saves input node features for a 7x7 window to CSV (via pandas)
  - Plots the 7x7 input window channels (R, H, D0, D1, boundary)
  - Plots the corresponding 6-channel output field (real/imag Ex, Ey, Ez)
- Writes outputs under:
  - `reference/analysis/figures/`
  - `reference/analysis/reports/`

You can run this script as:
- From repo root: `python reference/analysis/view_data.py`
- Or from `reference/analysis`: `python view_data.py`
"""

import os
import sys
import json
from glob import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas is required for `view_data.py` (for tabular inspection). "
        "Install it with `pip install pandas` or `conda install pandas`."
    ) from e


# ---------------------------------------------------------------------------
# Path setup: make sure we can import the reference data utilities
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../reference/analysis
REF_DIR = os.path.dirname(THIS_DIR)                          # .../reference
REPO_ROOT = os.path.dirname(REF_DIR)                         # repo root (.../GAT)

if REF_DIR not in sys.path:
    sys.path.append(REF_DIR)

from data_utils import FieldDataset, compute_dataset_stats, set_random_seed  # type: ignore

# Data directory (same as main/config: repo_root/data)
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Analysis output directories
FIG_DIR = os.path.join(THIS_DIR, "figures")
REPORT_DIR = os.path.join(THIS_DIR, "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def _get_file_list(max_files: int = 2):
    """Return a small list of data3d_*.mat files from DATA_DIR."""
    pattern = os.path.join(DATA_DIR, "data3d_*.mat")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No data3d_*.mat files found under {DATA_DIR}")
    return files[:max_files]


def _build_dataset(files, window_size=7, pd_size=6, Ng=24, R=3, E_scale=1.0,
                   num_samples=2000, normalize_output=False):
    """
    Build a small FieldDataset for analysis, mirroring train_base_model.py defaults.
    """
    print(f"Using {len(files)} files for statistics and dataset:")
    for fp in files:
        print(f"  - {fp}")

    # Compute normalization stats on the chosen files
    in_mean_np, in_std_np, out_mean_np, out_std_np = compute_dataset_stats(
        files,
        window_size=window_size,
        pd_size=pd_size,
        R=R,
        E_scale=E_scale,
    )
    in_mean = torch.tensor(in_mean_np, dtype=torch.float32)
    in_std = torch.tensor(in_std_np, dtype=torch.float32)
    out_mean = torch.tensor(out_mean_np, dtype=torch.float32)
    out_std = torch.tensor(out_std_np, dtype=torch.float32)

    print("Input mean:", in_mean.numpy())
    print("Input std:", in_std.numpy())
    print("Output mean:", out_mean.numpy())
    print("Output std:", out_std.numpy())

    dataset = FieldDataset(
        files,
        split="train",
        window_size=window_size,
        pd_size=pd_size,
        Ng=Ng,
        R=R,
        E_scale=E_scale,
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
        indices=None,
        num_samples=num_samples,
        normalize_output=normalize_output,
    )
    return dataset


def _save_input_dataframe(x_window: np.ndarray, sample_idx: int):
    """
    x_window: (7, 7, 5) array of input features in the main window.
    Saves a CSV with 49 rows (one per node) and 5 columns (R, H, D0, D1, B).
    """
    feature_names = ["R", "H", "D0", "D1", "B"]
    flat = x_window.reshape(-1, x_window.shape[-1])  # (49, 5)
    df = pd.DataFrame(flat, columns=feature_names)
    csv_path = os.path.join(REPORT_DIR, f"input_vectors_sample{sample_idx+1}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved input vectors CSV for sample {sample_idx} → {csv_path}")


def _plot_input_window(x_window: np.ndarray, sample_idx: int):
    """
    Plot the 7x7 input window channels (R, H, D0, D1, B).
    x_window: (7, 7, 5)
    """
    feature_names = ["R", "H", "D0", "D1", "B"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for i, name in enumerate(feature_names):
        ax = axes[i]
        im = ax.imshow(x_window[:, :, i], cmap="viridis", aspect="equal")
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Input 7x7 window features — sample {sample_idx}", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"inputs_sample{sample_idx+1}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved input window figure for sample {sample_idx} → {out_path}")


def _plot_output_field(y: np.ndarray, sample_idx: int):
    """
    Plot the 6-channel output field (Ex, Ey, Ez real/imag) as 3x2 subplots.
    y: (6, H, W)
    """
    components = ["Ex (Real)", "Ex (Imag)", "Ey (Real)", "Ey (Imag)", "Ez (Real)", "Ez (Imag)"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(6):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        im = ax.imshow(y[i], cmap="RdBu_r", aspect="auto")
        ax.set_title(components[i])
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Output field (6 channels) — sample {sample_idx}", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"fields_sample{sample_idx+1}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved output field figure for sample {sample_idx} → {out_path}")


def _save_shape_summary(data, dataset, sample_idx: int):
    """
    Save a small JSON summary of shapes and key parameters.
    """
    x = data.x
    y = data.y
    summary = {
        "sample_index": sample_idx,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "graph_shape": list(getattr(dataset, "graph_shape", [])),
        "window_size": getattr(dataset, "window_size", None),
        "pd_size": getattr(dataset, "pd_size", None),
        "Ng": getattr(dataset, "Ng", None),
        "R": getattr(dataset, "R", None),
    }
    summary_path = os.path.join(REPORT_DIR, f"summary_sample{sample_idx+1}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON for sample {sample_idx} → {summary_path}")


def main(num_samples: int = 2):
    """
    Entry point: build a small dataset, sample a few items, and visualize/inspect them.
    """
    print("=== Reference dataset viewer ===")
    print(f"Repo root:        {REPO_ROOT}")
    print(f"Reference dir:    {REF_DIR}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"Figures output:   {FIG_DIR}")
    print(f"Reports output:   {REPORT_DIR}")

    set_random_seed(42)

    files = _get_file_list(max_files=2)

    # Hyperparameters aligned with train_base_model.py
    WINDOW_SIZE = 7
    PD_SIZE = 6
    R = 3
    NG = 24
    E_SCALE = 1.0

    dataset = _build_dataset(
        files,
        window_size=WINDOW_SIZE,
        pd_size=PD_SIZE,
        Ng=NG,
        R=R,
        E_scale=E_SCALE,
        num_samples=2000,
        normalize_output=False,
    )

    if len(dataset) == 0:
        print("Dataset is empty; nothing to visualize.")
        return

    num_samples = min(num_samples, len(dataset))
    print(f"Inspecting first {num_samples} samples...")

    h, w = dataset.graph_shape  # (window_size + 2*pd_size, window_size + 2*pd_size)
    pd_size = dataset.pd_size
    window_size = dataset.window_size

    for idx in range(num_samples):
        data = dataset[idx]
        x = data.x.detach().cpu().numpy()          # (num_nodes, 5)
        y = data.y.detach().cpu().numpy()          # (6, H, W)

        # Reconstruct grid from flattened x (same as in FieldDataset.__getitem__)
        x_grid = x.reshape(h, w, -1)               # (H_grid, W_grid, 5)

        # Crop to main 7x7 window (exclude padding)
        x_window = x_grid[pd_size:pd_size + window_size,
                          pd_size:pd_size + window_size, :]

        print(f"\n--- Sample {idx} ---")
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("graph_shape:", dataset.graph_shape)
        print("window_size:", window_size, "pd_size:", pd_size, "Ng:", dataset.Ng)

        _save_input_dataframe(x_window, idx)
        _plot_input_window(x_window, idx)
        _plot_output_field(y, idx)
        _save_shape_summary(data, dataset, idx)

    print("\nDone. Inspect figures and reports in:")
    print(f"  - {FIG_DIR}")
    print(f"  - {REPORT_DIR}")


if __name__ == "__main__":
    main()

