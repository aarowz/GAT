# Experiments

This folder holds **standalone prototypes** that are not wired into the main GAT-Net training path (`main.py`, `gat_net/`).

## Jones matrix masking prototype (`metasurface_aug_test/`)

**Purpose:** Explore paper-inspired **physics-informed augmentation** for metasurfaces: treat each sample as a `20 × 6` Jones-style matrix (20 wavelength bins, six channels: three amplitudes and three phases), apply **masking strategies** similar to MetasurfaceViT pretraining, and train a small **reconstruction** model to fill masked values.

**Current state:**

- **Data:** Synthetic Jones-like tensors for fast iteration; swap-in loaders for real data can reuse the same masking API.
- **Masking:** Five mask types (`type1`–`type5`) in `metasurface_aug_test/masking.py`.
- **Training:** Lightweight MLP reconstructor in `metasurface_aug_test/model.py`; loss emphasizes masked regions with optional full-matrix regularization and **train-set channel balancing** (inverse std, clamped) so phase channels do not dominate amplitude channels.
- **Outputs:** `metasurface_aug_test/results/` — checkpoints, `train_history.json`, `summary.json` (unweighted and weighted masked L1), splits, and example dumps.

**Run (from repo root):**

```bash
python3 -m experiments.metasurface_aug_test.train_reconstruct
python3 -m experiments.metasurface_aug_test.evaluate
```

More detail lives in [`metasurface_aug_test/README.md`](metasurface_aug_test/README.md).
