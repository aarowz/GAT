# Metasurface Augmentation Quick Test

Standalone prototype for paper-inspired physics-informed masking on `20x6`
Jones-matrix-like samples.

This directory is intentionally isolated from the existing GAT-Net pipeline.

## What it does

- Generates synthetic Jones-style samples (`20 x 6`).
- Applies one of five masking strategies inspired by MetasurfaceViT.
- Trains a lightweight reconstruction model using L1 loss.
- Reports aggregate and per-mask-type metrics.

## Quick start

From repo root:

```bash
python -m experiments.metasurface_aug_test.train_reconstruct
python -m experiments.metasurface_aug_test.evaluate
```

Outputs are written under `experiments/metasurface_aug_test/results/`.
