# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAT-Net predicts electromagnetic field distributions (Ex, Ey, Ez — real and imaginary) from metasurface geometry encoded as graphs. Input is a 15×15 spatial block of metasurface cells; output is a 6-channel 360×360 E-field prediction.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the main model
python main.py

# Run Jones matrix augmentation experiment
python -m experiments.metasurface_aug_test.train_reconstruct
python -m experiments.metasurface_aug_test.evaluate
```

There is no test suite or linting configuration.

## Architecture

All hyperparameters live in `config.py` (root). The experiment has its own `experiments/metasurface_aug_test/config.py`.

### Data pipeline (`gat_net/dataset.py`)
- Loads `.mat` files from `data/` (11 files, ~2.1 GB, not committed)
- Each file yields ~1,550 random 15×15 block samples → ~17k total
- Each block becomes a PyG graph: nodes are metasurface cells with 5 features `[R, H, D_x, D_y, boundary]`; edges connect all cells within Euclidean distance ≤2
- Targets are 6-channel E-fields (real/imag of Ex, Ey, Ez) upsampled to 360×360

### Model (`gat_net/model.py`)
Three-stage pipeline:
1. **GAT**: GATv2Conv (hidden=200, heads=8, with edge attributes) → 1600-dim node embeddings
2. **GCN**: Two FC layers (1600→1600) with batch norm — transforms node features before spatial reshape
3. **CNN + PixelShuffle**: 5×5 conv blocks followed by learnable upsampling (factors of 2s and 3s) to reach 24× upsampling from 15→360

### Training (`gat_net/train.py`)
- Adam optimizer, lr=5e-4, weight_decay=1e-5
- ReduceLROnPlateau scheduler
- AMP (automatic mixed precision) when CUDA is available
- EarlyStopping with patience=15
- MSE loss on E-field predictions
- Saves best checkpoint to `outputs/checkpoints/`; generates visualizations to `outputs/figures/`

### Experiments (`experiments/metasurface_aug_test/`)
A standalone prototype exploring physics-informed data augmentation via Jones matrix masking. Independent of the main `gat_net` pipeline — uses synthetic data and a lightweight MLP. Five masking strategies (`masking.py`) simulate partial measurement scenarios. Channel balancing in the training loop (inverse std clamping) prevents phase components from dominating amplitude.

## Key Data Format

`.mat` files contain:
- `D` [2, H, W]: displacement
- `R` [H, W]: material property
- `H` [H, W]: height
- `Ex`, `Ey`, `Ez` [H_field, W_field]: complex E-field components (or a combined `E` array)

`outputs/` is gitignored for checkpoints and figures; `data/` and `*.mat` are gitignored entirely.
