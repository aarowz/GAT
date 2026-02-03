# Training outputs

This directory holds all outputs from training:

- **checkpoints/** – Saved model weights (`.pth` files)
- **figures/** – Loss curves and E-field prediction visualizations (`.png` files)

Generated when you run `python main.py`. Do not commit large files; they are listed in `.gitignore`.

If you have existing outputs in the project root (e.g. after transferring from a cluster), move them here:

- `gat_net_model.pth` → `outputs/checkpoints/`
- `training_losses.png`, `efield_*.png` → `outputs/figures/`
