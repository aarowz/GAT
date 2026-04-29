"""
Train masked reconstruction on synthetic Jones-matrix data.
"""

import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .masking import MASK_TYPES, apply_mask, sample_mask_type
from .model import JonesReconstructionMLP
from .synthetic_data import generate_synthetic_jones_data, split_dataset


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class JonesMaskedDataset(Dataset):
    def __init__(self, data: np.ndarray, seed: int = 42):
        self.data = data.astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        mtype = sample_mask_type(self.rng)
        out = apply_mask(sample, self.rng, mtype)
        return (
            torch.from_numpy(out.masked_input),
            torch.from_numpy(out.visible_mask),
            torch.from_numpy(out.target),
            mtype,
        )


def _collate(batch):
    masked, visible, target, mtypes = zip(*batch)
    return (
        torch.stack(masked, dim=0),
        torch.stack(visible, dim=0),
        torch.stack(target, dim=0),
        list(mtypes),
    )


def _compute_channel_weights(train_data: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute per-channel weights from train-set statistics.
    Uses inverse std so larger-scale channels are down-weighted.
    """
    if not cfg.use_channel_balance:
        return np.ones((cfg.num_channels,), dtype=np.float32)

    # train_data: (N, 20, 6) -> std over N and wavelength dimensions
    channel_std = train_data.std(axis=(0, 1)).astype(np.float32)
    inv = 1.0 / np.clip(channel_std, cfg.channel_balance_eps, None)
    weights = inv / np.mean(inv)
    weights = np.clip(weights, cfg.channel_weight_min, cfg.channel_weight_max)
    weights = (weights / np.mean(weights)).astype(np.float32)
    return weights


def _weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_weights: torch.Tensor,
    element_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weighted L1 where channel_weights shape is (6,).
    element_mask, if provided, must be broadcastable to pred (B, 20, 6).
    """
    weights = channel_weights.view(1, 1, -1)
    abs_err = torch.abs(pred - target)
    if element_mask is not None:
        weighted_num = (abs_err * element_mask * weights).sum()
        weighted_den = (element_mask * weights).sum().clamp_min(1.0)
    else:
        weighted_num = (abs_err * weights).sum()
        weighted_den = (torch.ones_like(abs_err) * weights).sum().clamp_min(1.0)
    return weighted_num / weighted_den


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    cfg: Config,
    channel_weights: torch.Tensor,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.set_grad_enabled(is_train):
        for masked, visible, target, _ in loader:
            masked = masked.to(device)
            visible = visible.to(device)
            target = target.to(device)

            pred = model(masked, visible)
            masked_region = 1.0 - visible
            l1_masked = _weighted_l1(
                pred=pred,
                target=target,
                channel_weights=channel_weights,
                element_mask=masked_region,
            )
            l1_full = _weighted_l1(
                pred=pred,
                target=target,
                channel_weights=channel_weights,
                element_mask=None,
            )
            loss = cfg.masked_loss_weight * l1_masked + cfg.full_loss_weight * l1_full

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = target.shape[0]
            total_loss += loss.item() * bs
            total_items += bs
    return total_loss / max(total_items, 1)


def train_main() -> dict:
    cfg = Config()
    _seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_data = generate_synthetic_jones_data(
        num_samples=cfg.num_samples,
        num_wavelengths=cfg.num_wavelengths,
        seed=cfg.seed,
    )
    train_data, val_data, test_data = split_dataset(all_data, cfg.train_split, cfg.val_split, cfg.seed)
    np.savez(
        os.path.join(cfg.output_dir, "splits.npz"),
        train=train_data,
        val=val_data,
        test=test_data,
    )

    train_ds = JonesMaskedDataset(train_data, seed=cfg.seed + 1)
    val_ds = JonesMaskedDataset(val_data, seed=cfg.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel_weights_np = _compute_channel_weights(train_data, cfg)
    channel_weights_t = torch.from_numpy(channel_weights_np).to(device)
    model = JonesReconstructionMLP(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, device, cfg, channel_weights_t)
        val_loss = _run_epoch(model, val_loader, None, device, cfg, channel_weights_t)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    out = {
        "config": asdict(cfg),
        "device": str(device),
        "mask_types": list(MASK_TYPES),
        "channel_weights": channel_weights_np.tolist(),
        "history": history,
        "best_val_loss": best_val,
    }
    with open(os.path.join(cfg.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    train_main()
