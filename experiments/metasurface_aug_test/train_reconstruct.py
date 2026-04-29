"""
Train masked reconstruction on synthetic Jones-matrix data.
"""

import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
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


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device, cfg: Config):
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
            masked_den = masked_region.sum().clamp_min(1.0)
            l1_masked = (torch.abs(pred - target) * masked_region).sum() / masked_den
            l1_full = F.l1_loss(pred, target)
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
    model = JonesReconstructionMLP(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, device, cfg)
        val_loss = _run_epoch(model, val_loader, None, device, cfg)
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
        "history": history,
        "best_val_loss": best_val,
    }
    with open(os.path.join(cfg.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    train_main()
