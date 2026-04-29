"""
Synthetic Jones-matrix-like data generator.
"""

import numpy as np


def _smooth_curve(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = rng.uniform(0.2, 1.5)
    b = rng.uniform(0.2, 1.5)
    p = rng.uniform(0.0, 2.0 * np.pi)
    y = 0.5 + 0.35 * np.sin(2.0 * np.pi * a * x + p) + 0.15 * np.cos(2.0 * np.pi * b * x)
    y += rng.normal(0.0, 0.03, size=n).astype(np.float32)
    y = np.clip(y, 0.0, 1.0)
    return (low + (high - low) * y).astype(np.float32)


def generate_synthetic_jones_data(
    num_samples: int,
    num_wavelengths: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Return array of shape (N, 20, 6):
      channels 0:3 amplitudes in [0, 1]
      channels 3:6 phases in [-pi, pi]
    """
    rng = np.random.default_rng(seed)
    all_samples = np.zeros((num_samples, num_wavelengths, 6), dtype=np.float32)

    for i in range(num_samples):
        amps = np.stack(
            [_smooth_curve(rng, num_wavelengths, 0.0, 1.0) for _ in range(3)],
            axis=1,
        )
        phases = np.stack(
            [_smooth_curve(rng, num_wavelengths, -np.pi, np.pi) for _ in range(3)],
            axis=1,
        )
        all_samples[i, :, :3] = amps
        all_samples[i, :, 3:] = phases
    return all_samples


def split_dataset(
    data: np.ndarray,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return data[train_idx], data[val_idx], data[test_idx]
