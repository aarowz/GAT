"""
Paper-inspired masking strategies for 20x6 Jones matrices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class MaskResult:
    masked_input: np.ndarray
    visible_mask: np.ndarray
    target: np.ndarray
    mask_type: str


MASK_TYPES = ("type1", "type2", "type3", "type4", "type5")


def _sample_component_pair(rng: np.random.Generator) -> Tuple[int, int]:
    # component k means amplitude channel k and phase channel k+3
    k = int(rng.integers(0, 3))
    return k, k + 3


def apply_mask(sample: np.ndarray, rng: np.random.Generator, mask_type: str) -> MaskResult:
    """
    sample: (20, 6)
    visible_mask: (20, 6), 1 for visible/kept, 0 for masked.
    """
    assert sample.ndim == 2 and sample.shape[1] == 6
    w = sample.shape[0]
    visible = np.zeros_like(sample, dtype=np.float32)

    wl = int(rng.integers(0, w))
    amp_channels = [0, 1, 2]

    if mask_type == "type1":
        # keep all six components at one random wavelength
        visible[wl, :] = 1.0
    elif mask_type == "type2":
        # keep all amplitudes across all wavelengths + phases at one wavelength
        visible[:, amp_channels] = 1.0
        visible[wl, 3:] = 1.0
    elif mask_type == "type3":
        # keep amplitude and phase of one component at one wavelength
        a_idx, p_idx = _sample_component_pair(rng)
        visible[wl, a_idx] = 1.0
        visible[wl, p_idx] = 1.0
    elif mask_type == "type4":
        # keep one component amplitudes across all wavelengths + phase at one wavelength
        a_idx, p_idx = _sample_component_pair(rng)
        visible[:, a_idx] = 1.0
        visible[wl, p_idx] = 1.0
    elif mask_type == "type5":
        # keep amplitude and phase of one component across all wavelengths
        a_idx, p_idx = _sample_component_pair(rng)
        visible[:, a_idx] = 1.0
        visible[:, p_idx] = 1.0
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    masked = sample * visible
    return MaskResult(masked_input=masked, visible_mask=visible, target=sample, mask_type=mask_type)


def sample_mask_type(rng: np.random.Generator, probs: Dict[str, float] | None = None) -> str:
    if probs is None:
        probs = {m: 1.0 / len(MASK_TYPES) for m in MASK_TYPES}
    p = np.array([probs[m] for m in MASK_TYPES], dtype=np.float32)
    p = p / p.sum()
    return str(rng.choice(np.array(MASK_TYPES), p=p))
