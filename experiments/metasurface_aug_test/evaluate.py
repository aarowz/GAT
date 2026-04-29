"""
Evaluate reconstruction model and write summary metrics.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .masking import MASK_TYPES, apply_mask
from .model import JonesReconstructionMLP


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, visible: torch.Tensor) -> float:
    mr = 1.0 - visible
    denom = mr.sum().clamp_min(1.0)
    return float(((pred - target).abs() * mr).sum().item() / denom.item())


def evaluate_main() -> dict:
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = np.load(os.path.join(cfg.output_dir, "splits.npz"))
    test_data = splits["test"].astype(np.float32)

    model = JonesReconstructionMLP(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    ckpt = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    rng = np.random.default_rng(cfg.seed + 1000)
    per_type_scores = {m: [] for m in MASK_TYPES}
    sample_examples = []

    with torch.no_grad():
        for idx, sample in enumerate(test_data):
            for mtype in MASK_TYPES:
                out = apply_mask(sample, rng, mtype)
                masked = torch.from_numpy(out.masked_input).unsqueeze(0).to(device)
                visible = torch.from_numpy(out.visible_mask).unsqueeze(0).to(device)
                target = torch.from_numpy(out.target).unsqueeze(0).to(device)

                pred = model(masked, visible)
                score = _masked_l1(pred, target, visible)
                per_type_scores[mtype].append(score)

                if idx < 3 and mtype == "type1":
                    sample_examples.append(
                        {
                            "sample_index": idx,
                            "mask_type": mtype,
                            "masked_l1": score,
                            "masked_input": out.masked_input.tolist(),
                            "prediction": pred.squeeze(0).cpu().numpy().tolist(),
                            "target": out.target.tolist(),
                            "visible_mask": out.visible_mask.tolist(),
                        }
                    )

    per_type_mean = {k: float(np.mean(v)) if v else None for k, v in per_type_scores.items()}
    aggregate = float(np.mean([x for v in per_type_scores.values() for x in v]))

    # Baseline: zero-fill predictor => output equals masked input
    baseline_scores = {m: [] for m in MASK_TYPES}
    for sample in test_data:
        for mtype in MASK_TYPES:
            out = apply_mask(sample, rng, mtype)
            pred = torch.from_numpy(out.masked_input).unsqueeze(0)
            target = torch.from_numpy(out.target).unsqueeze(0)
            visible = torch.from_numpy(out.visible_mask).unsqueeze(0)
            baseline_scores[mtype].append(_masked_l1(pred, target, visible))

    baseline_per_type = {k: float(np.mean(v)) if v else None for k, v in baseline_scores.items()}
    baseline_aggregate = float(np.mean([x for v in baseline_scores.values() for x in v]))

    summary = {
        "aggregate_masked_l1": aggregate,
        "per_mask_type_masked_l1": per_type_mean,
        "baseline_zero_fill_aggregate_masked_l1": baseline_aggregate,
        "baseline_zero_fill_per_mask_type_masked_l1": baseline_per_type,
        "improvement_vs_baseline": baseline_aggregate - aggregate,
        "num_test_samples": int(test_data.shape[0]),
    }

    with open(os.path.join(cfg.output_dir, cfg.summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(cfg.output_dir, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(sample_examples, f)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    evaluate_main()
