"""
Visualization module for GAT-Net predictions.

Generates visualizations comparing predicted vs ground truth E-field components,
showing both real and imaginary parts for Ex, Ey, and Ez.
Uses mesh refinement factor to upsample 15×15 to finer grid for display.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# When data has refinement <= 1 (same res as structure), use this for display so mesh shows
DEFAULT_DISPLAY_REFINEMENT = 24


def _upsample_for_mesh(arr, refinement, order=2):
    """
    Upsample array by mesh refinement factor for finer display.
    arr: (H, W) or (C, H, W). refinement: int (e.g. 24).
    Returns array of shape (H*refinement, W*refinement) or (C, H*ref, W*ref).
    """
    if refinement is None or refinement <= 1:
        return arr
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return zoom(arr, refinement, order=order)
    if arr.ndim == 3:
        return zoom(arr, (1, refinement, refinement), order=order)
    return arr


def visualize_efield_predictions(predictions, targets, num_samples=1, save_path='efield_predictions.png', mesh_refinement=None):
    """
    Visualize predicted E-field components vs ground truth.
    Uses a clean 3x2 layout (like efield_comparison) per figure: one row per
    field component (Ex, Ey, Ez), two columns (Predicted, Ground Truth).
    If mesh_refinement is set (e.g. 24), upsampled to finer grid for display.
    
    Args:
        predictions: Tensor of shape (batch_size, 6, H, W) - model predictions
        targets: Tensor of shape (batch_size, 6, H, W) - ground truth
        num_samples: Number of samples to visualize (default: 1 for clean layout)
        save_path: Path to save the visualization (default: 'efield_predictions.png')
        mesh_refinement: Optional int or array of ints (one per sample). If set, upsample for display.
    """
    # Convert to numpy and move to CPU if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    batch_size = min(predictions.shape[0], num_samples)
    field_names = ['Ex', 'Ey', 'Ez']
    figsize = (12, 10)
    fontsize_title = 14
    fontsize_label = 12
    
    for sample_idx in range(batch_size):
        pred = predictions[sample_idx]   # (6, H, W)
        target = targets[sample_idx]    # (6, H, W)
        ref = (mesh_refinement[sample_idx] if hasattr(mesh_refinement, '__len__') and not isinstance(mesh_refinement, (int, float)) else mesh_refinement) if mesh_refinement is not None else None
        if ref is None or int(ref) <= 1:
            ref = DEFAULT_DISPLAY_REFINEMENT
        pred = _upsample_for_mesh(pred, int(ref))
        target = _upsample_for_mesh(target, int(ref))
        
        # --- Real parts: 3 rows (Ex, Ey, Ez) x 2 cols (Predicted, Ground Truth) ---
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        for field_idx, field_name in enumerate(field_names):
            real_pred = pred[field_idx * 2]
            real_target = target[field_idx * 2]
            im1 = axes[field_idx, 0].imshow(real_pred, cmap='RdBu', aspect='auto')
            axes[field_idx, 0].set_title(f'{field_name} Real (Predicted)', fontsize=fontsize_title)
            axes[field_idx, 0].set_xlabel('Width', fontsize=fontsize_label)
            axes[field_idx, 0].set_ylabel('Height', fontsize=fontsize_label)
            plt.colorbar(im1, ax=axes[field_idx, 0], fraction=0.046, pad=0.04)
            im2 = axes[field_idx, 1].imshow(real_target, cmap='RdBu', aspect='auto')
            axes[field_idx, 1].set_title(f'{field_name} Real (Ground Truth)', fontsize=fontsize_title)
            axes[field_idx, 1].set_xlabel('Width', fontsize=fontsize_label)
            axes[field_idx, 1].set_ylabel('Height', fontsize=fontsize_label)
            plt.colorbar(im2, ax=axes[field_idx, 1], fraction=0.046, pad=0.04)
        plt.suptitle(f'E-Field Predictions vs Ground Truth (Real Parts) — Sample {sample_idx + 1}', fontsize=16)
        plt.tight_layout()
        out_real = save_path.replace('.png', '_real.png') if batch_size == 1 else save_path.replace('.png', f'_real_sample{sample_idx + 1}.png')
        plt.savefig(out_real, dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- Imaginary parts: same 3x2 layout ---
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        for field_idx, field_name in enumerate(field_names):
            imag_pred = pred[field_idx * 2 + 1]
            imag_target = target[field_idx * 2 + 1]
            im1 = axes[field_idx, 0].imshow(imag_pred, cmap='RdBu', aspect='auto')
            axes[field_idx, 0].set_title(f'{field_name} Imaginary (Predicted)', fontsize=fontsize_title)
            axes[field_idx, 0].set_xlabel('Width', fontsize=fontsize_label)
            axes[field_idx, 0].set_ylabel('Height', fontsize=fontsize_label)
            plt.colorbar(im1, ax=axes[field_idx, 0], fraction=0.046, pad=0.04)
            im2 = axes[field_idx, 1].imshow(imag_target, cmap='RdBu', aspect='auto')
            axes[field_idx, 1].set_title(f'{field_name} Imaginary (Ground Truth)', fontsize=fontsize_title)
            axes[field_idx, 1].set_xlabel('Width', fontsize=fontsize_label)
            axes[field_idx, 1].set_ylabel('Height', fontsize=fontsize_label)
            plt.colorbar(im2, ax=axes[field_idx, 1], fraction=0.046, pad=0.04)
        plt.suptitle(f'E-Field Predictions vs Ground Truth (Imaginary Parts) — Sample {sample_idx + 1}', fontsize=16)
        plt.tight_layout()
        out_imag = save_path.replace('.png', '_imaginary.png') if batch_size == 1 else save_path.replace('.png', f'_imaginary_sample{sample_idx + 1}.png')
        plt.savefig(out_imag, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"E-field visualizations saved:")
    print(f"  - {save_path.replace('.png', '_real.png')}")
    print(f"  - {save_path.replace('.png', '_imaginary.png')}")


def visualize_efield_comparison(predictions, targets, sample_idx=0, save_path='efield_comparison.png', mesh_refinement=None):
    """
    Create a detailed comparison visualization for a single sample.
    If mesh_refinement is set (e.g. 24), upsampled to finer grid for display.
    
    Args:
        predictions: Tensor of shape (batch_size, 6, H, W)
        targets: Tensor of shape (batch_size, 6, H, W)
        sample_idx: Index of sample to visualize (default: 0)
        save_path: Path to save the visualization
        mesh_refinement: Optional int or array (one per sample). If set, upsample for display.
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    pred = predictions[sample_idx]  # (6, H, W)
    target = targets[sample_idx]    # (6, H, W)
    # #region agent log
    _log = lambda **kw: open("/Users/az/GAT/.cursor/debug.log", "a").write(__import__("json").dumps({"sessionId":"debug-session","runId":"run1",**kw}) + "\n")
    _log(location="visualize.py:comparison", message="mesh_refinement received", hypothesisId="H3,H4", data={"mesh_refinement_is_none": mesh_refinement is None, "type": type(mesh_refinement).__name__, "sample_idx": sample_idx})
    # #endregion
    ref = (mesh_refinement[sample_idx] if hasattr(mesh_refinement, '__len__') and not isinstance(mesh_refinement, (int, float)) else mesh_refinement) if mesh_refinement is not None else None
    # #region agent log
    _log(location="visualize.py:ref", message="ref computed", hypothesisId="H3,H4", data={"ref": ref, "ref_is_none": ref is None, "pred_shape_before": list(pred.shape)})
    # #endregion
    if ref is None or int(ref) <= 1:
        ref = DEFAULT_DISPLAY_REFINEMENT
    pred = _upsample_for_mesh(pred, int(ref))
    target = _upsample_for_mesh(target, int(ref))
    # #region agent log
    _log(location="visualize.py:after_upsample", message="after upsample", hypothesisId="H4", data={"pred_shape_after": list(pred.shape)})
    # #endregion
    
    field_names = ['Ex', 'Ey', 'Ez']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for field_idx, field_name in enumerate(field_names):
        row = field_idx
        
        # Real part - Predicted
        im1 = axes[row, 0].imshow(pred[field_idx * 2], cmap='RdBu', aspect='auto')
        axes[row, 0].set_title(f'{field_name} Real (Predicted)')
        axes[row, 0].set_xlabel('Width')
        axes[row, 0].set_ylabel('Height')
        plt.colorbar(im1, ax=axes[row, 0])
        
        # Real part - Ground Truth
        im2 = axes[row, 1].imshow(target[field_idx * 2], cmap='RdBu', aspect='auto')
        axes[row, 1].set_title(f'{field_name} Real (Ground Truth)')
        axes[row, 1].set_xlabel('Width')
        axes[row, 1].set_ylabel('Height')
        plt.colorbar(im2, ax=axes[row, 1])
        
        # Real part - Error
        error_real = np.abs(pred[field_idx * 2] - target[field_idx * 2])
        im3 = axes[row, 2].imshow(error_real, cmap='hot', aspect='auto')
        axes[row, 2].set_title(f'{field_name} Real (Error)')
        axes[row, 2].set_xlabel('Width')
        axes[row, 2].set_ylabel('Height')
        plt.colorbar(im3, ax=axes[row, 2])
        
        # Imaginary part - Predicted
        im4 = axes[row, 3].imshow(pred[field_idx * 2 + 1], cmap='RdBu', aspect='auto')
        axes[row, 3].set_title(f'{field_name} Imaginary (Predicted)')
        axes[row, 3].set_xlabel('Width')
        axes[row, 3].set_ylabel('Height')
        plt.colorbar(im4, ax=axes[row, 3])
    
    plt.suptitle(f'E-Field Comparison: Sample {sample_idx+1}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed comparison saved to: {save_path}")
