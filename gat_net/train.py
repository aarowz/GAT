"""
Training module for GAT-Net.

Handles model training, validation, and loss tracking.
"""

import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from .visualize import visualize_efield_predictions, visualize_efield_comparison

try:
    import config
except ImportError:
    config = None


def _channel_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_indices=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE weighted by per-channel inverse std over the selected channels (balanced within that set)."""
    c = prediction.shape[1]
    if channel_indices is None:
        channel_indices = tuple(range(c))
    idx = list(channel_indices)
    pred_s = prediction[:, idx, :, :]
    targ_s = target[:, idx, :, :]
    per_ch_mse = ((pred_s - targ_s) ** 2).mean(dim=(0, 2, 3))
    weights = 1.0 / targ_s.std(dim=(0, 2, 3)).clamp(min=eps)
    weights = weights / weights.mean()
    return (per_ch_mse * weights).mean()


def _training_loss_channel_indices():
    """Channel indices for training/val loss; None => all output channels."""
    if config is None:
        return None
    return getattr(config, "LOSS_MSE_CHANNEL_INDICES", None)


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def train_gatnet(model, train_loader, val_loader, epochs=200, lr=5e-4, weight_decay=1e-5,
                 lr_factor=0.8, lr_patience=5, early_stopping_patience=15,
                 device='cuda', use_amp=True, print_freq=10, save_plot=True, save_visualizations=True,
                 figures_dir='outputs/figures'):
    """
    Train GAT-Net model.

    Training params aligned with reference train_base_model.py.
    
    Args:
        model: GATNet model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        lr_factor: ReduceLROnPlateau factor
        lr_patience: ReduceLROnPlateau patience
        early_stopping_patience: Early stopping patience
        device: Device to train on ('cuda' or 'cpu')
        use_amp: Use automatic mixed precision (fp16) when device is CUDA
        print_freq: Frequency of printing training progress
        save_plot: Whether to save loss plot
        save_visualizations: Whether to save E-field prediction visualizations
        figures_dir: Directory to save loss plot and E-field figures
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    train_losses = []
    val_losses = []

    loss_ch = _training_loss_channel_indices()
    print(f"Starting training on {device}" + (" (AMP enabled)" if use_amp else ""))
    if loss_ch is None:
        print("Loss: weighted MSE on all E-field output channels")
    else:
        print(f"Loss: weighted MSE on channels {loss_ch} only (viz still shows Ex, Ey, Ez)")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for batch_graph, batch_target in train_loader:
            batch_graph = batch_graph.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    prediction = model(batch_graph)
                    loss = _channel_weighted_mse(prediction, batch_target, loss_ch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                prediction = model(batch_graph)
                loss = _channel_weighted_mse(prediction, batch_target, loss_ch)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_graph, batch_target in val_loader:
                batch_graph = batch_graph.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        prediction = model(batch_graph)
                else:
                    prediction = model(batch_graph)
                loss = _channel_weighted_mse(prediction, batch_target, loss_ch)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        if early_stopping(avg_val_loss):
            print("\nEarly stopping triggered")
            break

        # Print progress
        if epoch % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.3e}")

    print("-" * 60)
    print(f"Training completed!")
    print(f"Final Train MSE: {train_losses[-1]:.6f}")
    print(f"Final Val MSE: {val_losses[-1]:.6f}")

    # Plot loss curves
    if save_plot:
        os.makedirs(figures_dir, exist_ok=True)
        loss_path = os.path.join(figures_dir, 'training_losses.png')
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        title = 'Training Progress'
        if loss_ch is not None:
            title += f' (channels {loss_ch})'
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        print(f"\nLoss plot saved to '{loss_path}'")
        plt.close()

    # Generate E-field prediction visualizations
    if save_visualizations:
        print("\nGenerating E-field prediction visualizations...")
        model.eval()

        # Collect validation predictions, targets, and mesh refinement
        val_predictions = []
        val_targets = []
        val_mesh = []
        
        with torch.no_grad():
            for batch_graph, batch_target in val_loader:
                batch_graph = batch_graph.to(device)
                batch_target = batch_target.to(device)
                prediction = model(batch_graph)
                val_predictions.append(prediction.cpu())
                val_targets.append(batch_target.cpu())
                # mesh: (batch_size, 1) -> collect for upsampling in viz
                if hasattr(batch_graph, 'mesh') and batch_graph.mesh is not None:
                    val_mesh.append(batch_graph.mesh.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = torch.cat(val_predictions, dim=0)
        all_targets = torch.cat(val_targets, dim=0)
        mesh_refinement = np.concatenate(val_mesh, axis=0).flatten() if val_mesh else None  # (N,) or None
        # Fallback: if batch didn't provide mesh (e.g. PyG batching), use config so mesh still plays in viz
        N = all_predictions.shape[0]
        if mesh_refinement is None and config is not None and hasattr(config, 'MESH_REFINEMENT_FACTOR'):
            mesh_refinement = np.full(N, getattr(config, 'MESH_REFINEMENT_FACTOR', 24), dtype=np.float32)
        
        # Generate visualizations (upsample by mesh refinement for finer display)
        os.makedirs(figures_dir, exist_ok=True)
        pred_path = os.path.join(figures_dir, 'efield_predictions.png')
        compare_path = os.path.join(figures_dir, 'efield_comparison.png')
        visualize_efield_predictions(
            all_predictions,
            all_targets,
            num_samples=1,
            save_path=pred_path,
            mesh_refinement=mesh_refinement,
        )
        visualize_efield_comparison(
            all_predictions,
            all_targets,
            sample_idx=0,
            save_path=compare_path,
            mesh_refinement=mesh_refinement,
        )

    return model, train_losses, val_losses
