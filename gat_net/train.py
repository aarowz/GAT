"""
Training module for GAT-Net.

Handles model training, validation, and loss tracking.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .visualize import visualize_efield_predictions, visualize_efield_comparison

try:
    import config
except ImportError:
    config = None


def train_gatnet(model, train_loader, val_loader, epochs=100, lr=5e-6, 
                 device='cuda', print_freq=10, save_plot=True, save_visualizations=True,
                 figures_dir='outputs/figures'):
    """
    Train GAT-Net model.
    
    Args:
        model: GATNet model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        print_freq: Frequency of printing training progress
        save_plot: Whether to save loss plot
        save_visualizations: Whether to save E-field prediction visualizations
        figures_dir: Directory to save loss plot and E-field figures (default: outputs/figures)
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for batch_graph, batch_target in train_loader:
            batch_graph = batch_graph.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            prediction = model(batch_graph)
            loss = F.mse_loss(prediction, batch_target)
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
                batch_graph = batch_graph.to(device)
                batch_target = batch_target.to(device)

                prediction = model(batch_graph)
                loss = F.mse_loss(prediction, batch_target)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

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
        plt.title('Training Progress', fontsize=14)
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
