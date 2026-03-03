"""
Main entry point for GAT-Net training.

This script:
1. Loads and processes the metasurface dataset
2. Splits data into training and validation sets
3. Initializes the GAT-Net model
4. Trains the model
5. Saves training results to outputs/checkpoints and outputs/figures
"""

import os
import torch
from torch.utils.data import DataLoader, random_split

from gat_net import MetasurfaceDataset, GATNet, train_gatnet, collate_graphs
import config


def main():
    """Main training pipeline."""
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Training on CPU is very slow (~2-3 sec/batch). On Explorer cluster, request a GPU node:")
        print("  srun --gres=gpu:1 -p gpu python main.py")
        print("  or use an interactive GPU session before running.")
    print("=" * 60)

    # Load dataset
    print(f"Loading dataset from: {config.DATA_FOLDER}")
    dataset = MetasurfaceDataset(
        data_folder=config.DATA_FOLDER,
        block_size=config.BLOCK_SIZE,
        num_blocks_per_metasurface=config.NUM_BLOCKS_PER_METASURFACE,
        mesh_refinement_factor=config.MESH_REFINEMENT_FACTOR,
        seed=config.SEED,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded! Please check your data folder path.")
        return

    print(f"Total samples loaded: {len(dataset)}")
    print("=" * 60)

    # Split dataset (70% train, 15% val, 15% test - matches reference)
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = int(config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED or 42)
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples (held out): {test_size}")
    print("=" * 60)

    # Create dataloaders (pin_memory + persistent_workers for faster GPU training)
    use_pin = getattr(config, 'PIN_MEMORY', True) and device.type == 'cuda'
    use_persistent = getattr(config, 'NUM_WORKERS', 0) > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=use_persistent,
    )

    # Create model
    print("Initializing GAT-Net model...")
    model = GATNet(
        input_dim=config.INPUT_DIM,
        gat_hidden=config.GAT_HIDDEN,
        gat_heads=config.GAT_HEADS,
        gcn_hidden=config.GCN_HIDDEN,
        cnn_hidden=config.CNN_HIDDEN,
        output_channels=config.OUTPUT_CHANNELS,
        grid_size=config.BLOCK_SIZE,
        mesh_refinement_factor=config.MESH_REFINEMENT_FACTOR,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    # Resume from checkpoint if configured and file exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'gat_net_model.pth')
    if getattr(config, 'RESUME_FROM_CHECKPOINT', False) and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Starting from scratch (no checkpoint or RESUME_FROM_CHECKPOINT=False)")

    # Train model
    print("Starting training...")
    trained_model, train_losses, val_losses = train_gatnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        lr_factor=config.LR_SCHEDULER_FACTOR,
        lr_patience=config.LR_SCHEDULER_PATIENCE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        device=device,
        use_amp=getattr(config, 'USE_AMP', True),
        print_freq=config.PRINT_FREQ,
        save_plot=config.SAVE_PLOT,
        save_visualizations=config.SAVE_VISUALIZATIONS,
        figures_dir=config.FIGURES_DIR
    )

    # Save model
    model_path = checkpoint_path
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
