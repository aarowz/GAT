# GAT-Net: Inverse Design of Multifunctional Metasurface

A PyTorch implementation for predicting electromagnetic field distributions from metasurface geometry using Graph Attention Networks (GAT).

## Overview

This project implements GAT-Net, a deep learning model that performs inverse design of metasurfaces. Given the geometry of a metasurface (material properties, height, displacement), the model predicts the resulting electromagnetic field distributions (Ex, Ey, Ez components).

## Project Structure

```
GAT/
├── gat_net/              # Main package directory
│   ├── __init__.py       # Package initialization and exports
│   ├── dataset.py        # Dataset loading and graph creation
│   ├── model.py          # GAT-Net model architecture
│   ├── train.py          # Training functions
│   ├── visualize.py      # E-field prediction visualizations
│   └── utils.py          # Utility functions (collate, etc.)
├── outputs/              # Training outputs (created at run time)
│   ├── checkpoints/      # Saved model weights (.pth)
│   └── figures/         # Loss curves and E-field visualizations (.png)
├── config.py             # Configuration parameters
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Architecture

The GAT-Net model consists of three main components:

1. **Graph Attention Network (GAT) Layer**: Processes the graph structure of the metasurface, learning node representations using attention mechanisms
2. **Graph Convolutional Network (GCN) Layers**: Fully connected layers that transform the learned features
3. **Convolutional Neural Network (CNN) Layers**: Spatial convolutions that predict the field distribution

### Data Flow

1. **Input**: 15×15 metasurface blocks with geometry data (R, H, D_x, D_y)
2. **Graph Creation**: Each pixel becomes a node with 7 features: [R, H, D_x, D_y, B, X, Y]
   - R: Material property
   - H: Height
   - D_x, D_y: Displacement components
   - B: Boundary indicator (1.0 for edges, 0.0 otherwise)
   - X, Y: Normalized spatial coordinates
3. **Graph Processing**: Nodes are connected if within distance ≤ 2
4. **Output**: 6-channel field prediction (real/imaginary parts of Ex, Ey, Ez)

## Installation

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `config.py` to adjust:

- Data folder path
- Model architecture (hidden dimensions, number of heads, etc.)
- Training hyperparameters (learning rate, batch size, epochs)
- Device settings (CPU/GPU)

### Running Training

```bash
python main.py
```

The script will:

1. Load and process .mat files from the specified data folder
2. Extract 15×15 blocks from each metasurface
3. Split data into training (80%) and validation (20%) sets
4. Train the GAT-Net model
5. Save the trained model and training loss plot

### Using Individual Components

You can also import and use components individually:

```python
from gat_net import MetasurfaceDataset, GATNet, train_gatnet

# Load dataset
dataset = MetasurfaceDataset(data_folder="path/to/data")

# Create model
model = GATNet()

# Train (with custom dataloaders)
train_gatnet(model, train_loader, val_loader, epochs=100)
```

## Module Documentation

### `gat_net/dataset.py`

**MetasurfaceDataset**: Handles loading .mat files and converting spatial data to graphs.

- `__init__()`: Initialize dataset with data folder path
- `create_graph()`: Convert a sample to graph representation
- `__getitem__()`: Get a graph sample by index

### `gat_net/model.py`

**GATNet**: The main model architecture.

- `__init__()`: Initialize model with configurable dimensions
- `forward()`: Forward pass through GAT → GCN → CNN layers

### `gat_net/train.py`

**train_gatnet()**: Training function with validation and loss tracking.

- Handles training loop, validation, and loss plotting
- Generates E-field prediction visualizations
- Returns trained model and loss history

### `gat_net/visualize.py`

**visualize_efield_predictions()**: Creates visualizations comparing predicted vs ground truth E-fields.

- Shows real and imaginary parts for Ex, Ey, Ez components
- Generates side-by-side comparison plots
- Saves visualization files automatically

**visualize_efield_comparison()**: Creates detailed comparison for a single sample.

- Shows predicted, ground truth, and error maps
- Useful for detailed analysis of model performance

### `gat_net/utils.py`

**collate_graphs()**: Custom collate function for batching graph data in DataLoader.

## Data Format

The code expects .mat files containing:

- `D`: Displacement data (shape: [2, H, W])
- `R`: Material property (shape: [H, W])
- `H`: Height data (shape: [H, W])
- `Ex`, `Ey`, `Ez`: Electric field components (shape: [H_field, W_field])

The field data can be either:

- Complex-valued arrays (will be split into real/imaginary parts)
- Real-valued arrays (imaginary part set to zero)

## Model Parameters

Default configuration:

- **GAT Layer**: 200 hidden units, 8 attention heads
- **GCN Layers**: 1600 hidden units (2 layers)
- **CNN Layers**: 64 → 128 → 64 → 6 channels
- **Total Parameters**: ~2-3 million (depending on configuration)

## Training Tips

1. **Learning Rate**: Start with 5e-6, adjust based on convergence
2. **Batch Size**: Adjust based on GPU memory (default: 32)
3. **Data Augmentation**: The code extracts random blocks, providing natural augmentation
4. **Monitoring**: Check `outputs/figures/training_losses.png` for convergence patterns

## Output

After training, all outputs are written under `outputs/`:

- **outputs/checkpoints/gat_net_model.pth** – Trained model weights
- **outputs/figures/training_losses.png** – Plot of training and validation losses
- **outputs/figures/efield_predictions_real.png** – Predicted vs ground truth E-field real parts
- **outputs/figures/efield_predictions_imaginary.png** – Predicted vs ground truth E-field imaginary parts
- **outputs/figures/efield_comparison.png** – Detailed comparison for a sample prediction

## Citation

If you use this code, please cite the original GAT-Net paper:

```
GAT-Net: Inverse Design of Multifunctional Metasurface Based on Graph Attention Network
```

## License

This implementation is provided for educational and research purposes.
