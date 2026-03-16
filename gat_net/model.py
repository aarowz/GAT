"""
GAT-Net model architecture.

The model consists of three main components:
1. Graph Attention Network (GAT) layer for learning node representations
2. Graph Convolutional Network (GCN) layers for feature transformation
3. Convolutional Neural Network (CNN) layers for spatial prediction
4. PixelShuffle upsampling (learnable) to target E-field resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv


def _factorize_scale(n: int):
    """Factorize upsampling factor into product of 2s and 3s (for PixelShuffle)."""
    factors = []
    for p in [3, 2]:
        while n % p == 0:
            factors.append(p)
            n //= p
    return factors, n


class UpBlock(nn.Module):
    """PixelShuffle upsampling block (learnable, from reference/base_model.py)."""
    def __init__(self, ch: int, out_ch: int, r: int, p_drop: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(ch, out_ch * (r ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(r)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p_drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class LastUpBlock(nn.Module):
    """Final upsampling block - no activation so output can span full E-field range."""
    def __init__(self, ch: int, out_ch: int, r: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, out_ch * (r ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(r)
        self.batch_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.batch_norm(x)
        return x


class GATNet(nn.Module):
    """
    GAT-Net: Graph Attention Network for Metasurface Inverse Design.
    
    Architecture:
    - GAT Layer: Processes graph structure with attention mechanism
    - GCN Layers: Fully connected layers for feature transformation
    - CNN Layers: Convolutional layers for spatial field prediction
    
    Input: Graph with 5-dimensional node features
    Output: 6-channel field prediction (real/imaginary parts of Ex, Ey, Ez)
    """

    def __init__(self, input_dim=5, gat_hidden=200, gat_heads=8, 
                 gcn_hidden=1600, cnn_hidden=64, output_channels=6,
                 grid_size=15, mesh_refinement_factor=24):
        """
        Initialize GAT-Net model.
        
        Args:
            input_dim: Number of input node features (default: 5)
            gat_hidden: Hidden dimension for GAT layer (default: 200)
            gat_heads: Number of attention heads (default: 8)
            gcn_hidden: Hidden dimension for GCN layers (default: 1600)
            cnn_hidden: Hidden dimension for CNN layers (default: 64)
            output_channels: Number of output channels (default: 6)
            grid_size: Spatial grid size from GNN (default: 15)
            mesh_refinement_factor: Upsample factor so output matches target grid (default: 24)
        """
        super().__init__()
        self.grid_size = grid_size
        self.output_grid_size = grid_size * mesh_refinement_factor

        # GAT Layer: Graph Attention with edge attributes
        self.gat1 = GATv2Conv(
            in_channels=input_dim,
            out_channels=gat_hidden,
            heads=gat_heads,
            concat=True,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=1
        )

        # GCN Layers: Fully connected layers
        self.gcn1 = nn.Linear(gat_hidden * gat_heads, gcn_hidden)
        self.gcn2 = nn.Linear(gcn_hidden, gcn_hidden)

        # Projection layer: Maps GCN output to CNN input dimension
        self.projection = nn.Linear(gcn_hidden, cnn_hidden)

        # CNN Layers: Spatial convolution for field prediction (output at coarse grid_size)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm2d(cnn_hidden),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm2d(cnn_hidden),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, output_channels, kernel_size=5, padding=2)
        )

        # PixelShuffle upsampling: learnable progressive upsampling (replaces bilinear)
        factors, remainder = _factorize_scale(mesh_refinement_factor)
        if remainder != 1:
            raise ValueError(
                f"mesh_refinement_factor={mesh_refinement_factor} must factor into 2s and 3s, "
                f"got remainder={remainder}"
            )
        self.upsample_blocks = nn.ModuleList()
        ch = output_channels
        for i, r in enumerate(factors):
            if i == len(factors) - 1:
                self.upsample_blocks.append(LastUpBlock(ch, ch, r))
            else:
                self.upsample_blocks.append(UpBlock(ch, ch, r, p_drop=0.1))

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(gat_hidden * gat_heads)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)

    def forward(self, batch_data):
        """
        Forward pass through the network.
        
        Args:
            batch_data: Batch of graph data from torch_geometric
            
        Returns:
            output: Tensor of shape (batch_size, 6, H, W) with field predictions
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        # GAT layer: Learn node representations with attention
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        # GCN layers: Transform features
        x = self.gcn1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.gcn2(x)
        x = F.relu(x)

        # Reshape for CNN: Convert graph nodes to spatial grid
        batch_size = batch.max().item() + 1 if batch is not None else 1
        nodes_per_graph = x.shape[0] // batch_size if batch is not None else x.shape[0]
        grid_size = int(np.sqrt(nodes_per_graph))

        x_spatial = []
        for b in range(batch_size):
            if batch is not None:
                mask = (batch == b)
                nodes = x[mask]
            else:
                nodes = x

            # Project to CNN input dimension
            nodes_projected = self.projection(nodes)  # (225, cnn_hidden)

            # Reshape to spatial grid
            nodes_reshaped = nodes_projected.view(grid_size, grid_size, -1)
            nodes_reshaped = nodes_reshaped.permute(2, 0, 1)  # (cnn_hidden, 15, 15)

            x_spatial.append(nodes_reshaped)

        x_spatial = torch.stack(x_spatial)  # (batch, cnn_hidden, grid_size, grid_size)
        
        # CNN layers: Predict field distribution at coarse grid
        output = self.cnn_layers(x_spatial)

        # PixelShuffle upsampling: learnable progressive upsampling to target resolution
        for up_block in self.upsample_blocks:
            output = up_block(output)

        return output
