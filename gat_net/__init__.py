"""
GAT-Net: Inverse Design of Multifunctional Metasurface Based on Graph Attention Network

A PyTorch implementation for predicting electromagnetic field distributions
from metasurface geometry using Graph Attention Networks.
"""

__version__ = "1.0.0"

from .dataset import MetasurfaceDataset
from .model import GATNet
from .train import train_gatnet
from .utils import collate_graphs
from .visualize import visualize_efield_predictions, visualize_efield_comparison

__all__ = [
    'MetasurfaceDataset',
    'GATNet',
    'train_gatnet',
    'collate_graphs',
    'visualize_efield_predictions',
    'visualize_efield_comparison',
]
