"""
Utility functions for GAT-Net.

Contains helper functions for data processing and model utilities.
"""

import torch
from torch_geometric.data import Batch


def collate_graphs(batch):
    """
    Custom collate function for batching graph data.
    
    Args:
        batch: List of (graph, target) tuples
        
    Returns:
        batched_graph: Batched graph data
        batched_targets: Stacked target tensors
    """
    graphs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    batched_targets = torch.stack(targets, dim=0)
    return batched_graph, batched_targets
