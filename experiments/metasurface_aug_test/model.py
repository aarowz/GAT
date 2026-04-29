"""
Lightweight reconstruction model for masked Jones matrices.
"""

import torch
from torch import nn


class JonesReconstructionMLP(nn.Module):
    def __init__(self, input_dim: int = 20 * 12, hidden_dim: int = 128, output_dim: int = 20 * 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, masked_input: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        # Concatenate values and mask so the model knows what was provided.
        x = torch.cat([masked_input, visible_mask], dim=-1)  # (B, 20, 12)
        b = x.shape[0]
        x = x.reshape(b, -1)
        y = self.net(x)
        return y.reshape(b, 20, 6)
