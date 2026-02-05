"""
Dataset module for loading and processing metasurface data.

This module handles:
- Loading .mat files containing metasurface geometry and field data
- Extracting 15x15 blocks from larger metasurfaces
- Converting spatial data to graph representations
"""

import os
import gc
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MetasurfaceDataset(Dataset):
    """
    Dataset for metasurface inverse design.
    
    Extracts 15x15 blocks from larger metasurfaces and converts them
    to graph representations suitable for GAT-Net.
    
    Each sample contains:
    - Node features: [R, H, D_x, D_y, B, X, Y] where:
        R: Material property
        H: Height
        D_x, D_y: Displacement components
        B: Boundary indicator (1.0 for edges, 0.0 otherwise)
        X, Y: Normalized spatial coordinates
    - Edges: Connections between nodes within distance ≤ 2
    - Target: 6-channel field data (real/imaginary parts of Ex, Ey, Ez)
    - Mesh: refinement factor (scalar) = E-field resolution / structure resolution (e.g. 2883/120 = 24).
      Aligns with actual physics: structure 120×120, E-field 2883×2883, so 24× per dimension.
    """

    def __init__(self, data_folder, block_size=15, num_blocks_per_metasurface=100):
        """
        Initialize the dataset.
        
        Args:
            data_folder: Path to folder containing .mat files
            block_size: Size of extracted blocks (default: 15x15)
            num_blocks_per_metasurface: Number of random blocks to extract per file
        """
        self.data_folder = data_folder
        self.block_size = block_size
        self.num_blocks_per_metasurface = num_blocks_per_metasurface
        self.samples = []
        self._load_and_process_data()

    def _load_and_process_data(self):
        """Load and process all .mat files in the data folder."""
        if not os.path.exists(self.data_folder):
            print(f"Directory {self.data_folder} does not exist!")
            return

        mat_files = [os.path.join(self.data_folder, f)
                    for f in os.listdir(self.data_folder) if f.endswith(".mat")]

        print(f"Found {len(mat_files)} .mat files")

        for file_idx, file_path in enumerate(mat_files):
            try:
                mat_data = scipy.io.loadmat(file_path)

                # Handle different E field formats
                if 'E' in mat_data:
                    Ex = mat_data['E'][0]
                    Ey = mat_data['E'][1]
                    Ez = mat_data['E'][2]
                else:
                    Ex = mat_data['Ex']
                    Ey = mat_data['Ey']
                    Ez = mat_data['Ez']

                D = mat_data["D"]
                R = mat_data["R"]
                H = mat_data["H"]

                H_total, W_total = D.shape[1], D.shape[2]

                # Extract random blocks
                for _ in range(self.num_blocks_per_metasurface):
                    i = np.random.randint(0, H_total - self.block_size + 1)
                    j = np.random.randint(0, W_total - self.block_size + 1)

                    D_x = D[0, i:i+self.block_size, j:j+self.block_size]
                    D_y = D[1, i:i+self.block_size, j:j+self.block_size]
                    R_patch = R[i:i+self.block_size, j:j+self.block_size]
                    H_patch = H[i:i+self.block_size, j:j+self.block_size]

                    # Extract corresponding fields
                    field_scale = Ex.shape[0] // H_total
                    field_i = i * field_scale
                    field_j = j * field_scale
                    field_size = self.block_size * field_scale

                    Ex_patch = Ex[field_i:field_i+field_size, field_j:field_j+field_size]
                    Ey_patch = Ey[field_i:field_i+field_size, field_j:field_j+field_size]
                    Ez_patch = Ez[field_i:field_i+field_size, field_j:field_j+field_size]

                    if field_scale > 1:
                        Ex_patch = Ex_patch[::field_scale, ::field_scale]
                        Ey_patch = Ey_patch[::field_scale, ::field_scale]
                        Ez_patch = Ez_patch[::field_scale, ::field_scale]

                    # Mesh = refinement factor (structure → E-field). e.g. 120×120 → 2883×2883 gives 24
                    self.samples.append({
                        'D_x': D_x, 'D_y': D_y, 'R': R_patch, 'H': H_patch,
                        'Ex': Ex_patch, 'Ey': Ey_patch, 'Ez': Ez_patch,
                        'mesh': field_scale
                    })

                print(f"File {file_idx+1}/{len(mat_files)}: Extracted {self.num_blocks_per_metasurface} blocks")
                del mat_data
                gc.collect()

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        print(f"Total samples: {len(self.samples)}")

    def create_graph(self, sample):
        """
        Convert a sample to a graph representation.
        
        Args:
            sample: Dictionary containing R, H, D_x, D_y, Ex, Ey, Ez
            
        Returns:
            data: torch_geometric Data object with node features and edges
            target: Tensor of shape (6, H, W) containing field predictions
        """
        R = sample['R']
        H = sample['H']
        D_x = sample['D_x']
        D_y = sample['D_y']

        height, width = R.shape
        node_features = []

        # Create node features [R, H, D_x, D_y, B, X, Y]
        for i in range(height):
            for j in range(width):
                B = 1.0 if (i == 0 or i == height-1 or j == 0 or j == width-1) else 0.0
                features = [
                    float(R[i, j]), float(H[i, j]),
                    float(D_x[i, j]), float(D_y[i, j]),
                    B, float(i)/height, float(j)/width
                ]
                node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Create edges with distance ≤ 2
        edges = []
        edge_attrs = []

        for i in range(height):
            for j in range(width):
                node_id = i * width + j
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            distance = np.sqrt(di**2 + dj**2)
                            if distance <= 2:
                                neighbor_id = ni * width + nj
                                edges.append([node_id, neighbor_id])
                                edge_attrs.append(distance)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).unsqueeze(-1)

        # Mesh: refinement factor (scalar) from actual data: E-field res / structure res (e.g. 24)
        mesh = torch.tensor([float(sample['mesh'])], dtype=torch.float32)

        # Create target (6 channels: real/imag for Ex, Ey, Ez)
        if np.iscomplexobj(sample['Ex']):
            target = torch.stack([
                torch.tensor(np.real(sample['Ex']), dtype=torch.float32),
                torch.tensor(np.imag(sample['Ex']), dtype=torch.float32),
                torch.tensor(np.real(sample['Ey']), dtype=torch.float32),
                torch.tensor(np.imag(sample['Ey']), dtype=torch.float32),
                torch.tensor(np.real(sample['Ez']), dtype=torch.float32),
                torch.tensor(np.imag(sample['Ez']), dtype=torch.float32),
            ])
        else:
            target = torch.stack([
                torch.tensor(sample['Ex'], dtype=torch.float32),
                torch.zeros_like(torch.tensor(sample['Ex'], dtype=torch.float32)),
                torch.tensor(sample['Ey'], dtype=torch.float32),
                torch.zeros_like(torch.tensor(sample['Ey'], dtype=torch.float32)),
                torch.tensor(sample['Ez'], dtype=torch.float32),
                torch.zeros_like(torch.tensor(sample['Ez'], dtype=torch.float32)),
            ])

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, mesh=mesh)
        return data, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.create_graph(self.samples[idx])
