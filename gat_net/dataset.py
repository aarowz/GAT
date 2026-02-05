"""
Dataset module for loading and processing metasurface data.

This module handles:
- Loading .mat files containing metasurface geometry and field data
- Preloading files into cache and sampling patch positions (like reference/data_utils.py)
- Converting spatial data to graph representations with shared graph structure
"""

import os
import random
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch_geometric.data import Data

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def _stack_E(Ex, Ey, Ez):
    """Stack E-field into 6 channels (real/imag for Ex, Ey, Ez)."""
    if np.iscomplexobj(Ex):
        E = np.stack((
            np.real(Ex), np.imag(Ex),
            np.real(Ey), np.imag(Ey),
            np.real(Ez), np.imag(Ez)
        ), axis=0).astype(np.float32)
    else:
        E = np.stack((
            Ex.astype(np.float32), np.zeros_like(Ex, dtype=np.float32),
            Ey.astype(np.float32), np.zeros_like(Ey, dtype=np.float32),
            Ez.astype(np.float32), np.zeros_like(Ez, dtype=np.float32)
        ), axis=0)
    return E


def _build_graph_structure(block_size, max_dist=2):
    """Build edge_index and edge_attr once for a block_size x block_size grid (distance <= max_dist)."""
    edges = []
    edge_attrs = []
    for i in range(block_size):
        for j in range(block_size):
            node_id = i * block_size + j
            for di in range(-max_dist, max_dist + 1):
                for dj in range(-max_dist, max_dist + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < block_size and 0 <= nj < block_size:
                        dist = np.sqrt(di**2 + dj**2)
                        if dist <= max_dist:
                            neighbor_id = ni * block_size + nj
                            edges.append([node_id, neighbor_id])
                            edge_attrs.append(dist)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).unsqueeze(-1)
    return edge_index, edge_attr


def _build_boundary_mask(block_size):
    """Boundary indicator: 1.0 on border, 0.0 inside (same for every sample)."""
    B = np.zeros((block_size, block_size), dtype=np.float32)
    B[0, :] = 1.0
    B[-1, :] = 1.0
    B[:, 0] = 1.0
    B[:, -1] = 1.0
    return B


class MetasurfaceDataset(Dataset):
    """
    Dataset for metasurface inverse design.
    Aligned with reference/data_utils.py: preload files, sample positions, shared graph structure.
    
    Each sample contains:
    - Node features: [R, H, D_x, D_y, B, X, Y]
    - Edges: built once (distance ≤ 2), reused for all samples
    - Target: 6-channel field (real/imag Ex, Ey, Ez)
    - Mesh: refinement factor (E-field res / structure res, e.g. 24)
    """

    def __init__(self, data_folder, block_size=15, num_blocks_per_metasurface=100, seed=None):
        """
        Args:
            data_folder: Path to folder containing .mat files
            block_size: Size of extracted blocks (15x15)
            num_blocks_per_metasurface: Number of random patch positions per file
            seed: Optional RNG seed for reproducible sampling
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.data_folder = data_folder
        self.block_size = block_size
        self.num_blocks_per_metasurface = num_blocks_per_metasurface
        self.data_cache = []
        self.sampled_positions = []  # (file_idx, i, j)

        self._preload_data()
        self._build_graph_structure()
        self._build_boundary_and_coords()

    def _preload_data(self):
        """Load each .mat file once into cache (like reference preload_data)."""
        if not os.path.exists(self.data_folder):
            print(f"Directory {self.data_folder} does not exist!")
            return
        mat_files = sorted([
            os.path.join(self.data_folder, f)
            for f in os.listdir(self.data_folder) if f.endswith(".mat")
        ])
        if not mat_files:
            print("No .mat files found.")
            return
        print(f"Preloading {len(mat_files)} .mat files...")
        for file_path in tqdm(mat_files):
            try:
                d = loadmat(file_path)
                if "E" in d:
                    Ex, Ey, Ez = d["E"][0], d["E"][1], d["E"][2]
                else:
                    Ex, Ey, Ez = d["Ex"], d["Ey"], d["Ez"]
                D = np.asarray(d["D"])
                R = np.asarray(d["R"], dtype=np.float32)
                H = np.asarray(d["H"], dtype=np.float32)
                H_total, W_total = D.shape[1], D.shape[2]
                field_scale = Ex.shape[0] // max(H_total, 1)
                E = _stack_E(Ex, Ey, Ez)  # (6, E_h, E_w)
                self.data_cache.append({
                    "E": E,
                    "R": R,
                    "H": H,
                    "D": D,
                    "field_scale": field_scale,
                    "H_total": H_total,
                    "W_total": W_total,
                })
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        if not self.data_cache:
            print("No files loaded.")
            return
        # Sample positions: num_blocks_per_metasurface random (i,j) per file
        for file_idx, data in enumerate(self.data_cache):
            H_total = data["H_total"]
            W_total = data["W_total"]
            valid_h = max(0, H_total - self.block_size + 1)
            valid_w = max(0, W_total - self.block_size + 1)
            if valid_h <= 0 or valid_w <= 0:
                continue
            for _ in range(self.num_blocks_per_metasurface):
                i = random.randint(0, valid_h - 1)
                j = random.randint(0, valid_w - 1)
                self.sampled_positions.append((file_idx, i, j))
        print(f"Total samples: {len(self.sampled_positions)}")

    def _build_graph_structure(self):
        """Build edge_index and edge_attr once (reused for all samples)."""
        self.edge_index, self.edge_attr = _build_graph_structure(self.block_size, max_dist=2)

    def _build_boundary_and_coords(self):
        """Precompute boundary mask and normalized coords for node features."""
        self._B = _build_boundary_mask(self.block_size)
        yy, xx = np.meshgrid(
            np.arange(self.block_size, dtype=np.float32) / self.block_size,
            np.arange(self.block_size, dtype=np.float32) / self.block_size,
            indexing="ij"
        )
        self._X_flat = xx.reshape(-1)
        self._Y_flat = yy.reshape(-1)
        self._B_flat = self._B.reshape(-1)

    def _get_patch(self, file_idx, i, j):
        """Return (R, H, D_x, D_y) patches and (E_patch, field_scale) for one sample."""
        data = self.data_cache[file_idx]
        bs = self.block_size
        scale = data["field_scale"]
        R_patch = data["R"][i : i + bs, j : j + bs]
        H_patch = data["H"][i : i + bs, j : j + bs]
        D = data["D"]
        D_x = D[0, i : i + bs, j : j + bs]
        D_y = D[1, i : i + bs, j : j + bs]
        # E-field: crop at fine resolution then downsample to block_size
        if scale >= 1:
            ei, ej = i * scale, j * scale
            size = bs * scale
            E_crop = data["E"][:, ei : ei + size, ej : ej + size]
            if scale > 1:
                E_patch = E_crop[:, ::scale, ::scale]
            else:
                E_patch = E_crop
        else:
            E_patch = data["E"][:, i : i + bs, j : j + bs]
        return R_patch, H_patch, D_x, D_y, E_patch, scale

    def _node_features_vectorized(self, R_patch, H_patch, D_x, D_y):
        """Build (N, 7) node features without Python loop."""
        R_flat = np.asarray(R_patch, dtype=np.float32).reshape(-1)
        H_flat = np.asarray(H_patch, dtype=np.float32).reshape(-1)
        D_x_flat = np.asarray(D_x, dtype=np.float32).reshape(-1)
        D_y_flat = np.asarray(D_y, dtype=np.float32).reshape(-1)
        x_np = np.stack(
            [R_flat, H_flat, D_x_flat, D_y_flat, self._B_flat, self._X_flat, self._Y_flat],
            axis=1
        )
        return torch.from_numpy(x_np)

    def __len__(self):
        return len(self.sampled_positions)

    def __getitem__(self, idx):
        file_idx, i, j = self.sampled_positions[idx]
        R_patch, H_patch, D_x, D_y, E_patch, field_scale = self._get_patch(file_idx, i, j)
        x = self._node_features_vectorized(R_patch, H_patch, D_x, D_y)
        target = torch.from_numpy(E_patch.copy())
        mesh = torch.tensor([float(field_scale)], dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            mesh=mesh,
        )
        return data, target
