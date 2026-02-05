import os
import torch
import numpy as np
import h5py
from scipy.io import loadmat
from glob import glob
from typing import Tuple, List, Optional
from torch_geometric.data import Data
from tqdm import tqdm
import random

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_unit_cell_field(file_path: str) -> Optional[np.ndarray]:
    try:
        with h5py.File(file_path, "r") as f:
            if "Ex" in f:
                Ex = f["Ex"][:]
            elif "Ex_data" in f:
                g = f["Ex_data"]
                if isinstance(g, h5py.Dataset):
                    Ex = g[:]
                elif "field" in g:
                    Ex = g["field"][:]
            else:
                raise KeyError("Cannot find Ex field data")
            
            if Ex.dtype.names and {"real", "imag"} <= set(Ex.dtype.names):
                Ex = Ex["real"] + 1j * Ex["imag"]
            elif Ex.dtype.names and {"r", "i"} <= set(Ex.dtype.names):
                Ex = Ex["r"] + 1j * Ex["i"]
                
            Ex = np.array(Ex)
            if Ex.ndim == 4:
                Ex = Ex.squeeze()
            
            field_real = np.real(Ex).astype(np.float32)
            field_imag = np.imag(Ex).astype(np.float32)
            return np.stack([field_real, field_imag], axis=0)
            
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def compute_dataset_stats(file_list: List[str], 
                         unit_cell_file: Optional[str] = None,
                         window_size: int = 7,
                         pd_size: int = 6,
                         R: int = 3,
                         E_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_base_channels = 4
    n_unit_cell_channels = 2 if unit_cell_file else 0
    n_in_channels = n_base_channels + n_unit_cell_channels
    
    sum_in = np.zeros(n_in_channels, dtype=np.float64)
    sum2_in = np.zeros(n_in_channels, dtype=np.float64)
    cnt_in = np.zeros(n_in_channels, dtype=np.float64)

    sum_out = np.zeros(6, dtype=np.float64)
    sum2_out = np.zeros(6, dtype=np.float64)
    cnt_out = np.zeros(6, dtype=np.float64)

    for fp in file_list:
        d = loadmat(fp)
        R_data = d['R'].astype(np.float64)
        H = d['H'].astype(np.float64)
        D = d['D'].astype(np.float64)
        Ex = d['Ex']; Ey = d['Ey']; Ez = d['Ez']

        # 基础特征统计
        xs = [R_data, H, D[0], D[1]]
        for i, arr in enumerate(xs):
            sum_in[i] += arr.sum()
            sum2_in[i] += (arr ** 2).sum()
            cnt_in[i] += arr.size
        if unit_cell_file:
            field_data = load_unit_cell_field(unit_cell_file)
            if field_data is not None:
                for i in range(2):
                    sum_in[n_base_channels + i] += field_data[i].sum()
                    sum2_in[n_base_channels + i] += (field_data[i] ** 2).sum()
                    cnt_in[n_base_channels + i] += field_data[i].size
        E6 = E_scale * np.stack((
            np.real(Ex), np.imag(Ex),
            np.real(Ey), np.imag(Ey),
            np.real(Ez), np.imag(Ez)
        ), axis=0).astype(np.float64)
        
        for c in range(6):
            sum_out[c] += E6[c].sum()
            sum2_out[c] += (E6[c] ** 2).sum()
            cnt_out[c] += E6[c].size
    mean_in = sum_in / np.maximum(cnt_in, 1.0)
    var_in = sum2_in / np.maximum(cnt_in, 1.0) - mean_in ** 2
    std_in = np.sqrt(np.maximum(var_in, 1e-6))

    mean_out = sum_out / np.maximum(cnt_out, 1.0)
    var_out = sum2_out / np.maximum(cnt_out, 1.0) - mean_out ** 2
    std_out = np.sqrt(np.maximum(var_out, 1e-6))
    mean_in = np.concatenate([mean_in, np.array([0.0])])
    std_in = np.concatenate([std_in, np.array([1.0])])
    
    return mean_in, std_in, mean_out, std_out

class FieldDataset(torch.utils.data.Dataset):
    """Build a dataset for field prediction task"""
    def __init__(self, file_list, split='train', train_ratio=0.7,
                 window_size=7, pd_size=6, Ng=24, R=3,
                 in_mean=None, in_std=None, out_mean=None, out_std=None,
                 E_scale=1.0, augment=True, indices=None, num_samples=20000,
                 normalize_output=True):
        super().__init__()
        self.file_list = file_list
        self.split = split
        self.window_size = window_size
        self.Ng = Ng
        self.R = R
        # pd_size is the padding size outside the main window
        self.pd_size = pd_size
        self.E_scale = E_scale
        self.augment = augment and (split == 'train')
        self.num_samples = num_samples  # fixed number of samples
        self.normalize_output = normalize_output  # normalize output

        # file indices: if indices are provided, use them strictly
        n_files = len(file_list)
        if indices is not None:
            self.file_indices = list(indices)
        else:
            n_train = int(n_files * train_ratio)
            if split == 'train':
                self.file_indices = list(range(n_train))
            elif split == 'val':
                self.file_indices = list(range(n_train, int(n_files * (train_ratio + (1 - train_ratio) / 2))))
            else:  # test
                self.file_indices = list(range(int(n_files * (train_ratio + (1 - train_ratio) / 2)), n_files))

        # normalize parameters (convert to 1D float32 Tensor on CPU to avoid device inconsistency)
        self.in_mean = torch.as_tensor(in_mean, dtype=torch.float32).view(-1).cpu() if in_mean is not None else None
        self.in_std = torch.as_tensor(in_std, dtype=torch.float32).view(-1).cpu() if in_std is not None else None
        self.out_mean = torch.as_tensor(out_mean, dtype=torch.float32).view(-1).cpu() if out_mean is not None else None
        self.out_std = torch.as_tensor(out_std, dtype=torch.float32).view(-1).cpu() if out_std is not None else None

        # preload data
        self.preload_data()

        # graph structure
        self.graph_shape = [self.window_size + 2 * self.pd_size, self.window_size + 2 * self.pd_size]
        self.build_graph_structure()

        # valid positions
        self.compute_valid_positions()

    def preload_data(self):
        self.data_cache = []
        print(f"Preloading {len(self.file_indices)} files for {self.split} set...")
        for idx in tqdm(self.file_indices):
            d = loadmat(self.file_list[idx])
            Ex = d['Ex']; Ey = d['Ey']; Ez = d['Ez']
            E = self.E_scale * np.stack((
                np.real(Ex), np.imag(Ex),
                np.real(Ey), np.imag(Ey),
                np.real(Ez), np.imag(Ez)
            ), axis=0).astype(np.float32)
            R = d['R'].astype(np.float32)
            H = d['H'].astype(np.float32)
            D = d['D'].astype(np.float32)
            self.data_cache.append({'E': E, 'R': R, 'H': H, 'D': D})
        self.str_shape = self.data_cache[0]['R'].shape

    def build_graph_structure(self):
        # boundary features
        zpd_size = self.R
        bnd_ = 2 * torch.ones(self.graph_shape)
        bnd_[:zpd_size, :] = 0
        bnd_[-zpd_size:, :] = 0
        bnd_[:, :zpd_size] = 0
        bnd_[:, -zpd_size:] = 0
        bnd_[zpd_size:self.pd_size, zpd_size:-zpd_size] = 1
        bnd_[-self.pd_size:-zpd_size, zpd_size:-zpd_size] = 1
        bnd_[zpd_size:-zpd_size, zpd_size:self.pd_size] = 1
        bnd_[zpd_size:-zpd_size, -self.pd_size:-zpd_size] = 1
        self.bnd_np = bnd_.numpy()

        # build edges (circular neighborhood)
        node_lst = []
        for i in range(self.graph_shape[0]):
            for j in range(self.graph_shape[1]):
                for k in range(-self.R, self.R + 1):
                    for l in range(-self.R, self.R + 1):
                        if k ** 2 + l ** 2 <= self.R ** 2:
                            if (0 <= i + k < self.graph_shape[0]) and (0 <= j + l < self.graph_shape[1]):
                                src = (i + k) * self.graph_shape[1] + (j + l)
                                dst = i * self.graph_shape[1] + j
                                node_lst.append([src, dst])
        nodes_arr = np.array(node_lst).T
        self.edge_index = torch.from_numpy(nodes_arr.astype(np.int64))

        # edge weights
        pos_y, pos_x = np.meshgrid(np.arange(self.graph_shape[1]),
                                   np.arange(self.graph_shape[0]))
        pos_single = torch.FloatTensor(
            np.stack((pos_x, pos_y), axis=0).transpose(1, 2, 0).reshape(-1, 2)
        )
        diff = pos_single[self.edge_index[1]] - pos_single[self.edge_index[0]]
        dist = torch.norm(diff, dim=1).clamp_min(0.5)
        self.edge_weight = (1.0 / dist).to(torch.float32)

        # main window mask
        mask_np = np.ones((self.graph_shape[0], self.graph_shape[1]), dtype=bool)
        mask_np[:self.pd_size, :] = False
        mask_np[-self.pd_size:, :] = False
        mask_np[:, :self.pd_size] = False
        mask_np[:, -self.pd_size:] = False
        self.mask_flat = mask_np.reshape(-1)

    def compute_valid_positions(self):
        """randomly generate fixed number of sample positions"""
        self.sampled_positions = []  # store (file_idx, r, c) tuples
        
        # compute valid sampling range for each file
        file_ranges = []
        for i, data in enumerate(self.data_cache):
            h, w = data['R'].shape
            valid_h = h - self.graph_shape[0] + 1
            valid_w = w - self.graph_shape[1] + 1
            if valid_h > 0 and valid_w > 0:
                file_ranges.append((i, valid_h, valid_w))
        
        if len(file_ranges) == 0:
            raise ValueError("No valid files to sample!")
        
        # randomly generate num_samples samples
        print(f"Randomly generating {self.num_samples} samples for {self.split} set...")
        for _ in range(self.num_samples):
            # randomly select a file
            file_idx, valid_h, valid_w = random.choice(file_ranges)
            # randomly select position
            r = random.randint(0, valid_h - 1)
            c = random.randint(0, valid_w - 1)
            self.sampled_positions.append((file_idx, r, c))
        
        self.total_samples = len(self.sampled_positions)
        print(f"{self.split} set generated, total {self.total_samples} samples")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # directly get sampled position
        file_idx, r0, c0 = self.sampled_positions[idx]
        data = self.data_cache[file_idx]

        # training augmentation: random small offset
        if self.augment:
            h, w = data['R'].shape
            valid_h = h - self.graph_shape[0]
            valid_w = w - self.graph_shape[1]
            r0 += np.random.randint(-2, 3)
            c0 += np.random.randint(-2, 3)
            r0 = max(0, min(r0, valid_h))
            c0 = max(0, min(c0, valid_w))

        r1 = r0 + self.graph_shape[0]
        c1 = c0 + self.graph_shape[1]

        # input features 5 channels: R, H, D[0], D[1], bnd
        rad_ = data['R'][r0:r1, c0:c1]
        hts_ = data['H'][r0:r1, c0:c1]
        dsp_ = data['D'][:, r0:r1, c0:c1]

        x_np = np.concatenate((
            np.expand_dims(rad_, 0),
            np.expand_dims(hts_, 0),
            dsp_,
            np.expand_dims(self.bnd_np, 0)
        ), axis=0).transpose(1, 2, 0).reshape(-1, 5).astype(np.float32)

        x = torch.from_numpy(x_np)

        # ensure x and mean/std are normalized on CPU to avoid device inconsistency
        if self.in_mean is not None:
            x = x.cpu()
            x = (x - self.in_mean) / (self.in_std + 1e-6)

        # target 6 channels, crop main window region and enlarge Ng
        rr0 = self.Ng * (r0 + self.pd_size)
        rr1 = self.Ng * (r0 + self.pd_size + self.window_size)
        cc0 = self.Ng * (c0 + self.pd_size)
        cc1 = self.Ng * (c0 + self.pd_size + self.window_size)

        y = torch.tensor(data['E'][:, rr0:rr1, cc0:cc1], dtype=torch.float32)
        # according to normalize_output flag decide whether to normalize
        if self.normalize_output and self.out_mean is not None:
            mean = self.out_mean.to(y.device).view(6, 1, 1)
            std = self.out_std.to(y.device).view(6, 1, 1)
            y = (y - mean) / (std + 1e-6)

        graph_data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_weight.unsqueeze(1),
            y=y,
            mask=torch.from_numpy(self.mask_flat)
        )
        return graph_data

# unit cell
def load_unit_cell_field_data(file_path: str, subtract_ref: bool = False) -> np.ndarray:
    """read field from Ex/Ey/Ez .mat file, return squeezed 2D complex matrix.
    if subtract_ref=True, subtract substrate reference field from T_reff_Ex/Ey/Ez.mat (if exists and size matches)
    """
    with h5py.File(file_path, "r") as f:
        # find top level key (maybe Ex_data / Ey_data / Ez_data)
        top_keys = list(f.keys())
        g = None
        for k in top_keys:
            if k.endswith("_data") or k in ["Ex_data", "Ey_data", "Ez_data"]:
                g = f[k]
                break
        if g is None:
            raise KeyError(f"No *_data group found in {file_path}, keys={top_keys}")

        arr = g["field"][()]

        # if (1,1) object array → dereference
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            ref = arr[0, 0]
            arr = f[ref][()]

        # if compound dtype (real/imag) → convert to complex
        if hasattr(arr, "dtype") and arr.dtype.names:
            names = set(arr.dtype.names)
            if {"real","imag"} <= names:
                arr = arr["real"] + 1j*arr["imag"]
            elif {"r","i"} <= names:
                arr = arr["r"] + 1j*arr["i"]

        field = np.array(arr).squeeze()  # remove (1,1), only keep (H,W)

    if subtract_ref:
        import re
        base = os.path.basename(file_path)
        comp_match = re.search(r'_(E[xyz])\.mat$', base, flags=re.IGNORECASE)
        if comp_match:
            comp = comp_match.group(1)
            ref_path = os.path.join(os.path.dirname(file_path), f'T_reff_{comp}.mat')
            if os.path.exists(ref_path):
                try:
                    ref = load_unit_cell_field_data(ref_path, subtract_ref=False)
                    if ref.shape == field.shape:
                        field = field - ref
                except Exception:
                    pass

    return field


def parse_unit_cell_filename(filename: str) -> dict:
    """parse unit cell filename, extract parameters
    for example: T_rad_6_hts_14_dx_-2_dy_-2_Ex.mat
    return: {'rad': 6, 'hts': 14, 'dx': -2, 'dy': -2, 'component': 'Ex'}
    """
    import re
    pattern = r'T_rad_(\d+)_hts_(\d+)_dx_(-?\d+)_dy_(-?\d+)_(\w+)\.mat'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return {
            'rad': int(match.group(1)),
            'hts': int(match.group(2)),
            'dx': int(match.group(3)),
            'dy': int(match.group(4)),
            'component': match.group(5)
        }
    else:
        raise ValueError(f"Cannot parse filename: {filename}")


class UnitCellDataset(torch.utils.data.Dataset):
    """unit cell dataset"""
    def __init__(self, data_dir: str, indices: Optional[List[int]] = None):
        """
        Args:
            data_dir: T_data folder path
            indices: list of indices to use
        """
        self.data_dir = data_dir
        
        # get all Ex files, for unique parameter combinations
        ex_files = sorted(glob(os.path.join(data_dir, '*_Ex.mat')))
        
        # parse all files to get parameter combinations
        self.samples = []
        for ex_file in ex_files:
            params = parse_unit_cell_filename(ex_file)
            # build Ex, Ey, Ez file paths
            base_name = os.path.basename(ex_file).replace('_Ex.mat', '')
            sample = {
                'params': params,
                'ex_file': ex_file,
                'ey_file': os.path.join(data_dir, f'{base_name}_Ey.mat'),
                'ez_file': os.path.join(data_dir, f'{base_name}_Ez.mat')
            }
            self.samples.append(sample)
        
        # if indices are provided, only keep corresponding samples
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        
        print(f"UnitCellDataset: found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load Ex, Ey, Ez field data
        ex_field = load_unit_cell_field_data(sample['ex_file'], subtract_ref=True)  # (H, W) complex
        ey_field = load_unit_cell_field_data(sample['ey_file'], subtract_ref=True)
        ez_field = load_unit_cell_field_data(sample['ez_file'], subtract_ref=True)
        
        # extract real and imag parts, build features (6, H, W)
        features = np.stack([
            np.real(ex_field),
            np.imag(ex_field),
            np.real(ey_field),
            np.imag(ey_field),
            np.real(ez_field),
            np.imag(ez_field)
        ], axis=0).astype(np.float32)
        
        # get structure parameters as extra information
        params = sample['params']
        
        return {
            'features': torch.from_numpy(features),  # (6, H, W)
            'params': params,
            'filename': os.path.basename(sample['ex_file'])
        }


class UnitCellGraphDataset(torch.utils.data.Dataset):
    """unit cell dataset - combine unit cell field data with overall structure data"""
    def __init__(self, 
                 t_data_dir: str,  # T_data folder (unit cell field data)
                 struct_data_files: List[str],  # data3d_*.mat file list (overall structure data)
                 indices: Optional[List[int]] = None, 
                 split: str = 'train',
                 window_size: int = 7,
                 pd_size: int = 6,
                 Ng: int = 24,
                 R: int = 3,
                 E_scale: float = 1.0,
                 in_mean=None, in_std=None,
                 out_mean=None, out_std=None,
                 augment: bool = False,
                 produce_unit_cell_features: bool = True,
                 unit_cell_mean: Optional[torch.Tensor] = None,
                 unit_cell_std: Optional[torch.Tensor] = None,
                 num_samples: int = 20000,
                 normalize_output: bool = True,
                 use_ex_only: bool = True):
        """
        Args:
            t_data_dir: T_data folder path (unit cell field data)
            struct_data_files: data3d_*.mat file list (overall structure data)
            indices: list of indices to use
            window_size: window size
            pd_size: padding size
            Ng: output grid size
            R: radius parameter
            E_scale: field scaling factor
            in_mean, in_std: input normalization parameters
            out_mean, out_std: output normalization parameters
            augment: whether to augment data
            normalize_output: whether to normalize output field
        """
        self.t_data_dir = t_data_dir
        self.struct_data_files = struct_data_files
        self.window_size = window_size
        self.pd_size = pd_size
        self.Ng = Ng
        self.R = R
        self.E_scale = E_scale
        self.augment = augment
        # when downstream uses embedding table, can only return ID without returning 6x49x49 features, save a lot of IO and copy
        self.produce_unit_cell_features = bool(produce_unit_cell_features)
        self.use_ex_only = use_ex_only  # whether to use Ex (2 channels) instead of Ex/Ey/Ez (6 channels)
        self.field_channels = 2 if use_ex_only else 6
        
        # normalization parameters (force conversion to 1D float32 Tensor to prevent indexing exceptions)
        self.in_mean = torch.as_tensor(in_mean, dtype=torch.float32).view(-1) if in_mean is not None else None
        self.in_std = torch.as_tensor(in_std, dtype=torch.float32).view(-1) if in_std is not None else None
        self.out_mean = torch.as_tensor(out_mean, dtype=torch.float32).view(-1) if out_mean is not None else None
        self.out_std = torch.as_tensor(out_std, dtype=torch.float32).view(-1) if out_std is not None else None

        # pre-cache common scalars to avoid repeated indexing in __getitem__ and avoid unexpected type issues
        if self.in_mean is not None and self.in_std is not None:
            assert self.in_mean.numel() >= 4 and self.in_std.numel() >= 4, "in_mean/in_std length insufficient, expected at least 4 dimensions (R,H,D0,D1)"
            self._in_r_mu = float(self.in_mean[0].item())
            self._in_h_mu = float(self.in_mean[1].item())
            self._in_d0_mu = float(self.in_mean[2].item())
            self._in_d1_mu = float(self.in_mean[3].item())
            self._in_r_std = float(self.in_std[0].item())
            self._in_h_std = float(self.in_std[1].item())
            self._in_d0_std = float(self.in_std[2].item())
            self._in_d1_std = float(self.in_std[3].item())
        else:
            self._in_r_mu = self._in_h_mu = self._in_d0_mu = self._in_d1_mu = 0.0
            self._in_r_std = self._in_h_std = self._in_d0_std = self._in_d1_std = 1.0
        
        # use FieldDataset to load overall structure data
        from utils.data_utils import FieldDataset
        self.field_dataset = FieldDataset(
            struct_data_files,
            split=split,
            window_size=window_size,
            pd_size=pd_size,
            Ng=Ng,
            R=R,
            E_scale=E_scale,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            augment=augment,
            indices=indices,
            num_samples=num_samples,
            normalize_output=normalize_output
        )
        
        # unit cell field data dictionary (quick lookup by parameters)
        # when only using embedding ID, can skip time-consuming cache building
        if self.produce_unit_cell_features:
            self._build_unit_cell_cache()
        else:
            self.unit_cell_cache = {}
    
    def _build_unit_cell_cache(self):
        """build unit cell field data cache, indexed by (rad, hts, dx, dy)"""
        print("building unit cell field data cache...")
        self.unit_cell_cache = {}
        
        # get all Ex files
        ex_files = sorted(glob(os.path.join(self.t_data_dir, '*_Ex.mat')))
        
        for ex_file in ex_files:
            try:
                if os.path.basename(ex_file).startswith('T_reff_'):
                    continue
                params = parse_unit_cell_filename(ex_file)
                key = (params['rad'], params['hts'], params['dx'], params['dy'])
                
                # load Ex, Ey, Ez (depending on configuration whether to load all)
                ex_field = load_unit_cell_field_data(ex_file, subtract_ref=True)
                
                if self.use_ex_only:
                    # only use Ex (2 channels: real, imag)
                    unit_cell_features = np.stack([
                        np.real(ex_field),
                        np.imag(ex_field)
                    ], axis=0).astype(np.float32)
                else:
                    # use Ex/Ey/Ez (6 channels)
                    base_name = os.path.basename(ex_file).replace('_Ex.mat', '')
                    ey_file = os.path.join(self.t_data_dir, f'{base_name}_Ey.mat')
                    ez_file = os.path.join(self.t_data_dir, f'{base_name}_Ez.mat')
                    
                    ey_field = load_unit_cell_field_data(ey_file, subtract_ref=True)
                    ez_field = load_unit_cell_field_data(ez_file, subtract_ref=True)
                    
                    unit_cell_features = np.stack([
                        np.real(ex_field),
                        np.imag(ex_field),
                        np.real(ey_field),
                        np.imag(ey_field),
                        np.real(ez_field),
                        np.imag(ez_field)
                    ], axis=0).astype(np.float32)
                
                self.unit_cell_cache[key] = torch.from_numpy(unit_cell_features)
            except Exception as e:
                print(f"warning: cannot load {ex_file}: {e}")
                continue
        
        print(f"unit cell cache built, {len(self.unit_cell_cache)} parameter combinations")
        
        # do global statistics and normalization on the entire library (preserve relative strength differences)
        if len(self.unit_cell_cache) > 0:
            all_features = torch.stack(list(self.unit_cell_cache.values()))  # (N, C, 49, 49)
            uc_mean = all_features.mean(dim=(0, 2, 3))  # (C,) global mean for each channel
            uc_std = all_features.std(dim=(0, 2, 3), unbiased=False)  # (C,) global std for each channel
            print(f"Unit cell library global mean (Ex only if self.use_ex_only else Ex/Ey/Ez): {uc_mean.numpy()}")
            print(f"Unit cell library global std (Ex only if self.use_ex_only else Ex/Ey/Ez): {uc_std.numpy()}")
            
            # immediately normalize cache (preserve relative strength differences)
            mean_view = uc_mean.view(self.field_channels, 1, 1)
            std_view = uc_std.view(self.field_channels, 1, 1)
            for key in self.unit_cell_cache:
                self.unit_cell_cache[key] = (self.unit_cell_cache[key] - mean_view) / (std_view + 1e-8)
            print("normalized unit cell cache (preserve relative strength information)")
    
    def normalize_unit_cell_cache(self, mean: torch.Tensor, std: torch.Tensor):
        """normalize all unit_cell_features in cache (in-place)"""
        if not self.produce_unit_cell_features or len(self.unit_cell_cache) == 0:
            return
        
        mean = mean.view(self.field_channels, 1, 1)
        std = std.view(self.field_channels, 1, 1)
        
        print(f"normalizing unit cell cache (total {len(self.unit_cell_cache)} units)...")
        for key in self.unit_cell_cache:
            self.unit_cell_cache[key] = (self.unit_cell_cache[key] - mean) / (std + 1e-6)
        self.unit_cell_mean = mean.view(self.field_channels)
        self.unit_cell_std = std.view(self.field_channels)
        print("normalization completed")
    
    def __len__(self):
        return len(self.field_dataset)
    
    def __getitem__(self, idx):
        # get base data from FieldDataset (contains structure parameters and target field)
        base_data = self.field_dataset[idx]
        
        num_nodes = base_data.x.size(0)
        
        # get corresponding unit cell field features for each node
        # base_data.x shape: (num_nodes, 5) - R, H, D0, D1, bnd
        unit_cell_features_list = []
        uc_id_list = []

        # control whether to only return ID (for embedding table)
        use_embedding_ids_only = not self.produce_unit_cell_features
        
        for i in range(num_nodes):
            # get structure parameters for the i-th node
            node_R = base_data.x[i, 0].item()
            node_H = base_data.x[i, 1].item()
            node_D0 = base_data.x[i, 2].item()
            node_D1 = base_data.x[i, 3].item()
            
            # de-normalize structure parameters
            if self.in_mean is not None and self.in_std is not None:
                node_R = node_R * self._in_r_std + self._in_r_mu
                node_H = node_H * self._in_h_std + self._in_h_mu
                node_D0 = node_D0 * self._in_d0_std + self._in_d0_mu
                node_D1 = node_D1 * self._in_d1_std + self._in_d1_mu
            
            # map continuous values to discrete parameter values
            rad = self._map_to_discrete_rad(node_R)
            hts = self._map_to_discrete_hts(node_H)
            dx = self._map_to_discrete_d(node_D0)
            dy = self._map_to_discrete_d(node_D1)
            
            # get unit cell field features from cache
            key = (rad, hts, dx, dy)
            if not use_embedding_ids_only:
                if key in self.unit_cell_cache:
                    # unit cell field features from cache (not normalized)
                    unit_cell_features = self.unit_cell_cache[key].clone()
                else:
                    # if not found, use zero tensor
                    unit_cell_features = torch.zeros(self.field_channels, 49, 49, dtype=torch.float32)
                unit_cell_features_list.append(unit_cell_features)
            # record discrete ID for the node (same as above de-normalization + discretization)
            uc_id_list.append(torch.tensor([rad, hts, dx, dy], dtype=torch.long))
        
        # stack unit cell features for all nodes: (num_nodes, C, 49, 49) C=2(Ex only) or 6(Ex/Ey/Ez)
        if not use_embedding_ids_only:
            unit_cell_features_all = torch.stack(unit_cell_features_list, dim=0)
        else:
            unit_cell_features_all = None

        # also keep unit cell discrete parameters as ID: (rad, hts, dx, dy)
        unit_cell_ids = torch.stack(uc_id_list, dim=0)
        
        # create new Data object
        # assemble Data (when using embedding table, do not add unit_cell_features key, avoid PyG concatenating None)
        kwargs = dict(
            x=base_data.x,
            edge_index=base_data.edge_index,
            edge_attr=base_data.edge_attr,
            y=base_data.y,
            mask=base_data.mask,
            num_nodes=base_data.num_nodes,
            unit_cell_ids=unit_cell_ids,
        )
        if unit_cell_features_all is not None:
            kwargs['unit_cell_features'] = unit_cell_features_all
        new_data = Data(**kwargs)
        
        return new_data
    
    def _map_to_discrete_rad(self, value):
        """map continuous R to nearest discrete value [6, 8, 10] (nearest neighbor rather than threshold)"""
        candidates = np.array([6, 8, 10], dtype=np.float32)
        idx = int(np.argmin(np.abs(candidates - float(value))))
        return int(candidates[idx])
    
    def _map_to_discrete_hts(self, value):
        """map continuous H to nearest discrete value [14, 16, 18]"""
        candidates = np.array([14, 16, 18], dtype=np.float32)
        idx = int(np.argmin(np.abs(candidates - float(value))))
        return int(candidates[idx])
    
    def _map_to_discrete_d(self, value):
        """map continuous D to nearest discrete value [-2, 0, 2]"""
        candidates = np.array([-2, 0, 2], dtype=np.float32)
        idx = int(np.argmin(np.abs(candidates - float(value))))
        return int(candidates[idx])