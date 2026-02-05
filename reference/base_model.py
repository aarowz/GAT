from calendar import firstweekday
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import math

class ImprovedGCNBlock(nn.Module):
    """GATCONV Residual Block"""
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.1):
        super().__init__()
        self.conv = GATConv(in_ch, out_ch, heads=1, concat=False, dropout=p_drop)
        # self.conv = GCNConv(in_ch, out_ch, add_self_loops=True, normalize=True)
        self.norm1 = nn.LayerNorm(out_ch)
        self.GN1 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.gelu = nn.GELU()
        # self.gelu = nn.GELU()
        self.norm2 = nn.LayerNorm(out_ch)
        self.GN2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        identity = x
        # out = self.conv(x, edge_index, edge_weight=edge_weight)
        out = self.conv(x, edge_index)
        #out = self.norm1(out)
        out = self.GN1(out)
        # out = self.leaky_relu(out)
        out = self.gelu(out)
        out = self.drop(out)
        if isinstance(self.proj, nn.Linear):
            identity = self.proj(identity)
        # return self.norm2(out + identity)
        return self.GN2(out + identity)
class UpBlock(nn.Module):
    """PixelShuffle UpBlock"""
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

class ResidualBlock(nn.Module):
    """Residual Block for CNN decoder"""
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p_drop)
        
        # 如果输入输出通道不同，需要1x1卷积投影
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # residual connection
        out = self.act(out)
        
        return out

def factorize_scale(Ng: int):
    """Factorize the upsampling factor into the product of 3 and 2"""
    factors = []
    n = Ng
    for p in [3, 2]:
        while n % p == 0:
            factors.append(p)
            n //= p
    return factors, n

class BaseGNNModel(nn.Module):
    """Base GNN Model (contains GNN and decoder)"""
    def __init__(self, 
                 in_channels: int = 5,
                 gnn_widths: list = [64, 128, 256],
                 mid_channels: int = None,  # 只在upsample模式下需要，默认None会自动设置为64
                 out_channels: int = 6,
                 Ng: int = 24,
                 p_drop: float = 0.1,
                 decoder_mode: str = 'direct'):  # 'direct' | 'upsample'
        super().__init__()
        assert decoder_mode in ('direct', 'upsample'), f"decoder_mode must be 'direct' or 'upsample', got {decoder_mode}"
        self.decoder_mode = decoder_mode
        self.Ng = Ng
        
        # mid_channels only needed in upsample mode
        if self.decoder_mode == 'upsample' and mid_channels is None:
            mid_channels = 64  # default value
            print(f"[BaseGNNModel] Upsample mode: using default mid_channels={mid_channels}")
        elif self.decoder_mode == 'direct' and mid_channels is not None:
            print(f"[BaseGNNModel] Direct mode: ignoring mid_channels parameter (not needed)")
        
        # GNN part
        feature_dim = 5
        chs = [feature_dim] + list(gnn_widths)
        self.blocks = nn.ModuleList([
            ImprovedGCNBlock(chs[i], chs[i + 1], p_drop=p_drop)
            for i in range(len(chs) - 1)
        ])
        
        last_dim = gnn_widths[-1]
        patch_size = Ng * Ng * out_channels  # 24*24*6 = 3456
        
        # decoder: different strategies based on mode
        if self.decoder_mode == 'direct':
            # mode 1: GNN directly outputs patch (if dimension matches) or through MLP mapping
            if last_dim == patch_size:
                # GNN last layer already outputs correct dimension, no MLP needed
                self.patch_mlp = None
                print(f"[BaseGNNModel] Direct mode: GNN directly outputs patch (last_dim={last_dim})")
            else:
                # MLP mapping to patch needed
                self.patch_mlp = nn.Sequential(
                    nn.Linear(last_dim, 1024),
                    nn.LayerNorm(1024),
                    nn.GELU(),
                    nn.Dropout(p_drop),
                    
                    nn.Linear(1024, 2048),
                    nn.LayerNorm(2048),
                    nn.GELU(),
                    nn.Dropout(p_drop),
                    
                    nn.Linear(2048, patch_size),
                )
                print(f"[BaseGNNModel] Direct mode: using MLP mapping ({last_dim} → {patch_size})")
            
            # CNN polish layer: smooth boundaries, refine details
        else:  # 'upsample'
            # mode 2: CNN progressive upsampling (PixelShuffle)
            # dimensionality reduction: last_dim → mid_channels
            self.dim_reduce = nn.Sequential(
                nn.Linear(last_dim, mid_channels),
                nn.LayerNorm(mid_channels),
                nn.GELU(),
                nn.Dropout(p_drop),
            )
            
            # calculate upsampling factor
            scale_factor = Ng  # 24
            factors, remainder = factorize_scale(scale_factor)
            if remainder != 1:
                raise ValueError(f"Ng={Ng} cannot be completely decomposed into the product of 2 and 3, remainder={remainder}")
            
            # build upsampling layers
            self.decoder_ups = nn.ModuleList()
            ch = mid_channels
            for factor in factors:
                self.decoder_ups.append(UpBlock(ch, ch, factor, p_drop))
            
            # final convolution layer: gradually reduce dimensionality to output channels
            self.decoder_head = nn.Sequential(
                ResidualBlock(mid_channels, mid_channels // 2, p_drop),
                ResidualBlock(mid_channels // 2, mid_channels // 4, p_drop),
                ResidualBlock(mid_channels // 4, mid_channels // 8, p_drop),
                nn.Conv2d(mid_channels // 8, out_channels, kernel_size=3, padding=1),
            )
        

    def forward(self, x, edge_index, edge_weight=None, data=None, window_size=7, pd_size=6):
        # GNN forward propagation
        for blk in self.blocks:
            x = blk(x, edge_index, edge_weight, data)
        
        # decoder: different paths based on mode
        batch_size = int(data.batch.max().item() + 1) if data is not None else 1
        main_window = x[data.mask]  # (B*49, last_dim)
        
        if self.decoder_mode == 'direct':
            # mode 1: GNN directly outputs or MLP maps to patch + CNN polish
            if self.patch_mlp is not None:
                # using MLP mapping
                patches = self.patch_mlp(main_window)  # (B*49, 24*24*6)
            else:
                # GNN directly outputs, no MLP needed
                patches = main_window  # (B*49, 24*24*6)
            
            # reshape: (B, 7, 7, 24*24*6) → (B, 7, 7, 24, 24, 6)
            patches = patches.view(batch_size, window_size, window_size, self.Ng, self.Ng, -1)
            
            # rearrange and concatenate: (B, 7, 7, 24, 24, 6) → (B, 6, 7*24, 7*24)
            # 先permute: (B, 6, 7, 24, 7, 24)
            patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
            # then view: (B, 6, 168, 168)
            H_full = window_size * self.Ng
            field = patches.view(batch_size, -1, H_full, H_full)
            
            # CNN polish: smooth boundaries and refine details
            field = self.polish_net(field)
            
        else:  # 'upsample'
            # mode 2: CNN progressive upsampling (PixelShuffle)
            # dimensionality reduction
            main_window = self.dim_reduce(main_window)  # (B*49, mid_channels)
            
            # reshape to spatial dimensions
            z_grid = main_window.view(batch_size, window_size, window_size, -1)  # (B, 7, 7, mid_channels)
            z_grid = z_grid.permute(0, 3, 1, 2).contiguous()  # (B, mid_channels, 7, 7)
            
            # progressive upsampling
            for up_block in self.decoder_ups:
                z_grid = up_block(z_grid)
            
            # final convolution mapping to output channels
            field = self.decoder_head(z_grid)  # (B, 6, 7*24, 7*24)
        
        return field
