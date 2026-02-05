import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from torch_geometric.loader import DataLoader
from models.base_model import BaseGNNModel
from utils.data_utils import FieldDataset
from utils.train_utils import load_checkpoint

def denorm(tensor, mean, std):
    """de-normalization function"""
    # Ensure mean and std are on the same device as the tensor
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    return tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

def visualize_results(true_field, pred_field, save_dir):
    """visualize true values, predicted values and errors"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    components = ['Ex (Real)', 'Ex (Imag)', 'Ey (Real)', 'Ey (Imag)', 'Ez (Real)', 'Ez (Imag)']
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)

    for i in range(6):
        # true values
        ax_true = fig.add_subplot(gs[0, i])
        im_true = ax_true.imshow(true_field[i], cmap='RdBu_r', aspect='auto')
        ax_true.set_title(f'True {components[i]}')
        ax_true.axis('off')
        divider = make_axes_locatable(ax_true)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_true, cax=cax)

        # predicted values
        ax_pred = fig.add_subplot(gs[1, i])
        im_pred = ax_pred.imshow(pred_field[i], cmap='RdBu_r', aspect='auto',
                                 vmin=true_field[i].min(), vmax=true_field[i].max())
        ax_pred.set_title(f'Predicted {components[i]}')
        ax_pred.axis('off')
        divider = make_axes_locatable(ax_pred)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_pred, cax=cax)

        # errors
        ax_err = fig.add_subplot(gs[2, i])
        error = np.abs(true_field[i] - pred_field[i])
        im_err = ax_err.imshow(error, cmap='hot', aspect='auto')
        ax_err.set_title(f'Abs Error {components[i]}')
        ax_err.axis('off')
        divider = make_axes_locatable(ax_err)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_err, cax=cax)

        # show MSE and MAE in error plot
        mse = np.mean((true_field[i] - pred_field[i]) ** 2)
        mae = np.mean(np.abs(true_field[i] - pred_field[i]))
        ax_err.text(
            0.02, 0.02,
            f"MSE: {mse:.4e}\nMAE: {mae:.4e}",
            transform=ax_err.transAxes,
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

    plt.suptitle('Field Prediction Results', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'field_prediction.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # hyperparameters - consistent with training
    WINDOW_SIZE = 7
    PD_SIZE = 6
    R = 3
    NG = 24
    E_SCALE = 1.0
    TRAIN_RATIO, VAL_RATIO = 0.7, 0.15  # remaining 0.15 for test

    # load data file list
    file_list = sorted(glob('./data/data3d_*.mat'))
    print(f"found {len(file_list)} data files")
    
    if len(file_list) == 0:
        raise FileNotFoundError("data files not found!")

    # global random split dataset - consistent with training
    rng = np.random.default_rng(seed=42)
    idx_all = np.arange(len(file_list))
    rng.shuffle(idx_all)
    n_train = int(len(idx_all) * TRAIN_RATIO)
    n_val = int(len(idx_all) * (TRAIN_RATIO + VAL_RATIO))
    
    test_idx = idx_all[n_val:]
    print(f"number of test samples: {len(test_idx)}")

    # load checkpoint
    checkpoint_path = './checkpoint/GNN/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # print checkpoint keys for debugging
    print("Checkpoint keys:", checkpoint.keys())

    # initialize model
    model = BaseGNNModel(
        in_channels=5,
        gnn_widths=[64, 128, 256],
        mid_channels=64,
        out_channels=6,
        Ng=24,
        p_drop=0.1
    ).to(device)

    # load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("model loaded successfully!")

    # get normalization parameters
    in_mean = checkpoint['in_mean']
    in_std = checkpoint['in_std']
    out_mean = checkpoint['out_mean']
    out_std = checkpoint['out_std']
    
    print(f"input mean: {in_mean.cpu().numpy()}")
    print(f"input std: {in_std.cpu().numpy()}")
    print(f"output mean: {out_mean.cpu().numpy()}")
    print(f"output std: {out_std.cpu().numpy()}")

    # build test dataset
    test_dataset = FieldDataset(
        file_list, split='test', augment=False,
        window_size=WINDOW_SIZE, pd_size=PD_SIZE, Ng=NG, R=R, E_scale=E_SCALE,
        in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std,
        indices=test_idx
    )
    print(f"number of test samples: {len(test_dataset)}")

    # use DataLoader to load data (batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # get first batch data
    data = next(iter(test_loader))
    data = data.to(device)

    # model inference
    with torch.no_grad():
        output = model(
            data.x, data.edge_index, data.edge_attr, data,
            window_size=WINDOW_SIZE, pd_size=PD_SIZE
        )

    # de-normalize
    # output shape is (1, 6, H, W), need to extract first sample
    pred_denorm = denorm(output[0].cpu(), out_mean.cpu(), out_std.cpu()).numpy()
    # data.y needs to be reshaped to (6, H, W)
    target = data.y.view(6, output.size(-2), output.size(-1))
    true_denorm = denorm(target.cpu(), out_mean.cpu(), out_std.cpu()).numpy()
    
    print(f"\npredicted output shape: {pred_denorm.shape}")
    print(f"true output shape: {true_denorm.shape}")

    # visualize
    visualize_results(true_denorm, pred_denorm, save_dir='./result/GNN/')

    # calculate MSE and MAE for each channel
    components = ['Ex (Real)', 'Ex (Imag)', 'Ey (Real)', 'Ey (Imag)', 'Ez (Real)', 'Ez (Imag)']
    print("\nError Statistics:")
    for i, comp in enumerate(components):
        mse = np.mean((true_denorm[i] - pred_denorm[i]) ** 2)
        mae = np.mean(np.abs(true_denorm[i] - pred_denorm[i]))
        print(f"{comp}: MSE = {mse:.6e}, MAE = {mae:.6e}")

    # plot MSE and MAE for each channel
    mse_values = [np.mean((true_denorm[i] - pred_denorm[i]) ** 2) for i in range(6)]
    mae_values = [np.mean(np.abs(true_denorm[i] - pred_denorm[i])) for i in range(6)]

    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(6)
    bar_width = 0.35

    mse_bars = ax.bar(index, mse_values, bar_width, label='MSE', color='b')
    mae_bars = ax.bar(index + bar_width, mae_values, bar_width, label='MAE', color='r')

    ax.set_xlabel('Field Components')
    ax.set_ylabel('Error')
    ax.set_title('MSE and MAE for Each Field Component')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Ex (Real)', 'Ex (Imag)', 'Ey (Real)', 'Ey (Imag)', 'Ez (Real)', 'Ez (Imag)'])
    ax.legend()

    plt.tight_layout()
    plt.savefig('./result/GNN/error.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
