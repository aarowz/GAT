import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from glob import glob
from tqdm import tqdm
from torch_geometric.loader import DataLoader  # new import path
from utils.data_utils import (
    set_random_seed,
    load_unit_cell_field,
    compute_dataset_stats,
    FieldDataset
)
from models.base_model import BaseGNNModel
from utils.train_utils import (
    save_checkpoint, 
    load_checkpoint,
    plot_training_history,
    compute_metrics,
    EarlyStopping
)
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    # add progress bar, set leave=True to keep progress
    pbar = tqdm(train_loader, desc='Training', leave=True)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # forward propagation
        output = model(
            batch.x, batch.edge_index, batch.edge_attr, batch,
            window_size=7, pd_size=6  # using default parameters
        )
        
        # adjust target tensor shape to match output
        target = batch.y.view(batch.num_graphs, 6, output.size(-2), output.size(-1))
        loss = criterion(output, target)
        
        # backward propagation
        loss.backward()
        optimizer.step()
        
        # update progress bar
        current_loss = loss.item()
        total_loss += current_loss * batch.num_graphs
        pbar.set_postfix({'loss': f'{current_loss:.6f}'})
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device, return_reinv_loss=False, out_mean=None, out_std=None):
    """
    validation function
    
    Args:
        return_reinv_loss: whether to return the loss after de-normalization
        out_mean, out_std: output normalization parameters, used for de-normalization calculation
    
    Returns:
        if return_reinv_loss=True, return (norm_loss, reinv_loss)
        else return norm_loss
    """
    model.eval()
    total_loss = 0
    total_reinv_loss = 0 if return_reinv_loss else None
    
    with torch.no_grad():
        # add progress bar, set leave=True to keep progress
        pbar = tqdm(val_loader, desc='Validation', leave=True)
        for batch in pbar:
            batch = batch.to(device)
            output = model(
                batch.x, batch.edge_index, batch.edge_attr, batch,
                window_size=7, pd_size=6  # using default parameters
            )
            # adjust target tensor shape to match output
            target = batch.y.view(batch.num_graphs, 6, output.size(-2), output.size(-1))
            loss = criterion(output, target)
            current_loss = loss.item()
            total_loss += current_loss * batch.num_graphs
            
            # calculate the loss after de-normalization
            if return_reinv_loss and out_mean is not None and out_std is not None:
                # de-normalize output and target
                out_mean_dev = out_mean.to(device).view(1, 6, 1, 1)
                out_std_dev = out_std.to(device).view(1, 6, 1, 1)
                
                output_reinv = output * out_std_dev + out_mean_dev
                target_reinv = target * out_std_dev + out_mean_dev
                
                reinv_loss = criterion(output_reinv, target_reinv)
                total_reinv_loss += reinv_loss.item() * batch.num_graphs
            
            pbar.set_postfix({'loss': f'{current_loss:.6f}'})
    
    avg_loss = total_loss / len(val_loader.dataset)
    
    if return_reinv_loss:
        avg_reinv_loss = total_reinv_loss / len(val_loader.dataset)
        return avg_loss, avg_reinv_loss
    else:
        return avg_loss

def main():
    # set random seed and device
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # hyperparameters
    WINDOW_SIZE = 7
    PD_SIZE = 6
    R = 3
    NG = 24
    E_SCALE = 1.0
    TRAIN_RATIO, VAL_RATIO = 0.7, 0.15  # remaining 0.15 for test
    BATCH_SIZE = 16
    NUM_SAMPLES_TRAIN = 14000  # number of random samples for training set
    NUM_SAMPLES_VAL = 3000     # number of random samples for validation set
    NUM_SAMPLES_TEST = 3000    # number of random samples for test set
    DECODER_MODE = 'upsample'    # 'direct': MLP directly outputs + CNN polish, 'upsample': CNN progressive upsampling (PixelShuffle)
    NORMALIZE_OUTPUT = False    # True: normalized training, False: original scale training

    # load data file list
    from glob import glob
    file_lst = sorted(glob('./data/data3d_*.mat'))
    print(f"found {len(file_lst)} data files")
    
    if len(file_lst) == 0:
        raise FileNotFoundError("data files not found!")

    # global random split dataset
    rng = np.random.default_rng(seed=42)
    idx_all = np.arange(len(file_lst))
    rng.shuffle(idx_all)
    n_train = int(len(idx_all) * TRAIN_RATIO)
    n_val = int(len(idx_all) * (TRAIN_RATIO + VAL_RATIO))
    
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:n_val]
    test_idx = idx_all[n_val:]

    train_files = [file_lst[i] for i in train_idx]
    val_files = [file_lst[i] for i in val_idx]
    test_files = [file_lst[i] for i in test_idx]
    print(f"dataset split -> training set: {len(train_files)}, validation set: {len(val_files)}, test set: {len(test_files)}")

    # calculate data statistics (only using training set)
    print("calculating normalization statistics for training set...")
    in_mean, in_std, out_mean, out_std = compute_dataset_stats(
        train_files, window_size=WINDOW_SIZE, pd_size=PD_SIZE, R=R, E_scale=E_SCALE
    )
    in_mean = torch.tensor(in_mean, dtype=torch.float32)
    in_std = torch.tensor(in_std, dtype=torch.float32)
    out_mean = torch.tensor(out_mean, dtype=torch.float32)
    out_std = torch.tensor(out_std, dtype=torch.float32)
    print(f"input mean: {in_mean.numpy()}, input std: {in_std.numpy()}")
    print(f"output mean: {out_mean.numpy()}, output std: {out_std.numpy()}")

    # build dataset
    train_dataset = FieldDataset(
        file_lst, split='train', augment=True,
        window_size=WINDOW_SIZE, pd_size=PD_SIZE, Ng=NG, R=R, E_scale=E_SCALE,
        in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std,
        indices=train_idx,
        num_samples=NUM_SAMPLES_TRAIN,
        normalize_output=NORMALIZE_OUTPUT
    )
    val_dataset = FieldDataset(
        file_lst, split='val', augment=False,
        window_size=WINDOW_SIZE, pd_size=PD_SIZE, Ng=NG, R=R, E_scale=E_SCALE,
        in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std,
        indices=val_idx,
        num_samples=NUM_SAMPLES_VAL,
        normalize_output=NORMALIZE_OUTPUT
    )
    test_dataset = FieldDataset(
        file_lst, split='test', augment=False,
        window_size=WINDOW_SIZE, pd_size=PD_SIZE, Ng=NG, R=R, E_scale=E_SCALE,
        in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std,
        indices=test_idx,
        num_samples=NUM_SAMPLES_TEST,
        normalize_output=NORMALIZE_OUTPUT
    )
    
    print(f"number of samples: training set {len(train_dataset)}, validation set {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    # build model
    model = BaseGNNModel(
        in_channels=5,
        #gnn_widths=[64, 256, 512, 1024, 6*24*24],  # GNN directly outputs
        gnn_widths=[64, 128, 256],  # GNN + MLP
        # mid_channels=64,  # only needed in upsample mode, not needed in direct mode
        out_channels=6,
        Ng=24,
        p_drop=0.1,
        decoder_mode=DECODER_MODE
    ).to(device)
    
    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5
    )
    
    # training settings
    num_epochs = 200
    early_stopping = EarlyStopping(patience=15)
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'test_loss': []}
    best_val_loss = float('inf')
    
    # 训练循环
    print("\n" + "="*60)
    print(f"training mode: {'normalized training' if NORMALIZE_OUTPUT else 'original scale training'}")
    print(f"decoder mode: {DECODER_MODE}")
    print("="*60 + "\n")
    print("starting training...\n")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        
        # update learning rate
        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]  # using get_last_lr() to get current learning rate
        
        # record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # check if it is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"found new best model! best validation loss: {best_val_loss:.6f}")
        
        # save checkpoint (containing normalization parameters)
        extra_data = {
            'in_mean': in_mean,
            'in_std': in_std,
            'out_mean': out_mean,
            'out_std': out_std
        }
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            save_dir='checkpoint/GNN',
            is_best=is_best,
            extra_data=extra_data
        )
        
        # print current epoch results
        print(f"training loss: {train_loss:.6f}")
        print(f"validation loss: {val_loss:.6f}")
        print(f"learning rate: {current_lr:.6f}")
        print("-" * 50)  # add separator to make output clearer
        



        if early_stopping(val_loss)[0]:
            print('\nEarly stopping triggered')
            break
        
    # final test: calculate loss based on NORMALIZE_OUTPUT mode
    if NORMALIZE_OUTPUT:
        # normalized mode: calculate both normalized and de-normalized loss
        test_loss, test_reinv_loss = validate(
            model, test_loader, criterion, device,
            return_reinv_loss=True,
            out_mean=out_mean,
            out_std=out_std
        )
        history['test_loss'].append(test_loss)
        history['test_reinv_loss'] = test_reinv_loss
        print(f"test loss (normalized):   {test_loss:.6f}")
        print(f"test loss (de-normalized): {test_reinv_loss:.6f}")
    else:
        # de-normalized mode: calculate loss based on original scale
        test_loss = validate(
            model, test_loader, criterion, device,
            return_reinv_loss=False
        )
        history['test_loss'].append(test_loss)
        print(f"test loss (original scale):  {test_loss:.6f}")
    print("\ntraining completed!")
    
    # save training history (CSV + PNG)
    os.makedirs('result/GNN', exist_ok=True)
    try:
        import csv
        csv_path = os.path.join('result/GNN', 'training_history.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
            for ep in range(len(history['train_loss'])):
                writer.writerow([
                    ep + 1,
                    history['train_loss'][ep],
                    history['val_loss'][ep],
                    history['lr'][ep]
                ])
        # save test loss
        import json
        with open(os.path.join('result/GNN', 'final_metrics.json'), 'w') as jf:
            if NORMALIZE_OUTPUT:
                json.dump({
                    'mode': 'normalized',
                    'test_loss_normalized': float(test_loss),
                    'test_loss_denormalized': float(test_reinv_loss)
                }, jf, indent=2)
            else:
                json.dump({
                    'mode': 'denormalized',
                    'test_loss': float(test_loss)
                }, jf, indent=2)
        print(f"training history saved to: {csv_path}")
    except Exception as e:
        print(f"failed to save CSV/JSON: {e}")
    
    # plot training history
    plot_training_history(history, save_path='result/GNN/training_history.png')
    
    # load best model for final evaluation
    best_model_path = os.path.join('checkpoint/GNN', 'best_model.pt')
    model, _, _, _ = load_checkpoint(model, None, best_model_path)
    
    # TODO: add model evaluation and prediction visualization code

if __name__ == '__main__':
    main()
