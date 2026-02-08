import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   save_dir: str,
                   is_best: bool = False,
                   extra_data: Optional[Dict] = None):
    """保存模型检查点"""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # 添加额外数据
    if extra_data is not None:
        state.update(extra_data)
    
    # 保存常规检查点
    if epoch > 0 and epoch % 50 == 0:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(state, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_model_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(state, best_model_path)

def load_checkpoint(model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   checkpoint_path: str) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 5))
    
    # 绘制训练损失和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 绘制学习率
    if 'lr' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_metrics(pred: torch.Tensor, 
                   target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """计算预测指标"""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    # 计算MSE
    mse = nn.MSELoss()(pred, target).item()
    
    # 计算MAE
    mae = nn.L1Loss()(pred, target).item()
    
    # 计算相关系数
    pred_mean = pred.mean()
    target_mean = target.mean()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    pred_std = pred.std()
    target_std = target.std()
    correlation = (covariance / (pred_std * target_std)).item()
    
    # 计算R²
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> Tuple[bool, bool]:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False, True
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, False
