import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from .regress import FeatureExtractor as RegFeatureExtractor, RegressorHead as RegHead


# ==================== ResNet 基础模块 ====================

class ResidualBlock1D(nn.Module):
    """
    1D 残差块，用于时序信号处理
    输入: (B, C, T) -> 输出: (B, C, T)
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class ResidualBlockDown1D(nn.Module):
    """
    带下采样的 1D 残差块
    输入: (B, in_ch, T) -> 输出: (B, out_ch, T//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接的投影
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


# ==================== 特征提取器 (ResNet) ====================

class FeatureExtractor(nn.Module):
    """
    基于 ResNet 的特征提取器
    输入: (B, T, C) -> 输出: (B, feat_dim)
    """
    def __init__(self, in_channels, feat_dim, hidden_dim=64, num_blocks=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 残差块堆叠（逐步下采样）
        self.layer1 = nn.Sequential(
            ResidualBlock1D(hidden_dim, dropout=dropout),
            ResidualBlock1D(hidden_dim, dropout=dropout),
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim, hidden_dim * 2, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 2, dropout=dropout),
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim * 2, hidden_dim * 4, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 4, dropout=dropout),
        )
        
        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, feat_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, T, C) -> 转换为 (B, C, T) 用于 Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局池化
        x = self.global_pool(x)  # (B, C, 1)
        x = x.squeeze(-1)       # (B, C)
        
        return self.fc(x)       # (B, feat_dim)


# ==================== 回归头 ====================

class RegressorHead(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, feat):
        return self.fc(feat)


# ==================== 评估函数 ====================

def compute_heading_mae(pred_heading, target_heading):
    """计算航向角平均绝对误差"""
    target_heading = target_heading.squeeze(-1) if target_heading.dim() > 1 else target_heading
    diff = pred_heading - target_heading
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return torch.abs(diff).mean()


class DualHeadingModel(torch.nn.Module):
    """
    双流航向预测模型（支持不确定性估计）：
    - 绝对航向头：输出 [预测值, log方差]
    - 相对航向头：输出 [预测值, log方差]
    """
    def __init__(self, in_channels, feat_dim=64):
        super().__init__()

        # 共享骨干网络
        self.feature_extractor = RegFeatureExtractor(in_channels, feat_dim)

        # [修改] output_dim=2 (均值, log方差)
        self.abs_head = RegHead(feat_dim, output_dim=2)
        self.rel_head = RegHead(feat_dim, output_dim=2)

    def forward(self, x):
        """
        Returns:
            pred_abs_mu: 绝对航向预测值
            pred_abs_logvar: 绝对航向不确定性 (log σ^2)
            pred_rel_mu: 相对航向预测值
            pred_rel_logvar: 相对航向不确定性 (log σ^2)
        """
        feat = self.feature_extractor(x)
        
        # 绝对航向输出 (B, 2)
        out_abs = self.abs_head(feat)
        pred_abs_mu = out_abs[:, 0:1]
        pred_abs_logvar = out_abs[:, 1:2]
        
        # 相对航向输出 (B, 2)
        out_rel = self.rel_head(feat)
        pred_rel_mu = out_rel[:, 0:1]
        pred_rel_logvar = out_rel[:, 1:2]

        return pred_abs_mu, pred_abs_logvar, pred_rel_mu, pred_rel_logvar


class DualHeadingLoss(torch.nn.Module):
    """
    双流航向损失函数：
    - 绝对航向：HeadingBinaryLoss
    - 相对航向：MSELoss
    - 总损失：loss_abs + w_rel * loss_rel
    """
    def __init__(self, num_bits=8, use_gray_code=True, quantizer=None, circular_weight=0.0, rel_weight=10.0):
        super().__init__()
        self.rel_weight = rel_weight

        # 绝对航向损失
        self.abs_loss = HeadingBinaryLoss(
            num_bits=num_bits,
            use_gray_code=use_gray_code,
            quantizer=quantizer,
            circular_weight=circular_weight
        )

        # 相对航向损失 (MSE)
        self.rel_loss = torch.nn.MSELoss()

    def forward(self, logits_abs, pred_rel, target_abs, target_rel):
        """
        Args:
            logits_abs: 绝对航向logits (batch_size, num_bits)
            pred_rel: 相对航向预测 (batch_size, 1)
            target_abs: 绝对航向标签 (batch_size, 1)
            target_rel: 相对航向标签 (batch_size, 1)

        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        # 绝对航向损失
        loss_abs = self.abs_loss(logits_abs, target_abs)

        # 相对航向损失
        loss_rel = self.rel_loss(pred_rel, target_rel)

        # 总损失
        total_loss = loss_abs + self.rel_weight * loss_rel

        loss_dict = {
            'total': total_loss.item(),
            'abs': loss_abs.item(),
            'rel': loss_rel.item(),
            'rel_weighted': (self.rel_weight * loss_rel).item()
        }

        return total_loss, loss_dict


class UncertaintyHeadingLoss(nn.Module):
    """
    [稳定版] 高斯负对数似然损失 (Gaussian NLL Loss + MSE Regularization)
    
    改进点：
    1. 增加 mse_weight：强制模型回归均值，防止通过调大方差"作弊"。
    2. 增加 log_var_clamp：限制方差范围，防止数值爆炸。
    """
    def __init__(self, rel_weight=10.0, mse_weight=1.0):
        super().__init__()
        self.rel_weight = rel_weight
        self.mse_weight = mse_weight  # 新增：MSE 正则权重 (推荐 1.0)
        
        # 简单的 MSE 用于正则化
        self.mse_loss = nn.MSELoss()

    def gaussian_nll(self, pred, target, log_var, is_periodic=False):
        # 1. 计算基础误差
        diff = pred - target
        if is_periodic:
            # 周期性 Wrap
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        mse_term = diff ** 2
        
        # 2. [关键] 方差截断 (Clamping)
        # min=-5 (sigma≈0.08) 防止除零/梯度爆炸
        # max=5  (sigma≈12)  防止模型躺平(方差过大)
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        
        # 3. NLL Loss 计算
        # Loss = 0.5 * exp(-s) * MSE + 0.5 * s
        loss_nll = 0.5 * torch.exp(-log_var) * mse_term + 0.5 * log_var
        
        return loss_nll.mean(), mse_term.mean()

    def forward(self, pred_abs_mu, pred_abs_logvar, pred_rel_mu, pred_rel_logvar, target_abs, target_rel):
        # --- 绝对航向 (Abs) ---
        loss_abs_nll, loss_abs_mse = self.gaussian_nll(
            pred_abs_mu, target_abs, pred_abs_logvar, is_periodic=True
        )
        
        # --- 相对航向 (Rel) ---
        loss_rel_nll, loss_rel_mse = self.gaussian_nll(
            pred_rel_mu, target_rel, pred_rel_logvar, is_periodic=False
        )
        
        # --- 总 Loss 组合 ---
        # 核心思想：同时优化 NLL (为了不确定性) 和 MSE (为了准度)
        # 如果不加 MSE，模型初期容易迷失方向
        loss_abs = loss_abs_nll + self.mse_weight * loss_abs_mse
        loss_rel = loss_rel_nll + self.mse_weight * loss_rel_mse
        
        total_loss = loss_abs + self.rel_weight * loss_rel
        
        # 返回 total_loss 以及拆解项供打印
        # 注意：这里返回的 loss_abs/loss_rel 已经是包含 MSE 正则的混合 Loss
        return total_loss, loss_abs, loss_rel