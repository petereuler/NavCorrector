import torch.nn as nn
import torch
import numpy as np


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


class ResidualBlockUp1D(nn.Module):
    """
    带上采样的 1D 残差块（用于 Decoder）
    输入: (B, in_ch, T) -> 输出: (B, out_ch, T*scale)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接的投影
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='linear', align_corners=False),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.upsample(x)
        out = self.relu(self.bn1(self.conv1(out)))
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