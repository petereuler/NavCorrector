import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEmbBase(nn.Module):
    """
    轻量级 1D CNN 编码器。
    输入: [B, 6, T]
    输出: [B, feat_dim] (L2 归一化)
    """

    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(x, dim=1, eps=1e-8)


class SupConHead(nn.Module):
    """用于 Stage 1 的投影头。"""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class IMUEmbDist(IMUEmbBase):
    """位移分支 IMU Embedding。"""

    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim)


class IMUEmbPose(IMUEmbBase):
    """姿态分支 IMU Embedding。"""

    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__(in_channels=in_channels, feat_dim=feat_dim)
