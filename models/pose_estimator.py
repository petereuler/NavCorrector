import torch
import torch.nn as nn
import torch.nn.functional as F
from .regress import FeatureExtractor as RegFeatureExtractor, RegressorHead as RegHead


def normalize_quaternion(q, eps=1e-8):
    return F.normalize(q, p=2, dim=-1, eps=eps)


def quaternion_geodesic_loss(pred, target, eps=1e-7):
    # q and -q represent the same rotation; use absolute dot
    dot = torch.abs(torch.sum(pred * target, dim=-1))
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    angle = 2.0 * torch.acos(dot)
    return angle.mean()


class PoseEstimator(nn.Module):
    def __init__(self, in_channels=6, feat_dim=64, output_dim=4, normalize_output=True):
        super().__init__()
        self.output_dim = output_dim
        self.normalize_output = normalize_output
        self.feature_extractor = RegFeatureExtractor(in_channels, feat_dim)
        self.head = RegHead(feat_dim, output_dim=output_dim)

    def forward(self, x):
        feat = self.feature_extractor(x)
        out = self.head(feat)
        if self.normalize_output and self.output_dim == 4:
            out = normalize_quaternion(out)
        return out
