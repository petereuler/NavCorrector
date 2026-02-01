import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsSupConLoss(nn.Module):
    """
    基于 4D 物理向量 (len, cos, sin, dz) 的监督对比损失（严格二值 mask）。
    """

    def __init__(self, temperature=0.07, th_len=0.1, th_angle_cos=0.99, th_dz=0.05):
        super().__init__()
        self.temperature = temperature
        self.th_len = th_len
        self.th_angle_cos = th_angle_cos
        self.th_dz = th_dz

    def forward(self, features, gt_phy):
        # features: [B, D]
        # gt_phy: [B, 4] -> (len, cos, sin, dz)
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 特征归一化，余弦相似度 = 点积
        feats = F.normalize(features, dim=1, eps=1e-8)
        sim = torch.matmul(feats, feats.T) / self.temperature  # [B, B]

        # 解析物理真值
        len_val = gt_phy[:, 0:1]
        cos_val = gt_phy[:, 1:2]
        sin_val = gt_phy[:, 2:3]
        dz_val = gt_phy[:, 3:4]

        # 条件1: 步长相似
        diff_len = torch.abs(len_val - len_val.T)
        c1 = diff_len < self.th_len

        # 条件2: 航向相似 (cos/sin 向量点积)
        v = torch.cat([cos_val, sin_val], dim=1)
        v = F.normalize(v, dim=1, eps=1e-8)
        dot = torch.matmul(v, v.T)
        c2 = dot > self.th_angle_cos

        # 条件3: 高程相似
        diff_dz = torch.abs(dz_val - dz_val.T)
        c3 = diff_dz < self.th_dz

        # 物理相似性 mask：三条件同时满足
        pos_mask = (c1 & c2 & c3).float()

        # 对角线置 0，自己不是自己的正样本
        eye = torch.eye(batch_size, device=device)
        pos_mask = pos_mask * (1.0 - eye)

        # 计算对比损失
        exp_sim = torch.exp(sim) * (1.0 - eye)  # 排除自身
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class PoseSupConLoss(nn.Module):
    """
    基于相对姿态(四元数)的监督对比损失。
    """

    def __init__(self, temperature=0.07, angle_th=0.05):
        super().__init__()
        self.temperature = temperature
        self.angle_th = angle_th

    def forward(self, features, gt_quat):
        # features: [B, D]
        # gt_quat: [B, 4]
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        feats = F.normalize(features, dim=1, eps=1e-8)
        sim = torch.matmul(feats, feats.T) / self.temperature

        # 计算四元数角度差作为姿态相似度
        q = F.normalize(gt_quat, dim=1, eps=1e-8)
        dot = torch.abs(torch.matmul(q, q.T))
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        diff_angle = 2.0 * torch.acos(dot)

        # 角度差小于阈值 -> 正样本
        pos_mask = (diff_angle < self.angle_th).float()

        eye = torch.eye(batch_size, device=device)
        pos_mask = pos_mask * (1.0 - eye)

        exp_sim = torch.exp(sim) * (1.0 - eye)
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
