import torch
import torch.nn as nn


class HybridIMUEmbed(nn.Module):
    """
    Hybrid Physical-Implicit Embedding.
    Output: [B, 128] soft bits via sigmoid.
    """

    def __init__(self, window_size=256, explicit_dims=None, latent_dim=32, feat_dim=128):
        super().__init__()
        explicit_dims = explicit_dims or {}
        self.explicit_dims = explicit_dims
        self.latent_dim = latent_dim
        self.window_size = window_size

        self.backbone = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3),
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

        self.heads = nn.ModuleDict()
        for key, dim in self.explicit_dims.items():
            self.heads[key] = nn.Linear(feat_dim, dim)

        self.fc_latent = nn.Linear(feat_dim, latent_dim)
        self.aux_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6 * window_size),
        )

        total_bits = sum(explicit_dims.values()) + latent_dim
        assert total_bits == 128, f"Total bits must be 128, got {total_bits}"

    def forward(self, x):
        # x: (B, 6, W)
        feat = self.backbone(x)
        feat = self.pool(feat).squeeze(-1)  # (B, D)

        split = {}
        explicit_parts = []
        for key, head in self.heads.items():
            out = torch.sigmoid(head(feat))
            split[key] = out
            explicit_parts.append(out)

        latent = torch.sigmoid(self.fc_latent(feat))
        split["latent"] = latent

        emb = torch.cat(explicit_parts + [latent], dim=1)
        recon = None
        if self.training:
            recon = self.aux_decoder(latent)
        return emb, split, recon


class HybridIMUEmbWrapper(nn.Module):
    """
    包装器：仅返回 embedding，用于 Stage2。
    """

    def __init__(self, hybrid_model):
        super().__init__()
        self.hybrid = hybrid_model

    def forward(self, x):
        emb, _, _ = self.hybrid(x)
        return emb
