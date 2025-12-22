import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, feat_dim):
        super().__init__()
        self.hidden_dim = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=in_channels,     # 对应原 in_channels = 输入通道数
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, feat_dim),
            nn.ReLU()
        )

    def forward(self, x):  # x: (B, T, C)
        output, (hn, _) = self.lstm(x)     # hn: (num_layers, B, hidden_dim)
        last_hidden = hn[-1]               # 取最后一层的隐藏状态: (B, hidden_dim)
        return self.fc(last_hidden)        # 输出 (B, feat_dim)

class RegressorHead(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, feat):
        return self.fc(feat)

class ResidualCVAE(nn.Module):
    def __init__(self, cond_dim, target_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.enc = nn.Sequential(
            nn.Linear(cond_dim + target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def encode(self, cond, residual):
        x = torch.cat([cond, residual], dim=-1)
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond, z):
        x = torch.cat([cond, z], dim=-1)
        return self.dec_fc(x)

    def forward(self, cond, residual):
        mu, logvar = self.encode(cond, residual)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(cond, z)
        return recon, mu, logvar

    def sample_posterior(self, cond, residual, num_samples: int):
        mu, logvar = self.encode(cond, residual)
        std = torch.exp(0.5 * logvar)
        N = mu.size(0)
        device = mu.device
        eps = torch.randn(num_samples, N, self.latent_dim, device=device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        cond_expand = cond.unsqueeze(0).expand(num_samples, N, self.cond_dim).reshape(-1, self.cond_dim)
        z_flat = z.reshape(-1, self.latent_dim)
        out = self.dec_fc(torch.cat([cond_expand, z_flat], dim=-1))
        return out.view(num_samples, N, self.target_dim)

def cvae_loss(recon, residual, mu, logvar):
    recon_loss = torch.mean((recon - residual) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld