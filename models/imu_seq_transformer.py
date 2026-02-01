import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, D)
        s_len = x.size(1)
        return x + self.pe[:s_len, :].unsqueeze(0)


class IMUSeqTransformer(nn.Module):
    """
    单流 Transformer：
    - emb: 冻结的 IMUEmbDist / IMUEmbPose
    - 输出维度由 output_dim 决定
    """

    def __init__(self, emb, output_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.emb = emb
        self.emb.eval()
        for p in self.emb.parameters():
            p.requires_grad = False

        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
        )
        self.trf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x_seq, src_key_padding_mask=None):
        # x_seq: (B, S, 6, W)
        bsz, seq_len, ch, win = x_seq.shape
        x_flat = x_seq.reshape(bsz * seq_len, ch, win)  # (B*S, 6, W)
        feat = self.emb(x_flat)  # (B*S, D)
        feat = feat.reshape(bsz, seq_len, -1)  # (B, S, D)
        feat = self.pos_enc(feat)
        out = self.trf(feat, src_key_padding_mask=src_key_padding_mask)
        return self.head(out)
