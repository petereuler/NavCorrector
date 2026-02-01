import torch
import torch.nn.functional as F


class HybridEmbeddingLoss(torch.nn.Module):
    """
    Hybrid loss:
    - Explicit BCE vs fractal-encoded bits
    - Implicit MSE for reconstruction
    """

    def __init__(self, coder, w_phy=1.0, w_aux=0.5):
        super().__init__()
        self.coder = coder
        self.w_phy = w_phy
        self.w_aux = w_aux

    def forward(self, model_output, gt_phy, raw_imu):
        emb, split, recon = model_output

        # Explicit BCE loss
        target_bits = self.coder.encode_physical(gt_phy)  # (B, 96)
        pred_bits = []
        for key in self.coder.config.keys():
            pred_bits.append(split[key])
        pred_bits = torch.cat(pred_bits, dim=1)
        loss_phy = F.binary_cross_entropy(pred_bits, target_bits)

        # Implicit auxiliary MSE loss
        if recon is None:
            loss_aux = torch.tensor(0.0, device=emb.device)
        else:
            raw_flat = raw_imu.reshape(raw_imu.size(0), -1)
            loss_aux = F.mse_loss(recon, raw_flat)

        return self.w_phy * loss_phy + self.w_aux * loss_aux
