import math
import torch


class JohnsonCoder:
    """
    N-bit Johnson code (twisted ring), 2N states, adjacent states differ by 1 bit.
    """

    def __init__(self, n_bits):
        if n_bits <= 0:
            raise ValueError("n_bits must be positive.")
        self.n_bits = int(n_bits)
        self.n_states = 2 * self.n_bits

    def encode(self, val_norm):
        """
        val_norm: Tensor in [0, 1), shape (B,) or scalar.
        returns: Tensor of shape (B, N) with 0/1 values.
        """
        if not torch.is_tensor(val_norm):
            val_norm = torch.tensor(val_norm, dtype=torch.float32)
        v = torch.clamp(val_norm, 0.0, 1.0 - 1e-8)
        idx = torch.floor(v * self.n_states).long()  # [0, 2N-1]
        bsz = idx.numel()
        code = torch.zeros(bsz, self.n_bits, device=idx.device, dtype=torch.float32)

        # Johnson sequence: 000..0, 000..1, 00..11, ..., 111..1, 111..0, ..., 100..0
        for i in range(bsz):
            k = int(idx[i].item())
            if k <= self.n_bits:
                ones = k
                if ones > 0:
                    code[i, self.n_bits - ones :] = 1.0
            else:
                ones = self.n_states - k
                if ones > 0:
                    code[i, :ones] = 1.0
        return code


class FractalEmbedder:
    """
    Fractal Johnson encoding with two levels (coarse + fine).
    config: dict like {'theta': [24, 24], 'len': [12, 12], 'dz': [12, 12]}
    ranges: dict like {'theta': (-pi, pi), 'len': (0, 5), 'dz': (-1, 1)}
    """

    def __init__(self, config, ranges=None):
        self.config = config
        self.ranges = ranges or {}
        self.coders = {}
        for key, (n1, n2) in config.items():
            self.coders[key] = (JohnsonCoder(n1), JohnsonCoder(n2))

    def _normalize(self, key, vals):
        if key in self.ranges:
            vmin, vmax = self.ranges[key]
            v = (vals - vmin) / (vmax - vmin + 1e-8)
            return torch.clamp(v, 0.0, 1.0 - 1e-8)
        # fallback: min-max within batch
        vmin = torch.min(vals)
        vmax = torch.max(vals)
        v = (vals - vmin) / (vmax - vmin + 1e-8)
        return torch.clamp(v, 0.0, 1.0 - 1e-8)

    def encode_physical(self, vals_dict):
        """
        vals_dict: dict of tensors (B,) for each key in config.
        returns: (B, total_bits)
        """
        parts = []
        for key, (n1, n2) in self.config.items():
            v = vals_dict[key].view(-1)
            v_norm = self._normalize(key, v)
            # coarse / fine split
            coarse_bins = 2 * n1
            coarse_idx = torch.floor(v_norm * coarse_bins) / coarse_bins
            residue = (v_norm - coarse_idx) * coarse_bins
            c1, c2 = self.coders[key]
            bits1 = c1.encode(coarse_idx)
            bits2 = c2.encode(residue)
            parts.append(torch.cat([bits1, bits2], dim=1))
        out = torch.cat(parts, dim=1)
        return out
