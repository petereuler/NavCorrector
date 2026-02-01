import random
import numpy as np
import torch
from torch.utils.data import Dataset


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_rotmat(q):
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


class SequenceDataset(Dataset):
    """
    单流训练数据集：
    返回:
        x_seq: [S, 6, W]
        y_seq: [S, D]
        padding_mask: [S] (True=padding)
    """

    def __init__(
        self,
        data_list,
        mode="dist",
        window_size=256,
        stride=64,
        seq_len=64,
        pad_cold_start=True,
        samples_per_epoch=20000,
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.seq_len = seq_len
        self.pad_cold_start = pad_cold_start
        self.samples_per_epoch = samples_per_epoch

        self.seqs = []
        self.token_starts = []
        self.index = []

        for seq in data_list:
            gyro, acc, pos, ori = seq
            imu = np.concatenate([gyro, acc], axis=1).astype(np.float32)
            n = imu.shape[0]
            max_start = n - window_size
            if max_start <= 0:
                continue
            starts = [t for t in range(0, max_start + 1, stride) if t + stride < n]
            if len(starts) == 0:
                continue

            seq_id = len(self.seqs)
            self.seqs.append((imu, pos, ori))
            self.token_starts.append(starts)
            num_tokens = len(starts)

            for s in range(0, num_tokens - seq_len + 1, seq_len):
                self.index.append((seq_id, s, seq_len, 0))

            if pad_cold_start:
                for k in (5, 10, 20):
                    if 0 < k <= num_tokens and k < seq_len:
                        pad_left = seq_len - k
                        self.index.append((seq_id, 0, k, pad_left))

        if len(self.index) == 0:
            raise RuntimeError("No valid sequence clips found.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        seq_id, start_idx, length, pad_left = self.index[random.randint(0, len(self.index) - 1)]
        imu, pos, ori = self.seqs[seq_id]
        starts = self.token_starts[seq_id]

        x_seq = np.zeros((self.seq_len, 6, self.window_size), dtype=np.float32)
        if self.mode == "dist":
            y_seq = np.zeros((self.seq_len, 3), dtype=np.float32)
        else:
            y_seq = np.zeros((self.seq_len, 4), dtype=np.float32)
        padding_mask = np.ones((self.seq_len,), dtype=bool)

        for i in range(length):
            tok = start_idx + i
            t = starts[tok]
            window = imu[t : t + self.window_size].copy()
            window[:, 0:3] = window[:, 0:3] / 2.0
            window[:, 3:6] = window[:, 3:6] / 9.8
            x_seq[pad_left + i] = window.T

            if self.mode == "dist":
                dp_world = pos[t + self.stride] - pos[t]
                q_abs = ori[t + self.stride].astype(np.float32)
                R_abs = quat_to_rotmat(q_abs)
                dp_body = (R_abs.T @ dp_world.reshape(3, 1)).reshape(-1)
                y_seq[pad_left + i] = dp_body.astype(np.float32)
            else:
                qa = ori[t].astype(np.float32)
                qb = ori[t + self.stride].astype(np.float32)
                q_rel = quat_mul(quat_conj(qa), qb)
                y_seq[pad_left + i] = q_rel.astype(np.float32)

            padding_mask[pad_left + i] = False

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y_seq),
            torch.from_numpy(padding_mask),
        )
