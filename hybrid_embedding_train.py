import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.dataset_RIDI import load_ridi_raw
from models.hybrid_imu_embed import HybridIMUEmbed
from utils.fractal_coder import FractalEmbedder
from utils.hybrid_loss import HybridEmbeddingLoss

# ======= 显式配置区 =======
RIDI_ROOT = "/home/admin407/code/zyshe/NavCorrector/RIDI"
ACC_SOURCE = "acce"
WINDOW_SIZE = 256
STRIDE = 256
SAMPLES_PER_EPOCH = 20000
BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-3
CKPT_DIR = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls"

# 训练分支： "nav" 或 "pose"
TRAIN_BRANCH = "nav"

# Fractal 编码配置
NAV_CONFIG = {"theta": (24, 24), "len": (12, 12), "dz": (12, 12)}
POSE_CONFIG = {"yaw": (16, 16), "roll": (16, 16), "pitch": (16, 16)}

NAV_RANGES = {"theta": (-np.pi, np.pi), "len": (0.0, 5.0), "dz": (-1.0, 1.0)}
POSE_RANGES = {"yaw": (-np.pi, np.pi), "roll": (-np.pi, np.pi), "pitch": (-np.pi, np.pi)}


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


def quat_to_euler(q):
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


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


class RandomWindowDataset(Dataset):
    def __init__(self, sequences, window_size=256, stride=256, samples_per_epoch=20000):
        self.samples_per_epoch = samples_per_epoch
        self.window_size = window_size
        self.stride = stride
        self.data = []
        for gyro, acc, pos, ori in sequences:
            n = min(len(gyro), len(acc), len(pos), len(ori))
            if n < window_size + stride:
                continue
            self.data.append(
                (
                    gyro[:n].astype(np.float32),
                    acc[:n].astype(np.float32),
                    pos[:n].astype(np.float32),
                    ori[:n].astype(np.float32),
                )
            )
        if len(self.data) == 0:
            raise RuntimeError("No valid sequences for random sampling.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        _ = idx
        seq_idx = random.randint(0, len(self.data) - 1)
        gyro, acc, pos, ori = self.data[seq_idx]
        max_start = len(gyro) - self.window_size - 1
        start = 0 if max_start <= 0 else random.randint(0, max_start)
        end = start + self.window_size

        gyro_win = gyro[start:end]
        acc_win = acc[start:end]
        imu = np.concatenate([gyro_win, acc_win], axis=1).astype(np.float32)
        imu = torch.from_numpy(imu).transpose(0, 1)  # [6, W]

        a = start + self.window_size // 2 - self.stride // 2
        b = start + self.window_size // 2 + self.stride // 2
        a = max(0, min(a, len(pos) - 1))
        b = max(0, min(b, len(pos) - 1))

        if TRAIN_BRANCH == "nav":
            dp_world = pos[b] - pos[a]
            q_abs = ori[b]
            R_abs = quat_to_rotmat(q_abs)
            dp_body = (R_abs.T @ dp_world.reshape(3, 1)).reshape(-1)
            theta = np.arctan2(dp_body[1], dp_body[0]).astype(np.float32)
            length = np.linalg.norm(dp_body[:2]).astype(np.float32)
            dz = dp_body[2].astype(np.float32)
            gt_phy = {
                "theta": torch.tensor(theta, dtype=torch.float32),
                "len": torch.tensor(length, dtype=torch.float32),
                "dz": torch.tensor(dz, dtype=torch.float32),
            }
        else:
            qa = ori[a]
            qb = ori[b]
            q_rel = quat_mul(quat_conj(qa), qb)
            roll, pitch, yaw = quat_to_euler(q_rel)
            gt_phy = {
                "yaw": torch.tensor(yaw, dtype=torch.float32),
                "roll": torch.tensor(roll, dtype=torch.float32),
                "pitch": torch.tensor(pitch, dtype=torch.float32),
            }

        return imu, gt_phy


def load_sequences():
    data_root = os.path.join(RIDI_ROOT, "data")
    list_path = os.path.join(data_root, "list_train_publish_v2.txt")
    if not os.path.exists(list_path):
        raise RuntimeError(f"RIDI list file not found: {list_path}")
    with open(list_path, "r") as f:
        names = [line.strip().split(",")[0] for line in f if line.strip()]
    seqs = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos, ori = load_ridi_raw(seq_dir, acc_source=ACC_SOURCE)
        seqs.append((gyro, acc, pos, ori))
    return seqs


def collate_fn(batch):
    imu_list, phy_list = zip(*batch)
    imu = torch.stack(imu_list, dim=0)
    keys = phy_list[0].keys()
    gt_phy = {k: torch.stack([p[k] for p in phy_list], dim=0) for k in keys}
    return imu, gt_phy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    seqs = load_sequences()
    ds = RandomWindowDataset(
        seqs,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)

    if TRAIN_BRANCH == "nav":
        config = NAV_CONFIG
        ranges = NAV_RANGES
        ckpt_path = os.path.join(CKPT_DIR, "hybrid_nav_best.pth")
    else:
        config = POSE_CONFIG
        ranges = POSE_RANGES
        ckpt_path = os.path.join(CKPT_DIR, "hybrid_pose_best.pth")

    coder = FractalEmbedder(config, ranges=ranges)
    model = HybridIMUEmbed(window_size=WINDOW_SIZE, explicit_dims={k: sum(v) for k, v in config.items()}).to(device)
    loss_fn = HybridEmbeddingLoss(coder=coder, w_phy=1.0, w_aux=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        count = 0
        for imu, gt_phy in dl:
            imu = imu.to(device)
            gt_phy = {k: v.to(device) for k, v in gt_phy.items()}
            out = model(imu)
            loss = loss_fn(out, gt_phy, imu)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * imu.size(0)
            count += imu.size(0)
        avg = total / max(count, 1)
        print(f"[Epoch {ep + 1}] Loss: {avg:.6f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
