import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.imu_embed import IMUEmbDist, IMUEmbPose, SupConHead
from utils.supcon_loss import PhysicsSupConLoss, PoseSupConLoss

# ======= 显式配置区 =======
# 训练分支配置：可选 "nav" 或 "pose"
TRAIN_BRANCH = "nav"

RIDI_ROOT = "/home/admin407/code/zyshe/NavCorrector/RIDI"
ACC_SOURCE = "acce"  # "acce" 或 "linacce"
WINDOW_SIZE = 256
STRIDE = 128
FEAT_DIM = 128
SAMPLES_PER_EPOCH = 40000
BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
TEMPERATURE = 0.07
TH_LEN = 0.25
TH_ANGLE_COS = 0.99
TH_DZ = 0.2
POSE_ANGLE_TH = 0.05
CKPT_DIR = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls"


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


def quat_to_yaw(q):
    """从四元数提取航向角 (yaw)。输入 q: [4] (w, x, y, z)."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_rotmat(q):
    """四元数转旋转矩阵。输入 q: [4] (w, x, y, z)."""
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
    """
    随机采样 IMU 窗口，返回物理监督向量。
    """

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
        if max_start <= 0:
            start = 0
        else:
            start = random.randint(0, max_start)
        end = start + self.window_size

        # window: [start, start+window)
        gyro_win = gyro[start:end]
        acc_win = acc[start:end]
        imu = np.concatenate([gyro_win, acc_win], axis=1).astype(np.float32)
        imu = torch.from_numpy(imu).transpose(0, 1)  # [6, W]

        # 保持与 stride 关系一致：以窗口中心为参考，a/b 相差 stride
        a = start + self.window_size // 2 - self.stride // 2
        b = start + self.window_size // 2 + self.stride // 2
        a = max(0, min(a, len(pos) - 1))
        b = max(0, min(b, len(pos) - 1))

        # 导航真值：世界位移旋到手机坐标系
        dp_world = pos[b] - pos[a]
        q_abs = ori[b]
        R_abs = quat_to_rotmat(q_abs)
        dp_body = (R_abs.T @ dp_world.reshape(3, 1)).reshape(-1)

        # 物理真值向量 (len, cos, sin, dz)
        len_xy = np.linalg.norm(dp_body[:2]).astype(np.float32)
        dz = dp_body[2].astype(np.float32)
        yaw = np.arctan2(dp_body[1], dp_body[0]).astype(np.float32)
        cos_val = np.cos(yaw).astype(np.float32)
        sin_val = np.sin(yaw).astype(np.float32)
        gt_phy = torch.tensor([len_xy, cos_val, sin_val, dz], dtype=torch.float32)

        # 相对姿态 q_rel = qa^-1 * qb
        qa = ori[a]
        qb = ori[b]
        q_rel = quat_mul(quat_conj(qa), qb)
        q_rel = torch.from_numpy(q_rel.astype(np.float32))
        return imu, gt_phy, q_rel


def load_ridi_sequences(ridi_root, acc_source="acce", use_train=True):
    data_root = os.path.join(ridi_root, "data")
    list_name = "list_train_publish_v2.txt" if use_train else "list_test_publish_v2.txt"
    list_path = os.path.join(data_root, list_name)
    if not os.path.exists(list_path):
        raise RuntimeError(f"RIDI list file not found: {list_path}")

    with open(list_path, "r") as f:
        names = [line.strip().split(",")[0] for line in f if line.strip()]

    sequences = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos, ori = load_ridi_raw(seq_dir, acc_source=acc_source)
        sequences.append((gyro, acc, pos, ori))

    if len(sequences) == 0:
        raise RuntimeError("No RIDI sequences loaded.")
    return sequences


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = load_ridi_sequences(RIDI_ROOT, acc_source=ACC_SOURCE, use_train=True)
    dataset = RandomWindowDataset(
        sequences,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if TRAIN_BRANCH == "nav":
        emb = IMUEmbDist(in_channels=6, feat_dim=FEAT_DIM).to(device)
        head = SupConHead(feat_dim=FEAT_DIM).to(device)
        criterion = PhysicsSupConLoss(
            temperature=TEMPERATURE,
            th_len=TH_LEN,
            th_angle_cos=TH_ANGLE_COS,
            th_dz=TH_DZ,
        ).to(device)
        ckpt_path = os.path.join(CKPT_DIR, "embed_dist_best.pth")
    elif TRAIN_BRANCH == "pose":
        emb = IMUEmbPose(in_channels=6, feat_dim=FEAT_DIM).to(device)
        head = SupConHead(feat_dim=FEAT_DIM).to(device)
        criterion = PoseSupConLoss(
            temperature=TEMPERATURE,
            angle_th=POSE_ANGLE_TH,
        ).to(device)
        ckpt_path = os.path.join(CKPT_DIR, "embed_pose_best.pth")
    else:
        raise ValueError(f"Unknown TRAIN_BRANCH: {TRAIN_BRANCH}")

    optimizer = optim.AdamW(list(emb.parameters()) + list(head.parameters()), lr=LR)

    best_loss = float("inf")
    os.makedirs(CKPT_DIR, exist_ok=True)
    for ep in range(EPOCHS):
        emb.train()
        head.train()
        total_loss = 0.0
        total_count = 0

        for imu, gt_phy, q_rel in loader:
            imu = imu.to(device)
            gt_phy = gt_phy.to(device)
            q_rel = q_rel.to(device)

            feat = emb(imu)
            proj = head(feat)
            if TRAIN_BRANCH == "nav":
                loss = criterion(proj, gt_phy)
            else:
                loss = criterion(proj, q_rel)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(emb.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            bs = imu.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        print(f"[Epoch {ep + 1}] Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(emb.state_dict(), ckpt_path)

    print(f"Best encoder saved to: {ckpt_path}")


if __name__ == "__main__":
    train()
