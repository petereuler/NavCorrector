import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.dataset_RIDI import load_ridi_raw, window_dataset as ridi_window
from models.imu_embed import IMUEmbDist, IMUEmbPose

# ======= 显式配置区 =======
RIDI_ROOT = "/home/admin407/code/zyshe/NavCorrector/RIDI"
ACC_SOURCE = "acce"
WINDOW_SIZE = 256
STRIDE = 256
SAMPLES = 20000
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAV_ENCODER_CKPT = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/embed_dist_best.pth"
POSE_ENCODER_CKPT = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/embed_pose_best.pth"

PROBE_EPOCHS = 30
PROBE_LR = 1e-3

KNN_K = 20
KNN_TH_LEN = 0.1
KNN_TH_ANGLE = 8.0 * np.pi / 180.0
KNN_TH_DZ = 0.05

OUTPUT_DIR = "/home/admin407/code/zyshe/NavCorrector/output/stage1_eval"


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


def quat_angle(q1, q2):
    q1 = q1 / (np.linalg.norm(q1, axis=1, keepdims=True) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2, axis=1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return 2.0 * np.arccos(dot)


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
    if len(seqs) == 0:
        raise RuntimeError("No sequences loaded.")
    return seqs


def build_samples(seqs, max_samples=SAMPLES):
    imu_list = []
    phy_list = []
    qrel_list = []
    for gyro, acc, pos, ori in seqs:
        [xg, xa], [_ylen, _yhead, _yabs, yori, yrel, ydp], _, _ = ridi_window(
            gyro,
            acc,
            pos,
            ori,
            mode="2d",
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            filter_window=0,
            smooth_heading=False,
            smooth_length=False,
            return_abs_heading=True,
            return_ori=True,
            return_rel_ori=True,
            return_delta_p=True,
            abs_heading_from_ori=False,
            align_heading_to_init_pose=False,
        )
        if xg.shape[0] == 0:
            continue
        for i in range(xg.shape[0]):
            gyro_win = xg[i]
            acc_win = xa[i]
            imu = np.concatenate([gyro_win, acc_win], axis=1).astype(np.float32)
            imu_list.append(imu.T)  # (6, W)

            dp_world = ydp[i]
            q_abs = yori[i]
            # dp_body = R_abs^T * dp_world
            w, x, y, z = q_abs
            ww, xx, yy, zz = w * w, x * x, y * y, z * z
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z
            R = np.array(
                [
                    [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
                    [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
                    [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
                ],
                dtype=np.float32,
            )
            dp_body = (R.T @ dp_world.reshape(3, 1)).reshape(-1)
            yaw = np.arctan2(dp_body[1], dp_body[0]).astype(np.float32)
            phy = np.array(
                [
                    np.linalg.norm(dp_body[:2]).astype(np.float32),
                    np.cos(yaw).astype(np.float32),
                    np.sin(yaw).astype(np.float32),
                    dp_body[2].astype(np.float32),
                ],
                dtype=np.float32,
            )
            phy_list.append(phy)
            qrel_list.append(yrel[i].astype(np.float32))

            if len(imu_list) >= max_samples:
                break
        if len(imu_list) >= max_samples:
            break

    imu_arr = np.stack(imu_list, axis=0)
    phy_arr = np.stack(phy_list, axis=0)
    qrel_arr = np.stack(qrel_list, axis=0)
    return imu_arr, phy_arr, qrel_arr


def linear_probe_nav(emb, phy):
    n = emb.shape[0]
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]

    x_tr = torch.from_numpy(emb[tr]).to(DEVICE)
    y_tr = torch.from_numpy(phy[tr]).to(DEVICE)
    x_va = torch.from_numpy(emb[va]).to(DEVICE)
    y_va = torch.from_numpy(phy[va]).to(DEVICE)

    head = nn.Linear(x_tr.shape[1], 4).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR)

    for _ in range(PROBE_EPOCHS):
        head.train()
        pred = head(x_tr)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(x_va)
        mse = F.mse_loss(pred, y_va).item()
    return mse


def linear_probe_pose(emb, qrel):
    n = emb.shape[0]
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]

    x_tr = torch.from_numpy(emb[tr]).to(DEVICE)
    y_tr = torch.from_numpy(qrel[tr]).to(DEVICE)
    x_va = torch.from_numpy(emb[va]).to(DEVICE)
    y_va = torch.from_numpy(qrel[va]).to(DEVICE)

    head = nn.Linear(x_tr.shape[1], 4).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR)

    for _ in range(PROBE_EPOCHS):
        head.train()
        pred = F.normalize(head(x_tr), dim=1, eps=1e-8)
        dot = torch.abs(torch.sum(pred * y_tr, dim=1))
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        loss = torch.mean(2.0 * torch.acos(dot))
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = F.normalize(head(x_va), dim=1, eps=1e-8)
        dot = torch.abs(torch.sum(pred * y_va, dim=1))
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        ang = torch.mean(2.0 * torch.acos(dot)).item()
    return ang


def knn_eval_nav(emb, phy):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T
    n = emb.shape[0]
    idx = np.argsort(-sim, axis=1)[:, 1 : KNN_K + 1]

    len_diff = []
    dz_diff = []
    ang_diff = []
    hit = []
    for i in range(n):
        nbr = idx[i]
        len_i, cos_i, sin_i, dz_i = phy[i]
        v_i = np.array([cos_i, sin_i], dtype=np.float32)
        for j in nbr:
            len_j, cos_j, sin_j, dz_j = phy[j]
            v_j = np.array([cos_j, sin_j], dtype=np.float32)
            len_diff.append(np.abs(len_i - len_j))
            dz_diff.append(np.abs(dz_i - dz_j))
            dot = np.clip(np.dot(v_i, v_j), -1.0, 1.0)
            ang = np.arccos(dot)
            ang_diff.append(ang)
            hit.append(
                (np.abs(len_i - len_j) < KNN_TH_LEN)
                and (ang < KNN_TH_ANGLE)
                and (np.abs(dz_i - dz_j) < KNN_TH_DZ)
            )
    return float(np.mean(len_diff)), float(np.mean(ang_diff)), float(np.mean(dz_diff)), float(np.mean(hit))


def knn_eval_pose(emb, qrel):
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T
    n = emb.shape[0]
    idx = np.argsort(-sim, axis=1)[:, 1 : KNN_K + 1]
    ang_diff = []
    for i in range(n):
        nbr = idx[i]
        q_i = qrel[i : i + 1]
        q_j = qrel[nbr]
        ang = quat_angle(q_i.repeat(len(nbr), axis=0), q_j)
        ang_diff.append(np.mean(ang))
    return float(np.mean(ang_diff))


def plot_pca(emb, color, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x = emb - emb.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    p = u[:, :2] * s[:2]
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(p[:, 0], p[:, 1], c=color, s=3, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pca.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seqs = load_sequences()
    imu, phy, qrel = build_samples(seqs)

    device = torch.device(DEVICE)

    # Nav encoder eval
    nav_enc = IMUEmbDist(in_channels=6, feat_dim=128).to(device)
    nav_enc.load_state_dict(torch.load(NAV_ENCODER_CKPT, map_location=device))
    nav_enc.eval()
    with torch.no_grad():
        emb = []
        for i in range(0, imu.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(imu[i : i + BATCH_SIZE]).to(device)
            emb.append(nav_enc(xb).cpu().numpy())
        emb = np.concatenate(emb, axis=0)
    nav_probe = linear_probe_nav(emb, phy)
    nav_knn = knn_eval_nav(emb, phy)
    plot_pca(emb, phy[:, 0], "nav_len")
    print(f"[Nav] Linear Probe MSE: {nav_probe:.6f}")
    print(f"[Nav] KNN mean |len| diff: {nav_knn[0]:.4f}, mean ang diff(rad): {nav_knn[1]:.4f}, mean |dz| diff: {nav_knn[2]:.4f}")
    print(f"[Nav] KNN hit ratio: {nav_knn[3]:.4f}")

    # Pose encoder eval
    pose_enc = IMUEmbPose(in_channels=6, feat_dim=128).to(device)
    pose_enc.load_state_dict(torch.load(POSE_ENCODER_CKPT, map_location=device))
    pose_enc.eval()
    with torch.no_grad():
        emb = []
        for i in range(0, imu.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(imu[i : i + BATCH_SIZE]).to(device)
            emb.append(pose_enc(xb).cpu().numpy())
        emb = np.concatenate(emb, axis=0)
    pose_probe = linear_probe_pose(emb, qrel)
    pose_knn = knn_eval_pose(emb, qrel)
    plot_pca(emb, quat_angle(qrel, qrel), "pose_dummy")
    print(f"[Pose] Linear Probe Angle Loss(rad): {pose_probe:.6f}")
    print(f"[Pose] KNN mean angle diff(rad): {pose_knn:.6f}")


if __name__ == "__main__":
    main()
