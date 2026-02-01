import os
import numpy as np
import torch

from data.dataset_RIDI import load_ridi_raw
from models.hybrid_imu_embed import HybridIMUEmbed, HybridIMUEmbWrapper
from models.imu_seq_transformer import IMUSeqTransformer
from src.inference import PDRInferenceEngine
from utils.visualization import plot_trajectory_comparison

# ======= 显式配置区 =======
RIDI_ROOT = "/home/admin407/code/zyshe/NavCorrector/RIDI"
ACC_SOURCE = "acce"
WINDOW_SIZE = 256
STRIDE = 256
SEQ_LEN = 16

CKPT_DIST = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/hybrid_nav_best.pth"
CKPT_POSE = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/hybrid_pose_best.pth"
CKPT_TRF_DIST = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/transformer_dist.pth"
CKPT_TRF_POSE = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/transformer_pose.pth"

OUTPUT_DIR = "/home/admin407/code/zyshe/NavCorrector/output/stage2_eval"


def load_test_sequences():
    data_root = os.path.join(RIDI_ROOT, "data")
    list_path = os.path.join(data_root, "list_test_publish_v2.txt")
    if not os.path.exists(list_path):
        raise RuntimeError(f"RIDI list file not found: {list_path}")
    with open(list_path, "r") as f:
        names = [line.strip().split(",")[0] for line in f if line.strip()]
    return names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    emb_dist = HybridIMUEmbWrapper(
        HybridIMUEmbed(window_size=WINDOW_SIZE, explicit_dims={"theta": 48, "len": 24, "dz": 24}, latent_dim=32)
    ).to(device)
    emb_pose = HybridIMUEmbWrapper(
        HybridIMUEmbed(window_size=WINDOW_SIZE, explicit_dims={"yaw": 32, "roll": 32, "pitch": 32}, latent_dim=32)
    ).to(device)
    emb_dist.hybrid.load_state_dict(torch.load(CKPT_DIST, map_location=device))
    emb_pose.hybrid.load_state_dict(torch.load(CKPT_POSE, map_location=device))

    dist_model = IMUSeqTransformer(emb=emb_dist, output_dim=4).to(device)
    pose_model = IMUSeqTransformer(emb=emb_pose, output_dim=4).to(device)
    dist_model.load_state_dict(torch.load(CKPT_TRF_DIST, map_location=device))
    pose_model.load_state_dict(torch.load(CKPT_TRF_POSE, map_location=device))
    dist_model.eval()
    pose_model.eval()

    engine = PDRInferenceEngine(
        dist_model=dist_model,
        pose_model=pose_model,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        seq_len=SEQ_LEN,
        device=device,
    )

    seq_names = load_test_sequences()
    for name in seq_names:
        seq_dir = os.path.join(RIDI_ROOT, "data", name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos, ori = load_ridi_raw(seq_dir, acc_source=ACC_SOURCE)
        imu_stream = np.concatenate([gyro, acc], axis=1).astype(np.float32)

        # 使用 GT 初始姿态对齐
        engine.reset(init_pos=pos[0], init_q=ori[0])

        pred_positions = []
        gt_positions = []
        step = 0
        for pred_pos in engine.process_stream(imu_stream):
            idx = min(step * STRIDE, pos.shape[0] - 1)
            gt_positions.append(pos[idx, :2].copy())
            pred_positions.append(pred_pos[:2].copy())
            step += 1

        if len(pred_positions) == 0:
            continue

        traj_gt = np.stack(gt_positions, axis=0)
        traj_pred = np.stack(pred_positions, axis=0)
        # 已用 GT 初始姿态/位置对齐，无需再平移

        errors = np.linalg.norm(traj_pred - traj_gt, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        print(f"[{name}] Traj RMSE: {rmse:.3f} m")

        plot_trajectory_comparison(
            traj_gt,
            traj_gt,
            traj_pred,
            OUTPUT_DIR,
            name,
        )


if __name__ == "__main__":
    main()
