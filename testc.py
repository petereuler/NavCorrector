import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from models.end2end_pdr import EndToEndPDR
from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window

# ================= 配置参数 =================
CONFIG = {
    'window_size': 200,    # 必须与 trainc.py 一致
    'stride': 10,          # 测试时可以密集一点，或者保持一致。用于轨迹积分时需注意匹配
    'batch_size': 64,
    'dataset': 'OXIOD',    # OXIOD, RONIN, SELFMADE
    'data_root': '/home/admin407/code/zyshe/NavCorrector/OXIOD', # 修改为你的数据路径
    'model_path': '/home/admin407/code/zyshe/NavCorrector/checkpoints/OXIOD_0115_1252/model_ep100.pth', # 【请修改】你的模型路径
    'gpu_id': 0,
    'test_file_idx': 0     # 在可视化环节，选择第几个验证文件进行画图
}

device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")

# ================= 工具函数 =================

def compute_errors(pred, gt):
    """计算简单的平均绝对误差 (MAE)"""
    return torch.mean(torch.abs(pred - gt)).item()

def angle_error(pred_vec, gt_vec):
    """
    计算两个单位向量的角度误差 (rad)
    pred_vec: (N, 2) [sin, cos]
    gt_vec: (N, 2)
    """
    # Dot product
    dot = torch.sum(pred_vec * gt_vec, dim=1)
    # Clamp for stability
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(dot)).item()

def align_trajectories(gt_traj, pred_traj):
    """
    使用 SVD 对齐两条轨迹 (用于去除全局坐标系偏差，公平对比形状)
    gt_traj: (N, 2)
    pred_traj: (N, 2)
    """
    # 1. 中心化
    gt_center = np.mean(gt_traj, axis=0)
    pred_center = np.mean(pred_traj, axis=0)
    gt_centered = gt_traj - gt_center
    pred_centered = pred_traj - pred_center
    
    # 2. 计算协方差矩阵 H
    H = np.dot(pred_centered.T, gt_centered)
    
    # 3. SVD
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 4. 旋转与平移预测轨迹
    pred_aligned = np.dot(pred_centered, R.T) + gt_center
    return pred_aligned

# ================= 加载单条序列用于可视化 =================

def load_single_sequence(dataset_name, data_root, file_index=0):
    """
    加载验证集中的某一个原始文件，用于生成完整轨迹
    """
    if dataset_name == 'OXIOD':
        # 这里硬编码了验证集列表，需与 training_utils 保持一致或读取配置
        # 为演示简单，我们直接列出几个验证文件
        val_files = [
            os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
            os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
            os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        ]
        gt_files = [f.replace("imu", "vi") for f in val_files]
        if file_index >= len(val_files): file_index = 0
        
        print(f"Loading Test Sequence: {val_files[file_index]}")
        gyro, acc, pos, ori = load_oxiod_raw(val_files[file_index], gt_files[file_index])
        
        # 使用对应的 window 函数切分，stride 必须与训练时的物理位移定义一致
        # 我们训练时 label 是 "stride 长度内的位移"，所以测试积分时也用这个 stride
        # 注意：可视化时我们可以用小一点的 stride 让轨迹更密，但需要对 step 进行缩放吗？
        # 不需要，因为 label 是 "pos[b] - pos[a]"，其中 b-a = stride。
        # 如果改变 stride，模型的 step 预测值物理意义会变（因为它学的是 stride 长度的位移）。
        # **关键**: 必须使用与训练相同的 stride 来切窗，才能直接累加 step 重建轨迹。
        inputs, labels, init_pos, _ = oxiod_window(
            gyro, acc, pos, ori, 
            window_size=CONFIG['window_size'], 
            stride=CONFIG['stride']
        )
        return inputs, labels, init_pos, pos[:, :2] # 返回 GT 完整轨迹用于对比

    elif dataset_name == 'RONIN':
        # RONIN 验证集
        seen_base = os.path.join(data_root, 'Data', 'seen_subjects_test_set')
        val_dirs = sorted([os.path.join(seen_base, d) for d in os.listdir(seen_base) if os.path.isdir(os.path.join(seen_base, d))])
        
        if file_index >= len(val_dirs): file_index = 0
        target_dir = val_dirs[file_index]
        print(f"Loading Test Sequence: {target_dir}")
        
        gyro, acc, pos, ori = load_ronin_raw(target_dir)
        inputs, labels, init_pos, _ = ronin_window(
            gyro, acc, pos, ori, mode='2d',
            window_size=CONFIG['window_size'], 
            stride=CONFIG['stride']
        )
        return inputs, labels, init_pos, pos[:, :2]

    else:
        # Selfmade
        files = []
        for r, d, fns in os.walk(data_root):
            for fn in fns:
                if fn.endswith('.csv'): files.append(os.path.join(r, fn))
        files = sorted(files)
        # 取最后 20%
        split = max(1, int(0.2 * len(files)))
        val_files = files[-split:]
        
        if file_index >= len(val_files): file_index = 0
        target_file = val_files[file_index]
        print(f"Loading Test Sequence: {target_file}")
        
        gyro, acc, pos, ori = load_selfmade_raw(target_file)
        inputs, labels, init_pos, _ = selfmade_window(
            gyro, acc, pos, ori, mode='2d',
            window_size=CONFIG['window_size'], stride=CONFIG['stride']
        )
        return inputs, labels, init_pos, pos[:, :2]


# ================= 主测试流程 =================

def test():
    # 1. 加载模型
    print(f"Loading model from {CONFIG['model_path']} ...")
    model = EndToEndPDR(in_dim=6).to(device)
    
    if os.path.exists(CONFIG['model_path']):
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print("Error: Model file not found!")
        return

    model.eval()

    # 2. 定性分析：轨迹重建 (Trajectory Reconstruction)
    # 加载一条完整的测试序列
    # inputs: [x_gyro, x_acc] (List of numpy arrays)
    # labels: [y_q, y_len, y_abs, y_rel]
    (x_gyro, x_acc), labels, init_pos, gt_full_traj = load_single_sequence(
        CONFIG['dataset'], CONFIG['data_root'], CONFIG['test_file_idx']
    )
    
    # 转换为 Tensor batch
    # Input shape: (N_windows, Window_Size, 6)
    if len(x_gyro) == 0:
        print("Sequence too short!")
        return

    # 拼接 Gyro, Acc
    x_batch = np.concatenate([x_gyro, x_acc], axis=-1) # (N, T, 6)
    # 如果 training_utils 里做了 permute (N,6,T)，这里也要保持一致
    # 假设 dataset 输出的是 (N, T, 3)，拼接后是 (N, T, 6)。
    # 检查 EndToEndPDR 输入要求：forward(x_raw) -> (B, T, 6)
    # 所以不需要 transpose。
    
    x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
    
    print(f"Running inference on {len(x_tensor)} windows...")
    
    with torch.no_grad():
        # 端到端推理
        # Phase 3 模式：use_gt_rotation=False
        outputs = model(x_tensor, use_gt_rotation=False)
        
        # 提取预测值
        pred_steps = outputs['step'].cpu().numpy().flatten() # (N,)
        pred_abs_vec = outputs['abs_vec'].cpu().numpy()      # (N, 2)
        pred_qs = outputs['q'].cpu().numpy()                 # (N, 4) 用于姿态分析
        
    # --- 轨迹积分 (Dead Reckoning) ---
    # Pos_k = Pos_{k-1} + Step_k * Direction_k
    # 初始位置设为 (0,0) 或 init_pos
    pred_traj = [np.array([0.0, 0.0])]
    curr_pos = np.array([0.0, 0.0])
    
    for i in range(len(pred_steps)):
        step_len = pred_steps[i]
        # pred_abs_vec 是 [sin, cos] -> dy, dx
        sin_theta, cos_theta = pred_abs_vec[i]
        
        # dx = step * cos, dy = step * sin
        dx = step_len * cos_theta
        dy = step_len * sin_theta
        
        curr_pos = curr_pos + np.array([dx, dy])
        pred_traj.append(curr_pos.copy())
        
    pred_traj = np.array(pred_traj)
    
    # --- GT 轨迹处理 ---
    # GT 轨迹是原始的密集点，我们需要下采样或者取对应点来对比
    # 或者直接画出原始 GT 轨迹（更准确）
    # 但为了计算 ATE，需要对齐点数。
    # 这里我们简单起见：画原始 GT 轨迹看形状，画预测轨迹看形状。
    
    # 对齐预测轨迹到 GT (SVD)
    # 由于点数不匹配（GT 是 dense 的，Pred 是 sparse window based），
    # 我们只对齐形状用于可视化。
    # 为了严谨，应该从 GT 中提取对应的 segment points。
    # dataset.py 里的 label 生成逻辑是：pb - pa。
    # 我们可以用 cumsum(labels[y_len] * labels[y_abs]) 来重建 GT 的稀疏轨迹
    
    gt_steps = labels[1].flatten()
    gt_headings = labels[2].flatten() # rad
    gt_sparse_traj = [np.array([0.0, 0.0])]
    curr_gt = np.array([0.0, 0.0])
    for i in range(len(gt_steps)):
        l = gt_steps[i]
        h = gt_headings[i]
        dx = l * np.cos(h)
        dy = l * np.sin(h)
        curr_gt = curr_gt + np.array([dx, dy])
        gt_sparse_traj.append(curr_gt.copy())
    gt_sparse_traj = np.array(gt_sparse_traj)
    
    # 现在 gt_sparse_traj 和 pred_traj 点数一一对应，可以对齐
    pred_aligned = align_trajectories(gt_sparse_traj, pred_traj)
    
    # 计算 ATE (Absolute Trajectory Error)
    ate = np.mean(np.linalg.norm(gt_sparse_traj - pred_aligned, axis=1))
    print(f"Sequence ATE: {ate:.4f} m")

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))
    
    # 1. 轨迹对比图
    plt.subplot(1, 2, 1)
    plt.plot(gt_sparse_traj[:, 0], gt_sparse_traj[:, 1], 'k-', label='Ground Truth (Sparse)', alpha=0.7)
    plt.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r--', label='Prediction (Aligned)', linewidth=2)
    plt.title(f"Trajectory Reconstruction (ATE: {ate:.2f}m)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # 2. 航向/步长 分析
    plt.subplot(2, 2, 2)
    plt.plot(gt_steps, label='GT Step', alpha=0.6)
    plt.plot(pred_steps, label='Pred Step', alpha=0.6)
    plt.title("Step Length Estimation")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # 计算 heading 角度用于显示
    pred_yaw = np.arctan2(pred_abs_vec[:, 0], pred_abs_vec[:, 1])
    gt_yaw = gt_headings
    plt.plot(np.degrees(gt_yaw), label='GT Heading', alpha=0.6)
    plt.plot(np.degrees(pred_yaw), label='Pred Heading', alpha=0.6)
    plt.title("Absolute Heading Estimation (deg)")
    plt.legend()
    plt.grid(True)
    
    save_path = f"test_result_{CONFIG['dataset']}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    # 3. 如果需要，可以把 PoseNet 的四元数误差也画出来
    # (Optional) ...

if __name__ == "__main__":
    test()