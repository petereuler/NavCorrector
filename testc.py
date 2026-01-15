import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import TensorDataset, DataLoader

from models.end2end_pdr import EndToEndPDR
from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window, get_yaw_from_quaternion
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window

# ================= 配置参数 =================
CONFIG = {
    'window_size': 200,    # 必须与 trainc.py 一致
    'stride': 10,          # 测试时的 stride
    'batch_size': 64,
    'dataset': 'OXIOD',    
    'data_root': '/home/admin407/code/zyshe/NavCorrector/OXIOD', 
    'model_path': 'checkpoints/OXIOD_0115_1311/model_ep100.pth', 
    'gpu_id': 0,
    'test_file_idx': 2,    
    
    # 可视化范围控制
    'vis_start_idx': 0,    
    'vis_end_idx': 200,     
}

device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")

# ================= 核心工具函数 =================

def generate_trajectory_2d(init_p, init_h, delta_l_list, delta_h_list):
    """
    使用步长（Δl）与绝对航向角（ψ）在平面内重建轨迹
    """
    trajectory = [init_p.copy()]
    current_p = init_p.copy()

    delta_l_list = np.squeeze(delta_l_list)
    delta_h_list = np.squeeze(delta_h_list)

    if delta_l_list.ndim == 0: delta_l_list = [delta_l_list]
    if delta_h_list.ndim == 0: delta_h_list = [delta_h_list]

    for dl, abs_h in zip(delta_l_list, delta_h_list):
        if hasattr(dl, 'item'): dl = dl.item()
        if hasattr(abs_h, 'item'): abs_h = abs_h.item()

        # dx = step * cos(global_heading)
        dx = dl * np.cos(abs_h)
        dy = dl * np.sin(abs_h)

        current_p = current_p + np.array([dx, dy])
        trajectory.append(current_p.copy())

    return np.array(trajectory)

def compute_cdf(errors):
    sorted_errors = np.sort(errors)
    probs = 1. * np.arange(len(errors)) / (len(errors) - 1)
    return sorted_errors, probs

# ================= 数据加载 =================

def load_single_sequence(dataset_name, data_root, file_index=0):
    """加载单条序列"""
    if dataset_name == 'OXIOD':
        val_files = [
            os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
            os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
            os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        ]
        gt_files = [f.replace("imu", "vi") for f in val_files]
        if file_index >= len(val_files): file_index = 0
        path = val_files[file_index]
        print(f"Loading: {path}")
        gyro, acc, pos, ori = load_oxiod_raw(path, gt_files[file_index])
        
    elif dataset_name == 'RONIN':
        seen_base = os.path.join(data_root, 'Data', 'seen_subjects_test_set')
        val_dirs = sorted([os.path.join(seen_base, d) for d in os.listdir(seen_base) if os.path.isdir(os.path.join(seen_base, d))])
        if file_index >= len(val_dirs): file_index = 0
        path = val_dirs[file_index]
        print(f"Loading: {path}")
        gyro, acc, pos, ori = load_ronin_raw(path)
    
    elif dataset_name == 'SELFMADE':
        files = []
        for r, d, fns in os.walk(data_root):
            for fn in fns:
                if fn.endswith('.csv'): files.append(os.path.join(r, fn))
        files = sorted(files)
        path = files[-1] 
        print(f"Loading: {path}")
        gyro, acc, pos, ori = load_selfmade_raw(path)

    # 切窗
    inputs, labels, init_pos, _ = oxiod_window(
        gyro, acc, pos, ori, 
        window_size=CONFIG['window_size'], 
        stride=CONFIG['stride']
    )
    
    # 提取 yaw_start (用于恢复全局坐标系)
    yaw_starts = []
    max_start = gyro.shape[0] - CONFIG['window_size'] - 1
    for i, idx in enumerate(range(0, max_start, CONFIG['stride'])):
        start_idx = idx + 1
        start_idx = max(0, min(start_idx, len(ori)-1))
        ys = get_yaw_from_quaternion(ori[start_idx])
        yaw_starts.append(ys)
    
    if len(yaw_starts) > len(labels[0]):
        yaw_starts = yaw_starts[:len(labels[0])]
        
    return inputs, labels, np.array(yaw_starts), init_pos, pos[:, :2] # 返回 init_pos

# ================= 主测试流程 =================

def test():
    # 1. 模型加载
    print(f"Loading model from {CONFIG['model_path']}")
    model = EndToEndPDR(in_dim=6).to(device)
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else:
        print("Model not found!"); return
    model.eval()

    # 2. 数据加载
    (x_gyro, x_acc), labels, gt_yaw_starts, init_pos, gt_full_traj = load_single_sequence(
        CONFIG['dataset'], CONFIG['data_root'], CONFIG['test_file_idx']
    )
    
    if len(x_gyro) == 0: return
    x_batch = np.concatenate([x_gyro, x_acc], axis=-1)
    x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)

    # 3. 推理
    print(f"Inference on {len(x_tensor)} windows...")
    with torch.no_grad():
        outputs = model(x_tensor, use_gt_rotation=False)
        pred_steps = outputs['step'].cpu().numpy().flatten()
        pred_abs_vec = outputs['abs_vec'].cpu().numpy() # [sin, cos]
        pred_abs_std = np.sqrt(outputs['abs_var'].cpu().numpy().flatten())

    # ================= 4. 轨迹生成 (同一起点，同一逻辑) =================
    
    # 准备航向数据 (Global Frame)
    # NavNet输出的是相对StartYaw的偏角，加上StartYaw即为Global Heading
    pred_local_headings = np.arctan2(pred_abs_vec[:, 0], pred_abs_vec[:, 1])
    pred_global_headings = pred_local_headings + gt_yaw_starts
    
    gt_local_headings = labels[2].flatten()
    gt_global_headings = gt_local_headings + gt_yaw_starts
    
    gt_steps_label = labels[1].flatten()

    # [关键] 统一使用 window_dataset 返回的 init_pos 作为起点
    start_p = init_pos
    
    # 生成预测轨迹
    traj_pred = generate_trajectory_2d(
        init_p=start_p, 
        init_h=0.0, 
        delta_l_list=pred_steps, 
        delta_h_list=pred_global_headings
    )

    # 生成真值轨迹 (Sparse)
    traj_gt = generate_trajectory_2d(
        init_p=start_p,
        init_h=0.0,
        delta_l_list=gt_steps_label,
        delta_h_list=gt_global_headings
    )

    # ================= 5. 切片与评估 =================
    
    start = CONFIG['vis_start_idx']
    end = CONFIG['vis_end_idx']
    if end == -1 or end > len(traj_pred):
        end = len(traj_pred)
    
    # 切片
    pred_slice = traj_pred[start:end]
    gt_slice = traj_gt[start:end]
    
    # 属性切片
    slice_steps = slice(start, end-1)
    p_steps_s = pred_steps[slice_steps]
    g_steps_s = gt_steps_label[slice_steps]
    
    p_head_s = pred_global_headings[slice_steps]
    g_head_s = gt_global_headings[slice_steps]
    p_head_std_s = pred_abs_std[slice_steps]
    
    # 计算误差
    min_len = min(len(pred_slice), len(gt_slice))
    pred_slice = pred_slice[:min_len]
    gt_slice = gt_slice[:min_len]
    
    error_vec = np.linalg.norm(gt_slice - pred_slice, axis=1)
    ate_rmse = np.sqrt(np.mean(error_vec**2))
    final_drift = error_vec[-1]
    
    print(f"Range: {start}-{end}, RMSE: {ate_rmse:.4f}m, Drift: {final_drift:.4f}m")

    # ================= 6. 绘图 =================
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3)

    # 1. 轨迹图
    ax1 = fig.add_subplot(gs[:, 0])
    # 画Dense GT作为背景 (可选)
    # ax1.plot(gt_full_traj[:, 0], gt_full_traj[:, 1], 'k-', alpha=0.1, label='Dense GT')
    
    ax1.plot(gt_slice[:, 0], gt_slice[:, 1], 'k-', linewidth=2, label='GT', alpha=0.6)
    ax1.plot(pred_slice[:, 0], pred_slice[:, 1], 'r--', linewidth=2, label='Pred')
    
    ax1.scatter(gt_slice[0, 0], gt_slice[0, 1], c='g', s=100, label='Start')
    ax1.scatter(gt_slice[-1, 0], gt_slice[-1, 1], c='b', marker='X', s=100, label='GT End')
    ax1.scatter(pred_slice[-1, 0], pred_slice[-1, 1], c='r', marker='X', s=100, label='Pred End')
    
    ax1.set_title(f'Trajectory (RMSE: {ate_rmse:.2f}m)')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 坐标分量
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.arange(len(gt_slice))
    ax2.plot(t, gt_slice[:, 0], 'k-', alpha=0.5, label='GT X')
    ax2.plot(t, pred_slice[:, 0], 'r--', alpha=0.8, label='Pred X')
    ax2.plot(t, gt_slice[:, 1], 'b-', alpha=0.5, label='GT Y')
    ax2.plot(t, pred_slice[:, 1], 'm--', alpha=0.8, label='Pred Y')
    ax2.set_title('Coordinate Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 步长
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(g_steps_s, 'k-', alpha=0.6, label='GT')
    ax3.plot(p_steps_s, 'r-', alpha=0.6, label='Pred')
    ax3.set_title('Step Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 航向
    ax4 = fig.add_subplot(gs[0, 2])
    # 简单unwrap绘图
    g_deg = np.degrees(g_head_s)
    p_deg = np.degrees(p_head_s)
    ax4.plot(g_deg, 'k-', alpha=0.6, label='GT')
    ax4.plot(p_deg, 'r--', alpha=0.6, label='Pred')
    
    std_deg = np.degrees(p_head_std_s)
    ax4.fill_between(range(len(p_deg)), p_deg-3*std_deg, p_deg+3*std_deg, color='r', alpha=0.2)
    ax4.set_title('Global Heading')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. CDF
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_err, probs = compute_cdf(error_vec)
    ax5.plot(sorted_err, probs, 'b-', linewidth=2)
    ax5.set_title('Error CDF')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    save_name = f"vis_{CONFIG['dataset']}_fixed_{start}_{end}.png"
    plt.savefig(save_name)
    print(f"Saved to {save_name}")

if __name__ == "__main__":
    test()