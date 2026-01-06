"""
结果处理和评估工具函数
包含用于测试结果处理、分析和保存的函数
"""

import os
import numpy as np
import pandas as pd


def compute_path_length(traj: np.ndarray) -> float:
    """计算轨迹的总路径长度"""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def reconstruct_from_absolute_angles(init_pos, step_lengths, absolute_angles):
    """
    根据步长和绝对角度重建轨迹
    
    Args:
        init_pos: 初始位置 (2,)
        step_lengths: 步长序列
        absolute_angles: 绝对角度序列
    
    Returns:
        traj: 重建的轨迹 (N, 2)
    """
    traj = [init_pos]
    curr_pos = init_pos.copy()
    # 确保长度一致
    n = min(len(step_lengths), len(absolute_angles))
    for i in range(n):
        l = step_lengths[i]
        theta = absolute_angles[i]
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        curr_pos[0] += dx
        curr_pos[1] += dy
        traj.append(curr_pos.copy())
    return np.array(traj)




def extract_ground_truth_positions(pos3d, window_size, stride, num_windows, start_index=None):
    """
    修正版：提取与预测步长严格对齐的真值位置
    """
    pos2d = pos3d[:, :2]
    
    # 如果没有指定 start_index，则根据 dataset_OXIOD 的逻辑计算默认偏移量
    if start_index is None:
        # 必须与 dataset_OXIOD.py 中的逻辑一致: mid = window//2 - stride//2
        mid = window_size // 2 - stride // 2
        start_index = mid
    
    # 1. 放入起始点 (对应预测轨迹的 init_pos)
    # 注意：这里我们直接从 pos2d 取真值，而不是用传入的 init_l，保证是绝对真值
    if start_index >= len(pos2d):
        return np.array([])
        
    gt_positions = [pos2d[start_index]] 
    
    # 2. 提取后续点
    # 预测的第 i 步是从 start_index + i*stride 到 start_index + (i+1)*stride
    # 所以轨迹点应该是序列: start, start+stride, start+2*stride...
    
    for i in range(num_windows):
        # 下一个点的位置索引
        frame_idx = start_index + (i + 1) * stride
        
        if frame_idx < len(pos2d):
            gt_positions.append(pos2d[frame_idx])
        else:
            break
            
    return np.array(gt_positions)

def save_results_to_csv(gt_vis, pred_vis, traj_pdr, dl, dh, pred_len, pred_head,
                       vis_num, base_name, output_dir):
    """
    保存测试结果到CSV文件
    
    Args:
        gt_vis: 真值轨迹可视化数据
        pred_vis: 预测轨迹可视化数据
        traj_pdr: PDR轨迹
        dl: 真值步长变化量
        dh: 真值航向角变化量
        pred_len: 预测步长
        pred_head: 预测航向角
        vis_num: 可视化数量
        base_name: 基础文件名
        output_dir: 输出目录
    """
    df_traj = pd.DataFrame({
        "step": np.arange(len(gt_vis)),
        "gt_x": gt_vis[:, 0],
        "gt_y": gt_vis[:, 1],
        "pred_x": pred_vis[:, 0],
        "pred_y": pred_vis[:, 1],
    })
    df_traj.to_csv(os.path.join(output_dir, f"{base_name}_trajectory.csv"), index=False)

    df_pdr_traj = pd.DataFrame({
        "step": np.arange(len(traj_pdr)),
        "pdr_x": traj_pdr[:, 0],
        "pdr_y": traj_pdr[:, 1],
    })
    df_pdr_traj.to_csv(os.path.join(output_dir, f"{base_name}_pdr_trajectory.csv"), index=False)

    df_time = pd.DataFrame({
        "step": np.arange(vis_num),
        "gt_dl": dl[:vis_num, 0],
        "pred_dl": pred_len[:vis_num, 0],
        "gt_dh": dh[:vis_num, 0],
        "pred_dh": pred_head[:vis_num, 0],
    })
    df_time.to_csv(os.path.join(output_dir, f"{base_name}_time_series.csv"), index=False)

