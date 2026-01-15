import numpy as np
import pandas as pd
import quaternion
from scipy.ndimage import gaussian_filter1d


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围，支持标量或数组。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def moving_average(x, k):
    """简易滑动平均滤波"""
    if k is None or k <= 1:
        return x
    k = int(k)
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=float) / float(k)
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return np.convolve(x, kernel, mode='same')
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.stack([np.convolve(x[:, i], kernel, mode='same') for i in range(x.shape[1])], axis=1)
    return x 


def get_yaw_from_quaternion(q_arr):
    """
    从四元数 [w, x, y, z] 中提取 Yaw 角
    公式: atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    """
    w, x, y, z = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw


def load_oxiod_raw(imu_data_filename, gt_data_filename):
    """加载 OxIOD 原始数据"""
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    # 切片去除开头结尾无效数据
    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]

    pos_data = gt_data[:, 2:5]
    # 四元数顺序：[w, x, y, z]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data


def window_dataset(gyro_data, acc_data, pos_data, ori_data, window_size=160, stride=36, 
                   filter_window=20, smooth_heading=True, heading_sigma=5, 
                   smooth_length=False, length_sigma=5, mode="2d"):
    """
    OXIOD 数据集切窗处理 (End-to-End PDR 适配版)
    
    Returns:
        inputs: [x_gyro, x_acc] (Raw Device Frame)
        labels: [y_q, y_len, y_head_abs, y_head_rel]
                其中 y_q 和 y_head_abs 均已对齐到 "窗口起始Yaw=0" 的坐标系
    """
    
    # 简单的预处理：位置平滑
    pos2d = pos_data[:, :2]
    if filter_window and filter_window > 1:
        pos2d = moving_average(pos2d, filter_window)

    x_gyro = []
    x_acc = []
    
    y_q = []        # 四元数标签 (Aligned)
    y_len = []      # 步长标签
    y_head_abs = [] # 绝对航向标签 (Aligned, rad)
    y_head_rel = [] # 相对航向标签 (rad)
    
    # 记录初始状态供参考
    idx_0 = 0
    a_0 = idx_0 + window_size // 2 - stride // 2
    a_0 = max(0, min(a_0, len(pos2d)-1))
    init_pos = pos2d[a_0, :]
    init_head = 0.0

    max_start = gyro_data.shape[0] - window_size - 1
    
    for i, idx in enumerate(range(0, max_start, stride)):
        # === 1. Input: Raw Data (Device Frame) ===
        # 不做任何旋转，保留最原始的 IMU 数据
        xg = gyro_data[idx + 1: idx + 1 + window_size, :]
        xa = acc_data [idx + 1: idx + 1 + window_size, :]
        
        # === 关键索引 ===
        start_idx = idx + 1
        mid_idx = idx + 1 + window_size // 2
        
        # 边界检查
        mid_idx = max(0, min(mid_idx, len(ori_data)-1))
        start_idx = max(0, min(start_idx, len(ori_data)-1))

        # === 2. 计算基准 Yaw (Base Yaw) ===
        # 以窗口第一帧的真值 Yaw 为基准
        # 我们的目标是让 PoseNet 学习：假设 Start时刻 Yaw=0，现在的 Pose 是什么
        yaw_start = get_yaw_from_quaternion(ori_data[start_idx])
        
        # 构建校准四元数: 绕 Z 轴旋转 -yaw_start
        # q = [cos(theta/2), 0, 0, sin(theta/2)]
        half_neg_yaw = -yaw_start / 2.0
        q_calib = np.quaternion(np.cos(half_neg_yaw), 0, 0, np.sin(half_neg_yaw))

        # === 3. Label: Aligned Quaternion ===
        # 获取窗口中心时刻的真值四元数 (World Frame)
        q_mid_raw = quaternion.from_float_array(ori_data[mid_idx])
        
        # 旋转到 Aligned Frame
        # 注意四元数乘法顺序：q_new = q_rot * q_orig
        q_mid_aligned = q_calib * q_mid_raw
        
        # === 4. Label: Navigation Targets ===
        a = idx + window_size // 2 - stride // 2
        b = idx + window_size // 2 + stride // 2
        
        a = max(0, min(a, len(pos2d)-1))
        b = max(0, min(b, len(pos2d)-1))
        
        pa = pos2d[a, :]
        pb = pos2d[b, :]
        
        # (1) 步长 (弦长) - 坐标系旋转不改变模长
        delta_len = np.linalg.norm(pb - pa)

        # (2) 绝对航向 (Aligned Frame)
        curr_diff = pb - pa
        if np.linalg.norm(curr_diff) < 1e-6:
            abs_heading_world = 0.0 # 静止时暂设为0
        else:
            # 计算 World Frame 下的位移方向
            abs_heading_world = np.arctan2(curr_diff[1], curr_diff[0])
            
        # 将航向对齐到 Aligned Frame (减去起始 Yaw)
        abs_heading_aligned = wrap_angle(abs_heading_world - yaw_start)

        # === 5. Appending ===
        x_gyro.append(xg)
        x_acc.append(xa)
        y_q.append(quaternion.as_float_array(q_mid_aligned))
        y_len.append(np.array([delta_len], dtype=np.float32))
        y_head_abs.append(np.array([abs_heading_aligned], dtype=np.float32))

    # 转为 numpy
    x_gyro = np.array(x_gyro)
    x_acc  = np.array(x_acc)
    y_q    = np.array(y_q) # (N, 4)
    y_len  = np.array(y_len)
    y_head_abs = np.array(y_head_abs)

    # 静止检测与处理
    if len(y_len) > 0 and len(y_head_abs) > 0:
        stationary_mask = np.abs(y_len.flatten()) < 0.01
        y_len[stationary_mask, 0] = 0.0
        # 静止时，位移方向无意义，保持原值或设为0均可
        # 但为了避免 loss 干扰，通常可以让 weight 为 0，或者这里不做特殊处理，依赖数据量

    # 平滑处理
    # 1. 平滑步长
    if smooth_length and len(y_len) > 0:
        y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
        y_len = y_len_smooth.reshape(-1, 1)
    
    # 2. 平滑绝对航向并计算相对航向
    if smooth_heading and len(y_head_abs) > 0:
        flat_head_abs = y_head_abs.flatten()
        unwrapped_head_abs = np.unwrap(flat_head_abs)
        smoothed_unwrapped_abs = gaussian_filter1d(unwrapped_head_abs, sigma=heading_sigma)

        # 计算相对航向 (差分)
        # Rel Heading = Heading_t - Heading_{t-1}
        # 注意：这是 Aligned Frame 下的差分，但角度差分本身就是相对量，
        # 所以它和 World Frame 下的差分是一样的 (常数 offset 被消掉了)
        rel_headings_unwrapped = np.zeros_like(smoothed_unwrapped_abs)
        rel_headings_unwrapped[1:] = np.diff(smoothed_unwrapped_abs)
        y_head_rel = rel_headings_unwrapped.reshape(-1, 1).astype(np.float32)

        # 重缠绝对航向
        y_head_abs_smooth = wrap_angle(smoothed_unwrapped_abs)
        y_head_abs = y_head_abs_smooth.reshape(-1, 1)

    else:
        y_head_rel = np.zeros((len(y_head_abs), 1), dtype=np.float32)
        if len(y_head_abs) > 1:
            unwrapped_abs = np.unwrap(y_head_abs.flatten())
            y_head_rel[1:, 0] = np.diff(unwrapped_abs)

    return [x_gyro, x_acc], [y_q, y_len, y_head_abs, y_head_rel], init_pos, init_head