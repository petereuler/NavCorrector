import numpy as np
import pandas as pd
import quaternion
from scipy.ndimage import gaussian_filter1d


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围，支持标量或数组。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotate_to_global(acc_data, gyro_data, ori_data):
    """
    利用四元数将加速度计和陀螺仪数据从设备坐标系旋转到世界坐标系

    参数:
    - acc_data: 加速度数据 (N, 3)，设备坐标系
    - gyro_data: 陀螺仪数据 (N, 3)，设备坐标系
    - ori_data: 姿态四元数 (N, 4)，格式为 [w, x, y, z]

    返回:
    - acc_global: 世界坐标系下的加速度 (N, 3)
    - gyro_global: 世界坐标系下的角速度 (N, 3)
    """
    acc_global = []
    gyro_global = []

    for i in range(len(acc_data)):
        # 获取当前时刻的四元数
        q = quaternion.from_float_array(ori_data[i])

        # 计算旋转矩阵 (从设备坐标系到世界坐标系)
        rotation_matrix = quaternion.as_rotation_matrix(q)

        # 旋转加速度向量
        acc_local = acc_data[i]
        acc_world = rotation_matrix @ acc_local
        acc_global.append(acc_world)

        # 旋转角速度向量 (注意：角速度的转换需要考虑四元数的共轭)
        gyro_local = gyro_data[i]
        gyro_world = rotation_matrix @ gyro_local
        gyro_global.append(gyro_world)

    return np.array(acc_global), np.array(gyro_global)

def moving_average(x, k):
    """简易滑动平均滤波，窗口 k>=1；k<=1 时原样返回。"""
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

def quaternion_to_euler(q):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    参数:
    - q (np.ndarray or quaternion.quaternion): 四元数，形状为 (4,)
    
    返回:
    - euler (np.ndarray): 对应的欧拉角 [roll, pitch, yaw]，单位是弧度
    """
    q = quaternion.from_float_array(q) if isinstance(q, np.ndarray) else q
    rotation_matrix = quaternion.as_rotation_matrix(q)
    
    # 从旋转矩阵提取欧拉角
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # roll (旋转绕X轴)
    pitch = np.arcsin(-rotation_matrix[2, 0])  # pitch (旋转绕Y轴)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # yaw (旋转绕Z轴)
    
    return np.array([roll, pitch, yaw])

def yaw_from_quaternion_array(ori_array):
    yaws = []
    for q in ori_array:
        e = quaternion_to_euler(q)
        yaws.append(e[2])
    return np.array(yaws)

def load_oxiod_raw(imu_data_filename, gt_data_filename):
    """
    加载 OxIOD 原始数据：IMU(gyro/acc) 与 GT 位置/姿态（3D）。

    参数:
    - imu_data_filename: IMU 数据的文件路径
    - gt_data_filename: 地面真实数据的文件路径

    返回:
    - gyro_data: 陀螺仪数据 (N, 3)
    - acc_data: 加速度数据 (N, 3)
    - pos_data: 位置数据 (N, 3)
    - ori_data: 姿态（四元数 [w, x, y, z]）(N, 4)
    """
    # 去除表头，防止训练中epoch第一轮读取表头
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    # 对数据进行切片以去除开头和结尾的无效数据
    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]

    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)  # 得到四元数顺序：[w, x, y, z]

    return gyro_data, acc_data, pos_data, ori_data

def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode = "2d", window_size = 160, stride = 36, filter_window = 20, smooth_heading = True, heading_sigma = 5, smooth_length = False, length_sigma = 5):
    mid = window_size // 2 - stride // 2
    if mode == "2d":
        # [新增] 将IMU数据旋转到世界坐标系
        acc_data, gyro_data = rotate_to_global(acc_data, gyro_data, ori_data)

        pos2d = pos_data[:, :2]
        if filter_window and filter_window > 1:
            pos2d = moving_average(pos2d, filter_window)

        x_gyro = []
        x_acc = []
        y_len = []
        y_head_abs = []  # 绝对航向标签
        y_head_rel = []  # 相对航向标签
        
        # 初始化
        # [修改] 使用绝对航向而非相对航向变化
        # 每个样本的标签是当前步的绝对位移方向

        # init_pos 取第一个窗口的起点 (a)
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        # 确保索引安全
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))

        init_pos = pos2d[a_0, :]

        # [修改] 对于绝对航向，我们不需要init_head，因为模型直接预测绝对航向
        init_head = 0.0  # 设为0，不再使用

        max_start = gyro_data.shape[0] - window_size - 1
        for i, idx in enumerate(range(0, max_start, stride)):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data [idx + 1: idx + 1 + window_size, :]
            
            x_gyro.append(xg)
            x_acc .append(xa)

            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2
            
            # 索引边界保护
            a = max(0, min(a, len(pos2d)-1))
            b = max(0, min(b, len(pos2d)-1))
            
            pa = pos2d[a, :]
            pb = pos2d[b, :]
            
            # 1. 步长 (弦长)
            delta_len = np.linalg.norm(pb - pa)

            # 2. [修改] 绝对航向：当前步的位移方向
            curr_diff = pb - pa
            # 处理静止情况，防止 NaN
            if np.linalg.norm(curr_diff) < 1e-6:
                abs_heading = 0.0  # 静止时设为0
            else:
                abs_heading = np.arctan2(curr_diff[1], curr_diff[0])

            y_len .append(np.array([delta_len], dtype=np.float32))
            y_head_abs.append(np.array([abs_heading], dtype=np.float32))
            # 相对航向将在平滑处理后计算

        x_gyro = np.array(x_gyro)
        x_acc  = np.array(x_acc)
        y_len  = np.array(y_len)
        y_head_abs = np.array(y_head_abs)

        # 在平滑之前进行数据清洗：基于步长判断静止状态
        # 如果步长绝对值小于阈值，说明处于静止状态，将步长和航向角都设为0
        if len(y_len) > 0 and len(y_head_abs) > 0:
            # 基于步长判断是否静止
            stationary_mask = np.abs(y_len.flatten()) < 0.01  # 步长小于1cm认为静止

            # 将静止状态的样本标签设为0
            y_len[stationary_mask, 0] = 0.0
            y_head_abs[stationary_mask, 0] = 0.0

        # 对步长进行平滑处理（提高真值轨迹的光滑性）
        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)
        
        # 对绝对航向进行平滑处理，并计算相对航向
        if smooth_heading and len(y_head_abs) > 0:
            flat_head_abs = y_head_abs.flatten()

            # 1. 解缠 (Unwrap): 消除 +/- pi 的跳变
            unwrapped_head_abs = np.unwrap(flat_head_abs)

            # 2. 平滑 (Smooth): 在连续空间进行高斯滤波
            smoothed_unwrapped_abs = gaussian_filter1d(unwrapped_head_abs, sigma=heading_sigma)

            # 3. 计算相对航向：在unwrap空间计算差分
            # 注意：第一个元素没有前一个值，设为0
            rel_headings_unwrapped = np.zeros_like(smoothed_unwrapped_abs)
            rel_headings_unwrapped[1:] = np.diff(smoothed_unwrapped_abs)

            # 4. 重缠绝对航向 (Rewrap): 变回 [-pi, pi] 范围
            y_head_abs_smooth = wrap_angle(smoothed_unwrapped_abs)
            y_head_abs = y_head_abs_smooth.reshape(-1, 1)

            # 5. 相对航向保持在连续空间（无需rewrap，因为是差分）
            y_head_rel = rel_headings_unwrapped.reshape(-1, 1).astype(np.float32)
        else:
            # 如果不平滑，直接计算相对航向
            y_head_rel = np.zeros((len(y_head_abs), 1), dtype=np.float32)
            if len(y_head_abs) > 1:
                # 在unwrap空间计算差分
                unwrapped_abs = np.unwrap(y_head_abs.flatten())
                y_head_rel[1:, 0] = np.diff(unwrapped_abs)

        return [x_gyro, x_acc], [y_len, y_head_abs, y_head_rel], init_pos, init_head

    elif mode == "3d":
        mid = window_size // 2 - stride // 2
        init_pos = pos_data[mid, :]
        init_euler = quaternion_to_euler(ori_data[mid, :])

        x_gyro = []
        x_acc = []
        y_delta_p = []
        y_delta_euler = []

        max_start = gyro_data.shape[0] - window_size - 1
        for idx in range(0, max_start, stride):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data [idx + 1: idx + 1 + window_size, :]
            
            x_gyro.append(xg)
            x_acc .append(xa)

            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2

            p_a = pos_data[a, :]
            p_b = pos_data[b, :]
            q_a = quaternion.from_float_array(ori_data[a, :])
            q_b = quaternion.from_float_array(ori_data[b, :])

            rotation_matrix = quaternion.as_rotation_matrix(q_a)
            delta_p = rotation_matrix.T @ (p_b - p_a)

            e_a = quaternion_to_euler(q_a)
            e_b = quaternion_to_euler(q_b)
            delta_euler = wrap_angle(e_b - e_a)

            y_delta_p.append(delta_p)
            y_delta_euler.append(delta_euler)

        x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
        x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
        y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
        y_delta_euler = np.reshape(y_delta_euler, (len(y_delta_euler), y_delta_euler[0].shape[0]))

        return [x_gyro, x_acc], [y_delta_p, y_delta_euler], init_pos, init_euler

    else:
        raise ValueError("mode must be '2d' or '3d'")