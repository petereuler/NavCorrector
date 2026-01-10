import os
import json
import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from RONIN.source.math_util import orientation_to_angles


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围"""
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


def _load_sequence(seq_path):
    """读取 RONIN 数据集的 HDF5 文件 (对齐到 Global Tango Frame)"""
    with open(os.path.join(seq_path, 'info.json')) as f:
        info = json.load(f)
    with h5py.File(os.path.join(seq_path, 'data.hdf5')) as f:
        ts = np.copy(f['synced/time'])
        gyro_uncalib = np.copy(f['synced/gyro_uncalib'])
        acce_uncalib = np.copy(f['synced/acce'])
        tango_pos = np.copy(f['pose/tango_pos'])
        if 'pose/tango_ori' in f.keys():
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
        else:
            init_tango_ori = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    
    # 标定
    gyro = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
    acce = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
    
    # 处理姿态源
    ori_src = info.get('ori_source', 'game_rv')
    with h5py.File(os.path.join(seq_path, 'data.hdf5')) as f:
        if ori_src == 'game_rv' and 'synced/game_rv' in f.keys():
            ori = np.copy(f['synced/game_rv'])
        elif 'synced/rv' in f.keys():
            ori = np.copy(f['synced/rv'])
        else:
            ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (ts.shape[0], 1))
            
    ori_q = quaternion.from_float_array(ori)
    
    # [对齐] IMU -> Tango 坐标系
    rot_imu_to_tango = quaternion.quaternion(*info.get('start_calibration', [1.0, 0.0, 0.0, 0.0]))
    init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
    ori_q = init_rotor * ori_q
    
    # 旋转 IMU 数据
    nz = np.zeros((gyro.shape[0], 1))
    gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
    glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
    glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
    
    dt = (ts[1:] - ts[:-1])[:, None]
    glob_v = (tango_pos[1:] - tango_pos[:-1]) / dt
    ts = ts[1:]
    
    return ts, np.concatenate([glob_gyro[1:], glob_acce[1:]], axis=1), glob_v[:, :2], quaternion.as_float_array(ori_q)[1:], tango_pos[1:]


def load_ronin_raw(seq_path):
    """加载 RONIN 原始数据"""
    ts, feat, vel2, ori, pos = _load_sequence(seq_path)
    gyro = feat[:, :3]
    acc = feat[:, 3:6]
    pos3 = pos
    
    # 提取 Yaw (从已对齐的四元数中提取)
    # 这代表了设备在 Global Tango Frame 下的真实朝向
    angles = orientation_to_angles(ori)
    yaw = angles[:, 0]
    
    return gyro, acc, pos3, yaw.reshape(-1, 1)


def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode="2d", window_size=200, stride=10, filter_window=10, smooth_heading=True, heading_sigma=1, smooth_length=False, length_sigma=1.0):
    """构建窗口化数据集 (修正静止航向问题)"""
    mid = window_size // 2 - stride // 2
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    if mode == "2d":
        pos2d = pos_data[:, :2]
        if filter_window and filter_window > 1:
            pos2d = moving_average(pos2d, filter_window)
        
        x_gyro = []
        x_acc = []
        y_len = []
        y_head_abs = []  # 绝对航向
        y_head_rel = []  # 相对航向
        
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        a_0 = max(0, min(a_0, len(pos2d)-1))
        init_pos = pos2d[a_0, :]
        init_head = 0.0

        max_start = gyro_data.shape[0] - window_size - 1
        for i, idx in enumerate(range(0, max_start, stride)):
            xg = gyro_data[idx + 1: idx + 1 + window_size, :]
            xa = acc_data[idx + 1: idx + 1 + window_size, :]
            x_gyro.append(xg)
            x_acc.append(xa)
            
            a = idx + window_size // 2 - stride // 2
            b = idx + window_size // 2 + stride // 2
            a = max(0, min(a, len(pos2d)-1))
            b = max(0, min(b, len(pos2d)-1))
            
            pa = pos2d[a, :]
            pb = pos2d[b, :]
            
            # 1. 步长 (弦长)
            delta_len = np.linalg.norm(pb - pa)
            
            # 2. [关键修正] 绝对航向计算
            curr_diff = pb - pa
            displacement = np.linalg.norm(curr_diff)
            
            # 阈值判断：如果位移小于 2cm，认为静止
            # 此时位移方向(Course)全是噪声，必须使用真实Yaw(Heading)代替
            if displacement < 0.1: 
                # 获取当前窗口中间时刻的索引
                center_k = (a + b) // 2
                center_k = max(0, min(center_k, len(ori_data)-1))
                # 使用真实 Yaw 作为标签 (保持朝向)
                abs_heading = ori_data[center_k, 0]
            else:
                # 运动状态：使用位移方向
                abs_heading = np.arctan2(curr_diff[1], curr_diff[0])
            
            y_len.append(np.array([delta_len], dtype=np.float32))
            y_head_abs.append(np.array([abs_heading], dtype=np.float32))

        x_gyro = np.array(x_gyro)
        x_acc = np.array(x_acc)
        y_len = np.array(y_len)
        y_head_abs = np.array(y_head_abs)

        # 数据清洗：静止处理
        if len(y_len) > 0 and len(y_head_abs) > 0:
            # 标记静止帧
            stationary_mask = np.abs(y_len.flatten()) < 0.1
            
            # 1. 步长设为 0 (正确)
            y_len[stationary_mask, 0] = 0.0
            
            # 2. [重要] 不要把航向设为 0！
            # 删除了 y_head_abs[stationary_mask, 0] = 0.0 这行有害代码
            # 因为我们在上面循环里已经填入了正确的 ori_data 真值

        # 平滑步长
        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)
        
        # 平滑航向角 (解决 Course 和 Heading 切换时的微小突变)
        if smooth_heading and len(y_head_abs) > 0:
            flat_head_abs = y_head_abs.flatten()
            unwrapped_head_abs = np.unwrap(flat_head_abs)
            smoothed_unwrapped_abs = gaussian_filter1d(unwrapped_head_abs, sigma=heading_sigma)
            
            rel_headings_unwrapped = np.zeros_like(smoothed_unwrapped_abs)
            rel_headings_unwrapped[1:] = np.diff(smoothed_unwrapped_abs)

            y_head_abs_smooth = wrap_angle(smoothed_unwrapped_abs)
            y_head_abs = y_head_abs_smooth.reshape(-1, 1)
            y_head_rel = rel_headings_unwrapped.reshape(-1, 1).astype(np.float32)
        else:
            y_head_rel = np.zeros((len(y_head_abs), 1), dtype=np.float32)
            if len(y_head_abs) > 1:
                unwrapped_abs = np.unwrap(y_head_abs.flatten())
                y_head_rel[1:, 0] = np.diff(unwrapped_abs)

        return [x_gyro, x_acc], [y_len, y_head_abs, y_head_rel], init_pos, init_head

    elif mode == "3d":
        raise ValueError("RONIN helper only provides 2d windows here")
    else:
        raise ValueError("mode must be '2d' or '3d'")