import os
import json
import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from RONIN.source.math_util import orientation_to_angles


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def moving_average(x, k):
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
    gyro = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
    acce = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
    ori_src = info.get('ori_source', 'game_rv')
    with h5py.File(os.path.join(seq_path, 'data.hdf5')) as f:
        if ori_src == 'game_rv' and 'synced/game_rv' in f.keys():
            ori = np.copy(f['synced/game_rv'])
        elif 'synced/rv' in f.keys():
            ori = np.copy(f['synced/rv'])
        else:
            ori = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (ts.shape[0], 1))
    ori_q = quaternion.from_float_array(ori)
    rot_imu_to_tango = quaternion.quaternion(*info.get('start_calibration', [1.0, 0.0, 0.0, 0.0]))
    init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
    ori_q = init_rotor * ori_q
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
    ts, feat, vel2, ori, pos = _load_sequence(seq_path)
    gyro = feat[:, :3]
    acc = feat[:, 3:6]
    pos3 = pos
    angles = orientation_to_angles(ori)
    yaw = angles[:, 0]
    
    # RONIN数据集降采样2（每隔一个样本取一个）
    # gyro = gyro[::2]
    # acc = acc[::2]
    # pos3 = pos3[::2]
    # yaw = yaw[::2]
    
    return gyro, acc, pos3, yaw.reshape(-1, 1)


def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode="2d", window_size=200, stride=10, filter_window=10, smooth_heading=True, heading_sigma=1, smooth_length=False, length_sigma=1.0):
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
        y_head_abs = []  # 绝对航向标签
        y_head_rel = []  # 相对航向标签
        
        # init_pos & init_head
        # [修改] 对于绝对航向，我们不需要init_head，因为模型直接预测绝对航向
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))
        init_pos = pos2d[a_0, :]
        
        init_head = 0.0  # [修改] 设为0，不再使用

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
            
            # 2. [修改] 绝对航向：当前步的位移方向
            curr_diff = pb - pa
            # 处理静止情况，防止 NaN
            if np.linalg.norm(curr_diff) < 1e-6:
                abs_heading = 0.0  # 静止时设为0
            else:
                abs_heading = np.arctan2(curr_diff[1], curr_diff[0])
            
            y_len.append(np.array([delta_len], dtype=np.float32))
            y_head_abs.append(np.array([abs_heading], dtype=np.float32))
            # 相对航向将在平滑处理后计算

        x_gyro = np.array(x_gyro)
        x_acc = np.array(x_acc)
        y_len = np.array(y_len)
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
        raise ValueError("RONIN helper only provides 2d windows here")
    else:
        raise ValueError("mode must be '2d' or '3d'")
