import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
from scipy.io import loadmat


def wrap_angle(angle):
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


def load_selfmade_raw(imu_or_all_path, gt_path=None, crop_head=0, crop_tail=0):
    """加载自制数据集 (支持 .mat 和 .csv)"""
    ext = os.path.splitext(imu_or_all_path)[1].lower()
    if ext == '.mat':
        mat = loadmat(imu_or_all_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        arr = None
        for k in keys:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.dtype.kind in ('f', 'i'):
                arr = v
                break
        if arr is None:
            raise RuntimeError('No numeric array found in MAT file')
        a = np.squeeze(arr)
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[0] == 11:
            channels = a
        elif a.shape[1] == 11:
            channels = a.T
        elif a.shape[0] > 11:
            channels = a[:11, :]
        elif a.shape[1] > 11:
            channels = a[:, :11].T
        else:
            raise RuntimeError('MAT array does not contain 11 channels')
        gyro = channels[0:3, :].T
        acc = channels[3:6, :].T
        heading_deg = channels[6, :]
        x = channels[9, :]
        y = channels[10, :]
    else:
        df_imu = pd.read_csv(imu_or_all_path)
        df_gt = None
        if gt_path is not None and gt_path != imu_or_all_path:
            df_gt = pd.read_csv(gt_path)
        if df_gt is None:
            values = df_imu.values
            gyro = values[:, 0:3]
            acc = values[:, 3:6]
            heading_deg = values[:, 6]
            x = values[:, 9]
            y = values[:, 10]
        else:
            v_imu = df_imu.values
            v_gt = df_gt.values
            gyro = v_imu[:, 0:3]
            acc = v_imu[:, 3:6]
            heading_deg = v_gt[:, 0]
            x = v_gt[:, 1]
            y = v_gt[:, 2]
    if crop_head or crop_tail:
        h = int(max(0, crop_head))
        t = int(max(0, crop_tail))
        if h > 0:
            gyro = gyro[h:]
            acc = acc[h:]
            heading_deg = heading_deg[h:]
            x = x[h:]
            y = y[h:]
        if t > 0:
            gyro = gyro[:-t]
            acc = acc[:-t]
            heading_deg = heading_deg[:-t]
            x = x[:-t]
            y = y[:-t]
    n = min(len(gyro), len(acc), len(heading_deg), len(x), len(y))
    gyro = gyro[:n]
    acc = acc[:n]
    heading_rad = np.radians(heading_deg[:n])
    pos3 = np.stack([x[:n], y[:n], np.zeros(n, dtype=float)], axis=1)
    ori_stub = heading_rad.reshape(-1, 1)
    
    return gyro, acc, pos3, ori_stub


def window_dataset(gyro_data, acc_data, pos_data, ori_data, window_size=160, stride=36, filter_window=5, smooth_heading=True, heading_sigma=3, smooth_length=False, length_sigma=1.0, mode="2d"):
    """
    SELFMADE 数据集切窗处理 (仅 2D)
    注意：为了兼容接口，保留 mode 参数，但内部不再处理 3d 逻辑
    """
    mid = window_size // 2 - stride // 2
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]
    
    pos2d = pos_data[:, :2]
    if filter_window and filter_window > 1:
        pos2d = moving_average(pos2d, filter_window)
    
    x_gyro = []
    x_acc = []
    y_len = []
    y_head_abs = []  # 绝对航向标签
    y_head_rel = []  # 相对航向标签

    # init_pos & init_head
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
        
        # 2. 绝对航向：当前步的位移方向
        curr_diff = pb - pa
        if np.linalg.norm(curr_diff) < 1e-6:
            abs_heading = 0.0  # 静止时设为0
        else:
            abs_heading = np.arctan2(curr_diff[1], curr_diff[0])
        
        y_len.append(np.array([delta_len], dtype=np.float32))
        y_head_abs.append(np.array([abs_heading], dtype=np.float32))
        
    x_gyro = np.array(x_gyro)
    x_acc = np.array(x_acc)
    y_len = np.array(y_len)
    y_head_abs = np.array(y_head_abs)

    # 静止检测
    if len(y_len) > 0 and len(y_head_abs) > 0:
        stationary_mask = np.abs(y_len.flatten()) < 0.01
        y_len[stationary_mask, 0] = 0.0
        y_head_abs[stationary_mask, 0] = 0.0

    # 平滑步长
    if smooth_length and len(y_len) > 0:
        y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
        y_len = y_len_smooth.reshape(-1, 1)
    
    # 平滑绝对航向并计算相对航向
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