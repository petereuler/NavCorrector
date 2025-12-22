import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
from scipy.io import loadmat


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


def yaw_from_positions(pos2d, min_displacement=0.05, smooth_window=5):
    """
    从位置计算 Course Angle，包含平滑和去噪
    """
    n = len(pos2d)
    if n < 2:
        return np.zeros(n)
        
    # 1. 强力平滑
    if smooth_window > 1:
        # 使用多次平滑以抑制高频噪声
        pos_smooth = moving_average(pos2d, smooth_window)
        pos_smooth = moving_average(pos_smooth, smooth_window)
    else:
        pos_smooth = pos2d
        
    # 2. 计算速度向量
    vel = np.gradient(pos_smooth, axis=0)
    
    # 3. 计算航向角，过滤微小位移
    yaws = np.zeros(n)
    
    # 初始角度
    if np.linalg.norm(vel[0]) > min_displacement:
        yaws[0] = np.arctan2(vel[0, 1], vel[0, 0])
        
    for i in range(1, n):
        v = vel[i]
        mag = np.linalg.norm(v)
        
        if mag > min_displacement:
            # 只有当速度足够大时才更新航向
            current_yaw = np.arctan2(v[1], v[0])
            yaws[i] = current_yaw
        else:
            # 否则保持上一个时刻的航向
            yaws[i] = yaws[i-1]
            
    # 4. 再次平滑角度 (在 sin/cos 域)
    # 这能消除某些跳变
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    cos_smooth = moving_average(cos_yaw, smooth_window)
    sin_smooth = moving_average(sin_yaw, smooth_window)
    yaws_smooth = np.arctan2(sin_smooth, cos_smooth)
            
    return yaws_smooth


def load_selfmade_raw(imu_or_all_path, gt_path=None, crop_head=0, crop_tail=0):
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
    
    # # SELFMADE数据集降采样2（每隔一个样本取一个）
    # gyro = gyro[::2]
    # acc = acc[::2]
    # pos3 = pos3[::2]
    # ori_stub = ori_stub[::2]
    
    return gyro, acc, pos3, ori_stub


def window_dataset(gyro_data, acc_data, pos_data, ori_data, mode="2d", window_size=160, stride=36, filter_window=5, smooth_heading=True, heading_sigma=3, smooth_length=False, length_sigma=1.0):
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
        y_head = []

        # init_pos & init_head
        idx_0 = 0
        a_0 = idx_0 + window_size // 2 - stride // 2
        b_0 = idx_0 + window_size // 2 + stride // 2
        a_0 = max(0, min(a_0, len(pos2d)-1))
        b_0 = max(0, min(b_0, len(pos2d)-1))
        init_pos = pos2d[a_0, :]
        
        diff_0 = pos2d[b_0] - pos2d[a_0]
        init_head = float(np.arctan2(diff_0[1], diff_0[0]))

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
            
            delta_len = np.linalg.norm(pb - pa)
            
            curr_diff = pb - pa
            if np.linalg.norm(curr_diff) < 1e-6:
                # Handle stationary case
                if i == 0:
                    curr_chord_angle = 0.0
                else:
                    # Use previous angle if stationary
                    # We need to compute prev angle first if we want to use it
                    pass
            else:
                curr_chord_angle = np.arctan2(curr_diff[1], curr_diff[0])
            
            if i == 0:
                delta_head = 0.0
                # Define for safety, though i=0 logic handles it
                pass 
            else:
                prev_a = a - stride
                if prev_a < 0:
                    delta_head = 0.0
                else:
                    prev_p = pos2d[prev_a]
                    prev_diff = pa - prev_p
                    if np.linalg.norm(prev_diff) < 1e-6:
                        # If previous step was stationary, assume continuity
                        prev_chord_angle_val = curr_chord_angle 
                    else:
                        prev_chord_angle_val = np.arctan2(prev_diff[1], prev_diff[0])
                    
                    # Handle current stationary case using prev angle
                    if np.linalg.norm(curr_diff) < 1e-6:
                         curr_chord_angle = prev_chord_angle_val

                    delta_head = wrap_angle(curr_chord_angle - prev_chord_angle_val)
            
            y_len.append(np.array([delta_len], dtype=np.float32))
            y_head.append(np.array([delta_head], dtype=np.float32))
            
        x_gyro = np.array(x_gyro)
        x_acc = np.array(x_acc)
        y_len = np.array(y_len)
        y_head = np.array(y_head)
        
        # 在平滑之前进行数据清洗：基于步长判断静止状态
        # 如果步长绝对值小于阈值，说明处于静止状态，将步长和航向角都设为0
        if len(y_len) > 0 and len(y_head) > 0:
            # 基于步长判断是否静止
            stationary_mask = np.abs(y_len.flatten()) < 0.01  # 步长小于1cm认为静止

            # 将静止状态的样本标签设为0
            y_len[stationary_mask, 0] = 0.0
            y_head[stationary_mask, 0] = 0.0

        # 对步长进行平滑处理（提高真值轨迹的光滑性）
        if smooth_length and len(y_len) > 0:
            y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
            y_len = y_len_smooth.reshape(-1, 1)
        
        # 对航向角进行平滑处理（提高真值轨迹的光滑性）
        if smooth_heading and len(y_head) > 0:
            y_head_smooth = gaussian_filter1d(y_head.flatten(), sigma=heading_sigma)
            y_head = y_head_smooth.reshape(-1, 1)
        
        return [x_gyro, x_acc], [y_len, y_head], init_pos, init_head
    elif mode == "3d":
        raise ValueError("SELFMADE only supports 2d mode here")
    else:
        raise ValueError("mode must be '2d' or '3d'")
