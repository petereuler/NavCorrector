import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def quat_conj(q):
    q = np.array(q, dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_rotmat(q):
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
    ], dtype=np.float32)


def moving_average(x, k):
    """简易滑动平均滤波，窗口 k>=1；k<=1 时原样返回。"""
    if k is None or k <= 1:
        return x
    k = int(k)
    kernel = np.ones(k, dtype=float) / float(k)
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)
    return x


def load_oxiod_raw(imu_data_filename, gt_data_filename, trim_head=1200, trim_tail=300):
    """
    加载 OxIOD 原始数据：IMU(gyro/acc) 与 GT 位置/姿态（XYZ + quaternion wxyz）。
    """
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    if trim_head > 0:
        imu_data = imu_data[trim_head:]
        gt_data = gt_data[trim_head:]
    if trim_tail > 0:
        imu_data = imu_data[:-trim_tail]
        gt_data = gt_data[:-trim_tail]

    m = min(len(imu_data), len(gt_data))
    imu_data = imu_data[:m]
    gt_data = gt_data[:m]

    gyro_data = imu_data[:, 4:7].astype(np.float32)
    acc_data = imu_data[:, 10:13].astype(np.float32)
    pos_data = gt_data[:, 2:5].astype(np.float32)
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1).astype(np.float32)
    ori_norm = np.linalg.norm(ori_data, axis=1, keepdims=True)
    ori_data = ori_data / np.clip(ori_norm, 1e-8, None)
    return gyro_data, acc_data, pos_data, ori_data


def get_oxiod_predefined_split_pairs(oxiod_root, split="train", sensor="syn"):
    """
    使用项目内预设划分返回 OXIOD 样本对，不读取数据集自带 Train/Test 文本。
    返回 [(name, imu_path, gt_path), ...]。
    """
    split = split.lower()
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported split: {split}")
    predefined_files = [
        os.path.join("handheld", "data1", "imu1.csv"),
        os.path.join("handheld", "data1", "imu3.csv"),
        os.path.join("handheld", "data1", "imu4.csv"),
        os.path.join("handheld", "data1", "imu7.csv"),
        os.path.join("handheld", "data2", "imu1.csv"),
        os.path.join("handheld", "data2", "imu2.csv"),
        os.path.join("handheld", "data2", "imu3.csv"),
        os.path.join("handheld", "data3", "imu2.csv"),
        os.path.join("handheld", "data3", "imu3.csv"),
        os.path.join("handheld", "data3", "imu4.csv"),
        os.path.join("handheld", "data3", "imu5.csv"),
        os.path.join("handheld", "data4", "imu2.csv"),
        os.path.join("handheld", "data4", "imu4.csv"),
        os.path.join("handheld", "data4", "imu5.csv"),
        os.path.join("handheld", "data5", "imu1.csv"),
        os.path.join("handheld", "data5", "imu2.csv"),
        os.path.join("handheld", "data5", "imu4.csv"),
    ]
    predefined_test = {
        os.path.join("handheld", "data1", "imu4.csv"),
        os.path.join("handheld", "data2", "imu2.csv"),
        os.path.join("handheld", "data3", "imu4.csv"),
        os.path.join("handheld", "data4", "imu5.csv"),
        os.path.join("handheld", "data5", "imu1.csv"),
    }

    pairs = []
    for rel_file in predefined_files:
        is_test = rel_file in predefined_test
        if (split == "train" and is_test) or (split == "test" and not is_test):
            continue
        rel_dir = os.path.dirname(rel_file)
        imu_name = os.path.basename(rel_file)
        imu_path = os.path.join(oxiod_root, rel_dir, sensor, imu_name)
        gt_path = os.path.join(oxiod_root, rel_dir, sensor, imu_name.replace("imu", "vi"))
        if not (os.path.exists(imu_path) and os.path.exists(gt_path)):
            continue
        name = rel_file.replace(".csv", "")
        pairs.append((name, imu_path, gt_path))
    return pairs


def get_oxiod_split_pairs(oxiod_root, split="train", sensor="syn"):
    """兼容旧调用名：使用预设划分。"""
    return get_oxiod_predefined_split_pairs(oxiod_root, split=split, sensor=sensor)


def window_dataset(
    gyro_data,
    acc_data,
    pos_data,
    ori_data,
    window_size=160,
    stride=36,
    filter_window=20,
    smooth_length=False,
    length_sigma=1.0,
    return_ori=False,
    return_rel_ori=False,
    return_delta_p=False,
    return_delta_p_world=False,
    flatten_world_z_for_body_label=False,
):
    m = min(gyro_data.shape[0], acc_data.shape[0], pos_data.shape[0], ori_data.shape[0])
    gyro_data = gyro_data[:m]
    acc_data = acc_data[:m]
    pos_data = pos_data[:m]
    ori_data = ori_data[:m]

    pos_xyz = pos_data
    if filter_window and filter_window > 1:
        pos_xyz = moving_average(pos_xyz, filter_window)

    imu_gyro = []
    imu_acc = []
    y_len = []
    y_ori = []
    y_rel = []
    y_dp = []
    y_dp_world = []

    start_0 = window_size // 2 - stride // 2
    end_0 = window_size // 2 + stride // 2
    start_0 = max(0, min(start_0, len(pos_xyz) - 1))
    end_0 = max(0, min(end_0, len(pos_xyz) - 1))
    init_pos = pos_xyz[start_0, :2]
    init_head = 0.0

    max_start = gyro_data.shape[0] - window_size - 1
    for idx in range(0, max_start, stride):
        gyro_window = gyro_data[idx + 1: idx + 1 + window_size, :]
        acc_window = acc_data[idx + 1: idx + 1 + window_size, :]
        imu_gyro.append(gyro_window)
        imu_acc.append(acc_window)

        start_idx = idx + window_size // 2 - stride // 2
        end_idx = idx + window_size // 2 + stride // 2
        start_idx = max(0, min(start_idx, len(pos_xyz) - 1))
        end_idx = max(0, min(end_idx, len(pos_xyz) - 1))

        pos_start_xyz = pos_xyz[start_idx, :]
        pos_end_xyz = pos_xyz[end_idx, :]
        delta_world = (pos_end_xyz - pos_start_xyz).astype(np.float32)
        delta_len = np.linalg.norm(delta_world)
        y_len.append(np.array([delta_len], dtype=np.float32))

        if return_delta_p:
            q_start = ori_data[start_idx].astype(np.float32)
            R_start = quat_to_rotmat(q_start)
            if flatten_world_z_for_body_label:
                delta_world_for_body = delta_world.copy()
                delta_world_for_body[2] = 0.0
            else:
                delta_world_for_body = delta_world
            dp_body = (R_start.T @ delta_world_for_body.reshape(3, 1)).reshape(3,)
            y_dp.append(dp_body.astype(np.float32))
        if return_delta_p_world:
            y_dp_world.append(delta_world.astype(np.float32))
        if return_ori:
            y_ori.append(ori_data[end_idx].astype(np.float32))
        if return_rel_ori:
            q_start = ori_data[start_idx].astype(np.float32)
            q_end = ori_data[end_idx].astype(np.float32)
            q_rel = quat_mul(quat_conj(q_start), q_end)
            y_rel.append(q_rel.astype(np.float32))

    x_gyro = np.array(imu_gyro)
    x_acc = np.array(imu_acc)
    y_len = np.array(y_len)
    if return_ori:
        y_ori = np.array(y_ori)
    if return_rel_ori:
        y_rel = np.array(y_rel)
    if return_delta_p:
        y_dp = np.array(y_dp)
    if return_delta_p_world:
        y_dp_world = np.array(y_dp_world)

    if smooth_length and len(y_len) > 0:
        y_len_smooth = gaussian_filter1d(y_len.flatten(), sigma=length_sigma)
        y_len = y_len_smooth.reshape(-1, 1)

    labels = [y_len]
    if return_ori:
        labels.append(y_ori)
    if return_rel_ori:
        labels.append(y_rel)
    if return_delta_p:
        labels.append(y_dp)
    if return_delta_p_world:
        labels.append(y_dp_world)
    return [x_gyro, x_acc], labels, init_pos, init_head
