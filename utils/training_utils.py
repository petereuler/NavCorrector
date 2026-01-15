"""
训练相关的工具函数
包含数据加载、数据增强、损失函数和可视化函数
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 假设这些 dataset 模块稍后会被更新以返回四元数 y_q
from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window


# ======= 损失函数 =======
def len_loss(pred, target):
    """步长回归损失函数"""
    return F.mse_loss(pred, target)


# ======= 数据加载函数 =======
def load_data_2d_oxiod(data_root, device, window_size=160, stride=32):
    """
    加载 OXIOD 数据集并分割为训练集和验证集
    
    返回 5 个张量: x, y_q (四元数), y_len, y_abs, y_rel
    """
    imu_files = [
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu7.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu3.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu4.csv'),
    ]
    gt_files = [f.replace("imu", "vi") for f in imu_files]

    val_set = set([
        os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data2', 'syn', 'imu2.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
        os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu5.csv'),
        os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu1.csv'),
        os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu4.csv'),
    ])

    # 增加 yq (quaternion) 列表
    xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr = [], [], [], [], [], []
    xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va = [], [], [], [], [], []

    for imu, gt in zip(imu_files, gt_files):
        gyro, acc, pos3d, ori = load_oxiod_raw(imu, gt)

        # 注意：这里假设 window_dataset 已经被修改为返回 [y_q, y_len, y_head_abs, y_head_rel]
        # dq 代表四元数标签 (Quaternion)
        [gx, ax], [dq, dl, dh_abs, dh_rel], _, _ = oxiod_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=20,
            smooth_heading=True,
            heading_sigma=1.25,
            smooth_length=False,
            length_sigma=1.0,
        )
        if imu in val_set:
            xg_va.append(gx)
            xa_va.append(ax)
            yq_va.append(dq)    # Append quaternion
            yl_va.append(dl)
            yh_abs_va.append(dh_abs)
            yh_rel_va.append(dh_rel)
        else:
            xg_tr.append(gx)
            xa_tr.append(ax)
            yq_tr.append(dq)    # Append quaternion
            yl_tr.append(dl)
            yh_abs_tr.append(dh_abs)
            yh_rel_tr.append(dh_rel)

    # 封装辅助函数处理张量转换
    def cat_and_to_device(xg_list, xa_list, yq_list, yl_list, ya_list, yr_list):
        if len(xg_list) == 0:
            return None
        
        # Input: (N, T, 6)
        xg = np.concatenate(xg_list, axis=0)
        xa = np.concatenate(xa_list, axis=0)
        x = np.concatenate([xg, xa], axis=-1)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        
        # Labels
        yq = torch.tensor(np.concatenate(yq_list, axis=0), dtype=torch.float32, device=device)
        yl = torch.tensor(np.concatenate(yl_list, axis=0), dtype=torch.float32, device=device)
        ya = torch.tensor(np.concatenate(ya_list, axis=0), dtype=torch.float32, device=device)
        yr = torch.tensor(np.concatenate(yr_list, axis=0), dtype=torch.float32, device=device)
        
        return x, yq, yl, ya, yr

    x_tr, yq_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr = cat_and_to_device(
        xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr
    )
    
    x_va, yq_va, ylen_va, yhead_abs_va, yhead_rel_va = cat_and_to_device(
        xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va
    )

    return (x_tr, x_va), (yq_tr, yq_va), (ylen_tr, ylen_va), \
           (yhead_abs_tr, yhead_abs_va), (yhead_rel_tr, yhead_rel_va)


def load_data_2d_selfmade(selfmade_root, device, window_size=160, stride=32):
    """
    加载 SELFMADE 数据集
    """
    files = []
    for r, d, fns in os.walk(selfmade_root):
        for fn in fns:
            if fn.lower().endswith('.csv') or fn.lower().endswith('.mat'):
                files.append(os.path.join(r, fn))
    
    # ... (省略 limit 处理代码，保持原样) ...
    limit = os.getenv('SELFMADE_LIMIT', None)
    if limit is not None:
        try:
            k = int(limit)
            if k > 0: files = files[:k]
        except Exception: pass
    
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No SELFMADE CSV files found")
    
    n = len(files)
    split = max(1, int(0.2 * n))
    val_set = set(files[-split:])
    
    xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr = [], [], [], [], [], []
    xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va = [], [], [], [], [], []
    
    for fp in files:
        gyro, acc, pos3d, ori = load_selfmade_raw(fp)
        
        # Unpack yq (dq)
        [gx, ax], [dq, dl, dh_abs, dh_rel], _, _ = selfmade_window(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=10,
            smooth_heading=True,
            heading_sigma=1.25,
            smooth_length=False,
            length_sigma=1.0,
        )
        if gx.shape[0] == 0:
            continue
        
        if fp in val_set:
            xg_va.append(gx); xa_va.append(ax); yq_va.append(dq); 
            yl_va.append(dl); yh_abs_va.append(dh_abs); yh_rel_va.append(dh_rel)
        else:
            xg_tr.append(gx); xa_tr.append(ax); yq_tr.append(dq);
            yl_tr.append(dl); yh_abs_tr.append(dh_abs); yh_rel_tr.append(dh_rel)
    
    if len(xg_tr) == 0: raise RuntimeError("Training set is empty!")

    def cat_and_to_device(xg_list, xa_list, yq_list, yl_list, ya_list, yr_list):
        xg = np.concatenate(xg_list, axis=0)
        xa = np.concatenate(xa_list, axis=0)
        x = np.concatenate([xg, xa], axis=-1)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        yq = torch.tensor(np.concatenate(yq_list, axis=0), dtype=torch.float32, device=device)
        yl = torch.tensor(np.concatenate(yl_list, axis=0), dtype=torch.float32, device=device)
        ya = torch.tensor(np.concatenate(ya_list, axis=0), dtype=torch.float32, device=device)
        yr = torch.tensor(np.concatenate(yr_list, axis=0), dtype=torch.float32, device=device)
        return x, yq, yl, ya, yr

    x_tr, yq_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr = cat_and_to_device(
        xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr
    )
    x_va, yq_va, ylen_va, yhead_abs_va, yhead_rel_va = cat_and_to_device(
        xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va
    )
    
    return (x_tr, x_va), (yq_tr, yq_va), (ylen_tr, ylen_va), \
           (yhead_abs_tr, yhead_abs_va), (yhead_rel_tr, yhead_rel_va)


def load_data_2d_ronin(ronin_root, device, window_size=160, stride=32):
    """
    加载 RONIN 数据集
    """
    train_dirs = []
    for subset in ['train_dataset_1', 'train_dataset_2']:
        base = os.path.join(ronin_root, 'Data', subset)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                d = os.path.join(base, name)
                if os.path.isdir(d): train_dirs.append(d)
    val_dirs = []
    seen_base = os.path.join(ronin_root, 'Data', 'seen_subjects_test_set')
    if os.path.isdir(seen_base):
        for name in sorted(os.listdir(seen_base)):
            d = os.path.join(seen_base, name)
            if os.path.isdir(d): val_dirs.append(d)

    xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr = [], [], [], [], [], []
    xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va = [], [], [], [], [], []

    for d in train_dirs:
        gyro, acc, pos3d, ori = load_ronin_raw(d)
        # Unpack yq (dq)
        [gx, ax], [dq, dl, dh_abs, dh_rel], _, _ = ronin_window(
            gyro, acc, pos3d, ori, mode='2d', window_size=window_size, stride=stride, filter_window=20,
            smooth_heading=True, heading_sigma=1.25, smooth_length=False, length_sigma=1.5,
        )
        if gx.shape[0] == 0: continue
        xg_tr.append(gx); xa_tr.append(ax); yq_tr.append(dq); 
        yl_tr.append(dl); yh_abs_tr.append(dh_abs); yh_rel_tr.append(dh_rel)
        
    for d in val_dirs:
        gyro, acc, pos3d, ori = load_ronin_raw(d)
        # Unpack yq (dq)
        [gx, ax], [dq, dl, dh_abs, dh_rel], _, _ = ronin_window(
            gyro, acc, pos3d, ori, mode='2d', window_size=window_size, stride=stride, filter_window=20,
            smooth_heading=True, heading_sigma=1, smooth_length=False, length_sigma=1.5,
        )
        if gx.shape[0] == 0: continue
        xg_va.append(gx); xa_va.append(ax); yq_va.append(dq); 
        yl_va.append(dl); yh_abs_va.append(dh_abs); yh_rel_va.append(dh_rel)

    def cat_and_to_device(xg_list, xa_list, yq_list, yl_list, ya_list, yr_list):
        if len(xg_list) == 0: return None
        xg = np.concatenate(xg_list, axis=0)
        xa = np.concatenate(xa_list, axis=0)
        x = np.concatenate([xg, xa], axis=-1)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        yq = torch.tensor(np.concatenate(yq_list, axis=0), dtype=torch.float32, device=device)
        yl = torch.tensor(np.concatenate(yl_list, axis=0), dtype=torch.float32, device=device)
        ya = torch.tensor(np.concatenate(ya_list, axis=0), dtype=torch.float32, device=device)
        yr = torch.tensor(np.concatenate(yr_list, axis=0), dtype=torch.float32, device=device)
        return x, yq, yl, ya, yr

    x_tr, yq_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr = cat_and_to_device(
        xg_tr, xa_tr, yq_tr, yl_tr, yh_abs_tr, yh_rel_tr
    )
    x_va, yq_va, ylen_va, yhead_abs_va, yhead_rel_va = cat_and_to_device(
        xg_va, xa_va, yq_va, yl_va, yh_abs_va, yh_rel_va
    )
    
    return (x_tr, x_va), (yq_tr, yq_va), (ylen_tr, ylen_va), \
           (yhead_abs_tr, yhead_abs_va), (yhead_rel_tr, yhead_rel_va)

