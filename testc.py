import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window, yaw_from_quaternion_array, moving_average
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window
from models.heading_classifier import (
    FeatureExtractor, RegressorHead,
    HeadingQuantizer, HeadingBinaryHead,
    DualHeadingModel,
)
from src.util import generate_trajectory_2d
from src.pdr import PDR
from utils.visualization import (
    wrap_angle,
    plot_trajectory_comparison,
    plot_heading_analysis,
    plot_time_series,
    plot_cumulative_series,
    plot_cumulative_error_series,
    plot_error_histogram,
    analyze_encoding_errors,
    plot_trajectory_with_quiver,
    plot_trajectory_turn_error_quiver
)
from utils.results import (
    compute_path_length,
    extract_ground_truth_positions,
    save_results_to_csv
)


# ===== 参数配置（必须与 trainc.py 一致）=====
window_size = 160
stride = 32
vis_num1 = 20000  # 当show_full_trajectory=False时，数据加载的最大长度限制
vis_num2 = 500    # 当show_full_trajectory=False时，可视化的最大长度限制
show_full_trajectory = False  # 设置为True时显示完整轨迹，忽略vis_num1和vis_num2限制
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
batch_size = 256
dataset = "OXIOD"

# 航向角量化参数（必须与 trainc.py 一致）
num_bits = 10  # 必须是 4 的倍数
num_bins = 2 ** num_bits  # 4096 个 bin
output_bits = num_bits
encoding_mode = 'binary_code'




def load_models(ckpt_dir, device):
    """加载预训练的双流航向回归模型"""
    input_dim = 6
    feature_dim = 64

    models = {
        'extractor_len': FeatureExtractor(input_dim, feature_dim).to(device),
        'reg_len': RegressorHead(feature_dim, 1).to(device),
    }

    # 双流航向模型 (直接回归)
    models['dual_heading'] = DualHeadingModel(
        in_channels=input_dim,
        feat_dim=feature_dim
    ).to(device)

    model_files = {
        'extractor_len': 'extractor_len.pth',
        'reg_len': 'reg_len.pth',
        'dual_heading': 'dual_heading_model.pth',
    }

    for model_name, filename in model_files.items():
        model_path = os.path.join(ckpt_dir, filename)
        if os.path.exists(model_path):
            models[model_name].load_state_dict(torch.load(model_path, map_location=device))
            models[model_name].eval()
            print(f"已加载模型: {filename}")
        else:
            print(f"警告: 未找到模型文件 {filename}")

    return models


def predict_in_batches(models, gx, ax, batch_size=256, fusion_alpha=0.1):
    """批量预测双流航向模型并进行互补滤波融合

    Args:
        models: 包含所有模型的字典
        gx: 陀螺仪数据
        ax: 加速度数据
        batch_size: 批次大小
        fusion_alpha: 互补滤波权重 (0-1)，控制绝对航向 vs 相对航向的比例

    Returns:
        pred_len: 预测步长
        pred_head_fused: 融合后的航向 (使用互补滤波)
        pred_head_abs: 绝对航向预测 (用于统计)
    """
    n = gx.shape[0]
    preds_len = []
    preds_head_fused = []
    preds_head_abs = []
    preds_binary_probs = []
    preds_binary_hard = []
    preds_logits = []

    models['extractor_len'].eval()
    models['reg_len'].eval()
    models['dual_heading'].eval()

    # 初始化融合航向 (第一个时刻使用绝对航向预测)
    fused_heading_prev = None

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            # 准备数据
            xb = torch.tensor(np.concatenate([gx[start:end], ax[start:end]], axis=-1),
                            dtype=torch.float32, device=device)

            # === 1. 步长预测 ===
            feat_l = models['extractor_len'](xb)
            pred_l = models['reg_len'](feat_l)

            # === 2. 双流航向预测 ===
            pred_abs, pred_rel = models['dual_heading'](xb)

            # === 3. 直接使用回归输出 ===
            pred_h_abs_batch = pred_abs.cpu().numpy()  # 绝对航向直接输出

            # 为兼容性创建虚拟的二进制编码统计
            pred_binary = np.zeros((pred_h_abs_batch.shape[0], num_bits))  # 虚拟二进制概率
            pred_binary_hard_batch = np.zeros((pred_h_abs_batch.shape[0], num_bits), dtype=np.int32)  # 虚拟硬解码

            # === 4. 互补滤波融合 ===
            batch_size_current = pred_h_abs_batch.shape[0]

            # 初始化融合航向数组
            fused_headings = np.zeros_like(pred_h_abs_batch)

            for i in range(batch_size_current):
                # 当前批次中的绝对航向预测
                abs_pred = pred_h_abs_batch[i, 0]

                # 相对航向预测 (弧度)
                rel_pred = pred_rel[i, 0].cpu().numpy()

                if fused_heading_prev is None:
                    # 第一个时刻：直接使用绝对航向
                    fused_heading = abs_pred
                else:
                    # 互补滤波: H_final[t] = (1-α) * (H_final[t-1] + ΔH_pred[t]) + α * H_abs_pred[t]
                    rel_integration = fused_heading_prev + rel_pred
                    fused_heading = (1 - fusion_alpha) * rel_integration + fusion_alpha * abs_pred

                fused_headings[i, 0] = fused_heading
                fused_heading_prev = fused_heading

            # === 5. 收集结果 ===
            preds_len.append(pred_l.cpu().numpy())
            preds_head_fused.append(fused_headings)
            preds_head_abs.append(pred_h_abs_batch.reshape(-1, 1))
            preds_binary_probs.append(pred_binary)
            preds_binary_hard.append(pred_binary_hard_batch)
            preds_logits.append(pred_abs.cpu().numpy())  # 使用pred_abs而不是logits_abs

            # 清理显存
            del xb, feat_l, pred_l, pred_abs, pred_rel

    # 合并所有批次的结果
    pred_len = np.concatenate(preds_len, axis=0)
    pred_head_fused = np.concatenate(preds_head_fused, axis=0)
    pred_head_abs = np.concatenate(preds_head_abs, axis=0)
    pred_binary_probs = np.concatenate(preds_binary_probs, axis=0)
    pred_binary_hard = np.concatenate(preds_binary_hard, axis=0)
    pred_logits = np.concatenate(preds_logits, axis=0)

    return pred_len, pred_head_fused, pred_head_abs, pred_binary_probs, pred_binary_hard, pred_logits


def main():
    project_dir = "/home/admin407/code/zyshe/NavCorrector"
    data_root = os.path.join(project_dir, "OXIOD")
    selfmade_root = os.path.join(project_dir, "SELFMADE")
    ronin_root = os.path.join(project_dir, "RONIN")
    ckpt_dir = os.path.join(project_dir, f"checkpoints_cls_{dataset}")
    output_dir = os.path.join(project_dir, f"output/testc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("双流航向回归测试（绝对航向 + 相对航向，直接回归）")
    print("="*60)
    print(f"  模型结构: DualHeadingModel (基于ResNet回归)")
    print(f"  绝对航向: 直接回归 [-π, π]")
    print(f"  相对航向: 直接回归 Δθ")
    print(f"  融合方式: 互补滤波 (α=0.1)")
    print("="*60)

    # 加载模型
    print("\n正在加载预训练双流模型...")
    models = load_models(ckpt_dir, device)
    print("模型加载完成！")

    # 测试文件列表
    if dataset == "SELFMADE" and os.path.isdir(selfmade_root):
        imu_files = []
        for r, d, fns in os.walk(selfmade_root):
            for fn in fns:
                if fn.lower().endswith('.csv') or fn.lower().endswith('.mat'):
                    imu_files.append(os.path.join(r, fn))
        imu_files = sorted(imu_files)
        gt_files = [None] * len(imu_files)
    elif dataset == "RONIN" and os.path.isdir(ronin_root):
        list_seen = os.path.join(ronin_root, 'lists', 'list_test_seen.txt')
        with open(list_seen) as f:
            names = [s.strip() for s in f.readlines() if len(s) > 0 and s[0] != '#']
        imu_files = [os.path.join(ronin_root, 'Data', 'seen_subjects_test_set', n) for n in names]
        gt_files = [None] * len(imu_files)
    else:
        imu_files = [
            os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu2.csv'),
            os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu5.csv'),
            os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu6.csv'),
            os.path.join(data_root, 'handheld', 'data3', 'syn', 'imu1.csv'),
            os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu1.csv'),
            os.path.join(data_root, 'handheld', 'data4', 'syn', 'imu3.csv'),
            os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu3.csv'),
            #os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu1.csv'),
        ]
        gt_files = [f.replace("imu", "vi") for f in imu_files]

    # 统计信息
    all_len_mae = []
    all_head_mae = []
    all_rmse = []

    # 逐文件测试
    for imu_file, gt_file in zip(imu_files, gt_files):
        print(f"\n正在处理文件: {os.path.basename(imu_file)}")
        
        if dataset == "RONIN":
            gyro, acc, pos3d, ori = load_ronin_raw(imu_file)
        elif dataset == "SELFMADE":
            gyro, acc, pos3d, ori = load_selfmade_raw(imu_file)
        else:
            gyro, acc, pos3d, ori = load_oxiod_raw(imu_file, gt_file)
            
        # 根据是否显示完整轨迹决定数据长度
        if show_full_trajectory:
            # 使用完整数据，不截断
            pass  # 保持原始长度
        else:
            # 使用限制长度（传统方式）
            gyro, acc, pos3d, ori = gyro[:vis_num1], acc[:vis_num1], pos3d[:vis_num1], ori[:vis_num1]
        pos2d = pos3d[:, :2]
        
        if dataset == "SELFMADE" or dataset == "RONIN":
            head = ori[:, 0]
        else:
            head = yaw_from_quaternion_array(ori)
            
        if dataset == "RONIN":
            window_fn = ronin_window
        elif dataset == "SELFMADE":
            window_fn = selfmade_window
        else:
            window_fn = oxiod_window
        
        # 根据数据集设置 filter_window（与训练时保持一致）
        if dataset == "RONIN":
            filter_window = 20  # RONIN 数据集不使用位置平滑
        elif dataset == "SELFMADE":
            filter_window = 20  # SELFMADE 数据集使用位置平滑
        else:
            filter_window = 20  # OXIOD 数据集使用位置平滑
        
        # 先获取平滑前的真值（用于对比）
        [gx_raw, ax_raw], [dl_raw, dh_abs_raw, dh_rel_raw], init_l_raw, init_h_raw = window_fn(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=filter_window,
            smooth_heading=False,  # 不平滑航向角
            smooth_length=False,    # 不平滑步长
        )

        # 再获取平滑后的真值（用于训练和评估）
        [gx, ax], [dl, dh_abs, dh_rel], init_l, init_h = window_fn(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=filter_window,
            smooth_heading=True,  # 启用航向角平滑，用于显示平滑后的轨迹
            heading_sigma=1.5,    # 航向角高斯平滑标准差
            smooth_length=False,   # 不平滑步长，只平滑航向
            length_sigma=1.0,    # 步长高斯平滑标准差
        )

        # 为兼容性，将绝对航向作为主要航向
        dh_raw = dh_abs_raw
        dh = dh_abs
        
        if gx.shape[0] == 0:
            print("窗口长度不足，跳过该序列")
            continue
        
        # 生成文件名前缀（提前定义，用于后续统计）
        if dataset == "RONIN":
            base_name = os.path.basename(imu_file)
        else:
            rel_path = os.path.relpath(imu_file, data_root)
            parts = rel_path.split(os.sep)
            base_name = f"{parts[-3]}_{parts[-1].split('.')[0]}"
            
        # 预测（双流航向 + 互补滤波融合）
        fusion_alpha = 0.1  # 互补滤波权重
        pred_len, pred_head_fused, pred_head_abs, pred_binary_probs, pred_binary_hard, pred_logits = predict_in_batches(
            models, gx, ax, batch_size=batch_size, fusion_alpha=fusion_alpha
        )

        # 为了兼容性，将融合结果作为主要预测结果
        pred_head_soft = pred_head_fused  # 用于轨迹重建的主要航向
        pred_head_hard = pred_head_abs    # 用于统计的绝对航向
    
        # 对齐数据长度
        min_len = min(len(dl), len(dh), len(pred_len), len(pred_head_soft), len(pred_head_hard))
        dl = dl[:min_len]
        dh = dh[:min_len]
        pred_len = pred_len[:min_len]
        pred_head_soft = pred_head_soft[:min_len]
        pred_head_hard = pred_head_hard[:min_len]

        # =======================================================
        # 【新增修复】强制对齐初始帧
        # 原因：消除第0步预测误差导致的整体轨迹旋转，确保对比公平
        # =======================================================
        if len(pred_head_soft) > 0:
            print(f"  > 执行初始对齐: 修正前第0步误差 {np.degrees(pred_head_soft[0,0] - dh[0,0]):.4f} deg")
            
            # 1. 强制第0步的航向变化完全等于真值
            # 这样在 heading_analysis.png 中，第0个点会完全重合
            pred_head_soft[0] = dh[0]
            
            # (可选) 如果你也想让硬解码对齐，加上这行
            pred_head_hard[0] = dh[0] 

            # (可选) 甚至可以对齐前几帧（例如前0.5秒），让模型“热身”
            # warmup_steps = 5
            # pred_head_soft[:warmup_steps] = dh[:warmup_steps]
        # =======================================================
        
        # 对齐平滑前的数据长度
        if dl_raw is not None and dh_raw is not None and len(dl_raw) > 0 and len(dh_raw) > 0:
            min_len_raw = min(len(dl_raw), len(dh_raw), min_len)
            dl_raw = dl_raw[:min_len_raw]
            dh_raw = dh_raw[:min_len_raw]
        else:
            dl_raw = None
            dh_raw = None
        
        # 对于回归方法，跳过编码错误分析（回归方法没有编码概念）
        print(f"  跳过编码错误分析（回归方法）")

        # [修改] 生成轨迹（基于步长+绝对航向）
        # 对于绝对航向，不再需要传入init_h（设为0即可）
        traj_gt = generate_trajectory_2d(init_l, 0.0, dl, dh[:len(dl)])
        traj_pred = generate_trajectory_2d(init_l, 0.0, pred_len, pred_head_soft[:len(pred_len)])
        
        # 生成平滑前的轨迹（用于对比）
        traj_gt_raw = None
        if dl_raw is not None and dh_raw is not None:
            traj_gt_raw = generate_trajectory_2d(init_l_raw, 0.0, dl_raw, dh_raw[:len(dl_raw)])

        # 计算 dataset_OXIOD 中使用的起始索引
        start_frame_idx = window_size // 2 - stride // 2  # 例如 160//2 - 32//2 = 64

        # 提取真值位置坐标
        # 注意：不需要传入 init_l 了，而是传入索引，这样更精准
        traj_gt_xy = extract_ground_truth_positions(
            pos3d, 
            window_size, 
            stride, 
            num_windows=len(dl), 
            start_index=start_frame_idx  # <--- 关键参数
)
        
        # 确保长度一致
        min_len = min(len(traj_gt), len(traj_gt_xy), len(traj_pred))
        traj_gt = traj_gt[:min_len]
        traj_gt_xy = traj_gt_xy[:min_len]
        traj_pred = traj_pred[:min_len]

        # 传统PDR对比
        pdr = PDR(initial_pos=init_l, initial_yaw=init_h)
        init_pos, init_yaw, pdr_dl, pdr_dh = pdr.get_step_and_heading_deltas(gyro, acc)
        traj_pdr = generate_trajectory_2d(init_pos, init_yaw, pdr_dl, pdr_dh[:len(pdr_dl)])
        

        # 截取可视化/评估数据
        if show_full_trajectory:
            # 使用完整轨迹进行评估
            if traj_gt_raw is not None:
                traj_len = min(len(traj_gt_raw), len(traj_pred))
                gt_vis = traj_gt_raw[:traj_len]  # 使用平滑前的真值
            else:
                traj_len = min(len(traj_gt), len(traj_pred))
                gt_vis = traj_gt[:traj_len]
        else:
            # 使用限制长度进行评估（传统方式）
            if traj_gt_raw is not None:
                traj_len = min(len(traj_gt_raw), len(traj_pred), vis_num2)
                gt_vis = traj_gt_raw[:traj_len]  # 使用平滑前的真值
            else:
                traj_len = min(len(traj_gt), len(traj_pred), vis_num2)
                gt_vis = traj_gt[:traj_len]
        pred_vis = traj_pred[:traj_len]

        # 计算评估指标（使用平滑前的真值）
        error = np.linalg.norm(gt_vis - pred_vis, axis=1)
        rmse = np.sqrt(np.mean((gt_vis - pred_vis) ** 2))
    
        # 使用平滑前的真值计算MAE
        dl_gt_for_error = dl_raw if dl_raw is not None else dl
        dh_gt_for_error = dh_raw if dh_raw is not None else dh

        if show_full_trajectory:
            # 使用完整数据计算MAE
            min_error_len = min(len(dl_gt_for_error), len(dh_gt_for_error), len(pred_len), len(pred_head_soft))
        else:
            # 使用限制长度计算MAE（传统方式）
            min_error_len = min(len(dl_gt_for_error), len(dh_gt_for_error), len(pred_len), len(pred_head_soft), vis_num2)

        len_mae = np.abs(pred_len[:min_error_len, 0] - dl_gt_for_error[:min_error_len, 0]).mean()
        head_mae = np.abs(wrap_angle(pred_head_soft[:min_error_len, 0] - dh_gt_for_error[:min_error_len, 0])).mean()
        
        gt_total_length = compute_path_length(gt_vis)
   
        # 打印评估结果
        print(f"[{base_name}] RMSE: {rmse:.4f}m")
        print(f"  Length MAE: {len_mae:.4f}m")
        print(f"  Heading MAE: {np.degrees(head_mae):.2f} deg")
        print(f"  Total path length: {gt_total_length:.2f}m")
        
        all_len_mae.append(len_mae)
        all_head_mae.append(head_mae)
        all_rmse.append(rmse)

        # 根据是否显示完整轨迹来决定可视化长度
        vis_len = len(dl) if show_full_trajectory else vis_num2
        print(f"可视化长度: {vis_len} (显示完整轨迹: {show_full_trajectory})")

        # 生成轨迹对比图（四个子图）
        plot_trajectory_comparison(traj_gt, traj_gt_xy, traj_pred, output_dir, base_name,
                                 traj_gt_raw=traj_gt_raw, traj_pdr=traj_pdr, vis_num=vis_len if show_full_trajectory else None)

        # ==================== [修改] 新增：真值 vs 预测 双箭头矢量图 ====================

        # 1. [修改] 准备预测值的绝对航向 (N,)
        # 对于绝对航向，pred_head_soft已经是绝对航向，直接使用
        vis_headings_pred = pred_head_soft[:len(pred_vis), 0]

        # 2. [修改] 准备真值的绝对航向 (N,)
        # 对于绝对航向，dh已经是绝对航向，直接使用
        vis_headings_gt = dh[:len(pred_vis), 0]
        
        # 3. 调用绘图
        plot_trajectory_with_quiver(
            positions=pred_vis, 
            headings_gt=vis_headings_gt, 
            headings_pred=vis_headings_pred,
            step_interval=1,    # 采样间隔，保持不变
            arrow_length=0.2,    # 箭头长度，稍微调小了一点
            output_file=os.path.join(output_dir, f"{base_name}_quiver.png")
        )
        print(f"  矢量航向对比图保存至: {base_name}_quiver.png")
        
        # =========================================================================

        # ==================== 新增：瞬时转向误差双箭头分析图 ====================
        
        # A. [修改] 准备基础数据 - 对于绝对航向
        # 截取长度对齐 (N)
        steps_len = len(pred_vis)
        abs_h_pred = pred_head_soft[:steps_len, 0]  # 预测的绝对航向
        abs_h_gt = dh[:steps_len, 0]                 # 真值的绝对航向

        # B. [修改] 预测绝对航向 (Red Arrow Data)
        vis_headings_pred = abs_h_pred

        # C. [修改] 计算"局部真值"航向 (Purple Arrow Data)
        # 对于绝对航向，我们直接使用真值的绝对航向作为局部真值
        # 因为绝对航向没有累积误差问题
        vis_headings_local_truth = abs_h_gt

        # 计算瞬时转向误差（用于打印信息）
        # 从绝对航向计算转向变化
        if len(abs_h_pred) > 1:
            turn_pred = np.diff(abs_h_pred)
            turn_gt = np.diff(abs_h_gt)
            turn_error = turn_gt - turn_pred
        
        # D. 调用绘图
        plot_trajectory_turn_error_quiver(
            positions=pred_vis,
            headings_pred=vis_headings_pred,        # 红色：模型想往哪走
            headings_local_truth=vis_headings_local_truth, # 紫色：模型该往哪走
            step_interval=1,    
            arrow_length=0.2,    # 稍微大一点以便观察分叉
            output_file=os.path.join(output_dir, f"{base_name}_turn_error_quiver.png")
        )
        print(f"  瞬时转向误差分析图保存至: {base_name}_turn_error_quiver.png")
        
        # 打印最大突变点，方便在Log中快速定位
        if len(abs_h_pred) > 1:
            max_err_idx = np.argmax(np.abs(turn_error))
            max_err_deg = np.degrees(np.abs(turn_error[max_err_idx]))
            print(f"  > 最大单步转向误差: {max_err_deg:.2f}° (at step {max_err_idx})")
        else:
            print(f"  > 数据长度不足，无法计算转向误差")
        
        # ====================================================================

        plot_heading_analysis(dh, pred_head_soft, pred_head_hard, vis_len,
                             os.path.join(output_dir, f"{base_name}_heading_analysis.png"),
                             dh_raw=dh_raw)

        plot_time_series(dl, dh, pred_len, pred_head_soft, vis_len,
                        os.path.join(output_dir, f"{base_name}_time_series.png"),
                        dl_raw=dl_raw, dh_raw=dh_raw)

        # plot_cumulative_series(dl, dh, pred_len, pred_head_soft, vis_num2, init_h,
        #                        os.path.join(output_dir, f"{base_name}_cumulative_series.png"),
        #                        dl_raw=dl_raw, dh_raw=dh_raw, init_h_raw=init_h_raw)

        # plot_cumulative_error_series(dl, dh, pred_len, pred_head_soft, vis_num2, init_h,
        #                              os.path.join(output_dir, f"{base_name}_cumulative_error.png"),
        #                              dl_raw=dl_raw, dh_raw=dh_raw, init_h_raw=init_h_raw)

        # plot_error_histogram(dl, dh, pred_len, pred_head_soft, vis_num2,
        #                     os.path.join(output_dir, f"{base_name}_error_histogram.png"),
        #                     quantizer, dl_raw=dl_raw, dh_raw=dh_raw)

        # 保存数据到CSV文件
        # save_results_to_csv(gt_vis, pred_vis, traj_pdr, dl, dh, pred_len, pred_head_soft,
        #                    vis_num2, base_name, output_dir)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 打印总体统计
    print("\n" + "="*60)
    print("总体统计:")
    print(f"  平均 RMSE: {np.mean(all_rmse):.4f}m")
    print(f"  平均 Length MAE: {np.mean(all_len_mae):.4f}m")
    print(f"  平均 Heading MAE: {np.degrees(np.mean(all_head_mae)):.2f} deg")
    print(f"  模型类型: 直接回归 (无量化)")
    print("="*60)

    print(f"\n测试完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
