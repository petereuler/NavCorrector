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
    analyze_encoding_errors
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
num_bits = 8  # 必须是 4 的倍数
num_bins = 2 ** num_bits  # 4096 个 bin
output_bits = num_bits
encoding_mode = 'binary_code'




def load_models(ckpt_dir, device):
    """加载预训练的模型和量化器"""
    input_dim = 6
    feature_dim = 64
    
    # 加载量化器（关键：必须使用训练时保存的量化器）
    quantizer_path = os.path.join(ckpt_dir, "quantizer.json")
    if os.path.exists(quantizer_path):
        quantizer = HeadingQuantizer(num_bins=num_bins, use_gray_code=True, )
        quantizer.load(quantizer_path)
    else:
        print(f"警告: 未找到量化器文件 {quantizer_path}，使用均匀量化")
        quantizer = HeadingQuantizer(num_bins=num_bins, use_gray_code=True, )
        # 手动设置均匀量化的边界
        quantizer.bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        quantizer.bin_centers = (quantizer.bin_edges[:-1] + quantizer.bin_edges[1:]) / 2
        quantizer.fitted = True
    
    models = {
        'extractor_len': FeatureExtractor(input_dim, feature_dim).to(device),
        'extractor_head': FeatureExtractor(input_dim, feature_dim).to(device),
        'reg_len': RegressorHead(feature_dim, 1).to(device),
    }
    
    # 使用改进的二进制分类头
    models['head'] = HeadingBinaryHead(feature_dim, num_bits=output_bits, hidden_dim=256, dropout=0.3).to(device)
    model_files = {
        'extractor_len': 'extractor_len.pth',
        'extractor_head': 'extractor_head_cls.pth',
        'reg_len': 'reg_len.pth',
        'head': 'cls_head.pth',
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(ckpt_dir, filename)
        if os.path.exists(model_path):
            models[model_name].load_state_dict(torch.load(model_path, map_location=device))
            models[model_name].eval()
            print(f"已加载模型: {filename}")
        else:
            print(f"警告: 未找到模型文件 {filename}")
    
    return models, quantizer


def predict_in_batches(models, quantizer, gx, ax, batch_size=256, temperature=1.0, return_binary=False):
    """批量预测 (集成 Soft Decoding)
    
    Args:
        return_binary: 是否返回二进制编码（用于统计）
    Returns:
        pred_len, pred_head_soft, pred_head_hard, (pred_binary_probs, pred_binary_hard, logits_h)
    """
    n = gx.shape[0]
    preds_len = []
    preds_head_soft = []
    preds_head_hard = []
    preds_binary_probs = []
    preds_binary_hard = []
    preds_logits = []

    models['extractor_len'].eval()
    models['reg_len'].eval()
    models['extractor_head'].eval()
    models['head'].eval()
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            # 准备数据
            xb = torch.tensor(np.concatenate([gx[start:end], ax[start:end]], axis=-1), 
                            dtype=torch.float32, device=device)
            
            # === 1. 步长预测 ===
            feat_l = models['extractor_len'](xb)
            pred_l = models['reg_len'](feat_l)
            
            # === 2. 航向预测 ===
            feat_h = models['extractor_head'](xb)
            logits_h = models['head'](feat_h)
            
            # === 3. 关键修改：区分硬解码和软解码 ===

            # A. 硬解码 (Hard Decode) - 用于统计 Bit 错误率
            probs = torch.sigmoid(logits_h)
            pred_binary = probs.cpu().numpy()
            pred_binary_hard_batch = (pred_binary > 0.5).astype(np.int32)
            
            # 使用 quantizer 的硬解码方法 (返回 Bin 中心)
            pred_h_hard_batch = quantizer.decode_from_binary_vector(pred_binary)

            # B. 软解码 (Soft Decode) - 用于生成高精度轨迹
            # 直接调用我们在 trainc.py 中使用的 soft expectation
            # 注意：这需要 logits，而不是 binary vector
            pred_h_soft_batch = quantizer.decode_soft_expectation(logits_h).cpu().numpy()

            # === 4. 收集结果 ===
            preds_len.append(pred_l.cpu().numpy())
            preds_head_soft.append(pred_h_soft_batch.reshape(-1, 1))
            preds_head_hard.append(pred_h_hard_batch.reshape(-1, 1))
            
            if return_binary:
                preds_binary_probs.append(pred_binary)
                preds_binary_hard.append(pred_binary_hard_batch)
                preds_logits.append(logits_h.cpu().numpy())
            
            # 清理显存 (对于大文件很重要)
            del xb, feat_l, pred_l, feat_h, logits_h
                
    pred_len = np.concatenate(preds_len, axis=0)
    pred_head_soft = np.concatenate(preds_head_soft, axis=0)
    pred_head_hard = np.concatenate(preds_head_hard, axis=0)
    
    if return_binary:
        pred_binary_probs = np.concatenate(preds_binary_probs, axis=0)
        pred_binary_hard = np.concatenate(preds_binary_hard, axis=0)
        pred_logits = np.concatenate(preds_logits, axis=0)
        return pred_len, pred_head_soft, pred_head_hard, pred_binary_probs, pred_binary_hard, pred_logits
    else:
        return pred_len, pred_head_soft, pred_head_hard


def main():
    project_dir = "/home/admin407/code/zyshe/Corrector"
    data_root = os.path.join(project_dir, "OXIOD")
    selfmade_root = os.path.join(project_dir, "SELFMADE")
    ronin_root = os.path.join(project_dir, "RONIN")
    ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
    output_dir = os.path.join(project_dir, f"output/testc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("航向角量化分类测试（自适应非均匀量化版）")
    print("="*60)
    print(f"  编码模式: {encoding_mode}")
    print(f"  位数: {num_bits} bits -> {num_bins} bins")
    print("="*60)

    # 加载模型和量化器
    print("\n正在加载预训练模型和量化器...")
    models, quantizer = load_models(ckpt_dir, device)
    print("模型加载完成！")
    
    # 打印量化器信息
    if quantizer.fitted:
        bin_widths = np.diff(quantizer.bin_edges)
        print(f"\n量化器信息:")
        print(f"  类型: {'自适应' if quantizer.adaptive else '均匀'}")
        print(f"  Bin 宽度范围: [{np.degrees(bin_widths.min()):.2f}deg, {np.degrees(bin_widths.max()):.2f}deg]")
        print(f"  Bin 宽度中位数: {np.degrees(np.median(bin_widths)):.2f}deg")

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
            os.path.join(data_root, 'handheld', 'data5', 'syn', 'imu1.csv'),
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
        [gx_raw, ax_raw], [dl_raw, dh_raw], init_l_raw, init_h_raw = window_fn(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=filter_window,
            smooth_heading=False,  # 不平滑航向角
            smooth_length=False,    # 不平滑步长
        )
            
        # 再获取平滑后的真值（用于训练和评估）
        [gx, ax], [dl, dh], init_l, init_h = window_fn(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=filter_window,
            smooth_heading=True,  # 启用航向角平滑，与训练时保持一致
            heading_sigma=1.5,    # 航向角高斯平滑标准差
            smooth_length=False,   # 不平滑步长，只平滑航向
            length_sigma=1.0,    # 步长高斯平滑标准差
        )
        
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
            
        # 预测（同时获取二进制编码用于统计）
        pred_len, pred_head_soft, pred_head_hard, pred_binary_probs, pred_binary_hard, pred_logits = predict_in_batches(
            models, quantizer, gx, ax, batch_size=batch_size, temperature=1.0, return_binary=True
        )
    
        # 对齐数据长度
        min_len = min(len(dl), len(dh), len(pred_len), len(pred_head_soft), len(pred_head_hard))
        dl = dl[:min_len]
        dh = dh[:min_len]
        pred_len = pred_len[:min_len]
        pred_head_soft = pred_head_soft[:min_len]
        pred_head_hard = pred_head_hard[:min_len]
        
        # 对齐平滑前的数据长度
        if dl_raw is not None and dh_raw is not None and len(dl_raw) > 0 and len(dh_raw) > 0:
            min_len_raw = min(len(dl_raw), len(dh_raw), min_len)
            dl_raw = dl_raw[:min_len_raw]
            dh_raw = dh_raw[:min_len_raw]
        else:
            dl_raw = None
            dh_raw = None
        
        # 编码错误统计和可视化（使用平滑前的真值）
        file_prefix = base_name
        # 使用平滑前的真值进行编码错误分析
        dh_gt_for_encoding = dh_raw if dh_raw is not None else dh
        dh_np = dh_gt_for_encoding[:len(pred_binary_probs)] if len(pred_binary_probs) <= len(dh_gt_for_encoding) else dh_gt_for_encoding
        if isinstance(dh_np, torch.Tensor):
            dh_np = dh_np.cpu().numpy()
        if dh_np.ndim > 1:
            dh_np = dh_np[:, 0] if dh_np.shape[1] == 1 else dh_np.flatten()
        
        analyze_encoding_errors(
            dh_np,
            pred_binary_probs,
            pred_binary_hard,
            quantizer,
            output_bits,
            output_dir,
            file_prefix
        )

        # 生成轨迹（基于步长+航向角累积）
        traj_gt = generate_trajectory_2d(init_l, init_h, dl, dh[:len(dl)])
        traj_pred = generate_trajectory_2d(init_l, init_h, pred_len, pred_head_soft[:len(pred_len)])
        
        # 生成平滑前的轨迹（用于对比）
        traj_gt_raw = None
        if dl_raw is not None and dh_raw is not None:
            traj_gt_raw = generate_trajectory_2d(init_l_raw, init_h_raw, dl_raw, dh_raw[:len(dl_raw)])

        # 提取真值位置坐标（基于窗口对应的真值位置）
        num_windows = len(dl)
        traj_gt_xy = extract_ground_truth_positions(pos3d, window_size, stride, num_windows, init_l)
        
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

        plot_heading_analysis(dh, pred_head_soft, pred_head_hard, vis_len,
                             os.path.join(output_dir, f"{base_name}_heading_analysis.png"),
                             quantizer, dh_raw=dh_raw)

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
    if quantizer.fitted:
        bin_widths = np.diff(quantizer.bin_edges)
        print(f"  量化精度范围: [{np.degrees(bin_widths.min())/2:.2f}deg, {np.degrees(bin_widths.max())/2:.2f}deg]")
    print("="*60)

    print(f"\n测试完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
