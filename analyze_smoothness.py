"""
分析真值轨迹的光滑性问题
检查训练时使用的真值航向角是否足够光滑，以及如何改进
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_OXIOD import load_oxiod_raw, window_dataset as oxiod_window, moving_average
from data.dataset_SELFMADE import load_selfmade_raw, window_dataset as selfmade_window
from data.dataset_RONIN import load_ronin_raw, window_dataset as ronin_window
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def smooth_heading(heading, method='gaussian', window=5, sigma=1.0):
    """
    对航向角进行平滑处理
    
    Args:
        heading: 航向角数组 (N,)
        method: 平滑方法 ('gaussian', 'moving_average', 'savgol')
        window: 平滑窗口大小
        sigma: 高斯滤波的标准差（仅用于 gaussian 方法）
    
    Returns:
        平滑后的航向角
    """
    if method == 'gaussian':
        return gaussian_filter1d(heading, sigma=sigma)
    elif method == 'moving_average':
        return moving_average(heading, window)
    elif method == 'savgol':
        # Savitzky-Golay 滤波器需要窗口大小为奇数
        window = window if window % 2 == 1 else window + 1
        return savgol_filter(heading, window, 3)  # 3阶多项式
    else:
        return heading


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi] 范围"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def analyze_heading_smoothness(data_root, output_dir, dataset="OXIOD"):
    """
    分析航向角的光滑性
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        dataset: 数据集名称 ('OXIOD', 'SELFMADE', 'RONIN')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"数据集: {dataset}")
    
    # 根据数据集选择加载函数和文件
    if dataset == "SELFMADE":
        # SELFMADE: 查找第一个 CSV 或 MAT 文件
        imu_files = []
        for r, d, fns in os.walk(data_root):
            for fn in fns:
                if fn.lower().endswith('.csv') or fn.lower().endswith('.mat'):
                    imu_files.append(os.path.join(r, fn))
        if len(imu_files) == 0:
            print(f"未找到 SELFMADE 数据文件在: {data_root}")
            return
        imu_file = imu_files[0]
        gt_file = None
        print(f"正在分析文件: {os.path.basename(imu_file)}")
        gyro, acc, pos3d, ori = load_selfmade_raw(imu_file)
        window_fn = selfmade_window
        filter_window = 20
        
    elif dataset == "RONIN":
        # RONIN: 查找第一个测试目录
        seen_base = os.path.join(data_root, 'Data', 'seen_subjects_test_set')
        if not os.path.isdir(seen_base):
            print(f"未找到 RONIN 测试目录: {seen_base}")
            return
        dirs = [os.path.join(seen_base, d) for d in sorted(os.listdir(seen_base)) 
                if os.path.isdir(os.path.join(seen_base, d))]
        if len(dirs) == 0:
            print(f"未找到 RONIN 数据目录在: {seen_base}")
            return
        ronin_dir = dirs[0]
        print(f"正在分析目录: {os.path.basename(ronin_dir)}")
        gyro, acc, pos3d, ori = load_ronin_raw(ronin_dir)
        window_fn = ronin_window
        filter_window = 0
        
    else:  # OXIOD
        imu_file = os.path.join(data_root, 'handheld', 'data1', 'syn', 'imu2.csv')
        gt_file = imu_file.replace("imu", "vi")
        
        if not os.path.exists(imu_file) or not os.path.exists(gt_file):
            print(f"文件不存在: {imu_file} 或 {gt_file}")
            return
        
        print(f"正在分析文件: {os.path.basename(imu_file)}")
        gyro, acc, pos3d, ori = load_oxiod_raw(imu_file, gt_file)
        window_fn = oxiod_window
        filter_window = 20
    
    # 限制数据长度用于可视化
    max_len = 50000
    gyro = gyro[:max_len]
    acc = acc[:max_len]
    pos3d = pos3d[:max_len]
    ori = ori[:max_len]
    
    # 根据数据集设置窗口参数
    if dataset == "RONIN":
        window_size = 160
        stride = 32
    else:
        window_size = 128
        stride = 16
    
    # 1. 原始方法：只平滑位置，然后计算航向角
    print("方法1: 只平滑位置（当前训练方法）")
    [gx1, ax1], [dl1, dh1], init_l1, init_h1 = window_fn(
        gyro, acc, pos3d, ori,
        mode="2d",
        window_size=window_size,
        stride=stride,
        filter_window=filter_window,  # 当前使用的平滑窗口
        smooth_heading=False,  # 不平滑航向角
        smooth_length=False,   # 不平滑步长
    )
    
    # 2. 增加位置平滑窗口（仅对非RONIN数据集）
    if dataset != "RONIN":
        print("方法2: 增加位置平滑窗口")
        [gx2, ax2], [dl2, dh2], init_l2, init_h2 = window_fn(
            gyro, acc, pos3d, ori,
            mode="2d",
            window_size=window_size,
            stride=stride,
            filter_window=filter_window * 2,  # 更大的平滑窗口
            smooth_heading=False,
            smooth_length=False,
        )
    else:
        # RONIN 不使用位置平滑，所以方法2和方法1相同
        print("方法2: RONIN数据集不使用位置平滑，与方法1相同")
        gx2, ax2, dl2, dh2, init_l2, init_h2 = gx1, ax1, dl1.copy(), dh1.copy(), init_l1, init_h1
    
    # 3. 对航向角本身进行平滑（在方法1的基础上）
    print("方法3: 对航向角进行高斯平滑")
    dh3 = smooth_heading(dh1.flatten(), method='gaussian', sigma=5).reshape(-1, 1)
    
    # 4. 对航向角使用 Savitzky-Golay 滤波
    print("方法4: 对航向角使用 Savitzky-Golay 滤波")
    dh4 = smooth_heading(dh1.flatten(), method='savgol', window=9).reshape(-1, 1)
    
    # 5. 组合方法：更大的位置平滑 + 航向角平滑
    print("方法5: 大位置平滑 + 航向角平滑")
    dh5 = smooth_heading(dh2.flatten(), method='gaussian', sigma=1.0).reshape(-1, 1)
    
    # 计算累积航向角（用于可视化）
    def cumulative_heading(dh, init_h):
        """计算累积航向角"""
        cum_h = [init_h]
        for d in dh:
            cum_h.append(wrap_angle(cum_h[-1] + d[0]))
        return np.array(cum_h[:-1])
    
    h1_cum = cumulative_heading(dh1, init_h1)
    h2_cum = cumulative_heading(dh2, init_h2)
    h3_cum = cumulative_heading(dh3, init_h1)
    h4_cum = cumulative_heading(dh4, init_h1)
    h5_cum = cumulative_heading(dh5, init_h2)
    
    # 计算航向角变化率（用于评估光滑性）
    def compute_smoothness_metric(dh):
        """计算光滑性指标：航向角变化的标准差"""
        return np.std(dh), np.abs(np.diff(dh.flatten())).mean()
    
    metrics = {}
    metrics['Method1-Original'] = compute_smoothness_metric(dh1)
    metrics['Method2-LargePosSmooth'] = compute_smoothness_metric(dh2)
    metrics['Method3-HeadingGaussian'] = compute_smoothness_metric(dh3)
    metrics['Method4-HeadingSG'] = compute_smoothness_metric(dh4)
    metrics['Method5-Combined'] = compute_smoothness_metric(dh5)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("航向角光滑性统计:")
    print("="*60)
    for name, (std_val, mean_diff) in metrics.items():
        print(f"{name:20s}: std={np.degrees(std_val):.4f}°, mean_diff={np.degrees(mean_diff):.4f}°")
    print("="*60)
    
    # 可视化
    vis_len = min(500, len(dh1))
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. 航向角变化量对比
    axes[0, 0].plot(np.degrees(dh1[:vis_len, 0]), label='Method1-Original', alpha=0.7, linewidth=1)
    axes[0, 0].plot(np.degrees(dh2[:vis_len, 0]), label='Method2-LargePosSmooth', alpha=0.7, linewidth=1)
    axes[0, 0].plot(np.degrees(dh3[:vis_len, 0]), label='Method3-HeadingGaussian', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Heading Change (dh) Comparison')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Heading Change (deg)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 累积航向角对比
    axes[0, 1].plot(np.degrees(h1_cum[:vis_len]), label='Method1-Original', alpha=0.7, linewidth=1)
    axes[0, 1].plot(np.degrees(h2_cum[:vis_len]), label='Method2-LargePosSmooth', alpha=0.7, linewidth=1)
    axes[0, 1].plot(np.degrees(h3_cum[:vis_len]), label='Method3-HeadingGaussian', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Cumulative Heading Comparison')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cumulative Heading (deg)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 航向角变化量的直方图
    axes[1, 0].hist(np.degrees(dh1[:vis_len, 0]), bins=50, alpha=0.5, label='Method1-Original', density=True)
    axes[1, 0].hist(np.degrees(dh3[:vis_len, 0]), bins=50, alpha=0.5, label='Method3-HeadingGaussian', density=True)
    axes[1, 0].set_title('Heading Change Distribution')
    axes[1, 0].set_xlabel('Heading Change (deg)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 航向角变化率的对比（一阶差分）
    diff1 = np.abs(np.diff(dh1[:vis_len, 0]))
    diff3 = np.abs(np.diff(dh3[:vis_len, 0]))
    axes[1, 1].plot(np.degrees(diff1), label='Method1-Original', alpha=0.7, linewidth=1)
    axes[1, 1].plot(np.degrees(diff3), label='Method3-HeadingGaussian', alpha=0.7, linewidth=1)
    axes[1, 1].set_title('Heading Change Rate (|diff(dh)|)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Change Rate (deg)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 所有方法的对比（航向角变化量）
    axes[2, 0].plot(np.degrees(dh1[:vis_len, 0]), label='Method1-Original', alpha=0.6, linewidth=1)
    axes[2, 0].plot(np.degrees(dh2[:vis_len, 0]), label='Method2-LargePosSmooth', alpha=0.6, linewidth=1)
    axes[2, 0].plot(np.degrees(dh3[:vis_len, 0]), label='Method3-HeadingGaussian', alpha=0.6, linewidth=1)
    axes[2, 0].plot(np.degrees(dh4[:vis_len, 0]), label='Method4-HeadingSG', alpha=0.6, linewidth=1)
    axes[2, 0].plot(np.degrees(dh5[:vis_len, 0]), label='Method5-Combined', alpha=0.6, linewidth=1)
    axes[2, 0].set_title('All Methods Comparison (dh)')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Heading Change (deg)')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 光滑性指标对比
    methods = list(metrics.keys())
    std_vals = [np.degrees(metrics[m][0]) for m in methods]
    mean_diffs = [np.degrees(metrics[m][1]) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[2, 1].bar(x - width/2, std_vals, width, label='Std of dh', alpha=0.7)
    axes[2, 1].bar(x + width/2, mean_diffs, width, label='Mean |diff(dh)|', alpha=0.7)
    axes[2, 1].set_title('Smoothness Metrics Comparison')
    axes[2, 1].set_xlabel('Method')
    axes[2, 1].set_ylabel('Value (deg)')
    axes[2, 1].set_xticks(x)
    # Short method names for x-axis labels
    method_labels = ['M1-Orig', 'M2-Pos', 'M3-Gauss', 'M4-SG', 'M5-Comb']
    axes[2, 1].set_xticklabels(method_labels, rotation=45, ha='right')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heading_smoothness_analysis.png'), dpi=150)
    plt.close()
    
    print(f"\n分析结果已保存到: {os.path.join(output_dir, 'heading_smoothness_analysis.png')}")
    
    # 保存统计数据
    with open(os.path.join(output_dir, 'smoothness_metrics.txt'), 'w') as f:
        f.write("航向角光滑性统计\n")
        f.write("="*60 + "\n")
        for name, (std_val, mean_diff) in metrics.items():
            f.write(f"{name:20s}: std={np.degrees(std_val):.4f}°, mean_diff={np.degrees(mean_diff):.4f}°\n")
        f.write("="*60 + "\n")
    
    return metrics


def main():
    project_dir = "/home/admin407/code/zyshe/Corrector"
    
    # 数据集选择：'OXIOD', 'SELFMADE', 'RONIN'
    dataset = os.getenv('DATASET', 'OXIOD').upper()
    
    if dataset == "SELFMADE":
        data_root = os.path.join(project_dir, "SELFMADE")
    elif dataset == "RONIN":
        data_root = os.path.join(project_dir, "RONIN")
    else:
        data_root = os.path.join(project_dir, "OXIOD")
        dataset = "OXIOD"
    
    output_dir = os.path.join(project_dir, "output", f"smoothness_analysis_{dataset.lower()}")
    
    print("="*60)
    print(f"航向角光滑性分析 - {dataset} 数据集")
    print("="*60)
    
    analyze_heading_smoothness(data_root, output_dir, dataset=dataset)


if __name__ == "__main__":
    main()

