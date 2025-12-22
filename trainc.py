import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import quaternion

from models.heading_classifier import (
    FeatureExtractor, RegressorHead,
    HeadingQuantizer, HeadingBinaryLoss,
    compute_bit_accuracy, compute_heading_mae, HeadingBinaryHead
)
from utils.training_utils import (
    len_loss,
    load_data_2d_oxiod,
    load_data_2d_selfmade,
    load_data_2d_ronin,
    plot_quantizer_analysis
)


# ======= 参数设置 =======
window_size = 160
stride = 32
batch_size = 64
feat_dim = 64
output_dim_len = 1

# 航向角量化参数
num_bits = 8  # 必须是 4 的倍数
num_bins = 2 ** num_bits
use_adaptive_quantization = False  # 启用自适应非均匀量化
# 计算输出位数
output_bits = num_bits

# 优化器参数
lr = 1e-4
weight_decay = 1e-4
epochs = 200

# 训练模式：'adaptive' (余弦退火+早停) 或 'fixed' (固定学习率+固定轮数)
train_mode = 'fixed'  # 'adaptive' or 'fixed'
early_stop_patience = 50  # 仅在 adaptive 模式下生效

# 数据增强
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
dataset = "OXIOD"

# 从环境变量读取
epochs = int(os.getenv('EPOCHS', epochs))


# ======= 训练函数 =======

def train_length_model(extractor, regressor, train_loader, val_loader, ckpt_dir, curve_dir):
    """训练步长回归模型"""
    optimizer = optim.AdamW(
        list(extractor.parameters()) + list(regressor.parameters()), 
        lr=lr, weight_decay=weight_decay
    )
    
    # 根据训练模式选择学习率调度器
    if train_mode == 'fixed':
        scheduler = None  # 固定学习率
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    ckpts = [os.path.join(ckpt_dir, f) for f in ["extractor_len.pth", "reg_len.pth"]]
    
    if os.path.exists(ckpts[0]) and os.path.exists(ckpts[1]):
        extractor.load_state_dict(torch.load(ckpts[0]))
        regressor.load_state_dict(torch.load(ckpts[1]))
        print("[Length] 发现已有最佳模型，跳过训练")
        return
    
    best_loss = float('inf')
    train_curve = []
    val_curve = []
    no_improve = 0
    
    mode_str = "Fixed LR" if train_mode == 'fixed' else "Adaptive (Cosine+EarlyStop)"
    print(f">>> 开始训练步长模型 (Regression, {mode_str})")
    for ep in range(epochs):
        t0 = time.time()
        extractor.train()
        regressor.train()
        total = 0.0
        cnt = 0
        
        for xb, yb_len, yb_head in train_loader:
            feat = extractor(xb)
            pred = regressor(feat)
            loss = len_loss(pred, yb_len)
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)
            optimizer.step()
            
            bs = xb.size(0)
            total += loss.item() * bs
            cnt += bs
        
        if scheduler is not None:
            scheduler.step()
        train_loss = total / max(cnt, 1)
        
        # 验证
        extractor.eval()
        regressor.eval()
        vtotal = 0.0
        vcnt = 0
        with torch.no_grad():
            for xb, yb_len, _ in val_loader:
                feat = extractor(xb)
                pred = regressor(feat)
                loss = len_loss(pred, yb_len)
                bs = xb.size(0)
                vtotal += loss.item() * bs
                vcnt += bs
        val_loss = vtotal / max(vcnt, 1)
        
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(extractor.state_dict(), ckpts[0])
            torch.save(regressor.state_dict(), ckpts[1])
            no_improve = 0
        else:
            no_improve += 1
        
        # 早停机制仅在 adaptive 模式下生效
        if train_mode == 'adaptive' and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
            
        if (ep + 1) % 10 == 0 or ep == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"[Length] Ep {ep+1}/{epochs} train={train_loss:.5f} val={val_loss:.5f} "
                  f"lr={current_lr:.2e} time={time.time()-t0:.1f}s")

    plt.figure()
    plt.plot(train_curve, label='train')
    plt.plot(val_curve, label='val')
    plt.title(f'Length Loss (MSE) - {mode_str}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(curve_dir, 'curve_length.png'))
    plt.close()


def train_heading_classifier(extractor, head, train_loader, val_loader,
                             ckpt_dir, curve_dir, quantizer):
    """训练航向分类模型 (Binary Output + Soft Decode Validation)"""
    optimizer = optim.AdamW(
        list(extractor.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    if train_mode == 'fixed':
        scheduler = None
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # 定义基础权重
    base_circular_weight = 1  # 建议稍微提高一点权重，因为现在Loss是可导的

    # 初始化 Loss (初始权重设为 0)
    criterion = HeadingBinaryLoss(
        num_bits=num_bits,
        use_gray_code=True,
        quantizer=quantizer,
        circular_weight=0.0
    )
    
    ckpts = [os.path.join(ckpt_dir, f) for f in ["extractor_head_cls.pth", "cls_head.pth"]]

    if os.path.exists(ckpts[0]) and os.path.exists(ckpts[1]):
        extractor.load_state_dict(torch.load(ckpts[0]))
        head.load_state_dict(torch.load(ckpts[1]))
        print("[Heading] 发现已有最佳模型，跳过训练")
        return

    best_mae = float('inf')
    train_curve = []
    val_curve = []
    val_mae_curve = []
    no_improve = 0

    mode_str = "Fixed LR" if train_mode == 'fixed' else "Adaptive (Cosine+EarlyStop)"
    print(f">>> 开始训练航向模型 (Binary Loss, Soft Val, {mode_str})")

    for ep in range(epochs):
        t0 = time.time()

        # Loss Warm-up: 前5轮不加几何约束，之后线性增加或固定
        if ep < 5:
            current_geo_weight = 0.0
        else:
            current_geo_weight = base_circular_weight
        criterion.circular_weight = current_geo_weight

        extractor.train()
        head.train()
        total_loss = 0.0
        total_bce = 0.0
        total_geo = 0.0
        cnt = 0

        for xb, _, yb_head in train_loader:
            feat = extractor(xb)
            logits = head(feat)

            # 获取详细 Loss
            loss, details = criterion(logits, yb_head, return_details=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_bce += details['bce'] * bs
            total_geo += details['geo'] * bs
            cnt += bs

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = total_loss / max(cnt, 1)
        avg_bce = total_bce / max(cnt, 1)
        avg_geo = total_geo / max(cnt, 1)

        # 验证 Loop
        extractor.eval()
        head.eval()
        vtotal = 0.0
        vcnt = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for xb, _, yb_head in val_loader:
                feat = extractor(xb)
                logits = head(feat)
                loss = criterion(logits, yb_head) # 验证集Loss只作参考

                bs = xb.size(0)
                vtotal += loss.item() * bs
                vcnt += bs

                all_logits.append(logits)
                all_targets.append(yb_head)

        val_loss = vtotal / max(vcnt, 1)

        # ==========================================
        # 关键修改：验证集使用 Soft Decoding 计算 MAE
        # ==========================================
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 使用我们在上一节实现的 decode_soft_expectation
        pred_heading_soft = quantizer.decode_soft_expectation(all_logits)

        # 计算 MAE (Pred 和 Target 都是 Tensor 且在 GPU 上)
        mae = compute_heading_mae(pred_heading_soft, all_targets)
        
        train_curve.append(avg_train_loss)
        val_curve.append(val_loss)
        val_mae_curve.append(mae.item())

        # 保存最佳模型
        if mae.item() < best_mae:
            best_mae = mae.item()
            torch.save(extractor.state_dict(), ckpts[0])
            torch.save(head.state_dict(), ckpts[1])
            no_improve = 0
        else:
            no_improve += 1

        if train_mode == 'adaptive' and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

        if (ep + 1) % 5 == 0 or ep == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            # 打印包含双流 Loss 的信息
            print(f"[Heading Ep {ep+1}] "
                  f"Loss: {avg_train_loss:.4f} (BCE_A:{avg_bce:.3f}, BCE_B:{avg_bce:.3f}, Geo:{avg_geo:.3f}, W:{current_geo_weight}) "
                  f"| Val MAE: {np.degrees(mae.item()):.2f}° "
                  f"| Time: {time.time()-t0:.1f}s")

    # 绘制曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(train_curve, label='train')
    axes[0, 0].plot(val_curve, label='val')
    axes[0, 0].set_title('Binary Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Validation Accuracy plot removed (not applicable for binary) or replaced
    axes[0, 1].text(0.5, 0.5, "Binary Output Mode\nAccuracy Metric N/A", ha='center')
    axes[0, 1].axis('off')
    
    axes[1, 0].plot([np.degrees(m) for m in val_mae_curve])
    axes[1, 0].set_title('Validation MAE (Soft Decoding)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (deg)')
    
    # 最佳 MAE 标记
    best_idx = np.argmin(val_mae_curve)
    axes[1, 0].scatter([best_idx], [np.degrees(val_mae_curve[best_idx])], 
                       color='green', s=100, zorder=5, label=f'Best: {np.degrees(val_mae_curve[best_idx]):.2f}deg')
    axes[1, 0].legend()
    
    # 绘制 bin 宽度分布（仅自适应量化）
    if use_adaptive_quantization and quantizer.fitted:
        bin_widths = np.diff(quantizer.bin_edges)
        axes[1, 1].bar(range(len(bin_widths)), np.degrees(bin_widths), alpha=0.7)
        axes[1, 1].axhline(y=np.degrees(2*np.pi/num_bins), color='r', linestyle='--', 
                          label=f'Uniform: {np.degrees(2*np.pi/num_bins):.2f}deg')
        axes[1, 1].set_title('Bin Width Distribution (Adaptive)')
        axes[1, 1].set_xlabel('Bin Index')
        axes[1, 1].set_ylabel('Width (deg)')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, f"Best MAE: {np.degrees(best_mae):.2f}deg\n"
                        f"Num Bits: {num_bits}\n",
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, 'curve_heading.png'))
    plt.close()
    
    print(f"\n最佳验证 MAE: {np.degrees(best_mae):.2f}°")


def main():
    project_dir = "/home/admin407/code/zyshe/Corrector"
    data_root = os.path.join(project_dir, "OXIOD")
    selfmade_root = os.path.join(project_dir, "SELFMADE")
    ronin_root = os.path.join(project_dir, "RONIN")
    ckpt_dir = os.path.join(project_dir, "checkpoints_cls")
    os.makedirs(ckpt_dir, exist_ok=True)
    curve_dir = os.path.join(project_dir, "output", f"trainc_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(curve_dir, exist_ok=True)
    
    quantizer_path = os.path.join(ckpt_dir, "quantizer.json")

    print("="*60)
    print("航向角量化分类训练（自适应非均匀量化版）")
    print("="*60)
    print(f"  位数: {num_bits} bits -> {num_bins} bins")
    print(f"  输出位数: {output_bits} bits")
    print(f"  自适应量化: {use_adaptive_quantization}")
    print(f"  损失函数: HeadingBinaryLoss (二进制编码)")
    print(f"  训练模式: {train_mode} ({'固定学习率+固定轮数' if train_mode == 'fixed' else '余弦退火+早停'})")
    print(f"  学习率: {lr}, 权重衰减: {weight_decay}, 轮数: {epochs}")
    print("="*60)

    # 加载数据
    print("\n📊 加载训练数据...")
    if dataset == "SELFMADE" and os.path.isdir(selfmade_root):
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va = load_data_2d_selfmade(selfmade_root, device, window_size, stride)
    elif dataset == "RONIN" and os.path.isdir(ronin_root):
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va = load_data_2d_ronin(ronin_root, device, window_size, stride)
    else:
        x_tr, ylen_tr, yhead_tr, x_va, ylen_va, yhead_va = load_data_2d_oxiod(data_root, device, window_size, stride)
        
    print(f"训练集: {x_tr.shape[0]} 样本")
    print(f"验证集: {x_va.shape[0]} 样本")
    
    # 打印航向角分布
    head_tr_np = yhead_tr.cpu().numpy().flatten()
    print(f"航向角范围: [{np.degrees(head_tr_np.min()):.1f}°, {np.degrees(head_tr_np.max()):.1f}°]")
    print(f"航向角标准差: {np.degrees(head_tr_np.std()):.1f}°")
    print(f"航向角中位数: {np.degrees(np.median(head_tr_np)):.1f}°")
    
    # 初始化量化器
    print("\n📐 初始化量化器...")
    quantizer = HeadingQuantizer(
        num_bins=num_bins,
        use_gray_code=True
    )
    
    # 检查是否已有保存的量化器
    if os.path.exists(quantizer_path):
        print(f"  发现已有量化器，从 {quantizer_path} 加载")
        quantizer.load(quantizer_path)
    else:
        # 仅使用训练集数据拟合量化器（防止数据泄漏）
        print("  使用训练集数据拟合量化器...")
        # 断言确保只使用训练集
        assert head_tr_np.shape[0] == len(x_tr), "量化器拟合数据必须仅包含训练集"
        quantizer.fit(head_tr_np)
        quantizer.save(quantizer_path)
    
    # 绘制量化器分析图
    plot_quantizer_analysis(quantizer, head_tr_np, curve_dir, num_bins)
    
    train_dataset = TensorDataset(x_tr, ylen_tr, yhead_tr)
    val_dataset = TensorDataset(x_va, ylen_va, yhead_va)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 模型初始化
    in_ch = x_tr.shape[-1]
    
    # 步长模型
    extractor_len = FeatureExtractor(in_channels=in_ch, feat_dim=feat_dim).to(device)
    reg_len = RegressorHead(feat_dim, output_dim_len).to(device)
    
    # 航向模型（使用改进的二进制分类头）
    extractor_head = FeatureExtractor(in_channels=in_ch, feat_dim=feat_dim).to(device)
    # 注意：这里输出维度是 output_bits
    head = HeadingBinaryHead(feat_dim, num_bits=output_bits, hidden_dim=256, dropout=0.3).to(device)

    # 训练
    print("\n🎯 训练步长模型")
    train_length_model(extractor_len, reg_len, train_loader, val_loader, ckpt_dir, curve_dir)
    
    print("\n🎯 训练航向分类模型 (Binary)")
    train_heading_classifier(extractor_head, head, train_loader, val_loader, 
                            ckpt_dir, curve_dir, quantizer)
    
    print("\n✅ 训练完成")
    print(f"   检查点保存在: {ckpt_dir}")
    print(f"   量化器保存在: {quantizer_path}")
    print(f"   训练曲线保存在: {curve_dir}")


if __name__ == "__main__":
    main()
