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
    HeadingQuantizer, DualHeadingModel, DualHeadingLoss,
    compute_heading_mae
)
from models.regress import FeatureExtractor as RegFeatureExtractor, RegressorHead as RegHead
from utils.training_utils import (
    len_loss,
    load_data_2d_oxiod,
    load_data_2d_selfmade,
    load_data_2d_ronin
)


# ======= 参数设置 =======
window_size = 160
stride = 32
batch_size = 64
feat_dim = 64
output_dim_len = 1

# 航向角量化参数
num_bits = 10  # 必须是 4 的倍数
num_bins = 2 ** num_bits
use_adaptive_quantization = False  # [修改] 绝对航向使用均匀量化，禁用自适应量化
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


def train_dual_heading_model(model, train_loader, val_loader,
                              ckpt_dir, curve_dir):
    """训练双流航向模型 (绝对航向 + 相对航向，直接回归)"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if train_mode == 'fixed':
        scheduler = None
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # 初始化损失函数 (直接MSE)
    abs_loss_fn = torch.nn.MSELoss()
    rel_loss_fn = torch.nn.MSELoss()
    rel_weight = 10.0  # 相对航向损失权重

    ckpt_path = os.path.join(ckpt_dir, "dual_heading_model.pth")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        print("[Dual Heading] 发现已有最佳模型，跳过训练")
        return

    best_mae = float('inf')
    train_curve = []
    val_curve = []
    val_mae_curve = []
    abs_loss_curve = []
    rel_loss_curve = []
    no_improve = 0

    mode_str = "Fixed LR" if train_mode == 'fixed' else "Adaptive (Cosine+EarlyStop)"
    print(f">>> 开始训练双流航向模型 (Direct Regression, {mode_str})")

    for ep in range(epochs):
        t0 = time.time()

        model.train()
        total_loss = 0.0
        total_abs_loss = 0.0
        total_rel_loss = 0.0
        cnt = 0

        for xb, _, yb_head_abs, yb_head_rel in train_loader:
            pred_abs, pred_rel = model(xb)

            # 计算双流损失
            loss_abs = abs_loss_fn(pred_abs, yb_head_abs)
            loss_rel = rel_loss_fn(pred_rel, yb_head_rel)
            loss = loss_abs + rel_weight * loss_rel

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_abs_loss += loss_abs.item() * bs
            total_rel_loss += loss_rel.item() * bs
            cnt += bs

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = total_loss / max(cnt, 1)
        avg_abs_loss = total_abs_loss / max(cnt, 1)
        avg_rel_loss = total_rel_loss / max(cnt, 1)

        # 验证 Loop
        model.eval()
        vtotal = 0.0
        vcnt = 0
        all_preds_abs = []
        all_targets_abs = []

        with torch.no_grad():
            for xb, _, yb_head_abs, yb_head_rel in val_loader:
                pred_abs, pred_rel = model(xb)

                # 直接计算MAE
                abs_mae = torch.abs(pred_abs - yb_head_abs).mean()

                bs = xb.size(0)
                vtotal += abs_mae.item() * bs
                vcnt += bs

                all_preds_abs.append(pred_abs)
                all_targets_abs.append(yb_head_abs)

        val_loss = vtotal / max(vcnt, 1)

        # 计算绝对航向MAE (考虑角度周期性)
        all_preds_abs = torch.cat(all_preds_abs, dim=0)
        all_targets_abs = torch.cat(all_targets_abs, dim=0)
        mae = compute_heading_mae(all_preds_abs, all_targets_abs)

        train_curve.append(avg_train_loss)
        val_curve.append(val_loss)
        val_mae_curve.append(mae.item())
        abs_loss_curve.append(avg_abs_loss)
        rel_loss_curve.append(avg_rel_loss)

        # 保存最佳模型
        if mae.item() < best_mae:
            best_mae = mae.item()
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if train_mode == 'adaptive' and no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

        if (ep + 1) % 5 == 0 or ep == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"[Dual Heading Ep {ep+1}] "
                  f"Loss: {avg_train_loss:.4f} (Abs:{avg_abs_loss:.4f}, Rel:{avg_rel_loss:.4f}) "
                  f"| Val MAE: {np.degrees(mae.item()):.2f}° "
                  f"| Time: {time.time()-t0:.1f}s")

    # 绘制曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(train_curve, label='train')
    axes[0, 0].plot(val_curve, label='val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    axes[0, 1].plot(abs_loss_curve, label='abs loss', color='blue')
    axes[0, 1].plot(rel_loss_curve, label='rel loss', color='red')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    axes[1, 0].plot([np.degrees(m) for m in val_mae_curve])
    axes[1, 0].set_title('Validation MAE (Abs Heading)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (deg)')

    # 最佳 MAE 标记
    best_idx = np.argmin(val_mae_curve)
    axes[1, 0].scatter([best_idx], [np.degrees(val_mae_curve[best_idx])],
                       color='green', s=100, zorder=5,
                       label=f'Best: {np.degrees(val_mae_curve[best_idx]):.2f}deg')
    axes[1, 0].legend()

    # 训练信息
    axes[1, 1].text(0.5, 0.5,
                    f"Dual Heading Model\n"
                    f"(Direct Regression)\n"
                    f"Best MAE: {np.degrees(best_mae):.2f}deg\n"
                    f"Rel Weight: {rel_weight}",
                    ha='center', va='center', fontsize=12,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, 'curve_dual_heading.png'))
    plt.close()

    print(f"\n最佳验证 MAE: {np.degrees(best_mae):.2f}°")


def main():
    project_dir = "/home/admin407/code/zyshe/NavCorrector"
    data_root = os.path.join(project_dir, "OXIOD")
    selfmade_root = os.path.join(project_dir, "SELFMADE")
    ronin_root = os.path.join(project_dir, "RONIN")
    ckpt_dir = os.path.join(project_dir, f"checkpoints_cls_{dataset}")
    os.makedirs(ckpt_dir, exist_ok=True)
    curve_dir = os.path.join(project_dir, "output", f"trainc_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(curve_dir, exist_ok=True)
    

    print("="*60)
    print("双流航向回归训练（绝对航向 + 相对航向，直接回归）")
    print("="*60)
    print(f"  模型结构: DualHeadingModel (基于ResNet回归)")
    print(f"  绝对航向: 直接回归 [-π, π]")
    print(f"  相对航向: 直接回归 Δθ")
    print(f"  损失函数: MSELoss (Abs) + 10×MSELoss (Rel)")
    print(f"  训练模式: {train_mode} ({'固定学习率+固定轮数' if train_mode == 'fixed' else '余弦退火+早停'})")
    print(f"  学习率: {lr}, 权重衰减: {weight_decay}, 轮数: {epochs}")
    print("="*60)

    # 加载数据
    print("\n📊 加载训练数据...")
    if dataset == "SELFMADE" and os.path.isdir(selfmade_root):
        x_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr, x_va, ylen_va, yhead_abs_va, yhead_rel_va = load_data_2d_selfmade(selfmade_root, device, window_size, stride)
    elif dataset == "RONIN" and os.path.isdir(ronin_root):
        x_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr, x_va, ylen_va, yhead_abs_va, yhead_rel_va = load_data_2d_ronin(ronin_root, device, window_size, stride)
    else:
        x_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr, x_va, ylen_va, yhead_abs_va, yhead_rel_va = load_data_2d_oxiod(data_root, device, window_size, stride)
        
    print(f"训练集: {x_tr.shape[0]} 样本")
    print(f"验证集: {x_va.shape[0]} 样本")
    
    # 打印航向角分布
    head_abs_tr_np = yhead_abs_tr.cpu().numpy().flatten()
    head_rel_tr_np = yhead_rel_tr.cpu().numpy().flatten()
    print(f"绝对航向范围: [{np.degrees(head_abs_tr_np.min()):.1f}°, {np.degrees(head_abs_tr_np.max()):.1f}°]")
    print(f"绝对航向标准差: {np.degrees(head_abs_tr_np.std()):.1f}°")
    print(f"相对航向范围: [{np.degrees(head_rel_tr_np.min()):.1f}°, {np.degrees(head_rel_tr_np.max()):.1f}°]")
    print(f"相对航向标准差: {np.degrees(head_rel_tr_np.std()):.1f}°")

    # 创建数据集 (双流航向标签)
    train_dataset = TensorDataset(x_tr, ylen_tr, yhead_abs_tr, yhead_rel_tr)
    val_dataset = TensorDataset(x_va, ylen_va, yhead_abs_va, yhead_rel_va)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 模型初始化
    in_ch = x_tr.shape[-1]
    
    # 步长模型
    extractor_len = RegFeatureExtractor(in_channels=in_ch, feat_dim=feat_dim).to(device)
    reg_len = RegHead(feat_dim, output_dim_len).to(device)
    
    # 双流航向模型 (绝对航向 + 相对航向，使用regress.py的结构)
    dual_heading_model = DualHeadingModel(
        in_channels=in_ch,
        feat_dim=feat_dim
    ).to(device)

    # 训练
    print("\n🎯 训练步长模型")
    train_length_model(extractor_len, reg_len, train_loader, val_loader, ckpt_dir, curve_dir)

    print("\n🎯 训练双流航向模型 (Abs + Rel, Direct Regression)")
    train_dual_heading_model(dual_heading_model, train_loader, val_loader,
                            ckpt_dir, curve_dir)
    
    print("\n✅ 训练完成")
    print(f"   检查点保存在: {ckpt_dir}")
    print(f"   训练曲线保存在: {curve_dir}")


if __name__ == "__main__":
    main()
