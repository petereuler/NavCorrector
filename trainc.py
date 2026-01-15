import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 引入我们刚才定好的模型
from models.end2end_pdr import EndToEndPDR

# 引入数据加载函数 (假设 utils 已按之前讨论修改好，或者我们在这里做适配)
from utils.training_utils import (
    load_data_2d_oxiod,
    load_data_2d_ronin,
    load_data_2d_selfmade
)

# ================= 配置参数 =================
CONFIG = {
    'window_size': 200,    # 根据你的数据集调整
    'stride': 10,          # 训练时的步进
    'batch_size': 64,
    'lr_pose': 1e-3,       # Phase 1 学习率
    'lr_nav': 1e-3,        # Phase 2 学习率
    'lr_joint': 1e-4,      # Phase 3 微调学习率
    'epochs_p1': 20,       # Phase 1: Pose Warmup
    'epochs_p2': 40,       # Phase 2: Nav Warmup
    'epochs_p3': 40,       # Phase 3: Joint Tuning
    'dataset': 'OXIOD',    # OXIOD, RONIN, SELFMADE
    'data_root': '/home/admin407/code/zyshe/NavCorrector/OXIOD', # 修改为你的数据路径
    'gpu_id': 0
}

device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")

# ================= 损失函数 =================
class PDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def gaussian_nll(self, pred_vec, target_vec, pred_var):
        """
        Gaussian Negative Log Likelihood Loss
        Loss = 0.5 * ( (y-y_hat)^2 / var + log(var) )
        pred_vec: (B, 2)
        target_vec: (B, 2)
        pred_var: (B, 1) scalar variance
        """
        # 计算向量的欧氏距离平方 ||y - y_hat||^2
        # 或者分别计算每个分量的 NLL。这里我们假设各分量共享方差，简化处理。
        diff_sq = torch.sum((pred_vec - target_vec)**2, dim=1, keepdim=True)
        
        # 加上 epsilon 防止 log(0) 或 除0
        # 模型输出的 var 已经是 softplus 处理过的，保证 > 0
        loss = 0.5 * (diff_sq / (pred_var + 1e-8) + torch.log(pred_var + 1e-8))
        return loss.mean()

    def pose_loss(self, pred_q, target_q, pred_q_var):
        """
        Pose NLL Loss based on Cosine Distance
        Dist = 1 - |<q1, q2>|
        """
        # Dot product
        dot = torch.sum(pred_q * target_q, dim=1, keepdim=True)
        # Manifold distance approximation: 1 - |dot|
        dist = 1.0 - torch.abs(dot)
        
        # NLL
        loss = 0.5 * (dist / (pred_q_var + 1e-8) + torch.log(pred_q_var + 1e-8))
        return loss.mean()

    def forward(self, preds, targets, mode='joint'):
        """
        preds: dict from model output
        targets: (q_gt, step_gt, abs_vec_gt, rel_vec_gt)
        mode: 'pose_only', 'nav_only', 'joint'
        """
        q_gt, step_gt, abs_gt, rel_gt = targets
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Pose Loss
        if mode in ['pose_only', 'joint']:
            l_pose = self.pose_loss(preds['q'], q_gt, preds['q_var'])
            loss_dict['loss_pose'] = l_pose
            total_loss += l_pose

        # 2. Nav Loss
        if mode in ['nav_only', 'joint']:
            # Step (MSE)
            l_step = self.mse(preds['step'], step_gt)
            
            # Abs Heading (NLL)
            l_abs = self.gaussian_nll(preds['abs_vec'], abs_gt, preds['abs_var'])
            
            # Rel Heading (NLL)
            l_rel = self.gaussian_nll(preds['rel_vec'], rel_gt, preds['rel_var'])
            
            loss_dict['loss_step'] = l_step
            loss_dict['loss_abs'] = l_abs
            loss_dict['loss_rel'] = l_rel
            
            total_loss += (l_step + l_abs + l_rel)

        return total_loss, loss_dict

# ================= 数据适配 =================
def prepare_data():
    print(f"Loading {CONFIG['dataset']} data...")
    
    if CONFIG['dataset'] == 'OXIOD':
        loader_func = load_data_2d_oxiod
    elif CONFIG['dataset'] == 'RONIN':
        loader_func = load_data_2d_ronin
    else:
        loader_func = load_data_2d_selfmade

    # [Fix] 正确解包 5 个返回值
    # 结构: (x_tr, x_va), (yq_tr, yq_va), (ylen_tr, ylen_va), (yabs_tr, yabs_va), (yrel_tr, yrel_va)
    (x_tr, x_va), (yq_tr, yq_va), (ylen_tr, ylen_va), \
    (yabs_rad_tr, yabs_rad_va), (yrel_rad_tr, yrel_rad_va) = \
        loader_func(CONFIG['data_root'], device, CONFIG['window_size'], CONFIG['stride'])

    def process_split(x, yq, yl, ya_rad, yr_rad):
        """
        处理单个数据集 (Train 或 Val)
        x: 已经是 (N, T, 6) 的 Tensor
        yq, yl, ya_rad, yr_rad: 对应标签 Tensor
        """
        # 1. 维度检查与调整 (以防万一 Channel First)
        # 现在的 training_utils 默认返回 (N, T, 6)，如果是 (N, 6, T) 则需要 permute
        if x.shape[1] == 6: 
            x = x.permute(0, 2, 1) # -> (N, T, 6)
        
        # 2. 转换角度为 sin/cos 向量 -> (N, 2)
        # 输入 ya_rad 是 (N, 1) -> 输出 (N, 2) [sin, cos]
        ya_vec = torch.cat([torch.sin(ya_rad), torch.cos(ya_rad)], dim=-1).float()
        yr_vec = torch.cat([torch.sin(yr_rad), torch.cos(yr_rad)], dim=-1).float()
        
        yq = yq.float()
        yl = yl.float()
        x = x.float()
        
        return TensorDataset(x, yq, yl, ya_vec, yr_vec)

    # 分别处理训练集和验证集
    train_ds = process_split(x_tr, yq_tr, ylen_tr, yabs_rad_tr, yrel_rad_tr)
    val_ds = process_split(x_va, yq_va, ylen_va, yabs_rad_va, yrel_rad_va)
    
    return train_ds, val_ds

# ================= 主训练循环 =================
def train():
    # 1. 准备
    save_dir = f"checkpoints/{CONFIG['dataset']}_{time.strftime('%m%d_%H%M')}"
    os.makedirs(save_dir, exist_ok=True)
    
    train_ds, val_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = EndToEndPDR(in_dim=6).to(device)
    criterion = PDRLoss().to(device)
    
    # 2. 定义优化器 (后续在 Loop 中动态调整参数组)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_pose'])
    
    total_epochs = CONFIG['epochs_p1'] + CONFIG['epochs_p2'] + CONFIG['epochs_p3']
    
    print(f"Start Training: Total {total_epochs} Epochs")
    print(f"Phase 1 (Pose): 0-{CONFIG['epochs_p1']}")
    print(f"Phase 2 (Nav): {CONFIG['epochs_p1']}-{CONFIG['epochs_p1']+CONFIG['epochs_p2']}")
    print(f"Phase 3 (Joint): {CONFIG['epochs_p1']+CONFIG['epochs_p2']}-{total_epochs}")

    loss_history = []

    for epoch in range(total_epochs):
        model.train()
        t0 = time.time()
        
        # === 阶段控制逻辑 ===
        if epoch < CONFIG['epochs_p1']:
            phase = 'Phase 1 (Pose)'
            mode = 'pose_only'
            # 冻结 Nav, 解冻 Pose
            for p in model.posenet.parameters(): p.requires_grad = True
            for p in model.navnet.parameters(): p.requires_grad = False
            lr = CONFIG['lr_pose']
            use_gt_rot = False # Pose阶段 Nav不工作，此参无所谓
            
        elif epoch < (CONFIG['epochs_p1'] + CONFIG['epochs_p2']):
            phase = 'Phase 2 (Nav)'
            mode = 'nav_only'
            # 冻结 Pose, 解冻 Nav
            for p in model.posenet.parameters(): p.requires_grad = False
            for p in model.navnet.parameters(): p.requires_grad = True
            lr = CONFIG['lr_nav']
            use_gt_rot = True # **关键**: 强行用 GT 姿态旋转数据喂给 NavNet
            
        else:
            phase = 'Phase 3 (Joint)'
            mode = 'joint'
            # 全部解冻
            for p in model.posenet.parameters(): p.requires_grad = True
            for p in model.navnet.parameters(): p.requires_grad = True
            lr = CONFIG['lr_joint']
            use_gt_rot = False # **关键**: 换回 PoseNet 的预测姿态，进行联调
            
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # === Batch 循环 ===
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # batch: [x_raw, y_q, y_len, y_abs_vec, y_rel_vec]
            x_raw, y_q, y_len, y_abs, y_rel = [b.to(device) for b in batch]
            
            # Forward
            # 注意: EndToEndPDR 的 forward 接口需支持 gt_q 和 use_gt_rotation
            preds = model(x_raw, gt_q=y_q, use_gt_rotation=use_gt_rot)
            
            # Loss
            loss, _ = criterion(preds, (y_q, y_len, y_abs, y_rel), mode=mode)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        
        print(f"Ep {epoch+1:3d} | {phase} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")
        
        # === 保存模型 ===
        # 在阶段切换点保存
        p1_end = CONFIG['epochs_p1']
        p2_end = p1_end + CONFIG['epochs_p2']
        
        if (epoch + 1) in [p1_end, p2_end, total_epochs]:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_ep{epoch+1}.pth"))
            print(f"Model saved to {save_dir}")

    # Plot
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss.png"))

if __name__ == "__main__":
    train()