import torch
import torch.nn as nn
import torch.nn.functional as F
from .regress import FeatureExtractor  # 复用你原有的 ResNet 提取器

# ==========================================
# 工具函数：几何运算 (完全可微)
# ==========================================

def quaternion_rotate(q, v):
    """
    使用四元数 q 旋转向量 v
    Args:
        q: (B, 4) or (B, T, 4) [w, x, y, z]
        v: (B, T, 3)
    Returns:
        v_rotated: (B, T, 3)
    """
    # 维度对齐
    if q.dim() == 2:
        q = q.unsqueeze(1) # (B, 1, 4)
    
    # 提取分量
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # 罗德里格斯旋转公式的四元数形式
    rx = (ww + xx - yy - zz) * vx + 2 * (xy - wz) * vy + 2 * (xz + wy) * vz
    ry = 2 * (xy + wz) * vx + (ww - xx + yy - zz) * vy + 2 * (yz - wx) * vz
    rz = 2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (ww - xx - yy + zz) * vz

    return torch.stack([rx, ry, rz], dim=-1)

def robust_normalize_vector(v, eps=1e-8):
    """显式归一化向量，防止除零"""
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / (norm + eps)

# ==========================================
# 模块 I: PoseNet (姿态机)
# ==========================================

class PoseNet(nn.Module):
    """
    模拟 game_rv 的物理过程：
    输入 Raw IMU -> 学习重力对齐(Pitch/Roll) 和 陀螺仪积分(Yaw) -> 输出 Global Attitude
    """
    def __init__(self, in_dim=6, hidden_dim=128):
        super().__init__()
        # 使用 GRU 捕捉时序积分特性 (比 CNN 更适合做姿态积分)
        self.rnn = nn.GRU(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        
        # 输出头
        self.fc_q = nn.Linear(hidden_dim, 4)
        # 姿态的不确定性 (用于 NLL Loss)，这里简化为标量方差
        self.fc_var = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # x: (B, T, 6)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        
        # 我们取窗口中心或最后一个时刻的特征来代表这个窗口的姿态
        # 假设取最后一个时刻 (如果是实时处理)
        feat = out[:, -1, :] 
        
        # 1. 四元数预测
        q = self.fc_q(feat)
        q = F.normalize(q, p=2, dim=-1) # 强制归一化
        
        # 2. 不确定性预测 (Softplus 保证 > 0)
        q_log_var = self.fc_var(feat)
        q_var = F.softplus(q_log_var) + 1e-6
        
        return q, q_var

# ==========================================
# 模块 II: NavNet (导航机)
# ==========================================

class NavNet(nn.Module):
    """
    输入旋转后的 Global IMU -> 预测运动参数 + 动态不确定性 (Q, R)
    """
    def __init__(self, in_dim=6, feat_dim=128):
        super().__init__()
        # 复用 ResNet1D 提取波形特征
        self.extractor = FeatureExtractor(in_dim, feat_dim)
        
        # --- Head 1: Step Length (标量) ---
        self.head_step = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # --- Head 2: Abs Heading (观测 Z) ---
        # 输出: [raw_vec_x, raw_vec_y, var_param]
        self.head_abs = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 3) 
        )
        
        # --- Head 3: Rel Heading (控制 U) ---
        # 输出: [raw_vec_x, raw_vec_y, var_param]
        self.head_rel = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x_rotated):
        # x_rotated: (B, T, 6)
        feat = self.extractor(x_rotated)
        
        # 1. 步长
        step = self.head_step(feat) # (B, 1)
        
        # 2. 绝对航向 (Abs) - 观测
        out_abs = self.head_abs(feat)
        abs_vec = robust_normalize_vector(out_abs[:, :2]) # sin, cos 归一化
        # 预测观测噪声 R (Softplus)
        abs_var = F.softplus(out_abs[:, 2:3]) + 1e-6 
        
        # 3. 相对航向 (Rel) - 预测
        out_rel = self.head_rel(feat)
        rel_vec = robust_normalize_vector(out_rel[:, :2]) # sin, cos 归一化
        # 预测过程噪声 Q (Softplus)
        rel_var = F.softplus(out_rel[:, 2:3]) + 1e-6

        return step, abs_vec, abs_var, rel_vec, rel_var

# ==========================================
# 模块 III: Differentiable EKF (核心创新)
# ==========================================

class DifferentiableEKF(nn.Module):
    """
    完全可微的 EKF 层。
    将 NavNet 输出的 Rel/Abs 及其方差作为输入，
    在计算图中执行滤波，允许梯度回传以优化方差参数 Q 和 R。
    """
    def __init__(self):
        super().__init__()
        # 初始协方差 P0 (可学习或固定)
        # 这里设为固定的大值，表示初始不确定
        self.register_buffer('init_P', torch.tensor([1.0]))

    def forward(self, u_vec, Q, z_vec, R, init_theta=None):
        """
        Args:
            u_vec: (B, T, 2) [sin, cos] 相对转角 (控制量)
            Q:     (B, T, 1) 过程噪声方差
            z_vec: (B, T, 2) [sin, cos] 绝对航向 (观测量)
            R:     (B, T, 1) 观测噪声方差
            init_theta: (B, 1) 初始航向，若为None则用第一帧观测初始化
        
        Returns:
            fused_theta: (B, T, 1) 滤波后的航向角
            P_history:   (B, T, 1) 协方差历史 (用于Loss正则化)
        """
        batch_size, seq_len, _ = u_vec.shape
        device = u_vec.device
        
        # 初始化状态
        if init_theta is None:
            # 使用第一帧的 Abs 观测作为初始状态
            init_theta = torch.atan2(z_vec[:, 0, 0], z_vec[:, 0, 1]).unsqueeze(1) # (B, 1)
        
        curr_theta = init_theta
        curr_P = self.init_P.expand(batch_size, 1) # (B, 1)
        
        fused_thetas = []
        P_history = []
        
        # --- EKF 循环 (时间步展开) ---
        for t in range(seq_len):
            # 1. 提取当前步的参数
            u_sin, u_cos = u_vec[:, t, 0:1], u_vec[:, t, 1:2]
            z_sin, z_cos = z_vec[:, t, 0:1], z_vec[:, t, 1:2]
            q_t = Q[:, t, :]
            r_t = R[:, t, :]
            
            # 2. 预测步 (Predict)
            # 状态预测: theta_k|k-1 = theta_k-1 + delta_theta
            # 利用 sin/cos 公式避免直接加角度导致的奇异性，这里为简便直接加角度
            # 先算出 delta_theta 的角度值
            delta_theta = torch.atan2(u_sin, u_cos)
            theta_pred = curr_theta + delta_theta
            
            # 协方差预测: P_k|k-1 = P_k-1 + Q
            P_pred = curr_P + q_t
            
            # 3. 更新步 (Update)
            # 计算卡尔曼增益: K = P_pred / (P_pred + R)
            K = P_pred / (P_pred + r_t + 1e-8)
            
            # 观测残差 (Innovation): y = z - theta_pred
            # 必须处理角度周期性 (-pi to pi)
            z_theta = torch.atan2(z_sin, z_cos)
            y = z_theta - theta_pred
            y = torch.atan2(torch.sin(y), torch.cos(y)) # Wrap angle
            
            # 状态更新: theta_k = theta_pred + K * y
            curr_theta = theta_pred + K * y
            
            # 协方差更新: P_k = (1 - K) * P_pred
            curr_P = (1 - K) * P_pred
            
            # 保存结果
            fused_thetas.append(curr_theta)
            P_history.append(curr_P)
            
        fused_thetas = torch.stack(fused_thetas, dim=1) # (B, T, 1)
        P_history = torch.stack(P_history, dim=1)       # (B, T, 1)
        
        return fused_thetas, P_history

# ==========================================
# 模块 IV: 端到端总成 (End-to-End PDR)
# ==========================================

class EndToEndPDR(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()
        self.posenet = PoseNet(in_dim)
        self.navnet = NavNet(in_dim)
        self.ekf = DifferentiableEKF()

    def forward(self, x_raw, gt_q=None, use_gt_rotation=False):
        """
        Args:
            x_raw: (B, T, 6) Raw IMU
            gt_q: (B, 4) or (B, T, 4) 真值四元数 (Phase 2 用)
            use_gt_rotation: Bool (Phase 2 用)
        Returns:
            字典包含所有中间变量和最终轨迹
        """
        # 1. PoseNet 估计姿态
        # PoseNet 输出的是窗口级特征，这里假设它输出序列中每一帧的姿态
        # 如果 PoseNet 也是 seq-to-seq，则 q_pred: (B, T, 4)
        # 之前的 PoseNet 定义是输出单帧，为了支持 EKF，我们需要 PoseNet 具备序列输出能力
        # **注意**: 上面的 PoseNet 定义取了 feat[:, -1, :]，这是一个窗口一个输出。
        # 为了做序列滤波，我们需要 NavNet 是 seq-to-seq 的。
        # 如果数据集是切窗的（例如 200 帧一个窗口），我们可以认为在这个窗口内做序列滤波。
        
        pred_q, pred_q_var = self.posenet(x_raw)
        
        # 2. 坐标系旋转
        if use_gt_rotation and gt_q is not None:
            q_for_rot = gt_q
            if q_for_rot.dim() == 2: q_for_rot = q_for_rot.unsqueeze(1) # Broadcast
        else:
            q_for_rot = pred_q
            # 扩展 q 到时间维度 (假设窗口内姿态变化主要由 Gyro 积分处理，
            # 这里 q_pred 代表窗口基准姿态)
            if q_for_rot.dim() == 2: q_for_rot = q_for_rot.unsqueeze(1)
        
        acc = x_raw[..., 3:]
        gyro = x_raw[..., :3]
        
        # 旋转到 Global (保留重力)
        acc_g = quaternion_rotate(q_for_rot, acc)
        gyro_g = quaternion_rotate(q_for_rot, gyro)
        x_rotated = torch.cat([gyro_g, acc_g], dim=-1)
        
        # 3. NavNet 预测 (假设 NavNet 也是 seq-to-seq，ResNet1D 需要调整 stride=1 才能做到)
        # 如果 NavNet 输出是 (B, 1, ...)，那 EKF 只能跑一步。
        # 为了 ICML 故事，建议 NavNet 输出序列 (B, T, ...)。
        # **暂时假设**: 你的 NavNet 结构输出的是 (B, 1, ...)，即一个窗口算一个位移。
        # **为了跑通 EKF**: 我们将 input 视为一个 Sequence (T个窗口组成的序列)，而不是 T 帧。
        # 在 `trainc.py` 里，我们需要构造 (Batch, Sequence_of_Windows, 6, Window_Size) 的数据。
        # **或者**: 简化版 ICML —— 在窗口内部不做 EKF，而是把 NavNet 的输出当成 EKF 的一步。
        # 也就是：T = Sequence Length (e.g. 10 个窗口连在一起)。
        
        # 这里按 NavNet 输出 (B, 1) 来写，外部循环调用或 Input 是 (B, Seq, Feat)
        # 为了兼容你现在的代码，我们假设 x_raw 是 (B, T, 6)，输出是一次性的。
        # **修正**: 为了实现 Differentiable EKF，必须是序列数据。
        # 我们假设输入 x_raw 已经是 (B, 6, window_size) 的格式，
        # NavNet 输出 (B, 1, ...)。
        # 这样 EKF 无法在模型内部展开 T 步，除非改变数据加载方式。
        
        # **关键决策**: 保持当前模型简单性，EKF 层放在 NavNet 之后。
        # 如果输入是单个窗口，EKF 只能做单步更新 (One-step EKF)。
        # 要做序列 EKF，需要在 trainc.py 里把多个窗口的输出拼起来喂给 EKF。
        # 或者，把 EndToEndPDR 设计为接收 (Batch, Sequence, Window_Size, 6)。
        
        step, abs_vec, abs_var, rel_vec, rel_var = self.navnet(x_rotated)
        
        # 此时 step: (B, 1), abs_vec: (B, 2), etc.
        # 我们在这里暂时不调用 ekf.forward 的循环，因为只有一个时间步。
        # 真正的序列训练需要在外部 Loop 或者改变 Input Shape。
        # 这里返回所有量供 trainc.py 处理。
        
        return {
            'q': pred_q, 'q_var': pred_q_var,
            'step': step,
            'abs_vec': abs_vec, 'abs_var': abs_var,
            'rel_vec': rel_vec, 'rel_var': rel_var
        }