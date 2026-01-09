import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from .regress import FeatureExtractor as RegFeatureExtractor, RegressorHead as RegHead


# ==================== ResNet 基础模块 ====================

class ResidualBlock1D(nn.Module):
    """
    1D 残差块，用于时序信号处理
    输入: (B, C, T) -> 输出: (B, C, T)
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class ResidualBlockDown1D(nn.Module):
    """
    带下采样的 1D 残差块
    输入: (B, in_ch, T) -> 输出: (B, out_ch, T//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接的投影
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


# ==================== 特征提取器 (ResNet) ====================

class FeatureExtractor(nn.Module):
    """
    基于 ResNet 的特征提取器
    输入: (B, T, C) -> 输出: (B, feat_dim)
    """
    def __init__(self, in_channels, feat_dim, hidden_dim=64, num_blocks=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 残差块堆叠（逐步下采样）
        self.layer1 = nn.Sequential(
            ResidualBlock1D(hidden_dim, dropout=dropout),
            ResidualBlock1D(hidden_dim, dropout=dropout),
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim, hidden_dim * 2, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 2, dropout=dropout),
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlockDown1D(hidden_dim * 2, hidden_dim * 4, stride=2, dropout=dropout),
            ResidualBlock1D(hidden_dim * 4, dropout=dropout),
        )
        
        # 全局平均池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, feat_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, T, C) -> 转换为 (B, C, T) 用于 Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局池化
        x = self.global_pool(x)  # (B, C, 1)
        x = x.squeeze(-1)       # (B, C)
        
        return self.fc(x)       # (B, feat_dim)


# ==================== 回归头 ====================

class RegressorHead(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, feat):
        return self.fc(feat)


# ==================== 格雷码工具函数 ====================

def binary_to_gray(n):
    """将二进制数转换为格雷码"""
    return n ^ (n >> 1)


def gray_to_binary(gray):
    """将格雷码转换为二进制数"""
    n = gray
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def int_to_binary_array(n, num_bits):
    """将整数转换为二进制数组"""
    return np.array([(n >> i) & 1 for i in range(num_bits - 1, -1, -1)], dtype=np.float32)


def binary_array_to_int(arr):
    """将二进制数组转换为整数"""
    result = 0
    for bit in arr:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(a, b):
    """计算两个整数的汉明距离"""
    xor = a ^ b
    dist = 0
    while xor:
        dist += xor & 1
        xor >>= 1
    return dist


def create_gray_code_table(num_bits):
    """
    创建格雷码查找表
    
    Returns:
        bin_to_gray: 二进制索引 -> 格雷码
        gray_to_bin: 格雷码 -> 二进制索引
    """
    num_codes = 2 ** num_bits
    bin_to_gray = np.array([binary_to_gray(i) for i in range(num_codes)], dtype=np.int64)
    gray_to_bin = np.zeros(num_codes, dtype=np.int64)
    for i, g in enumerate(bin_to_gray):
        gray_to_bin[g] = i
    return bin_to_gray, gray_to_bin




# ==================== 航向角量化器 ====================

class HeadingQuantizer:
    """
    航向角均匀量化器
    将连续的航向角（-π到π）均匀量化到离散的bin索引
    """
    def __init__(self, num_bins=256, use_gray_code=True):
        self.num_bins = num_bins
        self.num_bits = int(np.log2(num_bins))
        self.use_gray_code = use_gray_code
        self.adaptive = False  # HeadingQuantizer始终使用均匀量化

        # 输出位数等于输入位数
        self.code_bits = self.num_bits

        # 航向角范围：-π 到 π
        self.angle_range = 2 * np.pi
        self.bin_width = self.angle_range / num_bins

        # 计算bin边界和中心（用于可视化）
        self.bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)  # num_bins + 1 个边界
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # num_bins 个中心

        # 创建格雷码表（如果使用）
        if use_gray_code:
            self.bin_to_gray, self.gray_to_bin = create_gray_code_table(self.num_bits)
            # 预计算所有格雷码组合，用于软解码
            self.all_gray_codes = np.array([int_to_binary_array(self.bin_to_gray[i], self.num_bits)
                                           for i in range(self.num_bins)], dtype=np.float32)
        else:
            self.bin_to_gray = None
            self.gray_to_bin = None
            # 对于标准二进制编码，预计算所有可能组合
            self.all_gray_codes = np.array([int_to_binary_array(i, self.num_bits)
                                           for i in range(self.num_bins)], dtype=np.float32)

        self.fitted = False

    def fit(self, heading_data):
        """拟合量化器（对于均匀量化，实际上不需要拟合）"""
        self.fitted = True
        return self

    def encode(self, heading_angles):
        """
        将航向角编码为bin索引
        Args:
            heading_angles: numpy数组，弧度单位
        Returns:
            bin_indices: bin索引数组
        """
        # 将航向角映射到[0, 2π]范围
        angles_normalized = (heading_angles + np.pi) % (2 * np.pi)

        # 计算bin索引
        bin_indices = np.floor(angles_normalized / self.bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        return bin_indices

    def decode(self, bin_indices):
        """
        将bin索引解码为航向角
        Args:
            bin_indices: bin索引数组
        Returns:
            heading_angles: 航向角数组（弧度）
        """
        # 计算每个bin的中心角度
        angles_normalized = (bin_indices + 0.5) * self.bin_width

        # 映射回[-π, π]范围
        heading_angles = angles_normalized - np.pi

        return heading_angles

    def decode_from_binary_vector(self, binary_probs):
        """
        从二进制概率向量解码为航向角
        Args:
            binary_probs: 二进制概率数组，shape=(batch_size, num_bits)，值在[0,1]之间
        Returns:
            heading_angles: 航向角数组（弧度）
        """
        # 处理输入类型：转换为 numpy 数组（如果需要）
        if hasattr(binary_probs, 'cpu'):
            # PyTorch 张量
            binary_probs = binary_probs.cpu().numpy()

        # 将概率转换为二进制向量（阈值0.5）
        binary_vectors = (binary_probs >= 0.5).astype(int)

        # 将二进制向量转换为整数索引
        bin_indices = []
        for binary_vec in binary_vectors:
            if self.use_gray_code:
                # 如果使用格雷码，需要先转换为二进制
                gray_value = binary_array_to_int(binary_vec)
                bin_idx = self.gray_to_bin[gray_value]
            else:
                # 直接转换为整数
                bin_idx = binary_array_to_int(binary_vec)
            bin_indices.append(bin_idx)

        bin_indices = np.array(bin_indices)

        # 使用现有的decode方法
        return self.decode(bin_indices)

    def decode_soft_expectation(self, logits):
        """
        高级解码：利用Bit概率计算全Bin分布，再求角度期望。
        输入: logits (batch_size, num_bits)
        输出: heading_angles (batch_size,)
        """
        batch_size, num_bits = logits.shape
        device = logits.device

        # 转换为概率
        probs = torch.sigmoid(logits)  # (batch_size, num_bits)

        # 获取所有可能的编码组合
        all_codes = torch.tensor(self.all_gray_codes, device=device, dtype=torch.float32)  # (num_bins, num_bits)

        # 计算每个bin的log概率
        # P(Bin_i) = Product over bits: P(bit_j)^code_j * (1-P(bit_j))^(1-code_j)
        # 使用log概率避免数值下溢

        log_probs_1 = torch.log(probs + 1e-8).unsqueeze(1)  # (batch_size, 1, num_bits)
        log_probs_0 = torch.log(1 - probs + 1e-8).unsqueeze(1)  # (batch_size, 1, num_bits)
        codes_expanded = all_codes.unsqueeze(0)  # (1, num_bins, num_bits)

        # 计算每个bin的log概率：sum over bits [code_j * log(P_1) + (1-code_j) * log(P_0)]
        bin_log_probs = torch.sum(
            codes_expanded * log_probs_1 + (1 - codes_expanded) * log_probs_0,
            dim=2
        )  # (batch_size, num_bins)

        # 转换为概率
        bin_probs = torch.softmax(bin_log_probs, dim=1)  # (batch_size, num_bins)

        # 计算角度期望（使用向量平均处理周期性）
        angles_map = torch.tensor(self.bin_centers, device=device, dtype=torch.float32)  # (num_bins,)

        # 将角度转为单位向量
        sin_angles = torch.sin(angles_map)  # (num_bins,)
        cos_angles = torch.cos(angles_map)  # (num_bins,)

        # 计算加权平均
        sin_sum = torch.sum(bin_probs * sin_angles, dim=1)  # (batch_size,)
        cos_sum = torch.sum(bin_probs * cos_angles, dim=1)  # (batch_size,)

        # 从向量平均恢复角度
        pred_angles = torch.atan2(sin_sum, cos_sum)

        return pred_angles

    def encode_to_binary_vector(self, heading_angles):
        """
        将航向角编码为二进制向量
        Args:
            heading_angles: numpy数组，弧度单位
        Returns:
            binary_vectors: 二进制向量数组，shape=(batch_size, code_bits)
        """
        bin_indices = self.encode(heading_angles)

        binary_vectors = []
        for bin_idx in bin_indices:
            if self.use_gray_code:
                # 使用格雷码编码
                gray_code = self.bin_to_gray[bin_idx]
                binary_vec = int_to_binary_array(gray_code, self.num_bits)
            else:
                # 使用普通二进制编码
                binary_vec = int_to_binary_array(bin_idx, self.num_bits)

            binary_vectors.append(binary_vec)

        return np.array(binary_vectors)

    def get_soft_labels(self, heading_angles, sigma=0.1):
        """
        获取软标签（用于标签平滑）
        """
        bin_indices = self.encode(heading_angles)

        # 创建软标签
        soft_labels = np.zeros((len(heading_angles), self.num_bins), dtype=np.float32)

        for i, bin_idx in enumerate(bin_indices):
            # 高斯分布生成软标签
            distances = np.abs(np.arange(self.num_bins) - bin_idx)
            weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
            weights /= weights.sum()  # 归一化
            soft_labels[i] = weights

        return soft_labels


    def save(self, filepath):
        """保存量化器到文件"""
        state = {
            'num_bins': self.num_bins,
            'num_bits': self.num_bits,
            'use_gray_code': self.use_gray_code,
            'code_bits': self.code_bits,
            'angle_range': self.angle_range,
            'bin_width': self.bin_width,
            'bin_edges': self.bin_edges,
            'bin_centers': self.bin_centers,
            'adaptive': self.adaptive,
            'fitted': self.fitted
        }

        # 保存格雷码表（如果存在）
        if self.bin_to_gray is not None:
            state['bin_to_gray'] = self.bin_to_gray
            state['gray_to_bin'] = self.gray_to_bin

        with open(filepath, 'wb') as f:
            np.savez(f, **state)

    def load(self, filepath):
        """从文件加载量化器"""
        with open(filepath, 'rb') as f:
            state = np.load(f)

            self.num_bins = int(state['num_bins'])
            self.num_bits = int(state['num_bits'])
            self.use_gray_code = bool(state['use_gray_code'])
            self.code_bits = int(state['code_bits'])
            self.angle_range = float(state['angle_range'])
            self.bin_width = float(state['bin_width'])
            self.bin_edges = state['bin_edges']
            self.bin_centers = state['bin_centers']
            # 向后兼容：如果旧文件没有adaptive属性，默认设置为False
            self.adaptive = bool(state.get('adaptive', False))
            self.fitted = bool(state['fitted'])

            # 加载格雷码表（如果存在）
            if self.use_gray_code:
                self.bin_to_gray = state['bin_to_gray']
                self.gray_to_bin = state['gray_to_bin']
            else:
                self.bin_to_gray = None
                self.gray_to_bin = None

# ==================== 航向角分类头 ====================

class HeadingBinaryHead(torch.nn.Module):
    """
    航向角二进制编码头：
    - 输出 num_bits 个 logit（支持汉明码的14位编码）
    - 更深的网络
    - 残差连接
    - LayerNorm
    """
    def __init__(self, feat_dim, num_bits=8, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.num_bits = num_bits

        self.fc1 = torch.nn.Linear(feat_dim, hidden_dim)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = torch.nn.LayerNorm(hidden_dim // 2)
        self.fc_out = torch.nn.Linear(hidden_dim // 2, num_bits)

        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # 残差投影
        self.proj = torch.nn.Linear(feat_dim, hidden_dim)
        
    def forward(self, feat):
        # 第一层 + 残差
        h = self.fc1(feat)
        h = self.ln1(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        # 第二层 + 残差
        res = self.proj(feat)
        h = self.fc2(h) + res
        h = self.ln2(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        # 第三层
        h = self.fc3(h)
        h = self.ln3(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        return self.fc_out(h)

# ==================== 损失函数 ====================


# models/heading_classifier.py

class HeadingBinaryLoss(nn.Module):
    """航向角二进制编码损失（支持汉明码 + 累积误差约束）"""
    # 1. 修改 __init__ 方法，增加 cumulative_weight 参数
    def __init__(self, num_bits=8, use_gray_code=True, quantizer=None, 
                 circular_weight=0.1, cumulative_weight=1.0):
        super().__init__()
        self.num_bits = num_bits
        self.num_bins = 2 ** num_bits
        self.circular_weight = circular_weight
        self.cumulative_weight = cumulative_weight  # 新增：累积误差权重

        if quantizer is not None:
            self.quantizer = quantizer
            self.output_bits = quantizer.code_bits
        else:
            self.quantizer = HeadingQuantizer(
                num_bins=self.num_bins,
                use_gray_code=use_gray_code
            )
            self.output_bits = self.quantizer.code_bits

        all_codes = torch.tensor(self.quantizer.all_gray_codes, dtype=torch.float32)
        self.register_buffer('all_gray_codes', all_codes)

        bin_centers = torch.tensor(self.quantizer.bin_centers, dtype=torch.float32)
        self.register_buffer('bin_centers', bin_centers)

    def _differentiable_decode(self, logits):
        """内部使用的完全可微分的解码过程"""
        # (保持原有的 _differentiable_decode 代码不变)
        device = logits.device
        probs = torch.sigmoid(logits)
        log_probs_1 = torch.log(probs + 1e-8).unsqueeze(1)
        log_probs_0 = torch.log(1 - probs + 1e-8).unsqueeze(1)
        codes = self.all_gray_codes.to(device).unsqueeze(0)
        bin_log_probs = torch.sum(codes * log_probs_1 + (1 - codes) * log_probs_0, dim=2)
        bin_probs = torch.softmax(bin_log_probs, dim=1)
        bin_centers = self.bin_centers.to(device)
        sin_sum = torch.sum(bin_probs * torch.sin(bin_centers), dim=1)
        cos_sum = torch.sum(bin_probs * torch.cos(bin_centers), dim=1)
        pred_angles = torch.atan2(sin_sum, cos_sum)
        return pred_angles

    def forward(self, logits, target_heading, return_details=False):
        """
        增加累积误差项 (Bias Loss)
        """
        target_heading = target_heading.squeeze(-1)

        # 1. 基础 BCE Loss
        target_binary = self.quantizer.encode_to_binary_vector(target_heading.cpu().numpy())
        target_binary = torch.tensor(target_binary, device=logits.device, dtype=torch.float32)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target_binary, reduction='mean')

        # 获取软解码角度（用于几何约束和累积误差）
        if self.circular_weight > 0 or self.cumulative_weight > 0:
            pred_angles_soft = self._differentiable_decode(logits)
        else:
            pred_angles_soft = None

        # 2. 几何约束 Loss (Circular/Cosine Loss)
        if self.circular_weight > 0:
            circular_loss = 1.0 - torch.cos(pred_angles_soft - target_heading)
            circular_loss = circular_loss.mean()
        else:
            circular_loss = torch.tensor(0.0, device=logits.device)

        # 3. [新增] 累积误差 Loss (Cumulative/Bias Loss)
        if self.cumulative_weight > 0:
            # 计算预测值与真值的差值 (Diff)
            diff = pred_angles_soft - target_heading
            # 处理周期性，将差值限制在 [-pi, pi]
            diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
            
            # 核心逻辑：计算 Batch 内误差的均值，然后平方
            # 这惩罚了 Batch 整体的偏移 (Bias)，模拟累积误差
            bias_val = torch.mean(diff_wrapped)
            cumulative_loss = bias_val ** 2 
            # 也可以使用 torch.abs(bias_val)
        else:
            cumulative_loss = torch.tensor(0.0, device=logits.device)

        # 总 Loss
        loss = bce_loss + \
               self.circular_weight * circular_loss + \
               self.cumulative_weight * cumulative_loss

        if return_details:
            return loss, {
                "bce": bce_loss.item(), 
                "geo": circular_loss.item(),
                "cum": cumulative_loss.item() # 返回 cum loss 供打印
            }
        else:
            return loss


# ==================== 评估函数 ====================

def compute_bit_accuracy(logits, target_heading, quantizer):
    """计算比特级的准确率"""
    target_heading = target_heading.squeeze(-1)

    # 1. 获取目标的二进制向量 (batch_size, num_bits)
    target_binary = quantizer.encode_to_binary_vector(target_heading.cpu().numpy())
    target_binary = torch.tensor(target_binary, device=logits.device, dtype=torch.float32)

    # 2. 获取预测的二进制向量 (batch_size, num_bits)
    probs = torch.sigmoid(logits)
    pred_binary = (probs >= 0.5).float()

    # 3. 计算每个bit的准确率
    correct_bits = (pred_binary == target_binary).float()
    bit_acc = correct_bits.mean().item()

    # 4. 计算完全匹配率 (Exact Match Ratio) - 即预测出了完全正确的Bin
    # 只有当一行中所有bit都对，才算对
    row_correct = (correct_bits.sum(dim=1) == quantizer.num_bits).float()
    bin_acc = row_correct.mean().item()

    return {"bit_acc": bit_acc, "bin_acc": bin_acc}


def compute_heading_mae(pred_heading, target_heading):
    """计算航向角平均绝对误差"""
    target_heading = target_heading.squeeze(-1) if target_heading.dim() > 1 else target_heading
    diff = pred_heading - target_heading
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return torch.abs(diff).mean()


class DualHeadingModel(torch.nn.Module):
    """
    双流航向预测模型（使用regress.py中的回归结构）：
    - 共享骨干网络 (RegFeatureExtractor - ResNet)
    - 绝对航向头：RegHead (直接回归绝对航向)
    - 相对航向头：RegHead (直接回归相对航向变化)
    """
    def __init__(self, in_channels, feat_dim=64):
        super().__init__()

        # 共享骨干网络 (使用regress.py中的ResNet结构)
        self.feature_extractor = RegFeatureExtractor(in_channels, feat_dim)

        # 绝对航向头 (使用regress.py中的回归结构)
        self.abs_head = RegHead(feat_dim, output_dim=1)

        # 相对航向头 (使用regress.py中的回归结构)
        self.rel_head = RegHead(feat_dim, output_dim=1)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch_size, seq_len, in_channels)

        Returns:
            logits_abs: 绝对航向的二进制logits (batch_size, num_bits)
            pred_rel: 相对航向的预测值 (batch_size, 1)
        """
        # 提取特征
        feat = self.feature_extractor(x)  # (batch_size, feat_dim)

        # 绝对航向预测
        logits_abs = self.abs_head(feat)  # (batch_size, num_bits)

        # 相对航向预测
        pred_rel = self.rel_head(feat)  # (batch_size, 1)

        return logits_abs, pred_rel


class DualHeadingLoss(torch.nn.Module):
    """
    双流航向损失函数：
    - 绝对航向：HeadingBinaryLoss
    - 相对航向：MSELoss
    - 总损失：loss_abs + w_rel * loss_rel
    """
    def __init__(self, num_bits=8, use_gray_code=True, quantizer=None, circular_weight=0.0, rel_weight=10.0):
        super().__init__()
        self.rel_weight = rel_weight

        # 绝对航向损失
        self.abs_loss = HeadingBinaryLoss(
            num_bits=num_bits,
            use_gray_code=use_gray_code,
            quantizer=quantizer,
            circular_weight=circular_weight
        )

        # 相对航向损失 (MSE)
        self.rel_loss = torch.nn.MSELoss()

    def forward(self, logits_abs, pred_rel, target_abs, target_rel):
        """
        Args:
            logits_abs: 绝对航向logits (batch_size, num_bits)
            pred_rel: 相对航向预测 (batch_size, 1)
            target_abs: 绝对航向标签 (batch_size, 1)
            target_rel: 相对航向标签 (batch_size, 1)

        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        # 绝对航向损失
        loss_abs = self.abs_loss(logits_abs, target_abs)

        # 相对航向损失
        loss_rel = self.rel_loss(pred_rel, target_rel)

        # 总损失
        total_loss = loss_abs + self.rel_weight * loss_rel

        loss_dict = {
            'total': total_loss.item(),
            'abs': loss_abs.item(),
            'rel': loss_rel.item(),
            'rel_weighted': (self.rel_weight * loss_rel).item()
        }

        return total_loss, loss_dict


