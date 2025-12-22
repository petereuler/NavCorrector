import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class PDR:
    """行人航迹推算核心类（严格初始航向对齐）"""
    
    def __init__(self, initial_pos, initial_yaw, fs=100):
        # 初始化位置和航向
        self.current_pos = initial_pos[:2].copy()
        self.initial_yaw = initial_yaw
        self.current_heading = self.initial_yaw

        # 运动参数
        self.fs = fs
        self.dt = 1.0/fs

        # 轨迹存储
        self.positions = [self.current_pos.copy()]
        self.heading_history = [self.initial_yaw]

        # 步态检测相关
        self.acc_norm = None
        self.filtered_acc = None
        self.peak_indices = None


    def bandpass_filter(self, data, low=1, high=5):
        """带通滤波器"""
        nyq = 0.5 * self.fs
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data)

    def detect_steps(self, acc_data):
        """步态检测（存储中间数据）"""
        self.acc_norm = np.linalg.norm(acc_data, axis=1)
        self.filtered_acc = self.bandpass_filter(self.acc_norm)
        
        # 检测波峰并存储结果
        self.peak_indices, _ = find_peaks(
            self.filtered_acc, 
            height=0.2,
            distance=int(0.25*self.fs))
        return self.peak_indices

    def estimate_heading(self, gyro_data):
        """航向估计（带初始航向基准）"""
        headings = []
        current_yaw = self.initial_yaw
        
        for w_z in gyro_data[:, 2]:  # 仅使用Z轴角速度（绕垂直轴旋转）
            current_yaw += w_z * self.dt
            headings.append(current_yaw)

        self.heading_history = np.array(headings)
        return self.heading_history

    def estimate_stride(self, peak_indices, base_length=0.8):
        """
        基于 Weinberg 公式估计步长，并通过检测谷值计算峰谷差 Δa。
        
        Weinberg 公式：
            stride_length = k * (Δa)^(1/4)
        其中 Δa 为当前步的峰值与对应谷值的差值，k 为校准常数（这里用 base_length 表示）。
        
        参数：
        - acc_data: 原始加速度数据数组
        - peak_indices: 峰值所在的索引数组（原始数据中只包含峰值）
        - base_length: 校准常数 k，用于调整估计步长的大小（默认 0.65）
        
        返回：
        - 每一步的估计步长数组
        """
        # 检测谷值：对加速度数据取负值，找出局部最大值即为原信号的局部最小值（谷值）
        valley_indices, _ = find_peaks(-self.acc_norm)
        
        stride_lengths = []
        
        # 对于每个峰值，寻找对应的谷值
        for i, peak_idx in enumerate(peak_indices):
            # 尝试在该峰值与下一个峰值之间寻找谷值
            if i < len(peak_indices) - 1:
                # 找出在当前峰值之后且在下一个峰值之前的所有谷值
                candidate_valleys = valley_indices[(valley_indices > peak_idx) & (valley_indices < peak_indices[i+1])]
                if candidate_valleys.size > 0:
                    valley_idx = candidate_valleys[0]  # 取第一个谷值作为当前步的谷值
                else:
                    # 如果没有在两峰之间找到谷值，则尝试选取峰值之前最近的谷值
                    candidate_valleys = valley_indices[valley_indices < peak_idx]
                    valley_idx = candidate_valleys[-1] if candidate_valleys.size > 0 else peak_idx
            else:
                # 对于最后一个峰值，优先选取其后的谷值；如果没有，则选取之前的谷值
                candidate_valleys = valley_indices[valley_indices > peak_idx]
                if candidate_valleys.size > 0:
                    valley_idx = candidate_valleys[0]
                else:
                    candidate_valleys = valley_indices[valley_indices < peak_idx]
                    valley_idx = candidate_valleys[-1] if candidate_valleys.size > 0 else peak_idx
            
            # 计算峰谷差 Δa
            delta_a = self.acc_norm[peak_idx] - self.acc_norm[valley_idx]
            # 避免负值或者零的情况，可加个下限
            delta_a = max(delta_a, 1e-6)
            # 采用 Weinberg 公式计算步长
            stride = base_length * np.power(delta_a, 0.25)
            stride_lengths.append(stride)
        
        return np.array(stride_lengths)
    
    def update_position(self, steps, stride_lengths):
        """位置更新"""
        for i, idx in enumerate(steps):
            if idx >= len(self.heading_history):
                continue
                
            # 获取当前航向
            yaw = self.heading_history[idx]
            
            
            # 计算位移
            dx = stride_lengths[i] * np.cos(yaw)
            dy = stride_lengths[i] * np.sin(yaw)
            
            # 更新位置
            self.current_pos += np.array([dx, dy])
            self.positions.append(self.current_pos.copy())
            
        return np.array(self.positions)

    def get_step_and_heading_deltas(self, gyro_data, acc_data, base_length=0.65):
        acc_data = acc_data * 9.81  # 转换为 m/s²
        step_indices = self.detect_steps(acc_data)
        heading_series = self.estimate_heading(gyro_data)
        stride_lengths = self.estimate_stride(step_indices, base_length=base_length)
        step_headings = heading_series[step_indices]
        delta_headings = np.diff(step_headings, prepend=step_headings[0])
        init_pos = self.positions[0]
        init_head = step_headings[0]
        return init_pos, init_head, stride_lengths, delta_headings