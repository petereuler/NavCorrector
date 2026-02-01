import numpy as np
import torch


class PDRInferenceEngine:
    """
    双流推理引擎：
    - dist_model: 预测 dp_body
    - pose_model: 预测 q_rel
    """

    def __init__(
        self,
        dist_model,
        pose_model,
        window_size=256,
        stride=64,
        seq_len=64,
        device=None,
    ):
        self.dist_model = dist_model
        self.pose_model = pose_model
        self.window_size = window_size
        self.stride = stride
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.imu_buffer = []
        self.token_buffer = []
        self.current_pos = np.zeros(3, dtype=np.float32)
        self.current_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.dist_model.eval()
        self.pose_model.eval()
        for p in self.dist_model.parameters():
            p.requires_grad = False
        for p in self.pose_model.parameters():
            p.requires_grad = False

    def _normalize_imu(self, imu_window):
        w = imu_window.copy()
        w[:, 0:3] = w[:, 0:3] / 2.0
        w[:, 3:6] = w[:, 3:6] / 9.8
        return w


    def reset(self, init_pos=None, init_q=None):
        self.imu_buffer = []
        self.token_buffer = []
        if init_pos is None:
            self.current_pos = np.zeros(3, dtype=np.float32)
        else:
            self.current_pos = np.array(init_pos, dtype=np.float32).copy()
        if init_q is None:
            self.current_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            q = np.array(init_q, dtype=np.float32)
            self.current_q = q / (np.linalg.norm(q) + 1e-8)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q = np.array([w, x, y, z], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        return q

    def _quat_to_rotmat(self, q):
        w, x, y, z = q
        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z
        return np.array(
            [
                [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
            ],
            dtype=np.float32,
        )

    def _build_window(self):
        if len(self.imu_buffer) >= self.window_size:
            window = np.array(self.imu_buffer[-self.window_size :], dtype=np.float32)
        else:
            pad = self.window_size - len(self.imu_buffer)
            if len(self.imu_buffer) == 0:
                window = np.zeros((self.window_size, 6), dtype=np.float32)
            else:
                first = np.array(self.imu_buffer[0], dtype=np.float32)
                pad_block = np.repeat(first[None, :], pad, axis=0)
                window = np.concatenate([pad_block, np.array(self.imu_buffer, dtype=np.float32)], axis=0)
        return self._normalize_imu(window)

    def _push_token(self, token):
        self.token_buffer.append(token)
        if len(self.token_buffer) > self.seq_len:
            self.token_buffer.pop(0)

    def _build_seq(self):
        if len(self.token_buffer) < self.seq_len:
            pad = self.seq_len - len(self.token_buffer)
            zero = np.zeros((pad, 6, self.window_size), dtype=np.float32)
            seq = np.concatenate([zero, np.stack(self.token_buffer, axis=0)], axis=0)
            mask = np.ones((self.seq_len,), dtype=bool)
            mask[pad:] = False
        else:
            seq = np.stack(self.token_buffer, axis=0)
            mask = np.zeros((self.seq_len,), dtype=bool)
        return seq, mask

    def process_stream(self, imu_stream):
        for imu in imu_stream:
            self.imu_buffer.append(imu)
            if len(self.imu_buffer) % self.stride != 0:
                continue

            window = self._build_window()
            token = window.T  # (6, W)
            self._push_token(token)

            x_seq, mask = self._build_seq()
            x_seq = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)  # (1, S, 6, W)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)  # (1, S)

            with torch.no_grad():
                pred_nav = self.dist_model(x_seq, src_key_padding_mask=mask_t)[:, -1, :].squeeze(0).cpu().numpy()
                dp_body = np.array(
                    [
                        pred_nav[0] * pred_nav[1],
                        pred_nav[0] * pred_nav[2],
                        pred_nav[3],
                    ],
                    dtype=np.float32,
                )
                q_rel = self.pose_model(x_seq, src_key_padding_mask=mask_t)[:, -1, :].squeeze(0).cpu().numpy()
                q_rel = q_rel / (np.linalg.norm(q_rel) + 1e-8)

            self.current_q = self._quat_mul(self.current_q, q_rel)
            R = self._quat_to_rotmat(self.current_q)
            dp_world = (R @ dp_body.reshape(3, 1)).reshape(-1)
            self.current_pos += dp_world
            yield self.current_pos.copy()
