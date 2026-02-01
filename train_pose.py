import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset_RIDI import load_ridi_raw
from data.sequence_dataset import SequenceDataset
from models.hybrid_imu_embed import HybridIMUEmbed, HybridIMUEmbWrapper
from models.imu_seq_transformer import IMUSeqTransformer

# ======= 显式配置区 =======
RIDI_ROOT = "/home/admin407/code/zyshe/NavCorrector/RIDI"
ACC_SOURCE = "acce"
WINDOW_SIZE = 256
STRIDE = 256
SEQ_LEN = 16
PAD_COLD_START = True
SAMPLES_PER_EPOCH = 20000
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
LOG_EVERY = 50
CKPT_HYBRID = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/hybrid_pose_best.pth"
CKPT_TRF = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/transformer_pose.pth"

# Hybrid 显式维度
EXPLICIT_DIMS = {"yaw": 32, "roll": 32, "pitch": 32}


def load_sequences():
    data_root = os.path.join(RIDI_ROOT, "data")
    list_path = os.path.join(data_root, "list_train_publish_v2.txt")
    if not os.path.exists(list_path):
        raise RuntimeError(f"RIDI list file not found: {list_path}")
    with open(list_path, "r") as f:
        names = [line.strip().split(",")[0] for line in f if line.strip()]
    seqs = []
    for name in names:
        seq_dir = os.path.join(data_root, name)
        if not os.path.isdir(seq_dir):
            continue
        gyro, acc, pos, ori = load_ridi_raw(seq_dir, acc_source=ACC_SOURCE)
        seqs.append((gyro, acc, pos, ori))
    return seqs


def quat_cos_loss(q_pred, q_gt):
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + 1e-8)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + 1e-8)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
    return 1.0 - dot


def quat_mul_torch(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid = HybridIMUEmbed(
        window_size=WINDOW_SIZE,
        explicit_dims=EXPLICIT_DIMS,
        latent_dim=32,
        feat_dim=128,
    ).to(device)
    hybrid.load_state_dict(torch.load(CKPT_HYBRID, map_location=device))
    emb = HybridIMUEmbWrapper(hybrid)

    model = IMUSeqTransformer(emb=emb, output_dim=4).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    seqs = load_sequences()
    ds = SequenceDataset(
        seqs,
        mode="pose",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        seq_len=SEQ_LEN,
        pad_cold_start=PAD_COLD_START,
        samples_per_epoch=SAMPLES_PER_EPOCH,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    best = float("inf")
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        count = 0
        steps = 0
        for x_seq, y_seq, mask in dl:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)
            mask = mask.to(device)

            pred = model(x_seq, src_key_padding_mask=mask)
            pred = F.normalize(pred, dim=-1, eps=1e-8)
            valid = ~mask
            loss = quat_cos_loss(pred[valid], y_seq[valid]).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * x_seq.size(0)
            count += x_seq.size(0)
            steps += 1
            if steps % LOG_EVERY == 0:
                print(f"[Epoch {ep + 1} | Step {steps}] Loss: {total / max(count, 1):.6f}")

        avg = total / max(count, 1)
        print(f"[Epoch {ep + 1}] Loss: {avg:.6f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), CKPT_TRF)


if __name__ == "__main__":
    main()
