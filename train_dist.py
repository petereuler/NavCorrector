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
CKPT_HYBRID = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/hybrid_nav_best.pth"
CKPT_TRF = "/home/admin407/code/zyshe/NavCorrector/checkpoints_cls/transformer_dist.pth"

# Hybrid 显式维度
EXPLICIT_DIMS = {"theta": 48, "len": 24, "dz": 24}


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
        mode="dist",
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
            valid = ~mask
            length = pred[..., 0:1]
            cosv = pred[..., 1:2]
            sinv = pred[..., 2:3]
            dz = pred[..., 3:4]
            dp_body = torch.cat([length * cosv, length * sinv, dz], dim=-1)
            loss = F.mse_loss(dp_body[valid], y_seq[valid])

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
