"""
train_distinguisher.py

Trains a Gohr-style residual neural differential distinguisher on the
packed ASCON dataset produced by data_generator.py.

Input format on disk (from generate_dataset_streamed):
    <filename>_X.npy : (num_samples, 80) uint8, bit-packed
    <filename>_Y.npy : (num_samples,)     uint8, label (1 = real diff, 0 = random)

Each packed row unpacks to a (64, 10) float32 matrix:
    64 bit-positions (MSB-first per word) x 10 channels
    channels 0-4  = ciphertext1 words x0..x4
    channels 5-9  = ciphertext2 words x0..x4

Architecture: 1D-conv residual network operating along the 64 bit-position
axis, treating the 10 words as input channels -- this is the same layout
Gohr (2019) used for Speck, later adapted to ASCON by Bao et al. and
Shen et al. No aliasing with ascon_core.py / data_generator.py: this file
only consumes their outputs, it does not reimplement the permutation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import build_model, count_params, MODEL_REGISTRY


# =====================================================================
# Dataset: reads packed bytes lazily from the memmapped .npy files and
# unpacks each batch on the fly (keeps RAM flat regardless of dataset size)
# =====================================================================

class PackedAsconDataset(Dataset):
    def __init__(self, filename_prefix: str):
        self.X = np.load(f"{filename_prefix}_X.npy", mmap_mode="r")  # (N, 80) uint8
        self.Y = np.load(f"{filename_prefix}_Y.npy", mmap_mode="r")  # (N,)    uint8
        assert self.X.shape[0] == self.Y.shape[0]
        assert self.X.shape[1] == 80, "expected exactly 80 packed bytes per sample"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        packed = np.asarray(self.X[idx])          # (80,) uint8
        bits = np.unpackbits(packed).astype(np.float32)  # (640,)
        x = bits.reshape(64, 10)                  # (bit_position, channel)
        y = np.float32(self.Y[idx])
        return torch.from_numpy(x), torch.tensor(y)


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)          # (B, 64, 10)
    x = x.permute(0, 2, 1).contiguous()  # (B, 10, 64) -> channels-first for Conv1d
    y = torch.stack(ys, dim=0)
    return x, y


# =====================================================================
# Train / eval loops
# =====================================================================

def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser(description="Train ASCON neural distinguisher")
    parser.add_argument("--train_prefix", required=True, help="e.g. ascon_r4_train")
    parser.add_argument("--val_prefix", required=True, help="e.g. ascon_r4_val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--model_type", type=str, default="mlp",
                         choices=list(MODEL_REGISTRY.keys()),
                         help="Which architecture to train, e.g. 'mlp' or 'resnet_cnn'.")
    parser.add_argument("--width", type=int, default=32,
                         help="(resnet_cnn only) conv channel width")
    parser.add_argument("--num_blocks", type=int, default=5,
                         help="(resnet_cnn only) number of residual blocks")
    parser.add_argument("--hidden_dim", type=int, default=169,
                         help="(mlp only) hidden layer width -- default is parameter-matched to the default resnet_cnn")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = PackedAsconDataset(args.train_prefix)
    val_ds = PackedAsconDataset(args.val_prefix)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_batch, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )

    if args.model_type == "mlp":
        model = build_model("mlp", hidden_dim=args.hidden_dim).to(device)
    elif args.model_type == "resnet_cnn":
        model = build_model("resnet_cnn", width=args.width, num_blocks=args.num_blocks).to(device)
    else:
        model = build_model(args.model_type).to(device)

    print(f"Model: {args.model_type} | Trainable parameters: {count_params(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, val_loader, device, None, criterion)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # infer a round tag from train_prefix (e.g. "ascon_r3_train" -> "r3")
            round_tag = next((tok for tok in args.train_prefix.split("_")
                               if tok.startswith("r") and tok[1:].isdigit()), "r?")
            ckpt_name = f"{args.model_type}_{round_tag}.pt"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> New best val_acc={val_acc:.4f}, saved to {ckpt_path}")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
