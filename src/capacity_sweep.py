"""
mlp_capacity_sweep.py

Tests whether the gohr_cnn (~96.7% val_acc) vs mlp (~56-61% val_acc) gap on
the randomized-bit/randomized-word round-3 task is architectural (the CNN's
weight-sharing over bit position doing real work) or just a capacity
shortfall (the parameter-matched MLPBaseline, hidden_dim=169, simply being
too small).

Loads the existing ascon_r3_randbit_train / ascon_r3_randbit_val data
(the same data checkpoints_randbit/mlp_r3.pt and gohr_cnn_r3.pt were
trained on) and trains several MLPBaseline configs of increasing capacity
on it, from the original parameter-matched baseline up to ~10x larger.

Run from C:\\Users\\hp\\Documents\\ASCON:
    python mlp_capacity_sweep.py
"""

import time
import numpy as np
import torch
import torch.nn as nn

from models import MLPBaseline, count_params
from data_generator import unpack_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_split(prefix):
    """
    prefix e.g. 'ascon_r3_randbit_train' -> loads {prefix}_X.npy / _Y.npy,
    unpacks to (N, 64, 10) float32, then permutes to (N, 10, 64) to match
    the channels-first layout MLPBaseline/ResNetCNN/GohrCNN all expect
    (per models.py's dummy input shape (batch, 10, 64)).
    """
    X_packed = np.load(f"{prefix}_X.npy")
    Y = np.load(f"{prefix}_Y.npy")

    X = unpack_batch(X_packed)          # (N, 64, 10) float32
    X = np.transpose(X, (0, 2, 1))       # (N, 10, 64) channels-first

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    return X_t, Y_t


def train_one_config(hidden_dim, num_hidden_layers, X_train, y_train, X_val, y_val,
                      epochs=20, lr=1e-3, batch_size=512, dropout=0.2):
    model = MLPBaseline(
        input_dim=640, hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers, dropout=dropout,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_train.shape[0]
    X_val_dev, y_val_dev = X_val.to(DEVICE), y_val.to(DEVICE)

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_dev)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == y_val_dev).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        print(f"    epoch {epoch:>2}  train_loss={running_loss/n:.4f}  val_acc={val_acc:.4f}")

    n_params = count_params(model)
    return best_val_acc, best_epoch, n_params


if __name__ == "__main__":
    print("Loading randbit train/val data...")
    X_train, y_train = load_split("ascon_r3_randbit_train")
    X_val, y_val = load_split("ascon_r3_randbit_val")
    print(f"  train: {X_train.shape}, val: {X_val.shape}")

    # (hidden_dim, num_hidden_layers) configs, from the original
    # parameter-matched baseline up to ~10x+ larger. num_hidden_layers=3
    # matches the original architecture; a couple of deeper variants are
    # included to separate "wider" from "deeper" capacity increases.
    configs = [
        (169, 3),    # baseline -- matches original mlp_r3.pt / checkpoints_randbit/mlp_r3.pt
        (256, 3),
        (512, 3),
        (1024, 3),
        (2048, 3),
        (1024, 4),   # deeper variant at similar width to the 1024/3 config
        (2048, 5),   # widest + deepest tested
    ]

    print(f"\n{'hidden_dim':>10} {'layers':>7} {'params':>10} {'best_val_acc':>12} {'best_epoch':>10}")
    results = []
    for hidden_dim, n_layers in configs:
        print(f"\n--- config: hidden_dim={hidden_dim}, layers={n_layers} ---")
        t0 = time.time()
        val_acc, best_epoch, n_params = train_one_config(
            hidden_dim, n_layers, X_train, y_train, X_val, y_val
        )
        dt = time.time() - t0
        print(f"{hidden_dim:>10} {n_layers:>7} {n_params:>10,} {val_acc:>12.4f} {best_epoch:>10}  ({dt:.0f}s)")
        results.append((hidden_dim, n_layers, n_params, val_acc, best_epoch))

    print("\n=== SUMMARY ===")
    print(f"{'hidden_dim':>10} {'layers':>7} {'params':>10} {'best_val_acc':>12}")
    for hidden_dim, n_layers, n_params, val_acc, best_epoch in results:
        print(f"{hidden_dim:>10} {n_layers:>7} {n_params:>10,} {val_acc:>12.4f}")

    np.save(
        "mlp_capacity_sweep_results.npy",
        np.array([(h, l, p, v) for h, l, p, v, _ in results], dtype=np.float64),
    )
    print("\nSaved mlp_capacity_sweep_results.npy")
    print("Compare the largest configs' val_acc against gohr_cnn's ~0.967.")
    print("If even the ~10x-larger MLP stays well below that, the gap is architectural, not capacity.")
