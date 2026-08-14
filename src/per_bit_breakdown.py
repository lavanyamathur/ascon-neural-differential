# per_bit_breakdown.py
import numpy as np
import torch
from models import build_model
from train_distinguisher import PackedAsconDataset, collate_batch
from torch.utils.data import DataLoader

CHECKPOINT = "checkpoints_randbit/mlp_r3.pt"
VAL_PREFIX = "ascon_r3_randbit_val"

ckpt = torch.load(CHECKPOINT, map_location="cpu")
args = ckpt["args"]
print(f"Loaded checkpoint: model_type={args['model_type']}, val_acc={ckpt['val_acc']:.4f}, epoch={ckpt['epoch']}")

if args["model_type"] == "mlp":
    model = build_model("mlp", hidden_dim=args["hidden_dim"])
elif args["model_type"] == "resnet_cnn":
    model = build_model("resnet_cnn", width=args["width"], num_blocks=args["num_blocks"])
else:
    model = build_model(args["model_type"])

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ds = PackedAsconDataset(VAL_PREFIX)
meta = np.load(f"{VAL_PREFIX}_META.npy")  # (N, 2): word_idx, bit_idx -- only meaningful where Y==1
loader = DataLoader(ds, batch_size=2048, shuffle=False, collate_fn=collate_batch)

all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in loader:
        logits = model(x)
        preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).float()
        all_preds.append(preds.numpy())
        all_labels.append(y.numpy())

preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)
correct = (preds == labels)

real_mask = labels == 1
real_bits = meta[real_mask, 1]
real_words = meta[real_mask, 0]
real_correct = correct[real_mask]

print(f"\nOverall accuracy on real-diff samples: {real_correct.mean():.4f}")
print(f"Overall accuracy on all samples: {correct.mean():.4f}\n")

print("Per-bit accuracy (real-diff samples only, all words pooled):")
print(f"{'bit':>4} {'n':>6} {'accuracy':>10}")
for b in range(64):
    mask = real_bits == b
    n = mask.sum()
    acc = real_correct[mask].mean() if n > 0 else float('nan')
    marker = "  <-- " if acc > 0.9 else ("  ~" if acc > 0.6 else "")
    print(f"{b:>4} {n:>6} {acc:>10.4f}{marker}")

np.save("per_bit_accuracy_mlp.npy", np.array([
    (b, (real_bits == b).sum(), real_correct[real_bits == b].mean() if (real_bits == b).sum() > 0 else np.nan)
    for b in range(64)
]))
print("\nSaved per_bit_accuracy.npy")