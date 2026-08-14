"""
plot_bit_sweep.py

Reads bit_sweep_full_results.csv (produced by bit_sweep_full.py) and produces:
  1. A word x bit heatmap of correct-counts for each model (gohr_cnn, mlp),
     saved as PNG files.
  2. A printed summary table of any (word, bit) cells that are "hits"
     (correct_count far above the ~0-20 noise floor), so you can eyeball
     exactly which cells to call out in the writeup.

Usage:
    python plot_bit_sweep.py

Expects bit_sweep_full_results.csv to be in the same directory, with columns
that include at least: model, word, bit, correct (rename below if your
actual column names differ -- check with a quick
    python -c "import pandas as pd; print(pd.read_csv('bit_sweep_full_results.csv').columns.tolist())"
first if this errors on column names).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "bit_sweep_full_results.csv"
NOISE_THRESHOLD = 25  # cells with correct count above this are flagged as "hits"

df = pd.read_csv(CSV_PATH)
print("Columns found:", df.columns.tolist())
print(f"Total rows: {len(df)}")
print(df.head())

# Normalize column names in case of slight naming differences
col_map = {}
for c in df.columns:
    lc = c.strip().lower()
    if lc in ("model", "model_type"):
        col_map[c] = "model"
    elif lc in ("word", "word_idx"):
        col_map[c] = "word"
    elif lc in ("bit", "bit_idx"):
        col_map[c] = "bit"
    elif lc in ("correct", "correct_count", "n_correct"):
        col_map[c] = "correct"
df = df.rename(columns=col_map)

required = {"model", "word", "bit", "correct"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(
        f"CSV is missing expected columns {missing}. "
        f"Found columns: {df.columns.tolist()}. "
        f"Edit col_map above to match your actual CSV headers."
    )

models = df["model"].unique()
n_words = df["word"].max() + 1
n_bits = df["bit"].max() + 1

fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
axes = axes[0]

for ax, model in zip(axes, models):
    sub = df[df["model"] == model]
    grid = np.full((n_words, n_bits), np.nan)
    for _, row in sub.iterrows():
        grid[int(row["word"]), int(row["bit"])] = row["correct"]

    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=500)
    ax.set_title(f"model={model}")
    ax.set_xlabel("bit index (0-63)")
    ax.set_ylabel("word index (0-4)")
    ax.set_yticks(range(n_words))
    fig.colorbar(im, ax=ax, label="correct / 500")

plt.tight_layout()
out_path = "bit_sweep_heatmap.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved heatmap to {out_path}")

# --- Summary of notable hits ---
print(f"\n=== Cells with correct > {NOISE_THRESHOLD} (out of 500) ===")
hits = df[df["correct"] > NOISE_THRESHOLD].sort_values(
    ["bit", "word", "model"]
)
if hits.empty:
    print("No cells above threshold.")
else:
    print(hits.to_string(index=False))

# --- Specifically flag bit-0 and check if it's perfect across all words/models ---
print("\n=== bit == 0 rows (checking for word-invariant perfect distinguisher) ===")
bit0 = df[df["bit"] == 0].sort_values(["model", "word"])
print(bit0.to_string(index=False))

print("\n=== bit == 22 rows (secondary consistent signal) ===")
bit22 = df[df["bit"] == 22].sort_values(["model", "word"])
print(bit22.to_string(index=False))
