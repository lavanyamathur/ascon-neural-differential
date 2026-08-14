"""
saliency_analysis.py

RQ3: Interpretability pass on the round-3 distinguisher models.

Goal: for each architecture (gohr_cnn, resnet_cnn, mlp), find out which
input bits the model actually relies on when it correctly classifies a
real-differential round-3 pair, then map those bit-importance scores
back onto ASCON's real state layout (5 words x 64 bits, x2 ciphertexts)
and check whether high-importance bits cluster near the rotation
offsets used in the linear diffusion layer.

Input/data contract (matches data_generator.py + models.py exactly):
    - pack_state_to_bits() concatenates ciphertext-1's 5 words then
      ciphertext-2's 5 words -> unpack_batch() gives (batch, 64, 10):
          channel 0-4 = C1 words S[0..4]
          channel 5-9 = C2 words S[0..4]
      Bits are MSB-first: position index p (0..63) in that axis
      corresponds to actual bit (63 - p) of the word.
    - Models consume (batch, 10, 64) channels-first (collate_batch's
      permute) -- this script builds the tensor directly in that shape
      so no permute is needed downstream.

Method: gradient x input saliency (Simonyan et al.), computed per-sample
on a batch of correctly-classified real-diff (label=1) round-3 examples,
then averaged. This is the standard cheap-and-robust choice for bit-
vector inputs; if you want a second opinion, integrated gradients is
sketched at the bottom (off by default -- slower, ~20x more forward/
backward passes).

USAGE:
    1. Fill in CKPT_PATHS below with your actual round-3 checkpoint files.
    2. python saliency_analysis.py --model gohr_cnn
       (or: --model all   to run all three and produce a comparison plot)
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_generator import (
    ascon_permutation_vec,
    pack_state_to_bits,
    unpack_batch,
    DEFAULT_DELTA,
)
from models import build_model, MODEL_REGISTRY

# ---------------------------------------------------------------------
# FILL THESE IN with your actual saved round-3 checkpoint filenames.
# ---------------------------------------------------------------------
CKPT_PATHS = {
    "mlp": "checkpoints_randbit/mlp_r3.pt",
    "gohr_cnn": "checkpoints_randbit/gohr_cnn_r3.pt",
}
ROUNDS = 3
N_SAMPLES = 2000          # pool to draw correctly-classified real-diff examples from
N_SALIENCY_SAMPLES = 500  # how many correct real-diff examples to average saliency over
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ASCON linear-diffusion rotation offsets per word, for annotating results
ROTATION_OFFSETS = {
    0: (19, 28),
    1: (61, 39),
    2: (1, 6),
    3: (10, 17),
    4: (7, 41),
}


# =====================================================================
# Data: generate a round-3 batch directly (no need for a saved dataset)
# =====================================================================

def build_custom_delta(delta_word: int, delta_bit: int) -> np.ndarray:
    """Build a single-bit input difference: DEFAULT_DELTA's shape (5,)
    uint64 words, all zero except one bit set in one word.
    delta_bit is the actual bit position (0 = LSB .. 63 = MSB)."""
    if not (0 <= delta_word <= 4):
        raise ValueError(f"delta_word must be 0-4, got {delta_word}")
    if not (0 <= delta_bit <= 63):
        raise ValueError(f"delta_bit must be 0-63, got {delta_bit}")
    delta = np.zeros(5, dtype=np.uint64)
    delta[delta_word] = np.uint64(1) << np.uint64(delta_bit)
    return delta


def generate_round3_batch(n_samples: int, seed: int = 1234, delta: np.ndarray = None):
    """Same generation logic as data_generator.generate_dataset_streamed,
    but kept in memory (no disk streaming) since we only need a few
    thousand samples for saliency, not a training set.

    delta: (5,) uint64 array of word-wise XOR difference to apply for
    real-diff (label=1) samples. Defaults to DEFAULT_DELTA if None."""
    if delta is None:
        delta = DEFAULT_DELTA
    rng = np.random.default_rng(seed)

    indices = np.arange(n_samples, dtype=np.int64)
    Y = (indices % 2).astype(np.uint8)

    raw_p1 = rng.bytes(40 * n_samples)
    p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(n_samples, 5).T.astype(np.uint64)

    raw_p2_rand = rng.bytes(40 * n_samples)
    p2_rand = np.frombuffer(raw_p2_rand, dtype='>u8').reshape(n_samples, 5).T.astype(np.uint64)

    p2_real = p1 ^ delta[:, np.newaxis]
    label_mask = (Y == 1)
    p2 = np.where(label_mask, p2_real, p2_rand)

    c1 = ascon_permutation_vec(p1, ROUNDS)
    c2 = ascon_permutation_vec(p2, ROUNDS)

    X_packed = pack_state_to_bits(c1, c2)   # (n_samples, 80) uint8
    X = unpack_batch(X_packed)              # (n_samples, 64, 10) float32
    X = np.transpose(X, (0, 2, 1))          # -> (n_samples, 10, 64) channels-first

    return torch.from_numpy(X), torch.from_numpy(Y.astype(np.float32))


# =====================================================================
# Saliency: gradient x input, averaged over correctly-classified
# real-diff (label=1) samples
# =====================================================================

def compute_saliency(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor,
                      n_saliency_samples: int):
    model.eval()
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    with torch.no_grad():
        logits = model(X)
        preds = (torch.sigmoid(logits) > 0.5).float()

    # Only keep real-diff samples (label=1) the model got right
    correct_real = (preds == Y) & (Y == 1)
    idx = correct_real.nonzero(as_tuple=True)[0]

    if idx.numel() == 0:
        raise RuntimeError(
            "No correctly-classified real-diff samples found -- check that "
            "the checkpoint matches round 3 and the model architecture."
        )

    idx = idx[:n_saliency_samples]
    print(f"  using {idx.numel()} correctly-classified real-diff samples "
          f"(out of {int((Y == 1).sum().item())} real-diff samples in the pool)")

    X_sub = X[idx].clone().requires_grad_(True)

    logits_sub = model(X_sub)
    logits_sub.sum().backward()

    # gradient x input, absolute value, averaged over the batch
    saliency = (X_sub.grad * X_sub).abs().mean(dim=0)  # (10, 64)
    return saliency.detach().cpu().numpy()


# =====================================================================
# Map (10, 64) saliency array back to ASCON's real structure
# =====================================================================

def map_to_ascon_structure(saliency: np.ndarray):
    """
    saliency: (10, 64) array, channel-first (matches model input layout).
    Returns two (5, 64) arrays -- ciphertext-1 words and ciphertext-2
    words -- reindexed so column b is the actual ASCON bit position b
    (0 = LSB .. 63 = MSB), undoing the MSB-first packing.
    """
    c1_sal = saliency[0:5, :]   # channels 0-4
    c2_sal = saliency[5:10, :]  # channels 5-9

    # position index p in the array corresponds to actual bit (63 - p)
    # -> flip along the bit axis to get index = actual bit position
    c1_sal = c1_sal[:, ::-1]
    c2_sal = c2_sal[:, ::-1]

    return c1_sal, c2_sal


def report_top_bits(c1_sal: np.ndarray, c2_sal: np.ndarray, top_k: int = 15):
    print("\nTop bit positions by saliency (word, bit, score), C1 and C2 combined:")
    combined = []
    for word in range(5):
        for bit in range(64):
            combined.append((c1_sal[word, bit], "C1", word, bit))
            combined.append((c2_sal[word, bit], "C2", word, bit))
    combined.sort(key=lambda t: t[0], reverse=True)

    for score, which, word, bit in combined[:top_k]:
        near = ""
        offs = ROTATION_OFFSETS[word]
        candidates = {
            offs[0]: f"offset {offs[0]}",
            offs[1]: f"offset {offs[1]}",
            64 - offs[0]: f"wraparound of offset {offs[0]}",
            64 - offs[1]: f"wraparound of offset {offs[1]}",
        }
        best = min(candidates, key=lambda c: abs(bit - c))
        if abs(bit - best) <= 2:
            near = f"  <-- near {candidates[best]} for S[{word}]"
        print(f"  {which} S[{word}] bit {bit:2d}: {score:.5f}{near}")


def plot_heatmaps(c1_sal: np.ndarray, c2_sal: np.ndarray, model_name: str, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for ax, sal, label in zip(axes, [c1_sal, c2_sal], ["Ciphertext 1", "Ciphertext 2"]):
        im = ax.imshow(sal, aspect="auto", cmap="viridis")
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"S[{i}]" for i in range(5)])
        ax.set_title(f"{label} -- word x bit-position saliency ({model_name}, round {ROUNDS})")
        # mark rotation offsets
        for word in range(5):
            for off in ROTATION_OFFSETS[word]:
                ax.axvline(off, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Bit position (0 = LSB, 63 = MSB); dashed lines = rotation offsets")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  saved heatmap to {out_path}")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def run_for_model(name: str, delta: np.ndarray = None):
    print(f"\n=== {name} ===")
    ckpt_path = CKPT_PATHS[name]
    model = build_model(name).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # train_distinguisher.py saves a dict with metadata, not raw weights --
    # unwrap it. Falls back to treating ckpt as a raw state_dict if it's not
    # a dict-with-model_state_dict (e.g. if you saved differently elsewhere).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"  loaded checkpoint from epoch {ckpt.get('epoch')}, "
              f"val_acc={ckpt.get('val_acc'):.4f}")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    X, Y = generate_round3_batch(N_SAMPLES, delta=delta)
    saliency = compute_saliency(model, X, Y, N_SALIENCY_SAMPLES)
    c1_sal, c2_sal = map_to_ascon_structure(saliency)

    report_top_bits(c1_sal, c2_sal)
    plot_heatmaps(c1_sal, c2_sal, name, out_path=f"saliency_{name}_r{ROUNDS}.png")

    return c1_sal, c2_sal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()) + ["all"],
                         default="gohr_cnn")
    parser.add_argument("--delta_word", type=int, default=None,
                         help="Word index (0-4) for a custom single-bit input "
                              "difference, overriding DEFAULT_DELTA.")
    parser.add_argument("--delta_bit", type=int, default=None,
                         help="Bit position (0=LSB..63=MSB) within --delta_word "
                              "for the custom single-bit difference. Requires "
                              "--delta_word to also be set.")
    args = parser.parse_args()

    if (args.delta_word is None) != (args.delta_bit is None):
        parser.error("--delta_word and --delta_bit must be given together")

    custom_delta = None
    if args.delta_word is not None:
        custom_delta = build_custom_delta(args.delta_word, args.delta_bit)
        print(f"Using custom delta: word {args.delta_word}, bit {args.delta_bit}")

    if args.model == "all":
        results = {}
        for name in MODEL_REGISTRY:
            results[name] = run_for_model(name, delta=custom_delta)
    else:
        run_for_model(args.model, delta=custom_delta)
