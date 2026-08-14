"""
bootstrap_saliency.py

Bootstrap confidence intervals on per-bit saliency scores, to test
whether the top-ranked bits are statistically distinguishable from
the noise floor -- rather than just eyeballing "highest score in one
run."

Method: take the pool of correctly-classified real-diff samples used
for saliency (same pool compute_saliency() draws from), resample it
with replacement N_BOOTSTRAP times, recompute the per-bit saliency
score (gradient x input, averaged) each time, and build an empirical
95% CI per bit from the resulting distribution.

A bit is only reported as "real signal" if its CI does not overlap
with the CI of the bulk/median bit -- i.e. it's significantly above
the noise floor, not just numerically highest in a single point
estimate.

USAGE:
    python bootstrap_saliency.py --model gohr_cnn --delta_word 4 --delta_bit 0
    python bootstrap_saliency.py --model mlp --delta_word 4 --delta_bit 0
"""

import argparse
import numpy as np
import torch

from saliency_analysis import (
    CKPT_PATHS, ROUNDS, N_SAMPLES, DEVICE, ROTATION_OFFSETS,
    build_custom_delta, generate_round3_batch, map_to_ascon_structure,
)
from models import build_model, MODEL_REGISTRY

N_SALIENCY_SAMPLES = 500   # matches saliency_analysis.py
N_BOOTSTRAP = 200          # number of resamples
TOP_K_REPORT = 15


def load_model(name: str):
    model = build_model(name).to(DEVICE)
    ckpt = torch.load(CKPT_PATHS[name], map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"  loaded checkpoint from epoch {ckpt.get('epoch')}, "
              f"val_acc={ckpt.get('val_acc'):.4f}")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def get_correct_idx_and_data(model, X, Y):
    """One forward pass to find which samples the model gets right on
    real-diff (label=1) examples. Bootstrap resamples INDICES into this
    fixed correct set, not the raw dataset, so every resample is still
    guaranteed to be a correctly-classified real-diff sample."""
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    with torch.no_grad():
        logits = model(X)
        preds = (torch.sigmoid(logits) > 0.5).float()
    correct_real = (preds == Y) & (Y == 1)
    idx = correct_real.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise RuntimeError("No correctly-classified real-diff samples found.")
    idx = idx[:N_SALIENCY_SAMPLES]
    print(f"  correct-sample pool size for bootstrap: {idx.numel()}")
    return X, Y, idx


def saliency_for_indices(model, X, sample_idx):
    """Compute gradient x input saliency averaged over the given sample
    indices (with possible repeats, for bootstrap resampling)."""
    X_sub = X[sample_idx].clone().requires_grad_(True)
    logits_sub = model(X_sub)
    logits_sub.sum().backward()
    saliency = (X_sub.grad * X_sub).abs().mean(dim=0)  # (10, 64)
    return saliency.detach().cpu().numpy()


def run_bootstrap(name: str, delta: np.ndarray):
    print(f"\n=== bootstrap: {name} ===")
    model = load_model(name)
    X, Y = generate_round3_batch(N_SAMPLES, delta=delta)
    X, Y, base_idx = get_correct_idx_and_data(model, X, Y)
    n_correct = base_idx.numel()

    # point estimate (matches saliency_analysis.py's reported numbers)
    point_saliency = saliency_for_indices(model, X, base_idx)
    c1_point, c2_point = map_to_ascon_structure(point_saliency)

    # bootstrap resamples
    rng = np.random.default_rng(42)
    boot_scores = np.zeros((N_BOOTSTRAP, 10, 64), dtype=np.float32)
    for b in range(N_BOOTSTRAP):
        resample_pos = rng.integers(0, n_correct, size=n_correct)
        resample_idx = base_idx[resample_pos]
        sal = saliency_for_indices(model, X, resample_idx)
        boot_scores[b] = sal
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOTSTRAP}")

    # map each bootstrap sample through the same LSB-first reindexing
    # as map_to_ascon_structure, done here in bulk for speed
    boot_c1 = boot_scores[:, 0:5, :][:, :, ::-1]   # (N_BOOTSTRAP, 5, 64)
    boot_c2 = boot_scores[:, 5:10, :][:, :, ::-1]

    # combine C1 and C2 into one (N_BOOTSTRAP, 10, 64) "which x word x bit" array
    # for CI + ranking purposes, matching report_top_bits' combined list
    combined_boot = np.concatenate([boot_c1, boot_c2], axis=1)  # (N_BOOT, 10, 64) -> dims: [c1_w0..4, c2_w0..4]
    combined_point = np.concatenate(
        [c1_point[np.newaxis], c2_point[np.newaxis]], axis=1
    )[0]  # (10, 64)

    lower = np.percentile(combined_boot, 2.5, axis=0)   # (10, 64)
    upper = np.percentile(combined_boot, 97.5, axis=0)  # (10, 64)
    median = np.percentile(combined_boot, 50, axis=0)   # (10, 64)

    # noise floor: median bit's CI (use the median-ranked bit, not the
    # single lowest, so one dead bit doesn't set an artificially tight floor)
    flat_median = median.flatten()
    noise_floor_bit_flat_idx = np.argsort(flat_median)[len(flat_median) // 2]
    noise_floor_upper = upper.flatten()[noise_floor_bit_flat_idx]

    # rank bits by point estimate, report top K with CIs, and flag which
    # ones clear the noise floor (lower bound > noise floor's upper bound)
    which_labels = ["C1"] * 5 + ["C2"] * 5
    word_labels = list(range(5)) * 2

    flat_point = combined_point.flatten()
    order = np.argsort(flat_point)[::-1]

    print(f"\n  noise floor (median bit's 97.5th percentile): {noise_floor_upper:.5f}")
    print(f"  Top {TOP_K_REPORT} bits with 95% bootstrap CI "
          f"[point estimate, CI lower, CI upper], flag if CI clears noise floor:\n")

    for flat_idx in order[:TOP_K_REPORT]:
        ch = flat_idx // 64
        bit = flat_idx % 64
        which = which_labels[ch]
        word = word_labels[ch]
        pt = flat_point[flat_idx]
        lo = lower.flatten()[flat_idx]
        hi = upper.flatten()[flat_idx]
        clears = "REAL SIGNAL" if lo > noise_floor_upper else "not distinguishable from noise"
        print(f"  {which} S[{word}] bit {bit:2d}: point={pt:.5f}  "
              f"CI=[{lo:.5f}, {hi:.5f}]  -> {clears}")

    return combined_point, lower, upper, noise_floor_upper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="gohr_cnn")
    parser.add_argument("--delta_word", type=int, default=4)
    parser.add_argument("--delta_bit", type=int, default=0)
    args = parser.parse_args()

    delta = build_custom_delta(args.delta_word, args.delta_bit)
    run_bootstrap(args.model, delta)
