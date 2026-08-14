# differential_weight_probe.py
import numpy as np
from data_generator import ascon_permutation_vec

def hamming_weight_stats(word_idx, bit_idx, rounds=3, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    raw_p1 = rng.bytes(40 * n_samples)
    p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(n_samples, 5).T.astype(np.uint64)

    delta = np.zeros((5, n_samples), dtype=np.uint64)
    delta[word_idx, :] = np.uint64(1) << np.uint64(bit_idx)
    p2 = p1 ^ delta

    c1 = ascon_permutation_vec(p1, rounds)
    c2 = ascon_permutation_vec(p2, rounds)

    diff = c1 ^ c2  # (5, n_samples)
    # popcount per sample, summed across all 5 words -> total output Hamming weight
    weights = np.zeros(n_samples, dtype=np.int64)
    for w in range(5):
        col = diff[w].astype(np.uint64)
        for _ in range(64):
            weights += (col & np.uint64(1)).astype(np.int64)
            col >>= np.uint64(1)

    return weights.mean(), weights.std(), weights.min(), weights.max()

print(f"{'word':>4} {'bit':>4} {'mean_wt':>9} {'std':>7} {'min':>5} {'max':>5}")
for word_idx in [0, 4]:
    for bit_idx in [0, 1, 10, 21, 22, 23, 41, 42, 43, 44, 45, 32, 50, 63]:
        mean, std, mn, mx = hamming_weight_stats(word_idx, bit_idx)
        print(f"{word_idx:>4} {bit_idx:>4} {mean:>9.2f} {std:>7.2f} {mn:>5} {mx:>5}")