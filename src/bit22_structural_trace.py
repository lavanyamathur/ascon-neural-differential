"""
bit22_structural_trace.py

Step 3 (structural attribution): empirically measure, for every
(word, bit) single-bit input difference, how strongly it flips a
fixed set of "signature" output bits after ROUNDS of the ASCON
permutation. The signature bits are the ones your saliency runs
already identified as what the classifiers actually key on
(top entries from the mlp / gohr_cnn saliency reports).

This directly tests the hypothesis: is bit 22 "louder" because it
propagates unusually strongly into the model's existing blind spot
(the signature bits), rather than because of some standalone property
of position 22?

Run:
    python bit22_structural_trace.py
"""

import numpy as np
from ascon_core import ROUND_CONSTANTS, MASK64
from data_generator import ascon_permutation_vec

ROUNDS = 3
N_TRIALS = 20000  # random base-state pairs per (word, bit) config
SEED = 7

# Signature output bits the models were shown to rely on
# (word, actual_bit_position 0=LSB..63=MSB), taken from the saliency
# top-15 report (both C1 and C2 columns collapsed to word/bit -- the
# permutation output structure is the same for both ciphertext branches).
SIGNATURE_BITS = [
    (3, 52),
    (4, 55),
    (3, 10),
    (1, 34),
    (3, 32),
    (0, 14),
    (4, 14),
    (1, 10),
]

ROTATION_OFFSETS = {
    0: (19, 28),
    1: (61, 39),
    2: (1, 6),
    3: (10, 17),
    4: (7, 41),
}


def flip_rate_to_signature(delta_word: int, delta_bit: int, rng) -> dict:
    """For a single-bit input difference at (delta_word, delta_bit),
    run N_TRIALS random state pairs through ROUNDS of the permutation
    and measure, for each signature bit, the fraction of trials where
    that output bit differs between the two branches."""
    raw1 = rng.bytes(40 * N_TRIALS)
    p1 = np.frombuffer(raw1, dtype='>u8').reshape(N_TRIALS, 5).T.astype(np.uint64)

    delta = np.zeros((5, N_TRIALS), dtype=np.uint64)
    delta[delta_word, :] = np.uint64(1) << np.uint64(delta_bit)
    p2 = p1 ^ delta

    c1 = ascon_permutation_vec(p1, ROUNDS)
    c2 = ascon_permutation_vec(p2, ROUNDS)
    diff = c1 ^ c2  # (5, N_TRIALS)

    rates = {}
    for (sw, sb) in SIGNATURE_BITS:
        bitval = (diff[sw] >> np.uint64(sb)) & np.uint64(1)
        rates[(sw, sb)] = float(bitval.mean())
    return rates


def hamming_weight_control(delta_word: int, delta_bit: int, rng) -> float:
    raw1 = rng.bytes(40 * 4000)
    p1 = np.frombuffer(raw1, dtype='>u8').reshape(4000, 5).T.astype(np.uint64)
    delta = np.zeros((5, 4000), dtype=np.uint64)
    delta[delta_word, :] = np.uint64(1) << np.uint64(delta_bit)
    p2 = p1 ^ delta
    c1 = ascon_permutation_vec(p1, ROUNDS)
    c2 = ascon_permutation_vec(p2, ROUNDS)
    diff = (c1 ^ c2).astype(np.uint64)
    total = np.zeros(4000)
    for w in range(5):
        col = diff[w]
        # popcount via numpy bit_count if available (numpy>=2.0), else fallback
        if hasattr(np, "bit_count"):
            total += np.bit_count(col).astype(np.float64)
        else:
            total += np.array([bin(int(x)).count("1") for x in col], dtype=np.float64)
    return float(total.mean())


def main():
    rng = np.random.default_rng(SEED)

    print(f"Empirical signature-bit flip rates after {ROUNDS} rounds, "
          f"{N_TRIALS} trials per config.")
    print(f"Signature bits: {SIGNATURE_BITS}\n")

    # Test bit 22 vs its immediate neighbors, across all 5 words
    test_bits = [20, 21, 22, 23, 24, 42, 43, 44, 45]

    header = "word bit  " + "  ".join(f"S{w}b{b:02d}" for (w, b) in SIGNATURE_BITS) + "   sum   hamming_wt"
    print(header)
    print("-" * len(header))

    results = []
    for word in range(5):
        for bit in test_bits:
            rates = flip_rate_to_signature(word, bit, rng)
            s = sum(rates.values())
            hw = hamming_weight_control(word, bit, rng)
            row = f"{word:>4} {bit:>3}  " + "  ".join(f"{rates[k]:.3f}" for k in SIGNATURE_BITS) \
                  + f"  {s:.3f}  {hw:6.2f}"
            print(row)
            results.append((word, bit, s, hw))

    print("\n--- Ranking within each word (by summed signature-bit flip rate) ---")
    for word in range(5):
        word_rows = [r for r in results if r[0] == word]
        word_rows.sort(key=lambda r: -r[2])
        print(f"word {word}: " + ", ".join(f"bit{b}={s:.3f}" for (_, b, s, _) in word_rows))


if __name__ == "__main__":
    main()
