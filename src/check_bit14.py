"""
check_bit14.py

Follow-up to the bootstrap saliency result: bit 14 (across S[0], S[2],
S[4]) is now the CNN's strongest, statistically-real saliency signal
under the correct randbit checkpoint -- but the original reachability
rule never flagged it (S[3]#14 was only weakly biased, p=0.7506, not
deterministic, and that was a different word anyway).

This script runs a direct Monte Carlo empirical differential-probability
check at round 3, specifically for bit 14 in words 0, 2, and 4 (the
three words the CNN's saliency map lit up), under the confirmed-correct
word-4 / bit-0 injection convention. For context it also reports word 3
bit 14 (the original hand-picked bit) and word 1 bit 14, so all 5 words
are covered at this bit position.

A large sample size (500k, matching the earlier Monte Carlo check in
this project) is used so weak biases are distinguishable from chance
noise (chance = 0.5 flip rate; determinism = 0.0 or 1.0).

USAGE:
    python check_bit14.py
"""

import numpy as np

from data_generator import ascon_permutation_vec, DEFAULT_DELTA
from saliency_analysis import build_custom_delta, ROUNDS

N_PAIRS = 500_000
SEED = 7

WORDS_TO_CHECK = [0, 1, 2, 3, 4]
BIT_TO_CHECK = 14


def empirical_flip_rate(delta_word: int, delta_bit: int, n_pairs: int, seed: int):
    """Generate n_pairs of (p1, p2=p1^delta) round-ROUNDS outputs and
    return, for every (word, bit) position, the empirical probability
    that C1's bit differs from C2's bit at that position -- i.e. the
    output differential's per-bit flip rate. 0.5 = chance (no info),
    0.0 or 1.0 = fully deterministic (maximal info)."""
    rng = np.random.default_rng(seed)
    delta = build_custom_delta(delta_word, delta_bit)

    raw_p1 = rng.bytes(40 * n_pairs)
    p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(n_pairs, 5).T.astype(np.uint64)
    p2 = p1 ^ delta[:, np.newaxis]

    c1 = ascon_permutation_vec(p1, ROUNDS)
    c2 = ascon_permutation_vec(p2, ROUNDS)

    # c1, c2 shape: (5, n_pairs) uint64 words -> compute per-bit XOR flip rate
    diff = c1 ^ c2  # (5, n_pairs)
    flip_rate = np.zeros((5, 64), dtype=np.float64)
    for word in range(5):
        for bit in range(64):
            bit_vals = (diff[word] >> np.uint64(bit)) & np.uint64(1)
            flip_rate[word, bit] = bit_vals.mean()
    return flip_rate


def determinism_score(p: float) -> float:
    """0 at chance (p=0.5), 1 at full determinism (p=0.0 or 1.0)."""
    return 2 * abs(p - 0.5)


if __name__ == "__main__":
    print(f"Monte Carlo per-bit differential check, round {ROUNDS}, "
          f"{N_PAIRS} pairs, injection word 4 / bit 0\n")

    flip_rate = empirical_flip_rate(delta_word=4, delta_bit=0,
                                     n_pairs=N_PAIRS, seed=SEED)

    print(f"{'word':>6} {'bit':>4} {'flip_rate':>10} {'determinism':>12}  verdict")
    for word in WORDS_TO_CHECK:
        p = flip_rate[word, BIT_TO_CHECK]
        d = determinism_score(p)
        if d > 0.999:
            verdict = "DETERMINISTIC"
        elif d > 0.3:
            verdict = "biased"
        else:
            verdict = "~chance"
        print(f"S[{word}]  {BIT_TO_CHECK:4d} {p:10.4f} {d:12.4f}  {verdict}")

    print("\nFor context, also checking neighboring bits (13, 14, 15) "
          "in the three saliency-flagged words (0, 2, 4):")
    for word in (0, 2, 4):
        for bit in (13, 14, 15):
            p = flip_rate[word, bit]
            d = determinism_score(p)
            if d > 0.999:
                verdict = "DETERMINISTIC"
            elif d > 0.3:
                verdict = "biased"
            else:
                verdict = "~chance"
            print(f"S[{word}]  {bit:4d} {p:10.4f} {d:12.4f}  {verdict}")
