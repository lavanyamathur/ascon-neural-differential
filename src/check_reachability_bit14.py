"""
check_reachability_bit14.py

Test whether the existing reachability rule --
    "round-3 bit L of word w is deterministic iff round-2 columns
     {L, L+a, L+b} (a, b = word w's own rotation offsets) are still
     fully undiffused (flip rate 0.0) across all 5 words"
-- correctly predicts the two new deterministic bit-14 findings:
S[0]#14 and S[4]#14 (both p=0.0000 at round 3, confirmed by
check_bit14.py).

Word 0's rotation offsets: (19, 28) -> feeding columns {14, 33, 42}
Word 4's rotation offsets: (7, 41)  -> feeding columns {14, 21, 55}
(all mod 64)

For each feeding column, this checks the ROUND-2 (not round-3) output
flip rate across all 5 words. The rule predicts determinism only if
every one of those columns is fully undiffused (flip rate 0.0) in
every word at round 2.

USAGE:
    python check_reachability_bit14.py
"""

import numpy as np

from data_generator import ascon_permutation_vec
from saliency_analysis import build_custom_delta

N_PAIRS = 500_000
SEED = 7
ROUND2 = 2

# word -> (own rotation offsets a, b)
ROTATION_OFFSETS = {
    0: (19, 28),
    1: (61, 39),
    2: (1, 6),
    3: (10, 17),
    4: (7, 41),
}

WORDS_TO_TEST = {
    0: 14,  # S[0] bit 14 -- confirmed deterministic at round 3
    4: 14,  # S[4] bit 14 -- confirmed deterministic at round 3
}


def empirical_flip_rate_at_round(delta_word: int, delta_bit: int,
                                  n_pairs: int, seed: int, rounds: int):
    rng = np.random.default_rng(seed)
    delta = build_custom_delta(delta_word, delta_bit)

    raw_p1 = rng.bytes(40 * n_pairs)
    p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(n_pairs, 5).T.astype(np.uint64)
    p2 = p1 ^ delta[:, np.newaxis]

    c1 = ascon_permutation_vec(p1, rounds)
    c2 = ascon_permutation_vec(p2, rounds)

    diff = c1 ^ c2  # (5, n_pairs)
    flip_rate = np.zeros((5, 64), dtype=np.float64)
    for word in range(5):
        for bit in range(64):
            bit_vals = (diff[word] >> np.uint64(bit)) & np.uint64(1)
            flip_rate[word, bit] = bit_vals.mean()
    return flip_rate


if __name__ == "__main__":
    print(f"Reachability-rule check for S[0]#14 and S[4]#14, "
          f"round {ROUND2} feeding-column flip rates, {N_PAIRS} pairs\n")

    flip_rate_r2 = empirical_flip_rate_at_round(
        delta_word=4, delta_bit=0, n_pairs=N_PAIRS, seed=SEED, rounds=ROUND2
    )

    for word, bit in WORDS_TO_TEST.items():
        a, b = ROTATION_OFFSETS[word]
        feeding_cols = sorted(set([bit % 64, (bit + a) % 64, (bit + b) % 64]))
        print(f"--- S[{word}]#{bit} (round-3 deterministic) "
              f"-- word {word} offsets {ROTATION_OFFSETS[word]} "
              f"-> feeding round-2 columns {feeding_cols} ---")

        all_undiffused = True
        for col in feeding_cols:
            print(f"  round-2 flip rate at bit {col}, all 5 words:")
            for w in range(5):
                p = flip_rate_r2[w, col]
                status = "undiffused" if p < 1e-6 else f"ACTIVE (flip_rate={p:.4f})"
                print(f"    S[{w}] bit {col:2d}: {status}")
                if p >= 1e-6:
                    all_undiffused = False

        verdict = "RULE PREDICTS DETERMINISTIC (matches observation)" if all_undiffused \
            else "RULE PREDICTS ACTIVE (does NOT match -- exception, like S[3]#52/S[4]#55)"
        print(f"  => {verdict}\n")
