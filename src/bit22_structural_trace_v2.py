"""
bit22_structural_trace_v2.py

Corrected version of the structural trace.

Bug fix from v1: scoring by raw summed flip-rate treated a 0.5 (coin
flip / pure noise) as "high signal" and a 0.0 or 1.0 (fully
deterministic -- exploitable by a classifier) as "low signal", which
is backwards. This version scores by DETERMINISM: how far each rate
is from 0.5, i.e. score = 2 * |rate - 0.5|, which ranges 0 (pure
noise) to 1 (fully deterministic).

Also: v1's data showed S3b10 hitting EXACTLY 0.000 (fully
deterministic) at bit 22 and bit 42 specifically -- and 42 = 64-22,
i.e. the wraparound partner of 22. This version runs a dense sweep
over all 64 bits (not just the 9 spot-checked before) so we can see
whether that's a one-off coincidence or a real periodic/wraparound
structure, and explicitly reports the (b, 64-b) pairing pattern.

Run:
    python bit22_structural_trace_v2.py
"""

import numpy as np
from data_generator import ascon_permutation_vec

ROUNDS = 3
N_TRIALS_DENSE = 6000    # per (word, bit) config, dense 64-bit sweep
N_TRIALS_CONFIRM = 20000  # for confirming top hits with more trials
SEED = 7

SIGNATURE_BITS = [
    (3, 52), (4, 55), (3, 10), (1, 34),
    (3, 32), (0, 14), (4, 14), (1, 10),
]

WORDS_TO_SWEEP = [0, 4]  # the two words with existing saliency data to compare against


def flip_rates(delta_word, delta_bit, n_trials, rng):
    raw1 = rng.bytes(40 * n_trials)
    p1 = np.frombuffer(raw1, dtype='>u8').reshape(n_trials, 5).T.astype(np.uint64)
    delta = np.zeros((5, n_trials), dtype=np.uint64)
    delta[delta_word, :] = np.uint64(1) << np.uint64(delta_bit)
    p2 = p1 ^ delta
    c1 = ascon_permutation_vec(p1, ROUNDS)
    c2 = ascon_permutation_vec(p2, ROUNDS)
    diff = c1 ^ c2
    return {(sw, sb): float(((diff[sw] >> np.uint64(sb)) & np.uint64(1)).mean())
            for (sw, sb) in SIGNATURE_BITS}


def determinism_score(rates):
    """0 = pure noise (all rates ~0.5), higher = more deterministic."""
    return sum(2 * abs(r - 0.5) for r in rates.values())


def main():
    rng = np.random.default_rng(SEED)

    for word in WORDS_TO_SWEEP:
        print(f"\n{'='*70}\nWORD {word} -- dense sweep, all 64 bits, "
              f"{N_TRIALS_DENSE} trials each\n{'='*70}")

        scores = {}
        s3b10 = {}
        for bit in range(64):
            rates = flip_rates(word, bit, N_TRIALS_DENSE, rng)
            scores[bit] = determinism_score(rates)
            s3b10[bit] = rates[(3, 10)]

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        print("\nTop 10 bits by determinism score (2*|rate-0.5|, summed over signature bits):")
        for bit, sc in ranked[:10]:
            print(f"  bit {bit:2d}: score={sc:.3f}   S3b10_rate={s3b10[bit]:.3f}")

        print("\nWraparound-pair check (bit b vs 64-b), sorted by min(score_b, score_64-b):")
        pairs = []
        seen = set()
        for b in range(1, 64):
            partner = 64 - b
            if partner == b or partner not in scores or b in seen:
                continue
            seen.add(b); seen.add(partner)
            pairs.append((b, partner, scores[b], scores.get(partner, float('nan'))))
        pairs.sort(key=lambda t: -max(t[2], t[3]))
        for b, p, sb, sp in pairs[:12]:
            flag = "  <-- both high" if (sb > 1.0 and sp > 1.0) else ""
            print(f"  bit {b:2d} (score {sb:.3f})  <->  bit {p:2d} (score {sp:.3f}){flag}")

        # confirm the top hit with more trials
        top_bit = ranked[0][0]
        print(f"\nConfirming top bit ({top_bit}) with {N_TRIALS_CONFIRM} trials:")
        confirm_rates = flip_rates(word, top_bit, N_TRIALS_CONFIRM, rng)
        for k, v in confirm_rates.items():
            print(f"  S{k[0]}b{k[1]}: {v:.4f}")


if __name__ == "__main__":
    main()
