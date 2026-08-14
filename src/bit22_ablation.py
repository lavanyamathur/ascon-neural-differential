"""
bit22_ablation.py

Step 3 continued: isolate whether the observed b <-> 64-b wraparound
symmetry (and the bit-22 signal specifically) comes from the FINAL
linear diffusion layer alone, or depends on the nonlinear S-box
interaction across rounds.

Ablation: run the normal ASCON permutation for rounds 1..(ROUNDS-1)
unmodified, but on the LAST round, skip the substitution (S-box)
layer entirely -- only apply the round-constant XOR and the linear
diffusion layer. This isolates the final linear map L(x) = x ^
rotr(x,a) ^ rotr(x,c), which is the only piece we can derive
symmetry properties for by hand.

Interpretation:
  - If bit22's signal / the b<->64-b pairing SURVIVES ablation,
    the pattern is (at least partly) explained by the final linear
    layer alone -- a real mechanistic result.
  - If it DISAPPEARS under ablation, the pattern depends on the
    nonlinear S-box interaction across rounds and is NOT reducible
    to the linear layer -- also a real, reportable result, just a
    different one.

Run:
    python bit22_ablation.py
"""

import numpy as np
from ascon_core import ROUND_CONSTANTS, MASK64

ROUNDS = 3
N_TRIALS = 20000
SEED = 7

SIGNATURE_BITS = [
    (3, 52), (4, 55), (3, 10), (1, 34),
    (3, 32), (0, 14), (4, 14), (1, 10),
]

TEST_BITS = list(range(64))
WORDS_TO_SWEEP = [0, 4]


def rotr_vec(x, n):
    n %= 64
    return (x >> n) | (x << np.uint64(64 - n))


def ascon_permutation_vec_ablated(state, rounds, skip_sbox_last_round=True):
    """Same as data_generator.ascon_permutation_vec, but with an option
    to skip the substitution layer on the final round only."""
    assert state.shape[0] == 5
    assert state.dtype == np.uint64
    S = state.copy()
    MASK64_VEC = np.uint64(MASK64)

    round_indices = list(range(12 - rounds, 12))
    last_round = round_indices[-1]

    for r in round_indices:
        S[2] ^= np.uint64(ROUND_CONSTANTS[r])

        is_last = (r == last_round)
        if not (is_last and skip_sbox_last_round):
            # --- Substitution layer (normal) ---
            S[0] ^= S[4]; S[4] ^= S[3]; S[2] ^= S[1]
            T = [(S[i] ^ MASK64_VEC) & S[(i + 1) % 5] for i in range(5)]
            for i in range(5):
                S[i] ^= T[(i + 1) % 5]
            S[1] ^= S[0]; S[0] ^= S[4]; S[3] ^= S[2]; S[2] ^= MASK64_VEC
        # else: skip S-box entirely for this round (ablation)

        # --- Linear diffusion layer (always applied) ---
        S[0] ^= rotr_vec(S[0], 19) ^ rotr_vec(S[0], 28)
        S[1] ^= rotr_vec(S[1], 61) ^ rotr_vec(S[1], 39)
        S[2] ^= rotr_vec(S[2], 1)  ^ rotr_vec(S[2], 6)
        S[3] ^= rotr_vec(S[3], 10) ^ rotr_vec(S[3], 17)
        S[4] ^= rotr_vec(S[4], 7)  ^ rotr_vec(S[4], 41)

    return S


def flip_rates(delta_word, delta_bit, n_trials, rng, ablated):
    raw1 = rng.bytes(40 * n_trials)
    p1 = np.frombuffer(raw1, dtype='>u8').reshape(n_trials, 5).T.astype(np.uint64)
    delta = np.zeros((5, n_trials), dtype=np.uint64)
    delta[delta_word, :] = np.uint64(1) << np.uint64(delta_bit)
    p2 = p1 ^ delta
    c1 = ascon_permutation_vec_ablated(p1, ROUNDS, skip_sbox_last_round=ablated)
    c2 = ascon_permutation_vec_ablated(p2, ROUNDS, skip_sbox_last_round=ablated)
    diff = c1 ^ c2
    return {(sw, sb): float(((diff[sw] >> np.uint64(sb)) & np.uint64(1)).mean())
            for (sw, sb) in SIGNATURE_BITS}


def determinism_score(rates):
    return sum(2 * abs(r - 0.5) for r in rates.values())


def run_sweep(word, ablated, rng, n_trials):
    scores = {}
    for bit in TEST_BITS:
        rates = flip_rates(word, bit, n_trials, rng, ablated)
        scores[bit] = determinism_score(rates)
    return scores


def main():
    rng = np.random.default_rng(SEED)

    for word in WORDS_TO_SWEEP:
        print(f"\n{'='*70}\nWORD {word}\n{'='*70}")

        print("\n-- NORMAL (full S-box every round) --")
        normal_scores = run_sweep(word, ablated=False, rng=rng, n_trials=6000)
        normal_ranked = sorted(normal_scores.items(), key=lambda kv: -kv[1])
        print("Top 8 (excluding bit 0, the trained/known-trivial bit):")
        for bit, sc in [t for t in normal_ranked if t[0] != 0][:8]:
            print(f"  bit {bit:2d}: score={sc:.3f}")
        print(f"  (bit 22 rank: "
              f"{[b for b,_ in normal_ranked].index(22)+1} of 64, "
              f"score={normal_scores[22]:.3f})")

        print("\n-- ABLATED (S-box skipped on final round only) --")
        abl_scores = run_sweep(word, ablated=True, rng=rng, n_trials=6000)
        abl_ranked = sorted(abl_scores.items(), key=lambda kv: -kv[1])
        print("Top 8 (excluding bit 0):")
        for bit, sc in [t for t in abl_ranked if t[0] != 0][:8]:
            print(f"  bit {bit:2d}: score={sc:.3f}")
        print(f"  (bit 22 rank: "
              f"{[b for b,_ in abl_ranked].index(22)+1} of 64, "
              f"score={abl_scores[22]:.3f})")

        print("\n-- Wraparound pairs, ABLATED case (bit b vs 64-b) --")
        seen = set()
        pairs = []
        for b in range(1, 64):
            p = 64 - b
            if p == b or b in seen:
                continue
            seen.add(b); seen.add(p)
            pairs.append((b, p, abl_scores[b], abl_scores.get(p, float('nan'))))
        pairs.sort(key=lambda t: -max(t[2], t[3]))
        for b, p, sb, sp in pairs[:10]:
            flag = "  <-- both high" if (sb > 1.0 and sp > 1.0) else ""
            print(f"  bit {b:2d} (score {sb:.3f})  <->  bit {p:2d} (score {sp:.3f}){flag}")

        print(f"\nSUMMARY word {word}: bit22 score normal={normal_scores[22]:.3f} "
              f"vs ablated={abl_scores[22]:.3f}  "
              f"({'SURVIVES' if abl_scores[22] > 0.5 * normal_scores[22] else 'WEAKENS/DISAPPEARS'} under ablation)")


if __name__ == "__main__":
    main()
