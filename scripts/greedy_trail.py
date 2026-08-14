"""
Greedy differential trail (characteristic) search over the ASCON
permutation, built on top of the independently re-derived 5-bit S-box
DDT (compute_ddt.py).

Independent re-derivation -- rebuilds the S-box/DDT from scratch here
too (does not import or read any prior file), then greedily walks a
word-difference through N rounds of the ASCON linear layer + sbox
layer, picking the locally-best (max probability) output difference
at each of the 64 parallel S-box columns each round.

This is the standard "greedy characteristic" approach: NOT a proof of
the best possible trail (that needs MILP/SAT or branch-and-bound), but
a reproducible, from-scratch approximation of the round-by-round
probability decay, useful for sanity-checking the previously reported
2^-2 -> 2^-193 style result.

ASCON state: 5 words of WORD_BITS bits each (64 for full ASCON).
Sbox layer operates bitslice-wise across the 5 words: for bit column j
(0..63), the 5 input bits (x0[j],...,x4[j]) go through the same 5-bit
Sbox as compute_ddt.py.
Linear layer (Sigma functions) is ASCON's official rotation-XOR diffusion,
applied per word -- and since it's linear over GF(2), it applies directly
to *differences* the same way it applies to values.
"""

import json


WORD_BITS = 64

ROT = {
    0: (19, 28),
    1: (61, 39),
    2: (1, 6),
    3: (10, 17),
    4: (7, 41),
}


def sbox_bits(x0, x1, x2, x3, x4):
    x0 ^= x4
    x4 ^= x3
    x2 ^= x1
    t0, t1, t2, t3, t4 = x0, x1, x2, x3, x4
    t0 = 1 - t0
    t1 = 1 - t1
    t2 = 1 - t2
    t3 = 1 - t3
    t4 = 1 - t4
    t0 &= x1
    t1 &= x2
    t2 &= x3
    t3 &= x4
    t4 &= x0
    x0 ^= t1
    x1 ^= t2
    x2 ^= t3
    x3 ^= t4
    x4 ^= t0
    x1 ^= x0
    x0 ^= x4
    x3 ^= x2
    x2 = 1 - x2
    return x0, x1, x2, x3, x4


def int_to_bits(v):
    return ((v >> 4) & 1, (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1)


def bits_to_int(b0, b1, b2, b3, b4):
    return (b0 << 4) | (b1 << 3) | (b2 << 2) | (b3 << 1) | b4


def build_sbox():
    table = [0] * 32
    for x in range(32):
        table[x] = bits_to_int(*sbox_bits(*int_to_bits(x)))
    return table


def build_ddt(sbox):
    ddt = [[0] * 32 for _ in range(32)]
    for dx in range(32):
        for x in range(32):
            dy = sbox[x] ^ sbox[x ^ dx]
            ddt[dx][dy] += 1
    return ddt


def best_dy_for_dx(ddt, dx):
    """Return (best_dy, probability) for a given input difference dx.
    dx=0 always maps to dy=0 with probability 1 (no active input -> no
    active output). For dx!=0, pick the dy with max count (ties broken
    by smallest dy for determinism)."""
    if dx == 0:
        return 0, 1.0
    row = ddt[dx]
    best_count = max(row)
    best_dy = min(dy for dy, c in enumerate(row) if c == best_count)
    return best_dy, best_count / 32.0


def rotr(x, n, bits=WORD_BITS):
    mask = (1 << bits) - 1
    x &= mask
    return ((x >> n) | (x << (bits - n))) & mask


def linear_layer(words):
    """ASCON's linear diffusion layer. Applied identically to values or
    XOR-differences since it's GF(2)-linear (rotation + XOR only)."""
    out = [0] * 5
    for i in range(5):
        r1, r2 = ROT[i]
        out[i] = words[i] ^ rotr(words[i], r1) ^ rotr(words[i], r2)
    return out


def sbox_layer_on_difference(words, ddt):
    """Apply the sbox layer to a word-difference, greedily, column by
    column. Returns (output_words, round_probability)."""
    out = [0] * 5
    round_prob = 1.0
    nonzero_cols = 0
    for j in range(WORD_BITS):
        dx = 0
        for w in range(5):
            bit = (words[w] >> j) & 1
            dx |= (bit << w)
        # dx bit w corresponds to word w; convert to compute_ddt's
        # x0..x4 (MSB..LSB) convention: word0 -> bit4(MSB)... word4->bit0(LSB)
        dx_conv = 0
        for w in range(5):
            b = (dx >> w) & 1
            dx_conv |= (b << (4 - w))
        dy_conv, p = best_dy_for_dx(ddt, dx_conv)
        if dx_conv != 0:
            nonzero_cols += 1
            round_prob *= p
        # convert dy back to word bits
        for w in range(5):
            b = (dy_conv >> (4 - w)) & 1
            out[w] |= (b << j)
    return out, round_prob, nonzero_cols


def run_trail(start_words, n_rounds, ddt):
    import math
    words = list(start_words)
    cumulative_log2 = 0.0
    history = []
    for r in range(1, n_rounds + 1):
        words, round_prob, active_cols = sbox_layer_on_difference(words, ddt)
        # accumulate in log2 space to avoid float underflow at high round counts
        cumulative_log2 += math.log2(round_prob) if round_prob > 0 else float("-inf")
        words = linear_layer(words)
        hamming = sum(bin(w).count("1") for w in words)
        history.append({
            "round": r,
            "round_probability": round_prob,
            "cumulative_log2_probability": cumulative_log2,
            "active_sbox_columns": active_cols,
            "post_round_hamming_weight": hamming,
        })
    return history


def main():
    sbox = build_sbox()
    ddt = build_ddt(sbox)

    # Start with a single active bit: word 0, bit position 0.
    start = [1, 0, 0, 0, 0]
    n_rounds = 12  # full ASCON permutation p^12

    history = run_trail(start, n_rounds, ddt)

    print(f"Greedy trail from single-bit start (word0 bit0), {n_rounds} rounds:\n")
    print(f"{'round':>5} {'round_prob':>12} {'cum_prob (2^x)':>16} {'active_cols':>12} {'hamming_wt':>11}")
    for h in history:
        exp = h["cumulative_log2_probability"]
        print(f"{h['round']:>5} {h['round_probability']:>12.6f} {exp:>15.2f} "
              f"{h['active_sbox_columns']:>12} {h['post_round_hamming_weight']:>11}")

    with open("greedy_trail_result.json", "w") as f:
        json.dump({"start_words": start, "n_rounds": n_rounds, "history": history}, f, indent=2)
    print("\nWrote greedy_trail_result.json")


if __name__ == "__main__":
    main()
