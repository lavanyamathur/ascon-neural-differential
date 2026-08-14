"""
Bit-22 mechanism check — independent verification script.

Traces which output bits are perfectly deterministic (prob=1) after N rounds
of the ASCON permutation for a given single-bit input difference, then checks
which input bits collide with bit-0's known signature output positions
(word 3 / bit 10, word 3 / bit 32).

USAGE:
    python bit22_mechanism_check.py                  # uses built-in reference impl
    python bit22_mechanism_check.py --rounds 3 --samples 20000

If you want this to use YOUR pipeline's validated permutation instead of the
built-in reference implementation below (recommended before treating results
as final), replace the `permute()` function with an import from your own
module, e.g.:
    from ascon_core import ascon_permutation as permute
as long as it takes/returns 5 numpy uint64 arrays of shape (N,) in the same
word order your data_generator.py uses.
"""
import argparse
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

def rotr(x, r):
    r = np.uint64(r)
    return ((x >> r) | (x << (np.uint64(64) - r))) & MASK

def sbox_layer(x0, x1, x2, x3, x4):
    x0 = x0 ^ x4
    x4 = x4 ^ x3
    x2 = x2 ^ x1
    t0 = (~x0 & MASK) & x1
    t1 = (~x1 & MASK) & x2
    t2 = (~x2 & MASK) & x3
    t3 = (~x3 & MASK) & x4
    t4 = (~x4 & MASK) & x0
    x0 = x0 ^ t1
    x1 = x1 ^ t2
    x2 = x2 ^ t3
    x3 = x3 ^ t4
    x4 = x4 ^ t0
    x1 = x1 ^ x0
    x0 = x0 ^ x4
    x3 = x3 ^ x2
    x2 = (~x2) & MASK
    return x0, x1, x2, x3, x4

def lin_layer(x0, x1, x2, x3, x4):
    x0 = x0 ^ rotr(x0, 19) ^ rotr(x0, 28)
    x1 = x1 ^ rotr(x1, 61) ^ rotr(x1, 39)
    x2 = x2 ^ rotr(x2, 1)  ^ rotr(x2, 6)
    x3 = x3 ^ rotr(x3, 10) ^ rotr(x3, 17)
    x4 = x4 ^ rotr(x4, 7)  ^ rotr(x4, 41)
    return x0, x1, x2, x3, x4

FULL_RC = [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b]

def permute(x0, x1, x2, x3, x4, rounds, start_round=0):
    """NOTE: verify start_round matches the convention your data_generator.py
    uses (first `rounds` of 12 vs last `rounds` of 12) — swap start_round if
    your per_bit_breakdown.py numbers don't match."""
    rc = [np.uint64(c) for c in FULL_RC[start_round:start_round+rounds]]
    for r in range(rounds):
        x2 = x2 ^ rc[r]
        x0, x1, x2, x3, x4 = sbox_layer(x0, x1, x2, x3, x4)
        x0, x1, x2, x3, x4 = lin_layer(x0, x1, x2, x3, x4)
    return x0, x1, x2, x3, x4

def random_state(n, rng):
    return [rng.integers(0, 2**64, size=n, dtype=np.uint64) for _ in range(5)]

def flip_bit(words, word_idx, bit):
    out = [w.copy() for w in words]
    out[word_idx] = out[word_idx] ^ (np.uint64(1) << np.uint64(bit))
    return out

def deterministic_output_bits(word_idx, bit, rounds, n, rng, thresh=0.9):
    base = random_state(n, rng)
    flipped = flip_bit(base, word_idx, bit)
    o1 = permute(*base, rounds=rounds)
    o2 = permute(*flipped, rounds=rounds)
    hits = set()
    for w in range(5):
        diff = o1[w] ^ o2[w]
        bits = ((diff[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(np.float64)
        rate = bits.mean(axis=0)
        det = 2 * np.abs(rate - 0.5)
        for ob in range(64):
            if det[ob] > thresh:
                hits.add((w, ob))
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--word", type=int, default=4, help="word index the trained bit lives in")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Step 1: finding bit-0's deterministic output signature ({args.rounds} rounds)...")
    sig0 = deterministic_output_bits(args.word, 0, args.rounds, args.samples, rng)
    print(f"  bit 0 signature: {sorted(sig0)}")

    print(f"\nStep 2: scanning all 64 bits in word {args.word} for overlap with that signature...")
    overlap_bits = []
    for bit in range(64):
        hits = deterministic_output_bits(args.word, bit, args.rounds, args.samples, rng)
        overlap = hits & sig0
        if overlap and bit != 0:
            overlap_bits.append((bit, overlap, len(hits)))

    if not overlap_bits:
        print("  No other bit collides with bit 0's signature — mechanism not reproduced "
              "under this round/constant convention. Try --rounds or adjust start_round.")
    for bit, overlap, n_hits in overlap_bits:
        full = " (FULL MATCH)" if overlap == sig0 else ""
        print(f"  bit {bit:2d}: overlaps {sorted(overlap)}{full}  (total deterministic bits: {n_hits})")

if __name__ == "__main__":
    main()
