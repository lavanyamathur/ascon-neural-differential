"""
Exhaustive computation of the ASCON 5-bit S-box and its full 32x32
Differential Distribution Table (DDT).

Independent re-derivation -- does NOT read any prior results/ file.
Sbox is computed from ASCON's official bit-sliced substitution layer
(Dobraunig et al., spec Sec 5.2), not from a memorized lookup table,
so this is a from-scratch ground truth rather than a copy of old
numbers.

Bit convention (matches ascon.py's verified sbox() convention, see
shen-reproduction notes): x0 is the MOST significant bit.
  index = 16*x0 + 8*x1 + 4*x2 + 2*x3 + 1*x4
and the same convention is used to read the 5 output bits back into
an integer 0..31.
"""

import json
import csv
from itertools import product


def sbox_bits(x0, x1, x2, x3, x4):
    """ASCON's official 5-bit substitution layer, bit-sliced form.
    Operates on individual bits (ints 0/1), not word-batched arrays --
    this is the single-sample version of the same logic used in the
    project's ascon.py sbox(), for independent cross-check.
    """
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
    # index = 16*x0 + 8*x1 + 4*x2 + 2*x3 + x4  (x0 = MSB)
    return ((v >> 4) & 1, (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1)


def bits_to_int(b0, b1, b2, b3, b4):
    return (b0 << 4) | (b1 << 3) | (b2 << 2) | (b3 << 1) | b4


def build_sbox():
    table = [0] * 32
    for x in range(32):
        b = int_to_bits(x)
        y = sbox_bits(*b)
        table[x] = bits_to_int(*y)
    return table


def build_ddt(sbox):
    ddt = [[0] * 32 for _ in range(32)]
    for dx in range(32):
        for x in range(32):
            y1 = sbox[x]
            y2 = sbox[x ^ dx]
            dy = y1 ^ y2
            ddt[dx][dy] += 1
    return ddt


def main():
    sbox = build_sbox()
    print("S-box table (x -> S(x)):")
    print(sbox)

    ddt = build_ddt(sbox)

    # Sanity check against the previously-recorded dx=16 result:
    # dx=16 -> dy in {9,11,24,26} @ p=0.25 (i.e. count 8 out of 32)
    dx = 16
    hits = [(dy, ddt[dx][dy]) for dy in range(32) if ddt[dx][dy] > 0]
    print(f"\ndx={dx} nonzero row: {hits}")
    expected_dys = {9, 11, 24, 26}
    found_dys = {dy for dy, c in hits}
    print(f"Matches expected {{9,11,24,26}} @ count 8: "
          f"{found_dys == expected_dys and all(c == 8 for _, c in hits)}")

    # Write CSV: dx, dy, count, probability
    with open("ascon_sbox_ddt_fresh.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx", "dy", "count", "probability"])
        for dx in range(32):
            for dy in range(32):
                c = ddt[dx][dy]
                if c > 0:
                    w.writerow([dx, dy, c, c / 32.0])

    # Write JSON: sbox table + full ddt + summary of best (nonzero, dx!=0)
    best = []
    for dx in range(1, 32):
        row_max = max(ddt[dx])
        for dy in range(32):
            if ddt[dx][dy] == row_max:
                best.append({"dx": dx, "dy": dy, "count": row_max,
                             "probability": row_max / 32.0})
    # sort by probability desc, then dx
    best.sort(key=lambda r: (-r["probability"], r["dx"]))

    out = {
        "sbox": sbox,
        "ddt": ddt,
        "best_nonzero_dx_differentials": best[:10],
    }
    with open("ascon_sbox_ddt_fresh.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nTop differentials (dx!=0, max prob per row):")
    for r in best[:10]:
        print(r)

    print("\nWrote ascon_sbox_ddt_fresh.csv and ascon_sbox_ddt_fresh.json")


if __name__ == "__main__":
    main()
