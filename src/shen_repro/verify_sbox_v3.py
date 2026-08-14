"""
ASCON S-box verification script (v3 - correct bit-order convention).

Root cause of v1/v2's mismatches: the official truth table indexes its
32 inputs with x0 (word 0) as the MOST significant bit, i.e.
    index = 16*x0 + 8*x1 + 4*x2 + 2*x3 + x4
Both v1 and v2 built test inputs treating x0 as the LEAST significant
bit instead (index = x0 + 2*x1 + 4*x2 + 8*x3 + 16*x4), and read the
output back the same (wrong) way. Reversing both the input and output
bit order consistently (word k <-> bit (4-k) of the index) makes the
real sbox() from ascon.py match the official table on all 32 entries.

This confirms sbox() itself is correct -- v1/v2's mismatches were a
test-harness convention bug, not a cipher defect.

Array shape stays (n, 5): this matches ascon.py's own convention,
where sbox(x, t) indexes x[:, 0..4] as the 5 words across all rows
(samples) at once -- see permutation()/make_td_diff() in ascon.py,
which all pass state shaped (num_samples, 5).

Usage (from inside the shen_ascon folder, with ascon_env activated):
    python verify_sbox_v3.py
"""

import numpy as np
import ascon as ac

official_sbox = [
    0x04, 0x0b, 0x1f, 0x14, 0x1a, 0x15, 0x09, 0x02,
    0x1b, 0x05, 0x08, 0x12, 0x1d, 0x03, 0x06, 0x1c,
    0x1e, 0x13, 0x07, 0x0e, 0x00, 0x0d, 0x11, 0x18,
    0x10, 0x0c, 0x01, 0x19, 0x16, 0x0a, 0x0f, 0x17,
]

n = 32

# words-minor, samples-major (matches ascon.py's x[:, k] convention);
# word k gets bit (4-k) of i, since the official table treats x0 (word 0)
# as the MSB of the 5-bit input.
state = np.zeros((n, 5), dtype=np.uint64)
for i in range(n):
    for k in range(5):
        state[i, k] = np.uint64((i >> (4 - k)) & 1)

t = np.zeros((n, 5), dtype=np.uint64)
ac.sbox(state, t)
# sbox() mutates x (here: state) in place; t is scratch, not the output.

mismatches = []
for i in range(n):
    out = 0
    for k in range(5):
        bit = int(state[i, k]) & 1
        out |= bit << (4 - k)   # undo the same MSB-first convention on output
    expected = official_sbox[i]
    status = "OK" if out == expected else "MISMATCH"
    if out != expected:
        mismatches.append((i, expected, out))
    print(f"x={i:2d} (0b{i:05b})  expected={expected:2d} (0b{expected:05b})  "
          f"got={out:2d} (0b{out:05b})  {status}")

print()
if mismatches:
    print(f"{len(mismatches)} MISMATCHES found: {mismatches}")
else:
    print("ALL 32 S-BOX ENTRIES MATCH THE OFFICIAL TABLE.")
