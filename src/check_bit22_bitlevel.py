import numpy as np
from data_generator import pack_state_to_bits

base = np.random.default_rng(1).integers(0, 2**63, size=(5, 1), dtype=np.uint64)

for w in range(5):
    p1 = np.random.default_rng(0).integers(0, 2**63, size=(5, 1), dtype=np.uint64)
    p2 = p1.copy()
    p2[w, 0] ^= np.uint64(1) << np.uint64(22)

    a = np.asarray(pack_state_to_bits(base, p1)).astype(np.uint8)
    b = np.asarray(pack_state_to_bits(base, p2)).astype(np.uint8)

    # unpack to full bit-level resolution so we can see the EXACT bit,
    # not just which byte it falls in
    a_bits = np.unpackbits(a, axis=-1)
    b_bits = np.unpackbits(b, axis=-1)

    diff_bits = np.nonzero(a_bits != b_bits)[1]
    print(f"word {w}: n_bit_diff={len(diff_bits)}, bit_index={list(diff_bits)}")
