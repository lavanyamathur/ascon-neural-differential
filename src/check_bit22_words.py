import numpy as np
from data_generator import pack_state_to_bits

base = np.random.default_rng(1).integers(0, 2**63, size=(5, 1), dtype=np.uint64)

for w in range(5):
    p1 = np.random.default_rng(0).integers(0, 2**63, size=(5, 1), dtype=np.uint64)
    p2 = p1.copy()
    p2[w, 0] ^= np.uint64(1) << np.uint64(22)

    a = np.asarray(pack_state_to_bits(base, p1))
    b = np.asarray(pack_state_to_bits(base, p2))

    n_diff = (a != b).sum()
    idx = np.nonzero(a != b)
    print(f"word {w}: n_diff={n_diff}, idx={idx}")
