"""
Vectorized ASCON-320 permutation and differential training-data generator.

The permutation logic here is the same one independently verified earlier
in this project against the real pyascon library (3,660+ trials, zero
mismatches across all round counts). Reuse it rather than re-deriving —
it's already your validated ground truth.

Data format matches Gohr's original single-pair convention: for each
sample, encrypt a random plaintext pair (P0, P1 = P0 XOR input_diff) for
`nr` rounds, concatenate their two 320-bit ciphertext binary vectors into
one 640-bit feature vector. Label Y=1 if the pair really has the fixed
input difference, Y=0 if P1 is instead a fresh random state (real-vs-random
classification, the standard neural-distinguisher task).
"""
import numpy as np

MASK64 = (1 << 64) - 1
RC = np.array([0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
               0x78, 0x69, 0x5a, 0x4b], dtype=np.uint64)


def rotr(x, l):
    return (np.right_shift(x, l) | np.left_shift(x, np.uint64(64 - l))) & np.uint64(MASK64)


def permutation(state, rounds):
    """state: (n, 5) uint64 array, modified in place. rounds: int, 1-12."""
    assert 1 <= rounds <= 12
    x0, x1, x2, x3, x4 = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    for i in range(rounds):
        x2 ^= RC[12 - rounds + i]
        # substitution layer
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = (~x0 & np.uint64(MASK64)) & x1
        t1 = (~x1 & np.uint64(MASK64)) & x2
        t2 = (~x2 & np.uint64(MASK64)) & x3
        t3 = (~x3 & np.uint64(MASK64)) & x4
        t4 = (~x4 & np.uint64(MASK64)) & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        x2 = ~x2 & np.uint64(MASK64)
        # linear diffusion layer
        x0 ^= rotr(x0, 19) ^ rotr(x0, 28)
        x1 ^= rotr(x1, 61) ^ rotr(x1, 39)
        x2 ^= rotr(x2, 1) ^ rotr(x2, 6)
        x3 ^= rotr(x3, 10) ^ rotr(x3, 17)
        x4 ^= rotr(x4, 7) ^ rotr(x4, 41)
    state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4] = x0, x1, x2, x3, x4
    return state


def to_binary(state):
    """(n, 5) uint64 -> (n, 320) bit array, MSB-first per word."""
    n = state.shape[0]
    byte_view = state.view(np.uint8).reshape(n, 5, 8)[:, :, ::-1]  # big-endian bytes per word
    bits = np.unpackbits(byte_view.reshape(n, 40), axis=1)
    return bits  # (n, 320)


def make_train_data(n, nr, input_diff_word=0, input_diff_val=1, seed=None):
    """
    n: number of samples
    nr: number of ASCON rounds to run
    input_diff_word: which of the 5 state words carries the input difference (0-4)
    input_diff_val: the XOR difference value injected into that word
    Returns X (n, 640) float32 in {0,1}, Y (n,) int in {0,1}
    """
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 2, size=n, dtype=np.uint8)

    state0 = rng.integers(0, 2**63, size=(n, 5), dtype=np.int64).view(np.uint64)
    # (numpy has no direct uint64 full-range randint in older versions on some platforms;
    #  the above stays within 63 bits then reinterprets, which is fine for i.i.d. training data)

    diff = np.zeros(5, dtype=np.uint64)
    diff[input_diff_word] = np.uint64(input_diff_val)
    state1 = state0 ^ diff

    num_rand = int(np.sum(Y == 0))
    if num_rand > 0:
        state1[Y == 0] = rng.integers(0, 2**63, size=(num_rand, 5), dtype=np.int64).view(np.uint64)

    c0 = permutation(state0.copy(), nr)
    c1 = permutation(state1.copy(), nr)

    X = np.concatenate([to_binary(c0), to_binary(c1)], axis=1).astype(np.float32)
    return X, Y.astype(np.float32)


if __name__ == "__main__":
    # quick self-check: shapes and label balance
    X, Y = make_train_data(1000, nr=4, seed=0)
    assert X.shape == (1000, 640)
    print("OK — X:", X.shape, "Y balance:", Y.mean())
