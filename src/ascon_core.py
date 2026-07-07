"""
ascon_core.py

Standalone implementation of the ASCON permutation, the building block used
in all ASCON-based schemes (AEAD, hash, MAC). For differential cryptanalysis
research we work directly on this permutation, round-reduced, exactly as
prior neural-distinguisher papers on ASCON do (Bao et al. 2021/2023,
Shen et al. 2024). No key schedule / AEAD wrapping is needed for this:
the permutation itself is treated as a public, keyless transform and
analyzed for round-reduced differential distinguishability.

State: 320 bits = five 64-bit words [x0, x1, x2, x3, x4]
Full permutation: 12 rounds. Round-reduced versions used for cryptanalysis
take the LAST `rounds` rounds of the 12 (i.e. rounds are numbered so that
round constants match the spec regardless of how many rounds you run) —
this matches how the reference implementation and the literature define
round-reduced ASCON.
"""

from typing import List

MASK64 = 0xFFFFFFFFFFFFFFFF

# The 12 round constants, in order, for the full 12-round permutation.
# A run of `rounds` rounds uses the LAST `rounds` entries of this list,
# i.e. round constants for indices [12-rounds, 12).
ROUND_CONSTANTS = [0xf0 - r * 0x10 + r * 0x1 for r in range(12)]


def rotr(x: int, n: int) -> int:
    """64-bit rotate right."""
    x &= MASK64
    return ((x >> n) | (x << (64 - n))) & MASK64


def ascon_permutation(state: List[int], rounds: int) -> List[int]:
    """
    Apply the ASCON permutation for `rounds` rounds (round-reduced if
    rounds < 12) to a 5-word (320-bit) state. Returns a NEW list; does not
    mutate the input, so callers can reuse the same plaintext-derived state
    for both the "real" and "difference" branch without aliasing bugs.

    state: list of 5 ints, each a 64-bit word
    rounds: number of rounds to run (1..12)
    """
    assert len(state) == 5, "state must be 5 64-bit words"
    assert 1 <= rounds <= 12, "rounds must be between 1 and 12"

    S = list(state)  # copy, don't mutate caller's state

    for r in range(12 - rounds, 12):
        # --- add round constant (to word x2) ---
        S[2] ^= ROUND_CONSTANTS[r]

        # --- substitution layer (5-bit S-box, bitsliced across the 5 words) ---
        S[0] ^= S[4]
        S[4] ^= S[3]
        S[2] ^= S[1]
        T = [(S[i] ^ MASK64) & S[(i + 1) % 5] for i in range(5)]
        for i in range(5):
            S[i] ^= T[(i + 1) % 5]
        S[1] ^= S[0]
        S[0] ^= S[4]
        S[3] ^= S[2]
        S[2] ^= MASK64

        # --- linear diffusion layer ---
        S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28)
        S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39)
        S[2] ^= rotr(S[2], 1) ^ rotr(S[2], 6)
        S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17)
        S[4] ^= rotr(S[4], 7) ^ rotr(S[4], 41)

    return S


def random_state(rng) -> List[int]:
    """Generate a uniformly random 320-bit state as 5 64-bit words."""
    return [rng.getrandbits(64) for _ in range(5)]


def state_xor(a: List[int], b: List[int]) -> List[int]:
    return [x ^ y for x, y in zip(a, b)]


def state_to_bits(state: List[int]) -> List[int]:
    """Flatten a 5x64-bit state into a list of 320 bits (MSB-first per word),
    the standard format for feeding into a neural distinguisher."""
    bits = []
    for word in state:
        bits.extend([(word >> (63 - i)) & 1 for i in range(64)])
    return bits


if __name__ == "__main__":
    # quick smoke test: all-zero state, full 12 rounds, just confirm it runs
    # and is deterministic. Real validation happens in validate_ascon.py
    # against the official reference implementation.
    s0 = [0, 0, 0, 0, 0]
    out = ascon_permutation(s0, 12)
    print("12-round permutation of all-zero state:")
    print([hex(w) for w in out])
