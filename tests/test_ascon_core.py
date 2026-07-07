"""
test_ascon_core.py

Regression test for src/ascon_core.py.

Cross-checks our permutation against an inline copy of the official reference
implementation's round function (from pyascon / the NIST SP 800-232 reference
code), so this test has no network dependency and can run in CI.

If this test ever fails after modifying ascon_core.py, STOP — every dataset
generated with the broken code is invalid and needs to be regenerated.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ascon_core import ascon_permutation as impl_permutation

MASK64 = 0xFFFFFFFFFFFFFFFF


def _rotr(x, r):
    return (x >> r) | ((x << (64 - r)) & MASK64)


def _reference_permutation(state, rounds):
    """Inline copy of the reference round function, kept independent of
    ascon_core.py on purpose so this is a true cross-check, not a tautology."""
    S = list(state)
    for r in range(12 - rounds, 12):
        S[2] ^= (0xF0 - r * 0x10 + r * 0x1)
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
        S[0] ^= _rotr(S[0], 19) ^ _rotr(S[0], 28)
        S[1] ^= _rotr(S[1], 61) ^ _rotr(S[1], 39)
        S[2] ^= _rotr(S[2], 1) ^ _rotr(S[2], 6)
        S[3] ^= _rotr(S[3], 10) ^ _rotr(S[3], 17)
        S[4] ^= _rotr(S[4], 7) ^ _rotr(S[4], 41)
    return S


def test_matches_reference_across_random_states():
    rng = random.Random(1234)
    for _ in range(5000):
        state = [rng.getrandbits(64) for _ in range(5)]
        rounds = rng.randint(1, 12)
        assert impl_permutation(state, rounds) == _reference_permutation(state, rounds)


def test_all_zero_state_full_permutation():
    s0 = [0, 0, 0, 0, 0]
    assert impl_permutation(s0, 12) == _reference_permutation(s0, 12)


def test_does_not_mutate_input():
    s0 = [1, 2, 3, 4, 5]
    s0_copy = list(s0)
    impl_permutation(s0, 6)
    assert s0 == s0_copy, "ascon_permutation must not mutate the caller's state list"


if __name__ == "__main__":
    test_matches_reference_across_random_states()
    test_all_zero_state_full_permutation()
    test_does_not_mutate_input()
    print("All tests passed.")
