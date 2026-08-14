"""
trace_bit22.py

Purpose: confirm that "bit 22" as used in your sweeps (--delta_bit 22)
actually corresponds to the same physical state bit before AND after
whatever packing/unpacking happens between raw 64-bit words and the
flat vector the CNN/MLP consume.

The failure mode we're checking for: pack_state_to_bits (or
ascon_permutation_vec) reverses bit order within a word, or reindexes
words, such that "--delta_bit 22" in your CLI does NOT correspond to
bit 22 in the same convention everywhere. If that's happening, the
"bit 22 is louder" finding could really be "some OTHER bit is louder,
and it happens to get mislabeled as 22 somewhere in the pipeline."

Fill in the two imports below with your actual functions, then run:
    python trace_bit22.py
"""

import numpy as np

# --- EDIT THESE to match your actual project ---
# from data_generator import pack_state_to_bits, DEFAULT_DELTA
# from ascon_permutation import ascon_permutation_vec
PACK_FN = None   # e.g. pack_state_to_bits
PERM_FN = None   # e.g. ascon_permutation_vec (optional, for step 3)


def flip_bit_raw(state, word_idx, bit_idx):
    """state: np.uint64 array of shape (5,). Flip one bit in one word."""
    state = state.copy()
    state[word_idx] ^= np.uint64(1) << np.uint64(bit_idx)
    return state


def check_raw_flip(word_idx=4, bit_idx=22):
    """Step 1: confirm the raw numpy flip lands where we think, in isolation."""
    rng = np.random.default_rng(0)
    p1 = rng.integers(0, 2**63, size=5, dtype=np.uint64)
    p2 = flip_bit_raw(p1, word_idx, bit_idx)

    diff = p1 ^ p2
    nz_words = np.nonzero(diff)[0]
    print(f"--- STEP 1: raw numpy flip check (word={word_idx}, bit={bit_idx}) ---")
    print(f"  words that differ: {nz_words} (expected: [{word_idx}])")
    for w in nz_words:
        bit_positions = np.nonzero(
            np.unpackbits(diff[w].view(np.uint8))[::-1]
        )[0]
        print(f"  word {w}: differing bit positions = {list(bit_positions)} "
              f"(expected: [{bit_idx}])")
    if list(nz_words) == [word_idx]:
        print("  OK: only the target word differs.")
    else:
        print("  MISMATCH: unexpected word(s) differ -- check flip_bit_raw / word ordering.")


def check_packed_flip(word_idx=4, bit_idx=22):
    """
    Step 2: confirm that after your real pack_state_to_bits(), the SAME
    single bit differs at the position your model-facing code expects.
    This is the step that actually matters -- it's what the CNN/MLP see.
    """
    if PACK_FN is None:
        print("\n--- STEP 2: SKIPPED -- set PACK_FN to your real pack_state_to_bits ---")
        return

    rng = np.random.default_rng(0)
    p1 = rng.integers(0, 2**63, size=5, dtype=np.uint64)
    p2 = flip_bit_raw(p1, word_idx, bit_idx)

    packed1 = np.asarray(PACK_FN(p1))
    packed2 = np.asarray(PACK_FN(p2))

    diff = packed1 != packed2
    n_diff = diff.sum()
    diff_idx = np.nonzero(diff)[0]

    print(f"\n--- STEP 2: packed-vector flip check (word={word_idx}, bit={bit_idx}) ---")
    print(f"  packed vector length: {len(packed1)}")
    print(f"  number of differing positions: {n_diff} (expected: 1)")
    print(f"  differing index/indices: {list(diff_idx)}")

    if n_diff != 1:
        print("  MISMATCH: a single-bit raw flip produced more than one packed-bit "
              "difference (or zero) -- pack_state_to_bits is not a clean 1:1 bit map, "
              "or something upstream (e.g. the permutation being applied before packing) "
              "is spreading the difference. Re-check whether you're packing plaintext "
              "or ciphertext, and at which pipeline stage.")
    else:
        expected_naive = word_idx * 64 + bit_idx
        actual = diff_idx[0]
        print(f"  naive expected flat index (word*64+bit): {expected_naive}, actual: {actual}")
        if actual == expected_naive:
            print("  OK: matches naive word-major, bit-ascending layout.")
        else:
            print("  NOTE: doesn't match the naive layout -- not necessarily wrong, "
                  "but means your packing uses a different bit/word order "
                  "(e.g. MSB-first within word, or bit-major layout). Just make sure "
                  "this is CONSISTENT with how --delta_bit is interpreted elsewhere "
                  "in saliency_analysis.py, or your bit-22 label may not mean what "
                  "you think across different scripts.")


def check_multiple_bits_words():
    """Step 3: repeat the packed check for every word, at bit 22 specifically,
    since that's the bit under scrutiny -- confirms the mapping is consistent
    across all five words, not just word 4."""
    if PACK_FN is None:
        print("\n--- STEP 3: SKIPPED -- set PACK_FN to your real pack_state_to_bits ---")
        return
    print("\n--- STEP 3: bit 22 across all 5 words ---")
    for w in range(5):
        check_packed_flip(word_idx=w, bit_idx=22)


if __name__ == "__main__":
    check_raw_flip(word_idx=4, bit_idx=22)
    check_packed_flip(word_idx=4, bit_idx=22)
    check_multiple_bits_words()
