"""
multipair_generator.py

Extends data_generator.py to produce GROUPED multi-pair samples, for the
round-4 attention-based distinguisher (matching Shen et al.'s score-
distribution-over-multiple-ciphertext-pairs setup).

Does not modify data_generator.py or ascon_core.py -- imports and reuses
their vectorized primitives directly, so packing/permutation logic stays
in one place and stays bit-for-bit consistent with the validated single-
pair pipeline.

Group semantics:
  - Each GROUP has one label: 1 (real) or 0 (random).
  - A "real" group contains k pairs, each independently sampled, but all
    sharing the same fixed DEFAULT_DELTA -- this mirrors a chosen-plaintext
    attacker who fixes one input difference and collects k ciphertext pairs
    under it, which is what Shen et al.'s score-distribution method assumes.
  - A "random" group contains k independently sampled pairs with NO
    difference relationship at all (both plaintexts fully random per pair),
    matching the negative class in the single-pair generator.
  - k is fixed per dataset (pass k explicitly); vary k across separate
    dataset files for the k-ablation, rather than mixing group sizes
    within one file.
"""

import os
import numpy as np
from ascon_core import MASK64, ROUND_CONSTANTS  # noqa: F401 (kept for parity with data_generator.py)
from data_generator import (
    DEFAULT_DELTA,
    ascon_permutation_vec,
    pack_state_to_bits,
)


def generate_multipair_dataset_streamed(
    filename: str,
    num_groups: int,
    k: int,
    rounds: int,
    chunk_size: int = 20000,
    delta=None,
    seed=None,
):
    """
    Streams grouped multi-pair samples to disk.

    Output shapes:
      X: (num_groups, k, 80) uint8   -- k packed single-pair samples per group
      Y: (num_groups,)       uint8   -- one label per group

    chunk_size here counts GROUPS per chunk, not raw pairs (so actual pairs
    processed per chunk = chunk_size * k -- keep chunk_size smaller than in
    the single-pair generator if k is large, to bound memory the same way).
    """
    if delta is None:
        delta = DEFAULT_DELTA
    else:
        delta = np.array(delta, dtype=np.uint64)

    rng = np.random.default_rng(seed)

    x_path = f"{filename}_X.npy"
    y_path = f"{filename}_Y.npy"

    shape_x = (num_groups, k, 80)
    shape_y = (num_groups,)

    print(f"Allocating memory-mapped array storage on disk (k={k})...")
    X_mmap = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape_x)
    Y_mmap = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.uint8, shape=shape_y)

    groups_written = 0

    while groups_written < num_groups:
        current_chunk = min(chunk_size, num_groups - groups_written)
        total_pairs = current_chunk * k
        print(f"  -> Computing chunk: groups {groups_written} to "
              f"{groups_written + current_chunk} ({total_pairs} pairs)...")

        # One label per group, alternated for balance (mirrors single-pair generator)
        chunk_group_indices = np.arange(groups_written, groups_written + current_chunk, dtype=np.int64)
        Y_chunk = (chunk_group_indices % 2).astype(np.uint8)

        # Expand each group's label across its k pairs
        label_mask_flat = np.repeat(Y_chunk == 1, k)  # shape (total_pairs,)

        raw_p1 = rng.bytes(40 * total_pairs)
        p1 = np.frombuffer(raw_p1, dtype=">u8").reshape(total_pairs, 5).T.astype(np.uint64)

        raw_p2_rand = rng.bytes(40 * total_pairs)
        p2_rand = np.frombuffer(raw_p2_rand, dtype=">u8").reshape(total_pairs, 5).T.astype(np.uint64)

        p2_real = p1 ^ delta[:, np.newaxis]
        p2 = np.where(label_mask_flat, p2_real, p2_rand)

        c1 = ascon_permutation_vec(p1, rounds)
        c2 = ascon_permutation_vec(p2, rounds)

        X_chunk_packed_flat = pack_state_to_bits(c1, c2)          # (total_pairs, 80)
        X_chunk_packed = X_chunk_packed_flat.reshape(current_chunk, k, 80)

        X_mmap[groups_written:groups_written + current_chunk] = X_chunk_packed
        Y_mmap[groups_written:groups_written + current_chunk] = Y_chunk

        groups_written += current_chunk

    X_mmap.flush()
    Y_mmap.flush()
    del X_mmap, Y_mmap
    print(f"Successfully streamed {num_groups} groups (k={k}, rounds={rounds}) to disk.")


def unpack_multipair_batch(X_packed: np.ndarray) -> np.ndarray:
    """
    Restores full features for a batch of groups.

    Input shape:  (batch_size, k, 80) uint8
    Output shape: (batch_size, k, 64, 10) float32

    Reuses data_generator.unpack_batch per-pair by flattening the group
    dimension into the batch dimension, unpacking once, then reshaping back --
    avoids duplicating the unpacking logic.
    """
    from data_generator import unpack_batch

    batch_size, k, _ = X_packed.shape
    flat = X_packed.reshape(batch_size * k, 80)
    unpacked_flat = unpack_batch(flat)              # (batch_size*k, 64, 10)
    return unpacked_flat.reshape(batch_size, k, 64, 10)


if __name__ == "__main__":
    # Small smoke test -- not a full validation like data_generator's
    # validate_packed_stream(), just confirms shapes and round-trip.
    import time

    test_file = "test_multipair_smoke"
    num_groups, k, rounds = 200, 16, 4

    start = time.time()
    generate_multipair_dataset_streamed(test_file, num_groups, k, rounds, chunk_size=50, seed=7)
    elapsed = time.time() - start

    X = np.lib.format.open_memmap(f"{test_file}_X.npy", mode="r")
    Y = np.lib.format.open_memmap(f"{test_file}_Y.npy", mode="r")
    assert X.shape == (num_groups, k, 80), X.shape
    assert Y.shape == (num_groups,), Y.shape
    unpacked = unpack_multipair_batch(X[:5])
    assert unpacked.shape == (5, k, 64, 10), unpacked.shape
    print(f"Smoke test passed. Shapes OK. Generated in {elapsed:.4f}s.")

    del X, Y
    os.remove(f"{test_file}_X.npy")
    os.remove(f"{test_file}_Y.npy")
