"""
data_generator.py

Generates labeled training data for the neural differential distinguisher.
Imports constants from ascon_core.py -- does NOT modify or reimplement 
the reference cipher logic outside of this file.

Optimizations:
1. True 80-Byte Compression: Eliminates all padding overhead by flattening the 
   640-bit matrix (64 bit positions x 10 channels) into exactly 80 bytes per sample.
2. Memory-Mapped Stream Chunking: Streams data in iterative chunks via open_memmap.
"""

import os
import numpy as np
from ascon_core import ascon_permutation, MASK64, ROUND_CONSTANTS

# Input difference used by Shen et al. 2024 for their 4-round ASCON result:
DEFAULT_DELTA = np.array([0, 0, 0, 0, 1], dtype=np.uint64)


# =====================================================================
# Vectorized Logic for High-Throughput Batch Generation
# =====================================================================

def rotr_vec(x: np.ndarray, n: int) -> np.ndarray:
    """Vectorized 64-bit rotate right for numpy uint64 arrays."""
    n %= 64
    return (x >> n) | (x << np.uint64(64 - n))


def ascon_permutation_vec(state: np.ndarray, rounds: int) -> np.ndarray:
    """
    Vectorized version of the ASCON permutation operating on a numpy array
    of shape (5, batch_size) and dtype uint64.
    """
    assert state.shape[0] == 5
    assert state.dtype == np.uint64

    S = state.copy()
    MASK64_VEC = np.uint64(MASK64)

    for r in range(12 - rounds, 12):
        S[2] ^= np.uint64(ROUND_CONSTANTS[r])

        # --- Substitution layer ---
        S[0] ^= S[4]; S[4] ^= S[3]; S[2] ^= S[1]
        T = [(S[i] ^ MASK64_VEC) & S[(i + 1) % 5] for i in range(5)]
        for i in range(5):
            S[i] ^= T[(i + 1) % 5]
        S[1] ^= S[0]; S[0] ^= S[4]; S[3] ^= S[2]; S[2] ^= MASK64_VEC

        # --- Linear diffusion layer ---
        S[0] ^= rotr_vec(S[0], 19) ^ rotr_vec(S[0], 28)
        S[1] ^= rotr_vec(S[1], 61) ^ rotr_vec(S[1], 39)
        S[2] ^= rotr_vec(S[2], 1)  ^ rotr_vec(S[2], 6)
        S[3] ^= rotr_vec(S[3], 10) ^ rotr_vec(S[3], 17)
        S[4] ^= rotr_vec(S[4], 7)  ^ rotr_vec(S[4], 41)

    return S


def pack_state_to_bits(state_c1: np.ndarray, state_c2: np.ndarray) -> np.ndarray:
    """
    Extracts MSB-first bits, concatenates into 10 channels, and packs them
    contiguously into exactly 80 bytes per sample without any padding.
    
    Input shapes: (5, chunk_size) each
    Output shape: (chunk_size, 80) dtype uint8
    """
    chunk_size = state_c1.shape[1]
    shifts = np.arange(63, -1, -1, dtype=np.uint64)[:, np.newaxis, np.newaxis]
    
    # Extract bits -> shape (64, 5, chunk_size)
    bits_c1 = (state_c1[np.newaxis, :, :] >> shifts) & np.uint64(1)
    bits_c2 = (state_c2[np.newaxis, :, :] >> shifts) & np.uint64(1)
    
    # Transpose to (chunk_size, 64, 5)
    c1_bitmatrix = bits_c1.transpose(2, 0, 1).astype(np.uint8)
    c2_bitmatrix = bits_c2.transpose(2, 0, 1).astype(np.uint8)
    
    # Combine into a flat feature sequence of 640 bits per sample -> shape (chunk_size, 640)
    X_flat = np.concatenate([c1_bitmatrix, c2_bitmatrix], axis=-1).reshape(chunk_size, 640)
    
    # Pack 640 bits perfectly into 80 bytes
    X_packed = np.packbits(X_flat, axis=-1)
    return X_packed


def unpack_batch(X_packed: np.ndarray) -> np.ndarray:
    """
    Restores full features into the structural (batch_size, 64, 10) matrix.
    
    Input shape:  (batch_size, 80) uint8
    Output shape: (batch_size, 64, 10) float32
    """
    batch_size = X_packed.shape[0]
    # Unpack 80 bytes back to 640 bits
    X_flat = np.unpackbits(X_packed, axis=-1).astype(np.float32)
    # Reshape back to our required spatial layout
    return X_flat.reshape(batch_size, 64, 10)


# =====================================================================
# Main Stream Dataset Generation Entry Point
# =====================================================================

def generate_dataset_streamed(filename: str, num_samples: int, rounds: int, 
                              chunk_size: int = 200000, delta=None, seed=None):
    """
    Streams the perfectly packed results into 80-byte wide arrays on disk.
    """
    if delta is None:
        delta = DEFAULT_DELTA
    else:
        delta = np.array(delta, dtype=np.uint64)

    rng = np.random.default_rng(seed)

    x_path = f"{filename}_X.npy"
    y_path = f"{filename}_Y.npy"

    shape_x = (num_samples, 80)  # EXACTLY 80 bytes per sample
    shape_y = (num_samples,)

    print(f"Allocating memory-mapped array storage on disk...")
    X_mmap = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.uint8, shape=shape_x)
    Y_mmap = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.uint8, shape=shape_y)

    samples_written = 0

    while samples_written < num_samples:
        current_chunk = min(chunk_size, num_samples - samples_written)
        print(f"  -> Computing chunk: samples {samples_written} to {samples_written + current_chunk}...")

        chunk_indices = np.arange(samples_written, samples_written + current_chunk, dtype=np.int64)
        Y_chunk = (chunk_indices % 2).astype(np.uint8)

        raw_p1 = rng.bytes(40 * current_chunk)
        p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(current_chunk, 5).T.astype(np.uint64)

        raw_p2_rand = rng.bytes(40 * current_chunk)
        p2_rand = np.frombuffer(raw_p2_rand, dtype='>u8').reshape(current_chunk, 5).T.astype(np.uint64)

        p2_real = p1 ^ delta[:, np.newaxis]
        label_mask = (Y_chunk == 1)
        p2 = np.where(label_mask, p2_real, p2_rand)

        c1 = ascon_permutation_vec(p1, rounds)
        c2 = ascon_permutation_vec(p2, rounds)

        X_chunk_packed = pack_state_to_bits(c1, c2)

        X_mmap[samples_written:samples_written + current_chunk] = X_chunk_packed
        Y_mmap[samples_written:samples_written + current_chunk] = Y_chunk

        samples_written += current_chunk

    X_mmap.flush()
    Y_mmap.flush()
    
    del X_mmap, Y_mmap
    print(f"Successfully streamed and packed {num_samples} samples to disk.")


def validate_packed_stream():
    """
    Validates that the 80-byte contiguous packing perfectly matches the scalar code.
    """
    print("Executing strict packing validation cross-check...")
    filename = "test_validation_stream"
    num_samples = 100
    rounds = 4
    seed = 54321

    rng = np.random.default_rng(seed)
    chunk_size = 40
    
    captured_p1 = []
    captured_p2 = []
    captured_y = []
    
    samples_written = 0
    while samples_written < num_samples:
        current_chunk = min(chunk_size, num_samples - samples_written)
        chunk_indices = np.arange(samples_written, samples_written + current_chunk, dtype=np.int64)
        Y_chunk = (chunk_indices % 2).astype(np.uint8)
        
        raw_p1 = rng.bytes(40 * current_chunk)
        p1 = np.frombuffer(raw_p1, dtype='>u8').reshape(current_chunk, 5).T.astype(np.uint64)
        raw_p2_rand = rng.bytes(40 * current_chunk)
        p2_rand = np.frombuffer(raw_p2_rand, dtype='>u8').reshape(current_chunk, 5).T.astype(np.uint64)
        
        p2_real = p1 ^ DEFAULT_DELTA[:, np.newaxis]
        p2 = np.where((Y_chunk == 1), p2_real, p2_rand)
        
        captured_p1.append(p1)
        captured_p2.append(p2)
        captured_y.append(Y_chunk)
        samples_written += current_chunk
        
    final_p1 = np.concatenate(captured_p1, axis=1)
    final_p2 = np.concatenate(captured_p2, axis=1)
    final_y = np.concatenate(captured_y)

    generate_dataset_streamed(filename, num_samples, rounds, chunk_size=chunk_size, seed=seed)

    X_mmap = np.lib.format.open_memmap(f"{filename}_X.npy", mode='r')
    Y_mmap = np.lib.format.open_memmap(f"{filename}_Y.npy", mode='r')

    for i in range(num_samples):
        unpacked_x = unpack_batch(X_mmap[i:i+1])[0] 
        label = Y_mmap[i]

        assert label == final_y[i], f"Label mismatch at {i}"

        c1_scalar = ascon_permutation(list(final_p1[:, i]), rounds)
        c2_scalar = ascon_permutation(list(final_p2[:, i]), rounds)

        def scalar_to_bits(w):
            return [(w >> (63 - b)) & 1 for b in range(64)]

        cols = [scalar_to_bits(w) for w in c1_scalar] + [scalar_to_bits(w) for w in c2_scalar]
        expected_x = np.array(cols, dtype=np.float32).T

        assert np.array_equal(unpacked_x, expected_x), f"Bit-packing data verification failure at sample {i}"

    del X_mmap, Y_mmap
    os.remove(f"{filename}_X.npy")
    os.remove(f"{filename}_Y.npy")
    print("Success! Memory-mapped packing outputs match scalar reference bit-for-bit.")


if __name__ == "__main__":
    validate_packed_stream()

    import time
    print("\nBenchmarking high-throughput streaming (1,000,000 samples)...")
    start = time.time()
    generate_dataset_streamed("ascon_r4_1m", num_samples=1000000, rounds=4, chunk_size=250000, seed=42)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.4f} seconds!")
    x_size_mb = os.path.getsize("ascon_r4_1m_X.npy") / (1024 * 1024)
    print(f"Generated X file size on disk: {x_size_mb:.2f} MB (Target hit: exactly 76.29 MB per 1M samples)")
    
    os.remove("ascon_r4_1m_X.npy")
    os.remove("ascon_r4_1m_Y.npy")