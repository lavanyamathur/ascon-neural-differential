"""
make_data.py

Generates the train/val packed datasets used by train_distinguisher.py.
Run this once before training any model in the architecture zoo.

Usage:
    python make_data.py
    python make_data.py --rounds 4 --train_samples 500000 --val_samples 50000

Produces (in the current folder):
    ascon_r{rounds}_train_X.npy / _Y.npy
    ascon_r{rounds}_val_X.npy   / _Y.npy
"""

import argparse
from data_generator import generate_dataset_streamed


def main():
    parser = argparse.ArgumentParser(description="Generate ASCON distinguisher train/val data")
    parser.add_argument("--rounds", type=int, default=4,
                         help="Number of ASCON permutation rounds (round-reduced). Default 4.")
    parser.add_argument("--train_samples", type=int, default=500_000)
    parser.add_argument("--val_samples", type=int, default=50_000)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--val_seed", type=int, default=2,
                         help="Must differ from --train_seed so val data isn't a copy of train data.")
    parser.add_argument("--chunk_size", type=int, default=200_000)
    args = parser.parse_args()

    assert args.train_seed != args.val_seed, "train_seed and val_seed must differ"

    train_prefix = f"ascon_r{args.rounds}_train"
    val_prefix = f"ascon_r{args.rounds}_val"

    print(f"Generating TRAIN set: {args.train_samples:,} samples, "
          f"{args.rounds} rounds, seed={args.train_seed}")
    generate_dataset_streamed(
        train_prefix, num_samples=args.train_samples, rounds=args.rounds,
        chunk_size=args.chunk_size, seed=args.train_seed,
    )

    print(f"\nGenerating VAL set: {args.val_samples:,} samples, "
          f"{args.rounds} rounds, seed={args.val_seed}")
    generate_dataset_streamed(
        val_prefix, num_samples=args.val_samples, rounds=args.rounds,
        chunk_size=args.chunk_size, seed=args.val_seed,
    )

    print(f"\nDone. Train with:")
    print(f"  python train_distinguisher.py --train_prefix {train_prefix} "
          f"--val_prefix {val_prefix} --model_type mlp")
    print(f"  python train_distinguisher.py --train_prefix {train_prefix} "
          f"--val_prefix {val_prefix} --model_type resnet_cnn")


if __name__ == "__main__":
    main()
