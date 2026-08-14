"""
Rounds sweep: train each of the four architectures at each round count in
ROUNDS_TO_TEST, log validation accuracy, save results.csv + a plot.

This is intentionally the FIRST experiment in the sequence (cheap, and
gives you the accuracy-vs-round curve that everything else — bit-size
sweep, attention model, RNN baseline — gets compared against).

Usage:
    python rounds_sweep.py --quick      # small run to sanity-check the pipeline
    python rounds_sweep.py              # full run

Adjust N_TRAIN / N_VAL / EPOCHS for your compute budget — the --quick flag
uses small numbers so you can confirm everything runs end-to-end in a few
minutes before committing to a multi-hour full run.
"""
import argparse
import csv
import time

import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

from ascon_perm import make_train_data
from architectures import ARCHITECTURES

ROUNDS_TO_TEST = [1, 2, 3, 4, 5, 6]
BATCH_SIZE = 5000


def cyclic_lr(num_epochs, high_lr, low_lr):
    def sched(i):
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return sched


def run_sweep(n_train, n_val, epochs, out_csv='results_rounds_sweep.csv'):
    rows = []
    for nr in ROUNDS_TO_TEST:
        print(f"\n{'='*60}\nROUNDS = {nr}\n{'='*60}")
        X_train, Y_train = make_train_data(n_train, nr=nr, seed=1000 + nr)
        X_val, Y_val = make_train_data(n_val, nr=nr, seed=2000 + nr)

        for arch_name, builder in ARCHITECTURES.items():
            t0 = time.time()
            model = builder(input_dim=X_train.shape[1])
            lr = LearningRateScheduler(cyclic_lr(epochs, 0.002, 0.0001))
            hist = model.fit(
                X_train, Y_train,
                epochs=epochs, batch_size=BATCH_SIZE, shuffle=True,
                validation_data=(X_val, Y_val),
                callbacks=[lr], verbose=0,
            )
            best_val_acc = max(hist.history['val_acc'])
            elapsed = time.time() - t0
            print(f"  {arch_name:10s}  best_val_acc={best_val_acc:.4f}  ({elapsed:.0f}s)")
            rows.append({
                'rounds': nr,
                'architecture': arch_name,
                'best_val_acc': best_val_acc,
                'n_train': n_train,
                'n_val': n_val,
                'epochs': epochs,
                'train_time_sec': round(elapsed, 1),
            })
            # write incrementally so a crash doesn't lose earlier results
            _write_csv(rows, out_csv)

    print(f"\nDone. Results in {out_csv}")
    return rows


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(csv_path='results_rounds_sweep.csv', out_png='rounds_sweep.png'):
    import csv as csv_mod
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {}
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            data.setdefault(row['architecture'], []).append(
                (int(row['rounds']), float(row['best_val_acc']))
            )

    plt.figure(figsize=(7, 5))
    for arch, points in data.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker='o', label=arch)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='chance (50%)')
    plt.axhline(0.6925, color='red', linestyle=':', linewidth=1, label="Shen et al. 4-round (69.25%, multi-pair)")
    plt.xlabel('Rounds')
    plt.ylabel('Validation accuracy')
    plt.title('Accuracy vs. rounds — ASCON-320 single-pair distinguisher')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Plot saved to {out_png}")


if __name__ == '__main__':
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='small run to sanity-check the pipeline')
    parser.add_argument('--tag', default=None, help='label for this run, e.g. "quick" or "full_v1" (default: timestamp)')
    args = parser.parse_args()

    tag = args.tag or ('quick' if args.quick else datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs('results', exist_ok=True)
    out_csv = f'results/results_rounds_sweep_{tag}.csv'
    out_png = f'results/rounds_sweep_{tag}.png'

    if args.quick:
        rows = run_sweep(n_train=5000, n_val=2000, epochs=3, out_csv=out_csv)
    else:
        rows = run_sweep(n_train=10**6, n_val=10**5, epochs=20, out_csv=out_csv)

    plot_results(csv_path=out_csv, out_png=out_png)
