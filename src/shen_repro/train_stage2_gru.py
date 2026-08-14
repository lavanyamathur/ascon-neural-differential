"""
train_stage2_gru.py

Drop-in replacement for the stage-2 aggregator, using a Bidirectional GRU
instead of attention pooling or sort+MLP.

Same contract as train_stage2_attention.py: stage-1 (best4.h5) is reused
UNCHANGED and frozen, so any accuracy delta vs. the baseline / attention
version is attributable purely to the aggregator architecture.

Design choice: GRU over the k per-pair scores, NOT sorted first (same
reasoning as the attention script -- the aggregator should learn its own
useful ordering/weighting rather than have one imposed). Bidirectional so
the aggregator sees each score with both left and right context, since the
sequence order itself is arbitrary and there's no reason to privilege
forward-only context.

This is meant as a cheap sanity-check architecture as much as a real
contender: if GRU beats chance while attention doesn't (or vice versa),
that's informative about whether the problem is the aggregation mechanism
itself or something upstream (stage-1 signal, data pipeline).
"""

import argparse
import csv
import os
import time
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Input, Reshape, Bidirectional, GRU,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

import ascon as ac

WDIR = "./saved_model/"


def make_gru_aggregator(group_size, gru_units=32, embed_dim=32, reg_param=10**-5):
    inp = Input(shape=(group_size,))
    tokens = Reshape((group_size, 1))(inp)                  # (batch, k, 1)
    tokens = Dense(embed_dim, activation="relu", kernel_regularizer=l2(reg_param))(tokens)
    pooled = Bidirectional(GRU(gru_units))(tokens)           # (batch, 2*gru_units)
    out = Dense(1, activation="sigmoid", kernel_regularizer=l2(reg_param))(pooled)
    return Model(inputs=inp, outputs=out)


def log_run_to_csv(csv_path, model_name, round_num, params, train_samples,
                    val_acc, val_loss, train_time_min, notes=""):
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "round", "params", "train_samples",
                              "val_acc", "val_loss", "train_time_min", "notes"])
        writer.writerow([model_name, round_num, params, train_samples,
                          f"{val_acc:.4f}", f"{val_loss:.4f}",
                          f"{train_time_min:.2f}", notes])


def make_checkpoint(datei):
    return ModelCheckpoint(datei, monitor="val_loss", save_best_only=True)


def train(group_size, num_rounds, train_groups, eval_groups, num_epochs=50, batch_size=10000):
    print(f"=== GRU aggregator, group_size={group_size} ===")
    stage1 = load_model(WDIR + "best4.h5")

    print("Generating training groups...")
    X, Y = ac.make_td_diff(train_groups, group_size, num_rounds)
    X = stage1.predict(X, batch_size=10000, verbose=0).reshape(train_groups, group_size)

    print("Generating eval groups...")
    X_eval, Y_eval = ac.make_td_diff(eval_groups, group_size, num_rounds)
    X_eval = stage1.predict(X_eval, batch_size=10000, verbose=0).reshape(eval_groups, group_size)

    model_s = make_gru_aggregator(group_size)
    model_s.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_s.summary()

    check = make_checkpoint(WDIR + f"best4_s={group_size}_gru.h5")
    bs = batch_size if group_size <= 128 else min(batch_size, 256)

    start_time = time.time()
    h = model_s.fit(
        X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
        validation_data=(X_eval, Y_eval), callbacks=[check],
    )
    train_time_min = (time.time() - start_time) / 60

    best_epoch = int(np.argmax(h.history["val_accuracy"]))
    best_val_acc = h.history["val_accuracy"][best_epoch]
    best_val_loss = h.history["val_loss"][best_epoch]

    log_run_to_csv(
        csv_path="./results/metrics/r4_architecture_comparison.csv",
        model_name=f"bigru_pool_k={group_size}",
        round_num=num_rounds,
        params=model_s.count_params(),
        train_samples=train_groups,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        train_time_min=train_time_min,
        notes="no-sort, BiGRU aggregator, stage1=best4.h5 frozen",
    )

    print("Best validation accuracy:", best_val_acc)
    return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, choices=[128, 4096], required=True)
    args = parser.parse_args()

    if args.group_size == 128:
        train(group_size=128, num_rounds=4, train_groups=78125, eval_groups=7812)
    else:
        train(group_size=4096, num_rounds=4, train_groups=2000, eval_groups=500)
