"""
train_stage2_transformer.py

Drop-in replacement for the stage-2 aggregator: a small Transformer encoder
(self-attention among all k tokens) followed by mean-pooling, instead of
single-query attention pooling or GRU.

Same contract as the other stage-2 scripts: stage-1 (best4.h5) is reused
UNCHANGED and frozen.

Difference vs. train_stage2_attention.py's LearnedQueryPool: that script
does pooling-by-attention (1 query attends over k tokens, O(k)). This
script does full self-attention among all k tokens first (O(k^2) per
attention layer), which lets tokens exchange information with each other
directly before pooling -- strictly more expressive, at the cost of
quadratic memory/compute. This is why group_size=4096 defaults to a
smaller batch size below (4096^2 attention weights per sample is heavy).
If OOM at k=4096, drop --group_size to 128 only, or reduce ff_dim /
num_heads.

No positional encoding, same reasoning as the attention script: the k
pairs are exchangeable, order carries no information.
"""

import argparse
import csv
import os
import time
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Input, Reshape, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Dropout, Add,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

import ascon as ac

WDIR = "./saved_model/"


def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1, reg_param=10**-5):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    ff = Dense(ff_dim, activation="relu", kernel_regularizer=l2(reg_param))(x)
    ff = Dense(embed_dim, kernel_regularizer=l2(reg_param))(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    return x


def make_transformer_aggregator(group_size, embed_dim=32, num_heads=4, ff_dim=64,
                                 num_layers=2, reg_param=10**-5):
    inp = Input(shape=(group_size,))
    tokens = Reshape((group_size, 1))(inp)
    tokens = Dense(embed_dim, activation="relu", kernel_regularizer=l2(reg_param))(tokens)

    x = tokens
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim, num_heads, ff_dim, reg_param=reg_param)

    pooled = GlobalAveragePooling1D()(x)
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
    print(f"=== Transformer aggregator, group_size={group_size} ===")
    stage1 = load_model(WDIR + "best4.h5")

    print("Generating training groups...")
    X, Y = ac.make_td_diff(train_groups, group_size, num_rounds)
    X = stage1.predict(X, batch_size=10000, verbose=0).reshape(train_groups, group_size)

    print("Generating eval groups...")
    X_eval, Y_eval = ac.make_td_diff(eval_groups, group_size, num_rounds)
    X_eval = stage1.predict(X_eval, batch_size=10000, verbose=0).reshape(eval_groups, group_size)

    model_s = make_transformer_aggregator(group_size)
    model_s.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_s.summary()

    check = make_checkpoint(WDIR + f"best4_s={group_size}_transformer.h5")
    # full self-attention is O(k^2) -- much smaller batch at k=4096 or you'll OOM
    bs = batch_size if group_size <= 128 else min(batch_size, 64)

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
        model_name=f"transformer_pool_k={group_size}",
        round_num=num_rounds,
        params=model_s.count_params(),
        train_samples=train_groups,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        train_time_min=train_time_min,
        notes="no-sort, 2-layer self-attn + mean pool, stage1=best4.h5 frozen",
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
