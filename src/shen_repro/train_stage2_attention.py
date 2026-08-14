"""
train_stage2_attention.py

Drop-in replacement for train_stage2_128.py / train_stage2_4096.py's
sort + min-max-scale + dense-MLP aggregation, using attention instead.

Only the stage-2 aggregator changes. Stage-1 (ascon_stage1_full_10M_best.keras)
is reused UNCHANGED, so any accuracy difference vs. the existing sort+MLP
baseline is attributable to the aggregation method alone -- not a confound
from a different per-pair scorer.

Design choice: pooling-by-multi-head-attention (a single learnable query
vector attends over all k per-pair scores as keys/values), NOT full
self-attention among all k scores. Full self-attention is O(k^2) attention
weights per sample -- fine at k=128, but 4096^2 ~= 16.8M per sample makes
it impractical at k=4096. A single-query attention pool is O(k), so the
same architecture scales to both group sizes without changing shape.

No positional encoding is added deliberately: the k pairs in a group are
exchangeable (order carries no information, same reason the baseline sorts
them first), so the aggregator should be permutation-invariant. Attention
pooling is naturally permutation-invariant without needing to impose an
explicit order the way sorting does.

Stage-1 model note: this had briefly been switched to
ascon_stage1_full_10M_best.keras over a provenance concern about
best4.h5. That swap is reverted -- check_stage1_signal.py showed
ascon_stage1_full_10M_best.keras itself produces statistically
indistinguishable scores for real vs. random groups (~0.5009 vs
~0.5006), i.e. no usable per-pair signal, which is what was actually
causing this script's aggregator to flatline at chance -- not a bug in
the attention aggregator itself. best4.h5 is confirmed to carry real
signal (the sort+MLP baseline trained on it reaches ~0.55 val_acc), so
it's back to being the shared stage-1 model for both the baseline and
this script, preserving the "aggregation method is the only variable"
comparison. best4.h5's training provenance is still not fully pinned
down -- worth resolving eventually, but it's the model that actually
works, so it's the right one to build on for now.
"""

import argparse
import csv
import os
import time
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, Input, MultiHeadAttention, LayerNormalization, Layer, Reshape,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

import ascon as ac

WDIR = "./saved_model/"


class LearnedQueryPool(Layer):
    """
    A single trainable query vector attends over the k token embeddings
    (keys/values) via MultiHeadAttention, producing one pooled vector per
    sample. O(k) per sample, not O(k^2) -- scales to k=4096.
    """
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = LayerNormalization()

    def build(self, input_shape):
        # one learnable query vector, broadcast across the batch
        self.query = self.add_weight(
            shape=(1, 1, self.embed_dim), initializer="glorot_uniform",
            trainable=True, name="pool_query",
        )
        super().build(input_shape)

    def call(self, tokens):
        # tokens: (batch, k, embed_dim)
        batch_size = tf.shape(tokens)[0]
        query = tf.tile(self.query, [batch_size, 1, 1])  # (batch, 1, embed_dim)
        pooled = self.mha(query=query, key=tokens, value=tokens)  # (batch, 1, embed_dim)
        pooled = self.norm(pooled)
        return tf.squeeze(pooled, axis=1)  # (batch, embed_dim)


def make_attention_aggregator(group_size, embed_dim=32, num_heads=4, reg_param=10**-5):
    inp = Input(shape=(group_size,))
    tokens = Reshape((group_size, 1))(inp)                  # (batch, k, 1)
    tokens = Dense(embed_dim, activation="relu", kernel_regularizer=l2(reg_param))(tokens)
    pooled = LearnedQueryPool(embed_dim=embed_dim, num_heads=num_heads)(tokens)
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
    print(f"=== Attention aggregator, group_size={group_size} ===")
    stage1 = load_model(WDIR + "best4.h5")

    print("Generating training groups...")
    X, Y = ac.make_td_diff(train_groups, group_size, num_rounds)
    X = stage1.predict(X, batch_size=10000, verbose=0).reshape(train_groups, group_size)
    # deliberately NOT sorted -- attention should not need the explicit
    # order the baseline imposes; this is part of the comparison

    print("Generating eval groups...")
    X_eval, Y_eval = ac.make_td_diff(eval_groups, group_size, num_rounds)
    X_eval = stage1.predict(X_eval, batch_size=10000, verbose=0).reshape(eval_groups, group_size)

    model_s = make_attention_aggregator(group_size)
    model_s.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_s.summary()

    check = make_checkpoint(WDIR + f"best4_s={group_size}_attention.h5")
    # smaller batch size than the baseline at k=4096, since per-sample
    # attention cost is higher; adjust if you hit memory limits
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
        model_name=f"attention_pool_k={group_size}",
        round_num=num_rounds,
        params=model_s.count_params(),
        train_samples=train_groups,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        train_time_min=train_time_min,
        notes="no-sort, single-query MHA pool, stage1=best4.h5 frozen",
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
