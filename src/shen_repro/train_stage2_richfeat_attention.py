"""
train_stage2_richfeat_attention.py

Novel variant, not from Shen et al. Every other stage-2 script (baseline
sort+MLP, attention, GRU, transformer) feeds the aggregator ONE scalar
number per pair -- stage-1's final sigmoid score. That's a massive
bottleneck: stage-1 collapses a 320-dim input down to 1 float before the
aggregator ever sees it, so almost all information stage-1 computed
internally is thrown away before stage-2 gets a chance to use it.

This script instead chops stage-1 just before its final Dense(1, sigmoid)
layer and feeds the aggregator stage-1's PENULTIMATE-layer activations for
each pair (a vector, not a scalar). Stage-1 is still frozen and unchanged
-- this isn't retraining stage-1, just reading out an earlier layer of it.
The aggregator (same LearnedQueryPool attention pool as
train_stage2_attention.py) then has k x d-dimensional tokens to work with
instead of k x 1, where d is whatever width stage-1's second-to-last layer
happens to be.

IMPORTANT -- verify the layer index before trusting this:
Run stage1.summary() once and confirm layers[-2] is the Dense layer
feeding into the final sigmoid, not a Dropout/BatchNorm/etc. If stage-1
has a Dropout or BatchNorm right before the output, change
FEATURE_LAYER_INDEX below to point at the actual last Dense layer (e.g.
-3). Getting this wrong doesn't crash anything, it just means you're
feeding a less-meaningful vector to the aggregator.

If this beats the scalar-score versions, that's a legitimate, reportable
finding: it shows compressing to a single per-pair score is losing signal
that a richer per-pair representation can recover, independent of which
aggregator architecture is used on top.
"""

import argparse
import csv
import os
import time
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, Input, MultiHeadAttention, LayerNormalization, Layer,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf

import ascon as ac

WDIR = "./saved_model/"
FEATURE_LAYER_INDEX = -2  # check with stage1.summary() before trusting this


class LearnedQueryPool(Layer):
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = LayerNormalization()

    def build(self, input_shape):
        self.query = self.add_weight(
            shape=(1, 1, self.embed_dim), initializer="glorot_uniform",
            trainable=True, name="pool_query",
        )
        super().build(input_shape)

    def call(self, tokens):
        batch_size = tf.shape(tokens)[0]
        query = tf.tile(self.query, [batch_size, 1, 1])
        pooled = self.mha(query=query, key=tokens, value=tokens)
        pooled = self.norm(pooled)
        return tf.squeeze(pooled, axis=1)


def make_richfeat_aggregator(group_size, feature_dim, embed_dim=32, num_heads=4, reg_param=10**-5):
    # input is already (batch, group_size, feature_dim) -- no Reshape needed,
    # stage-1's penultimate layer is already a real vector per pair
    inp = Input(shape=(group_size, feature_dim))
    tokens = Dense(embed_dim, activation="relu", kernel_regularizer=l2(reg_param))(inp)
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
    print(f"=== Rich-feature attention aggregator, group_size={group_size} ===")
    stage1 = load_model(WDIR + "best4.h5")
    stage1.summary()
    print(f"Using layer index {FEATURE_LAYER_INDEX}: {stage1.layers[FEATURE_LAYER_INDEX].name} "
          f"-- CONFIRM this is a Dense layer above before trusting results.")

    feature_extractor = Model(inputs=stage1.input, outputs=stage1.layers[FEATURE_LAYER_INDEX].output)
    feature_dim = feature_extractor.output_shape[-1]
    print(f"Per-pair feature dim: {feature_dim}")

    print("Generating training groups...")
    X, Y = ac.make_td_diff(train_groups, group_size, num_rounds)
    X = feature_extractor.predict(X, batch_size=10000, verbose=0).reshape(train_groups, group_size, feature_dim)

    print("Generating eval groups...")
    X_eval, Y_eval = ac.make_td_diff(eval_groups, group_size, num_rounds)
    X_eval = feature_extractor.predict(X_eval, batch_size=10000, verbose=0).reshape(eval_groups, group_size, feature_dim)

    model_s = make_richfeat_aggregator(group_size, feature_dim)
    model_s.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_s.summary()

    check = make_checkpoint(WDIR + f"best4_s={group_size}_richfeat.h5")
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
        model_name=f"richfeat_attention_k={group_size}",
        round_num=num_rounds,
        params=model_s.count_params(),
        train_samples=train_groups,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        train_time_min=train_time_min,
        notes=f"no-sort, attention pool over stage1 layer[{FEATURE_LAYER_INDEX}] "
              f"({feature_dim}-dim per pair), stage1=best4.h5 frozen",
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
