"""
gohr_cnn.py

Gohr-style plain CNN distinguisher, following the architecture from
Gohr 2019 ("Improving Attacks on Round-Reduced Speck32/64 using Deep
Learning"). This is the middle model in the architecture zoo: more
structure-aware than MLPBaseline, but without the residual/skip
connections used in ResNetCNN (models.py).

Input: (batch, 10, 64) float32 -- same channels-first format ResNetCNN
and MLPBaseline consume (10 word-channels x 64 bit-positions), produced
by collate_batch's permute. No reshaping needed here.

Output: a single raw logit per sample (matches MLPBaseline/ResNetCNN --
no sigmoid applied, so this is compatible with BCEWithLogitsLoss).

Default widths (base_width=18, fc_width=128, num_conv_layers=3) were
chosen to land within ~5% of MLPBaseline (166,973 params) and ResNetCNN
(167,713 params) -- see count_params() output below. Re-check with
count_params() if you change num_conv_layers or seq_len.
"""

import torch
import torch.nn as nn


class GohrCNN(nn.Module):
    """
    Plain (non-residual) Gohr-style CNN.

    Structure:
      1. A kernel-size-1 "word mixing" conv that expands the 10 input
         channels to `base_width` channels at each of the 64 bit
         positions (lets the network combine bits from different words
         at the same position before looking at neighboring positions).
      2. `num_conv_layers` stacked kernel-size-3 convs (padding=1, so
         length stays 64), each followed by BatchNorm + ReLU. No skip
         connections -- that's the key difference from ResNetCNN.
      3. Flatten and pass through two FC hidden layers with BatchNorm +
         ReLU, then a single linear output (raw logit).
    """

    def __init__(
        self,
        in_channels: int = 10,
        seq_len: int = 64,
        base_width: int = 18,
        num_conv_layers: int = 3,
        fc_width: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_conv = nn.Conv1d(in_channels, base_width, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(base_width)

        conv_blocks = []
        for _ in range(num_conv_layers):
            conv_blocks.append(
                nn.Conv1d(base_width, base_width, kernel_size=3, padding=1)
            )
            conv_blocks.append(nn.BatchNorm1d(base_width))
            conv_blocks.append(nn.ReLU(inplace=True))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        flat_dim = base_width * seq_len
        self.fc1 = nn.Linear(flat_dim, fc_width)
        self.fc1_bn = nn.BatchNorm1d(fc_width)
        self.fc2 = nn.Linear(fc_width, fc_width)
        self.fc2_bn = nn.BatchNorm1d(fc_width)
        self.fc_out = nn.Linear(fc_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 10, 64) -- already channels-first, no permute needed
        x = self.relu(self.input_bn(self.input_conv(x)))
        x = self.conv_blocks(x)
        x = self.dropout(x)

        x = x.flatten(1)

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)

        return self.fc_out(x).squeeze(-1)  # raw logit, matches MLPBaseline/ResNetCNN


if __name__ == "__main__":
    # Smoke test + parameter count check, mirroring models.py's own
    # __main__ block so the numbers are directly comparable.
    from models import count_params, build_model, MODEL_REGISTRY

    dummy = torch.randn(8, 10, 64)

    model = GohrCNN()
    out = model(dummy)
    assert tuple(out.shape) == (8,), f"unexpected shape {out.shape}"
    gohr_params = count_params(model)

    print(f"{'model':<12} {'output shape':<16} {'params':>12}")
    print("-" * 42)
    print(f"{'gohr_cnn':<12} {str(tuple(out.shape)):<16} {gohr_params:>12,}")

    for name in MODEL_REGISTRY:
        m = build_model(name)
        o = m(dummy)
        print(f"{name:<12} {str(tuple(o.shape)):<16} {count_params(m):>12,}")
