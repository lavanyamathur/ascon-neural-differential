"""
rnn.py

Bidirectional LSTM differential distinguisher for round-reduced ASCON.
Consumes the same (batch, 10, 64) tensor as every other model in
src/models/ (10 word-channels x 64 bit-positions), so it drops into
the existing training loop and MODEL_REGISTRY unchanged.

Structural bias this adds: recurrence over the bit-position axis, the
third inductive bias in the comparison alongside convolution
(cnn_gohr.py / resnet.py) and attention (transformer.py). mlp.py has
none of the three -- it's the structure-blind floor all of these are
measured against.
"""

import torch
import torch.nn as nn


class RNNDistinguisher(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_dim: int = 32,
                 num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):
        # x arrives as (B, 10, 64) channels-first; flip to (B, 64, 10)
        # so bit-position is the time axis and word-channel is the
        # per-timestep feature vector.
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)  # mean-pool over the bit-position/time axis
        return self.head(pooled).squeeze(-1)


if __name__ == "__main__":
    dummy = torch.randn(8, 10, 64)
    model = RNNDistinguisher()
    out = model(dummy)
    assert tuple(out.shape) == (8,), f"unexpected output shape {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RNNDistinguisher: output shape {tuple(out.shape)}, "
          f"{n_params:,} trainable params")
