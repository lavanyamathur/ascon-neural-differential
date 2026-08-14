"""
transformer.py

Transformer-encoder differential distinguisher for round-reduced ASCON.
Consumes the same (batch, 10, 64) tensor as every other model in
src/models/ (10 word-channels x 64 bit-positions, produced by
data_generator.py + train.py's collate step) so it drops into the
existing training loop and MODEL_REGISTRY with no changes elsewhere.

Structural bias this adds to the comparison: self-attention lets any
two bit-positions interact within a single layer, unlike cnn_gohr.py /
resnet.py whose receptive field only grows with depth. mlp.py has no
positional bias at all; this and rnn.py are the two new inductive
biases being tested against it.
"""

import torch
import torch.nn as nn


class TransformerDistinguisher(nn.Module):
    def __init__(self, in_channels: int = 10, d_model: int = 24,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 48, seq_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x arrives as (B, 10, 64) channels-first; flip to (B, 64, 10)
        # so bit-position is the sequence axis and word-channel is the
        # per-token feature vector.
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)  # pool over bit-position axis
        return self.head(x).squeeze(-1)


if __name__ == "__main__":
    # quick shape/param sanity check, same pattern as the models.py
    # __main__ block used elsewhere in this project
    dummy = torch.randn(8, 10, 64)
    model = TransformerDistinguisher()
    out = model(dummy)
    assert tuple(out.shape) == (8,), f"unexpected output shape {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerDistinguisher: output shape {tuple(out.shape)}, "
          f"{n_params:,} trainable params")
