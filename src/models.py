"""
models.py

All architectures for the ASCON neural distinguisher comparison live here,
so train_distinguisher.py can switch between them with a single flag and
every model trains on the exact same data pipeline.

Each model takes the same input: a (batch, 64, 10) float32 tensor
(64 bit-positions x 10 word-channels, unpacked from data_generator.py's
80-byte packed format) and outputs a single logit per sample (real-diff
vs random).

Currently implemented:
    - MLPBaseline   : the "sees the bits, understands no structure" floor.
                      No convolution, no positional assumptions -- everything
                      the CNN later gets credit for has to come from actually
                      using the (bit-position, word) structure, not just from
                      having more parameters than this model.

Parameter budget note: MLPBaseline's hidden width is chosen so its total
parameter count is in the same ballpark as the CNN models you'll add next
(Gohr-style, ResNet-style). Check param counts with count_params() below
and adjust `hidden_dim` if they drift more than ~5% apart once the CNNs
are built -- fair comparison depends on this.
"""

import torch
import torch.nn as nn
from gohr_cnn import GohrCNN


def count_params(model: nn.Module) -> int:
    """Total trainable parameter count -- use this to keep the zoo matched."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPBaseline(nn.Module):
    """
    Plain fully-connected network. Flattens the (64, 10) input into a single
    640-length vector and never looks at it as spatial/structured data again.

    This is intentionally NOT a single-layer model (that would be too weak
    to be an interesting floor) and NOT deeper than ~4 hidden layers (that
    risks overfitting a fairly simple binary task and makes param-matching
    against the CNNs harder to reason about). Three hidden layers with
    BatchNorm + Dropout is the standard "serious but structure-blind"
    baseline used in the neural-distinguisher literature (e.g. Gohr 2019
    reports a comparable MLP baseline alongside the CNN).

    Args:
        input_dim: flattened input size (64 * 10 = 640 by default)
        hidden_dim: width of each hidden layer -- tune this to match the
                    CNN zoo's parameter count once those are built
        num_hidden_layers: number of hidden layers (3 is the default and
                    recommended starting point)
        dropout: dropout probability between hidden layers
    """

    def __init__(self, input_dim: int = 640, hidden_dim: int = 169,
                 num_hidden_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = [nn.Flatten()]

        in_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # output logit

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x arrives as (B, 10, 64) from collate_batch's channels-first
        # permute (done so the CNNs can consume it directly). The MLP
        # doesn't care about channel order since it flattens everything
        # anyway, so no extra permute needed here.
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------
# Registry: lets train_distinguisher.py select a model by name from the
# command line instead of editing code. Add new architectures here as
# you build them (Gohr-style CNN, ResNet-style, multi-pair CNN, ...).
# ---------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """One residual block: two 1D convolutions with a skip connection
    around them. Used by ResNetCNN below."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ResNetCNN(nn.Module):
    """
    Gohr-style 1D-conv network with residual (skip) connections, reading
    along the 64 bit-position axis with the 10 words as input channels.
    This is the CNN half of the architecture zoo -- MLPBaseline above is
    its structure-blind counterpart, meant to be parameter-matched against
    this model once both are finalized.
    """

    def __init__(self, in_channels: int = 10, width: int = 32,
                 num_blocks: int = 5, seq_len: int = 64, dense_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * seq_len, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


MODEL_REGISTRY = {
    "mlp": MLPBaseline,
    "resnet_cnn": ResNetCNN,
    "gohr_cnn": GohrCNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Quick sanity check + parameter-matching report: build every model in
    # the registry, confirm the output shape, and print param counts side
    # by side so you can see at a glance whether the zoo is still fair.
    dummy = torch.randn(8, 10, 64)  # (batch, channels, bit-positions)

    print(f"{'model':<12} {'output shape':<16} {'params':>12}")
    print("-" * 42)
    counts = {}
    for name in MODEL_REGISTRY:
        model = build_model(name)
        out = model(dummy)
        assert tuple(out.shape) == (8,), f"{name} produced unexpected shape {out.shape}"
        counts[name] = count_params(model)
        print(f"{name:<12} {str(tuple(out.shape)):<16} {counts[name]:>12,}")

    if len(counts) > 1:
        lo, hi = min(counts.values()), max(counts.values())
        drift = (hi - lo) / lo * 100
        print(f"\nParam count spread: {drift:.1f}% "
              f"({'OK, within ~5% target' if drift <= 5 else 'too wide -- adjust hidden_dim/width to match'})")
