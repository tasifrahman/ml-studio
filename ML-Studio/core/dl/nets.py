"""The actual ``torch.nn`` modules.

Kept deliberately small.  Every model here is a plain multilayer perceptron with
one extra idea: **categorical columns get learned embeddings instead of one-hot
columns**.  A 30-category column costs 30 input features as one-hot, but an 8-wide
embedding vector as an embedding — and the network gets to learn that "granite" and
"basalt" are near each other rather than being told they are orthogonal.

The embedding input is recovered from the one-hot block the workbench preprocessor
already produced: columns are named ``column=category``, so a group of them is one
categorical variable.  That keeps this file independent of the preprocessor — it
just needs names it can read, which is also what makes the pipeline pickleable and
the column names meaningful in every importance chart.
"""
from __future__ import annotations

import math

import torch
from torch import nn

ACTIVATIONS = ("relu", "gelu", "tanh", "leaky_relu", "silu")


def activation_layer(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
    }.get(str(name).lower(), nn.ReLU)()


def embedding_dim_for(cardinality: int, cap: int = 16) -> int:
    """How wide an embedding a categorical variable of this size deserves.

    The rule of thumb from the fast.ai tabular work: ``min(cap, round(1.6 * card**0.56))``.
    It grows sub-linearly, so a 100-category column does not get a 100-wide vector.
    """
    card = max(2, int(cardinality))
    return int(max(2, min(int(cap or 16), round(1.6 * card ** 0.56))))


def parse_hidden(text, fallback=(64, 32)) -> tuple[int, ...]:
    """``"128, 64, 32"`` -> ``(128, 64, 32)``.  Junk is ignored, not fatal."""
    if isinstance(text, (list, tuple)):
        tokens = list(text)
    else:
        tokens = str(text or "").replace(";", ",").replace("x", ",").split(",")
    sizes = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        try:
            size = int(float(token))
        except (TypeError, ValueError):
            continue
        if size >= 1:
            sizes.append(min(size, 4096))
    return tuple(sizes) or tuple(fallback)


class TabularMLP(nn.Module):
    """MLP over a numeric block plus one embedding per categorical variable."""

    def __init__(
        self,
        n_numeric: int,
        cat_sizes=(),
        hidden=(64, 32),
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = False,
        embed_dim: int = 0,
        max_embed_dim: int = 16,
        out_dim: int = 1,
    ):
        super().__init__()
        self.cat_sizes = [int(c) for c in cat_sizes]
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.activation = activation
        self.batch_norm = bool(batch_norm)
        self.n_numeric = int(n_numeric)
        self.out_dim = int(out_dim)

        self.embeddings = nn.ModuleList()
        embed_total = 0
        for card in self.cat_sizes:
            width = int(embed_dim) if int(embed_dim or 0) > 0 else embedding_dim_for(card, max_embed_dim)
            self.embeddings.append(nn.Embedding(card, width))
            embed_total += width
            # Small init: embeddings start near zero so the numeric block is not
            # swamped before the first few updates.
            nn.init.normal_(self.embeddings[-1].weight, mean=0.0, std=0.05)

        dims = [self.n_numeric + embed_total, *self.hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(activation_layer(activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dims[-1], self.out_dim)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x_num]
        if len(self.embeddings) and x_cat is not None:
            for i, table in enumerate(self.embeddings):
                parts.append(table(x_cat[:, i]))
        hidden = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(self.trunk(hidden))


class AutoencoderNet(nn.Module):
    """Symmetric encoder/decoder used for unsupervised work.

    Compresses the prepared matrix into a small latent code and reconstructs it.
    The latent code is a *learned* non-linear compression, which is the whole point
    next to PCA: it can follow a curved manifold, and the reconstruction error is
    itself a usable anomaly score.
    """

    def __init__(
        self,
        n_features: int,
        hidden=(64, 32),
        latent: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = tuple(hidden)
        self.latent = int(latent)
        self.dropout = float(dropout)
        self.activation = activation
        self.batch_norm = bool(batch_norm)

        def block(sizes, final_activation: bool):
            layers: list[nn.Module] = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if batch_norm and i < len(sizes) - 2:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                if i < len(sizes) - 2 or final_activation:
                    layers.append(activation_layer(activation))
                if dropout and i < len(sizes) - 2:
                    layers.append(nn.Dropout(float(dropout)))
            return nn.Sequential(*layers) if layers else nn.Identity()

        enc_sizes = [self.n_features, *self.hidden, self.latent]
        self.encoder = block(enc_sizes, final_activation=False)
        dec_sizes = [self.latent, *reversed(self.hidden), self.n_features]
        self.decoder = block(dec_sizes, final_activation=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        # x_cat is accepted and ignored so the shared training loop can call every
        # network with the same (numeric, categorical) signature.
        return self.decoder(self.encoder(x))


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def architecture_note(n_numeric: int, cat_sizes, hidden, out_dim: int, params: int = 0) -> str:
    """A short human-readable description, shown in the GUI log and the report."""
    bits = [f"{n_numeric} numeric inputs"]
    if cat_sizes:
        bits.append(f"{len(cat_sizes)} categorical -> embeddings "
                    f"({', '.join(str(c) for c in cat_sizes)})")
    trunk = " -> ".join(str(h) for h in hidden) or "(no hidden layer)"
    tail = f" · {params:,} weights" if params else ""
    return f"{' + '.join(bits)} | {trunk} -> {out_dim}{tail}"
