"""Deep learning for the workbench: CPU-only PyTorch models for tabular data.

The models here are ordinary scikit-learn estimators, so they plug into the
existing preprocessor pipeline and inherit cross-validation, tuning, permutation
importance, the HTML report, the saved bundle and the exported script.

Importing this package does **not** import torch.  Torch is an optional
dependency, and the rest of the workbench has to keep working without it — the
model registry imports these classes lazily, and ``is_available`` only offers the
neural networks when torch can actually be imported.
"""
from __future__ import annotations

from .runtime import (
    cpu_count,
    default_threads,
    describe_runtime,
    observing,
    torch_available,
    torch_problem,
)

__all__ = [
    "cpu_count",
    "default_threads",
    "describe_runtime",
    "observing",
    "torch_available",
    "torch_problem",
    "TorchMLPRegressor",
    "TorchMLPClassifier",
    "TorchAutoencoder",
    "describe_trained",
]


def __getattr__(name):
    """Lazy re-export so ``import core.dl`` never pulls torch in by itself."""
    if name in ("TorchMLPRegressor", "TorchMLPClassifier", "TorchAutoencoder", "describe_trained"):
        from . import estimators

        return getattr(estimators, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
