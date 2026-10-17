"""scikit-learn-compatible wrappers around the torch networks.

Everything in this workbench — cross-validation, tuning, feature-subset search,
the report, the saved bundle, the exported script — talks to models through the
scikit-learn estimator API.  So rather than build a parallel deep-learning path
with its own GUI, these classes implement ``fit`` / ``predict`` / ``predict_proba``
and slot into the existing :class:`~core.preprocess.WorkbenchPreprocessor` pipeline.
The upshot is that a neural network here gets cross-validation, permutation
importance, tuning, a report and a reusable bundle for free.

Three deliberate choices, all aimed at a 4-core CPU with no GPU:

* **The model is stored as a ``state_dict``, not a live module.**  Bundles stay
  small and load without depending on torch's module pickling.
* **Training is bounded three ways**: early stopping on a validation split, a
  ``max_seconds`` wall-clock budget, and a small default architecture.  A fit that
  goes wrong finishes slightly worse instead of hanging the window.
* **Categoricals become embeddings.**  The preprocessor's one-hot block is named
  ``column=category``; a group of those is recognised and replaced by a learned
  embedding table.  High-cardinality columns stop costing a feature each, and the
  network can learn that two categories are similar.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from . import runtime
from .nets import AutoencoderNet, TabularMLP, architecture_note, parameter_count, parse_hidden

DEFAULT_HIDDEN = "64, 32"


def _as_frame(X, names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    array = np.asarray(X)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    columns = list(names) if names is not None and len(names) == array.shape[1] else \
        [f"x{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.astype("float64").fillna(0.0).to_numpy(dtype="float32", copy=True)


class _TorchBase(BaseEstimator):
    """Shared plumbing: parameter handling, categorical discovery, training loop."""

    _task = "regression"

    def __init__(
        self,
        hidden_layers: str = DEFAULT_HIDDEN,
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        use_embeddings: bool = True,
        embedding_dim: int = 0,
        epochs: int = 120,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        patience: int = 12,
        val_fraction: float = 0.15,
        threads: int = 4,
        max_seconds: int = 180,
        random_state: int = 42,
    ):
        # sklearn convention: store every parameter untouched, validate in fit().
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.patience = patience
        self.val_fraction = val_fraction
        self.threads = threads
        self.max_seconds = max_seconds
        self.random_state = random_state

    # ------------------------------------------------------------------ schema
    def _discover_schema(self, frame: pd.DataFrame) -> None:
        """Find one-hot groups (``col=value``) and turn them into embedding slots."""
        columns = [str(c) for c in frame.columns]
        values = _numeric_matrix(frame)
        groups: dict[str, list[int]] = {}
        if self.use_embeddings:
            for index, name in enumerate(columns):
                if "=" in name:
                    groups.setdefault(name.rsplit("=", 1)[0], []).append(index)

        positions: list[list[int]] = []
        for _, idx in groups.items():
            # Only treat it as a categorical if the block really is one-hot: a
            # numeric column whose *name* happens to contain "=" must stay numeric.
            if len(idx) >= 2 and np.isin(values[:, idx], (0.0, 1.0)).all():
                positions.append(sorted(idx))

        claimed = {i for idx in positions for i in idx}
        self.cat_positions_ = positions
        self.cat_names_ = [columns[idx[0]].rsplit("=", 1)[0] for idx in positions]
        self.numeric_positions_ = [i for i in range(len(columns)) if i not in claimed]
        self.feature_names_in_ = columns
        self.cat_sizes_ = [len(idx) + 1 for idx in positions]   # +1 = the "unseen" row
        self.n_features_in_ = len(columns)

    def _split(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepared matrix -> (numeric block, categorical index block)."""
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        if len(getattr(self, "feature_names_in_", [])) and list(frame.columns) != list(self.feature_names_in_):
            raise ValueError(
                "these columns are not the ones the model was trained on: "
                f"expected {len(self.feature_names_in_)} columns, got {len(frame.columns)}"
            )
        values = _numeric_matrix(frame)
        positions = getattr(self, "cat_positions_", [])
        if not positions:
            return values, None
        cat = np.zeros((len(frame), len(positions)), dtype="int64")
        for j, idx in enumerate(positions):
            block = values[:, idx]
            rows = block.argmax(axis=1)
            rows[block.sum(axis=1) <= 0.0] = len(idx)   # all-zero row = unseen category
            cat[:, j] = rows
        if self.numeric_positions_:
            numeric = values[:, self.numeric_positions_]
        else:
            numeric = np.zeros((len(frame), 0), dtype="float32")
        return numeric, cat

    # ------------------------------------------------------------------- loop
    def _build_optimizer(self, module):
        import torch

        name = str(self.optimizer).lower()
        lr = float(self.learning_rate)
        wd = float(self.weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(module.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(module.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(module.parameters(), lr=lr, weight_decay=wd)

    def _train_module(self, module, loss_fn, train, valid, train_weights=None,
                      valid_weights=None):
        """Mini-batch loop with early stopping, a time budget and live progress.

        ``train``/``valid`` are tuples ``(x_numeric, x_categorical, y)`` of tensors.
        Returns the loss history and a note explaining why training stopped.
        """
        import torch

        runtime.configure_threads(self.threads)
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        x_num, x_cat, y = train
        n = len(y)
        epochs = max(1, int(self.epochs or 1))
        batch = max(1, min(int(self.batch_size or 64), n))
        patience = max(0, int(self.patience or 0))
        # BatchNorm cannot normalise a batch of one, so drop a trailing singleton.
        drop_last = bool(self.batch_norm) and n > batch
        optimizer = self._build_optimizer(module)
        rng = np.random.RandomState(int(self.random_state))

        history = {"epoch": [], "train_loss": [], "val_loss": []}
        best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
        started = time.time()
        stopped = f"ran all {epochs} epochs"

        for epoch in range(1, epochs + 1):
            runtime.check_cancel()
            module.train()
            order = rng.permutation(n)
            total, seen = 0.0, 0
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                if drop_last and len(idx) < 2:
                    continue
                optimizer.zero_grad()
                out = module(x_num[idx], x_cat[idx] if x_cat is not None else None)
                loss = loss_fn(out, y[idx])
                # Flatten first.  Regression losses come back as (batch, 1) while the
                # weights are (batch,), and multiplying those two shapes broadcasts
                # into a (batch, batch) outer product — which then cancels the
                # weights back out of the mean, silently ignoring them.
                loss = loss.reshape(-1)
                if train_weights is not None:
                    w = train_weights[idx].reshape(-1)
                    loss = (loss * w).sum() / w.sum().clamp(min=1e-12)
                else:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(idx)
                seen += len(idx)
            train_loss = total / max(1, seen)

            val_loss = None
            if valid is not None:
                module.eval()
                with torch.no_grad():
                    vx, vc, vy = valid
                    out = module(vx, vc if vc is not None else None)
                    vloss = loss_fn(out, vy).reshape(-1)
                    # The validation loss has to use the same weighting as the
                    # training loss.  Otherwise the epoch we restore at the end is
                    # the one that fits the *down-weighted* rows best, which is the
                    # opposite of what the weights asked for.
                    if valid_weights is not None:
                        vw = valid_weights.reshape(-1)
                        val_loss = float((vloss * vw).sum() / vw.sum().clamp(min=1e-12))
                    else:
                        val_loss = float(vloss.mean())

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            monitored = train_loss if val_loss is None else val_loss

            if monitored < best_loss - 1e-7:
                best_loss, best_epoch, stale = monitored, epoch, 0
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            else:
                stale += 1

            runtime.report(
                epoch, epochs,
                f"epoch {epoch}/{epochs} · loss {train_loss:.4f}"
                + (f" · val {val_loss:.4f}" if val_loss is not None else "")
                + (f" · best {best_loss:.4f}" if best_state is not None else ""),
            )

            if patience and stale >= patience:
                stopped = f"early stopping at epoch {epoch} (no improvement for {patience})"
                break
            if runtime.budget_exceeded(started, self.max_seconds):
                stopped = (f"stopped at epoch {epoch}: the {int(self.max_seconds)}s time budget "
                           "ran out — lower the epochs or the layer sizes for a full run")
                break

        if best_state is not None:
            module.load_state_dict(best_state)

        self.history_ = history
        self.stopped_reason_ = stopped
        self.best_epoch_ = best_epoch
        self.n_epochs_ = len(history["epoch"])
        self.train_seconds_ = round(time.time() - started, 2)
        self.best_loss_ = None if best_state is None else float(best_loss)
        self.n_parameters_ = parameter_count(module)
        return module

    # ------------------------------------------------------------ persistence
    def _finalise(self, module, out_dim: int, arch: dict | None = None):
        module.eval()
        self.state_dict_ = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        self.module_ = None
        self.arch_ = arch or {
            "kind": type(module).__name__,
            "n_numeric": int(getattr(module, "n_numeric", 0)),
            "cat_sizes": list(getattr(module, "cat_sizes", [])),
            "hidden": list(getattr(module, "hidden", ())),
            "dropout": float(getattr(module, "dropout", 0.0)),
            "activation": str(getattr(module, "activation", "relu")),
            "batch_norm": bool(getattr(module, "batch_norm", False)),
            "out_dim": int(out_dim),
        }
        self.architecture_ = architecture_note(
            self.arch_["n_numeric"], self.arch_["cat_sizes"], self.arch_["hidden"],
            out_dim, self.n_parameters_)
        return self

    def _module(self):
        """Rebuild the network from ``arch_`` + ``state_dict_`` (never pickled live)."""
        if getattr(self, "module_", None) is not None:
            return self.module_
        import torch

        arch = self.arch_
        if arch.get("kind") == "AutoencoderNet":
            module = AutoencoderNet(
                n_features=arch.get("n_features", 0), hidden=tuple(arch.get("hidden", ())),
                latent=arch.get("latent", 8), dropout=arch.get("dropout", 0.0),
                activation=arch.get("activation", "relu"), batch_norm=arch.get("batch_norm", False))
        else:
            module = TabularMLP(
                n_numeric=arch["n_numeric"], cat_sizes=arch["cat_sizes"],
                hidden=tuple(arch["hidden"]), dropout=arch["dropout"],
                activation=arch["activation"], batch_norm=arch["batch_norm"],
                embed_dim=int(getattr(self, "embedding_dim", 0) or 0), out_dim=arch["out_dim"])
        module.load_state_dict(self.state_dict_)
        module.eval()
        self.module_ = module
        return module

    def __getstate__(self):
        state = dict(self.__dict__)
        state["module_"] = None            # rebuilt from state_dict_ on demand
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.module_ = None

    # ---------------------------------------------------------------- helpers
    def _forward(self, X) -> np.ndarray:
        import torch

        module = self._module()
        x_num, x_cat = self._split(X)
        with torch.no_grad():
            tn = torch.as_tensor(x_num)
            tc = None if x_cat is None else torch.as_tensor(x_cat)
            out = module(tn, tc)
            return out.detach().cpu().numpy()

    def _validation_split(self, n: int, y: np.ndarray):
        """Hold out a slice for early stopping; never on tiny data."""
        fraction = float(self.val_fraction or 0.0)
        if fraction <= 0 or n < 25:
            return None, np.arange(n)
        from sklearn.model_selection import train_test_split

        idx = np.arange(n)
        stratify = None
        if self._task in ("binary", "multiclass"):
            counts = pd.Series(y).value_counts()
            if len(counts) and counts.min() >= 2:
                stratify = y
        try:
            train_idx, val_idx = train_test_split(
                idx, test_size=min(max(fraction, 0.05), 0.4),
                random_state=int(self.random_state), stratify=stratify)
        except ValueError:
            train_idx, val_idx = train_test_split(
                idx, test_size=min(max(fraction, 0.05), 0.4), random_state=int(self.random_state))
        return val_idx, train_idx

    def training_summary(self) -> str:
        """One line for the GUI log / report: what was trained and how it stopped."""
        if not hasattr(self, "history_"):
            return "not trained yet"
        return (f"{self.architecture_} · {self.n_epochs_} epochs "
                f"({self.train_seconds_}s) · {self.stopped_reason_}")


class TorchMLPRegressor(RegressorMixin, _TorchBase):
    """Feed-forward network for regression, with embeddings for categoricals."""

    _task = "regression"

    def __init__(
        self,
        hidden_layers: str = DEFAULT_HIDDEN,
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        use_embeddings: bool = True,
        embedding_dim: int = 0,
        loss: str = "mse",
        standardize_target: bool = True,
        epochs: int = 120,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        patience: int = 12,
        val_fraction: float = 0.15,
        threads: int = 4,
        max_seconds: int = 180,
        random_state: int = 42,
    ):
        super().__init__(
            hidden_layers=hidden_layers, activation=activation, dropout=dropout,
            batch_norm=batch_norm, use_embeddings=use_embeddings, embedding_dim=embedding_dim,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, optimizer=optimizer, patience=patience,
            val_fraction=val_fraction, threads=threads, max_seconds=max_seconds,
            random_state=random_state)
        self.loss = loss
        self.standardize_target = standardize_target

    def fit(self, X, y, sample_weight=None):
        import torch

        runtime.check_cancel()
        frame = _as_frame(X)
        self._discover_schema(frame)
        y = np.asarray(pd.Series(y).astype("float64").fillna(0.0), dtype="float32")

        if self.standardize_target:
            self.y_mean_ = float(np.mean(y))
            self.y_std_ = float(np.std(y)) or 1.0
            y_scaled = (y - self.y_mean_) / self.y_std_
        else:
            self.y_mean_, self.y_std_ = 0.0, 1.0
            y_scaled = y

        x_num, x_cat = self._split(frame)
        val_idx, train_idx = self._validation_split(len(y), y)

        def tensors(idx):
            return (torch.as_tensor(x_num[idx]),
                    None if x_cat is None else torch.as_tensor(x_cat[idx]),
                    torch.as_tensor(y_scaled[idx]).reshape(-1, 1))

        weights = None
        if sample_weight is not None:
            w = np.asarray(pd.Series(sample_weight).astype("float64").fillna(0.0), dtype="float32")
            weights = torch.as_tensor(w)

        module = TabularMLP(
            n_numeric=x_num.shape[1], cat_sizes=self.cat_sizes_,
            hidden=parse_hidden(self.hidden_layers), dropout=float(self.dropout),
            activation=self.activation, batch_norm=bool(self.batch_norm),
            embed_dim=int(self.embedding_dim or 0), out_dim=1)
        loss_fn = (torch.nn.HuberLoss(reduction="none")
                   if str(self.loss).lower() == "huber" else torch.nn.MSELoss(reduction="none"))
        self._train_module(module, loss_fn, tensors(train_idx),
                           None if val_idx is None else tensors(val_idx),
                           None if weights is None else weights[train_idx],
                           None if (weights is None or val_idx is None) else weights[val_idx])
        return self._finalise(module, 1)

    def predict(self, X):
        out = self._forward(X).reshape(-1)
        return out * self.y_std_ + self.y_mean_


class TorchMLPClassifier(ClassifierMixin, _TorchBase):
    """Feed-forward network for classification, with embeddings for categoricals."""

    _task = "binary"

    def __init__(
        self,
        hidden_layers: str = DEFAULT_HIDDEN,
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = False,
        use_embeddings: bool = True,
        embedding_dim: int = 0,
        class_weight: str | None = None,
        label_smoothing: float = 0.0,
        epochs: int = 120,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        patience: int = 12,
        val_fraction: float = 0.15,
        threads: int = 4,
        max_seconds: int = 180,
        random_state: int = 42,
    ):
        super().__init__(
            hidden_layers=hidden_layers, activation=activation, dropout=dropout,
            batch_norm=batch_norm, use_embeddings=use_embeddings, embedding_dim=embedding_dim,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, optimizer=optimizer, patience=patience,
            val_fraction=val_fraction, threads=threads, max_seconds=max_seconds,
            random_state=random_state)
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def fit(self, X, y, sample_weight=None):
        import torch

        runtime.check_cancel()
        frame = _as_frame(X)
        self._discover_schema(frame)

        # Preserve the label dtype: the app compares predictions against the
        # original target column, so a 0/1 target must come back as 0/1 and not
        # as the strings "0"/"1" (which would silently score zero accuracy).
        y_series = pd.Series(y)
        codes, uniques = pd.factorize(y_series, sort=True)
        codes = np.asarray(codes, dtype="int64")
        if (codes < 0).any():
            # A missing label becomes its own class instead of crashing the fit.
            uniques = list(uniques) + ["(missing)"]
            codes = np.where(codes < 0, len(uniques) - 1, codes)
        self.classes_ = np.asarray(uniques)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("the target column has only one distinct value — nothing to classify")
        self.n_classes_ = n_classes
        self._task = "binary" if n_classes <= 2 else "multiclass"

        x_num, x_cat = self._split(frame)
        val_idx, train_idx = self._validation_split(len(codes), codes)

        def tensors(idx):
            return (torch.as_tensor(x_num[idx]),
                    None if x_cat is None else torch.as_tensor(x_cat[idx]),
                    torch.as_tensor(codes[idx]))

        weights = None
        if sample_weight is not None:
            w = np.asarray(pd.Series(sample_weight).astype("float64").fillna(0.0), dtype="float32")
            weights = torch.as_tensor(w)

        class_weights = None
        if str(self.class_weight or "").lower() == "balanced":
            counts = np.bincount(codes, minlength=n_classes).astype("float64")
            counts[counts == 0] = 1.0
            balance = len(codes) / (n_classes * counts)
            class_weights = torch.as_tensor(balance, dtype=torch.float32)

        module = TabularMLP(
            n_numeric=x_num.shape[1], cat_sizes=self.cat_sizes_,
            hidden=parse_hidden(self.hidden_layers), dropout=float(self.dropout),
            activation=self.activation, batch_norm=bool(self.batch_norm),
            embed_dim=int(self.embedding_dim or 0), out_dim=n_classes)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="none",
            label_smoothing=float(self.label_smoothing or 0.0))
        self._train_module(module, loss_fn, tensors(train_idx),
                           None if val_idx is None else tensors(val_idx),
                           None if weights is None else weights[train_idx],
                           None if (weights is None or val_idx is None) else weights[val_idx])
        return self._finalise(module, n_classes)

    def predict_proba(self, X):
        logits = self._forward(X)
        if logits.shape[1] == 1:                      # defensive: a 1-class fit
            p = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.column_stack([1 - p, p])
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def feature_importances_(self):
        """Not defined for a neural network — the app falls back to permutation."""
        raise AttributeError("a neural network has no native feature_importances_")


class TorchAutoencoder(TransformerMixin, _TorchBase):
    """Unsupervised non-linear compression, with optional clustering of the code.

    Sits where PCA sits: ``transform`` gives a low-dimensional learned embedding of
    the prepared matrix.  It also clusters that embedding with KMeans and exposes
    ``labels_``, so it can be used directly as a deep-clustering method — which is
    the more useful thing on a dataset where the clusters are not linearly
    separable in the original features.
    """

    _task = "unsupervised"

    def __init__(
        self,
        hidden_layers: str = "64, 32",
        latent_dim: int = 8,
        n_clusters: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        epochs: int = 150,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        patience: int = 15,
        val_fraction: float = 0.15,
        threads: int = 4,
        max_seconds: int = 180,
        random_state: int = 42,
    ):
        super().__init__(
            hidden_layers=hidden_layers, activation=activation, dropout=dropout,
            batch_norm=batch_norm, use_embeddings=False, embedding_dim=0,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, optimizer=optimizer, patience=patience,
            val_fraction=val_fraction, threads=threads, max_seconds=max_seconds,
            random_state=random_state)
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        import torch

        runtime.check_cancel()
        frame = _as_frame(X)
        self._discover_schema(frame)
        # An autoencoder reconstructs the whole prepared matrix, embeddings included:
        # there is no target to protect, so nothing is dropped.
        values = _numeric_matrix(frame)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = [str(c) for c in frame.columns]

        latent = max(1, min(int(self.latent_dim or 8), max(1, values.shape[1] - 1) or 1))
        val_idx, train_idx = self._validation_split(len(values), np.zeros(len(values)))

        def tensors(idx):
            x = torch.as_tensor(values[idx])
            return x, None, x                     # the target *is* the input

        module = AutoencoderNet(
            n_features=values.shape[1], hidden=parse_hidden(self.hidden_layers),
            latent=latent, dropout=float(self.dropout), activation=self.activation,
            batch_norm=bool(self.batch_norm))
        self._train_module(module, torch.nn.MSELoss(reduction="none"),
                           tensors(train_idx), None if val_idx is None else tensors(val_idx))
        self._finalise(module, values.shape[1], arch={
            "kind": "AutoencoderNet",
            "n_features": int(values.shape[1]),
            "latent": int(latent),
            "hidden": list(parse_hidden(self.hidden_layers)),
            "dropout": float(self.dropout),
            "activation": str(self.activation),
            "batch_norm": bool(self.batch_norm),
            "n_numeric": int(values.shape[1]),
            "cat_sizes": [],
            "out_dim": int(values.shape[1]),
        })
        self.architecture_ = (f"autoencoder {values.shape[1]} -> "
                              f"{' -> '.join(str(h) for h in parse_hidden(self.hidden_layers))}"
                              f" -> {latent} -> ... -> {values.shape[1]} · "
                              f"{self.n_parameters_:,} weights")

        self.embedding_ = self._encode(values)
        self.reconstruction_error_ = self._reconstruction_error(values)
        self._fit_clusters()
        return self

    # ------------------------------------------------------------------ latent
    def _encode(self, values: np.ndarray) -> np.ndarray:
        import torch

        module = self._module()
        with torch.no_grad():
            return module.encode(torch.as_tensor(values)).detach().cpu().numpy()

    def _reconstruction_error(self, values: np.ndarray) -> np.ndarray:
        import torch

        module = self._module()
        with torch.no_grad():
            rebuilt = module(torch.as_tensor(values)).detach().cpu().numpy()
        return np.mean((rebuilt - values) ** 2, axis=1)

    def _prepared(self, X) -> pd.DataFrame:
        """Validate that ``X`` is the prepared matrix this autoencoder was fitted on.

        ``transform`` and ``reconstruction_error`` deliberately take the *encoded*
        matrix — by the time the app calls them the pipeline has already run the
        preprocessor.  Passing the raw frame instead is an easy mistake, and pandas'
        own complaint about it ("could not convert string to float") says nothing
        about the real problem, so check the columns and say something useful.
        """
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        expected = list(getattr(self, "feature_names_in_", []) or [])
        if expected and list(frame.columns) != expected:
            preview = ", ".join(str(c) for c in expected[:4])
            raise ValueError(
                f"this autoencoder works on the {len(expected)} prepared columns it was "
                f"fitted on ({preview}{'…' if len(expected) > 4 else ''}), but was given "
                f"{len(frame.columns)} columns — pass the output of the pipeline's "
                "'prepare' step rather than the raw frame"
            )
        return frame

    def transform(self, X):
        return self._encode(_numeric_matrix(self._prepared(X)))

    def reconstruction_error(self, X) -> np.ndarray:
        """Per-row squared reconstruction error — usable as an anomaly score."""
        return self._reconstruction_error(_numeric_matrix(self._prepared(X)))

    def _fit_clusters(self):
        from sklearn.cluster import KMeans

        k = int(self.n_clusters or 0)
        if k < 2 or len(self.embedding_) < k:
            self.labels_ = None
            self.cluster_model_ = None
            return
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=int(self.random_state))
        self.labels_ = kmeans.fit_predict(self.embedding_)
        self.cluster_model_ = kmeans

    def predict(self, X):
        """Cluster assignment for new rows, via the fitted KMeans on the latent code."""
        if getattr(self, "cluster_model_", None) is None:
            raise ValueError("this autoencoder was not asked to cluster (n_clusters < 2)")
        return self.cluster_model_.predict(self.transform(X))


def describe_trained(estimator) -> dict:
    """Small summary the GUI and the report can show for any of the DL models."""
    out = {}
    for key in ("architecture_", "n_epochs_", "train_seconds_", "stopped_reason_",
                "n_parameters_", "best_loss_"):
        value = getattr(estimator, key, None)
        if value is not None:
            out[key] = value
    return out
