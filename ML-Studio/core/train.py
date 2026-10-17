"""Splitting, cross-validation, evaluation and the on-disk model bundle.

A trained model is only reusable if the numbers you measured came from the same
arithmetic that will be applied later.  So a bundle stores the whole pipeline —
formulas, imputation values, encoders, scaler statistics and the estimator — and
prediction goes through ``bundle["pipeline"]`` and nothing else.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import threading
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from . import metrics as M
from . import models as MODELS
from .preprocess import PreprocessSpec, build_pipeline

BUNDLE_FORMAT = "ml-workbench-bundle"
BUNDLE_VERSION = 1


class Cancelled(Exception):
    """Raised inside a long job when the user presses Cancel."""


def check_cancel(cancel: threading.Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled("cancelled by user")


# --------------------------------------------------------------------------
# split description
# --------------------------------------------------------------------------


@dataclass
class TrainSpec:
    target: str | None = None
    features: list | None = None
    task: str = "regression"
    test_size: float = 0.2
    shuffle: bool = True
    stratify: bool = False
    random_state: int = 42
    cv_folds: int = 5
    primary_metric: str = "r2"
    scale_target: bool = False
    drop_missing_target: bool = True
    sample_weight_column: str | None = None
    group_column: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrainSpec":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    target_name: str
    task: str
    labels: list = field(default_factory=list)
    w_train: pd.Series | None = None
    w_test: pd.Series | None = None
    dropped: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def resolve_features(df: pd.DataFrame, spec: TrainSpec) -> list:
    """Feature columns = everything the user chose, minus the target."""
    if spec.task == "unsupervised":
        cols = list(spec.features) if spec.features else [c for c in df.columns]
    else:
        cols = list(spec.features) if spec.features else [c for c in df.columns if c != spec.target]
    return [c for c in cols if c in df.columns and c != spec.target]


def make_split(df: pd.DataFrame, spec: TrainSpec) -> Split:
    """Split the frame into train/test, keeping the index for traceability."""
    features = resolve_features(df, spec)
    if not features:
        raise ValueError("no feature columns selected")

    if spec.task == "unsupervised":
        X = df[features].copy()
        y = pd.Series(np.zeros(len(df)), index=df.index, name="cluster")
        empty = X.iloc[0:0]
        return Split(X, empty, y, y.iloc[0:0], features, "", spec.task, [], None, None,
                     {"unsupervised": "no split is used for clustering / dimensionality reduction"})

    if not spec.target or spec.target not in df.columns:
        raise ValueError(f"target column {spec.target!r} is not in the data")

    data = df[features + [spec.target]].copy()
    dropped = {}
    before = len(data)
    if spec.drop_missing_target:
        data = data[data[spec.target].notna()]
        dropped["rows missing the target"] = before - len(data)
    if data.empty:
        raise ValueError("every row is missing the target value")

    X = data[features]
    y = data[spec.target]
    if spec.scale_target and pd.api.types.is_numeric_dtype(y):
        y = (y - y.mean()) / (y.std() or 1.0)

    weights = None
    if spec.sample_weight_column and spec.sample_weight_column in df.columns:
        weights = df.loc[data.index, spec.sample_weight_column]

    strat = None
    if spec.stratify and spec.task in ("binary", "multiclass"):
        counts = y.value_counts()
        strat = y if counts.min() >= 2 else None
        if strat is None:
            dropped["stratified split"] = "disabled: a class has fewer than 2 rows"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(spec.test_size), random_state=int(spec.random_state),
        shuffle=bool(spec.shuffle), stratify=strat,
    )
    w_train = w_test = None
    if weights is not None:
        w_train = weights.loc[X_train.index]
        w_test = weights.loc[X_test.index]

    labels = sorted(set(y.dropna().unique().tolist())) if spec.task in ("binary", "multiclass") else []
    if spec.task == "binary" and len(labels) != 2:
        raise ValueError(
            f"a binary task needs exactly 2 distinct target values, found {len(labels)}: {labels[:6]}"
        )
    if spec.task == "multiclass" and len(labels) < 3:
        dropped["multiclass"] = f"only {len(labels)} class(es) present — consider the binary task"

    return Split(X_train, X_test, y_train, y_test, features, spec.target, spec.task,
                 labels, w_train, w_test, dropped)


def cv_splitter(spec: TrainSpec, y, n_rows: int):
    folds = max(2, min(int(spec.cv_folds or 5), n_rows // 2 or 2))
    if spec.task in ("binary", "multiclass") and spec.stratify:
        counts = pd.Series(y).value_counts()
        if counts.min() >= folds:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=spec.random_state), folds
        # too few of some class for stratified folds
        return KFold(n_splits=folds, shuffle=True, random_state=spec.random_state), folds
    return KFold(n_splits=folds, shuffle=True, random_state=spec.random_state), folds


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def build_train_pipeline(pre_spec: PreprocessSpec | dict, model_key: str, model_params: dict, task: str):
    estimator = MODELS.build_model(model_key, model_params)
    return build_pipeline(pre_spec, estimator, task=task), estimator


def _proba_of(estimator, X):
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except Exception:
            return None
    if hasattr(estimator, "decision_function"):
        try:
            scores = np.asarray(estimator.decision_function(X))
            if scores.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1 - p, p])
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        except Exception:
            return None
    return None


def metrics_for(task: str, n_classes: int, chosen: list | None = None) -> list:
    """Every metric worth computing for this task.

    Probability metrics only appear when the estimator can actually produce
    probabilities, which is decided at evaluation time, not here.
    """
    keys = list(M.REGRESSION) if task == "regression" else list(M.CLASSIFICATION) + list(M.PROBA_METRICS)
    if chosen:
        keys = [k for k in chosen if k in keys] + [k for k in keys if k not in chosen]
    return keys


def cross_validate_pipeline(
    pipe,
    X: pd.DataFrame,
    y: pd.Series,
    spec: TrainSpec,
    weights=None,
    progress=None,
    cancel: threading.Event | None = None,
) -> dict:
    """Manual k-fold loop: gives per-fold progress and allows cancellation.

    Returns ``{"folds": [ {metric: value, ...}, ... ], "summary": {metric: (mean, std)},
    "primary": {"mean":…, "std":…, "per_fold": […]}}``.
    """
    splitter, folds = cv_splitter(spec, y, len(X))
    per_fold: list[dict] = []
    keys = metrics_for(spec.task, len(set(pd.Series(y).unique())))
    for i, (tr, va) in enumerate(splitter.split(X, y)):
        check_cancel(cancel)
        if progress:
            progress(i, folds, f"fold {i + 1}/{folds}")
        model = clone(pipe)
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = pd.Series(y).iloc[tr], pd.Series(y).iloc[va]
        if weights is not None:
            try:
                model.fit(Xtr, ytr, model__sample_weight=pd.Series(weights).iloc[tr])
            except Exception:
                model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        proba = _proba_of(model, Xva) if spec.task != "regression" else None
        result = M.evaluate(spec.task, yva, pred, proba,
                            sample_weight=None if weights is None else pd.Series(weights).iloc[va])
        per_fold.append({k: v for k, v in result.items() if not k.startswith("_")})

    summary = {}
    for key in keys:
        values = np.array([f.get(key, np.nan) for f in per_fold], dtype=float)
        finite = values[np.isfinite(values)]
        summary[key] = (float(finite.mean()) if finite.size else float("nan"),
                        float(finite.std()) if finite.size else float("nan"))
    primary = spec.primary_metric
    primary_values = [f.get(primary, np.nan) for f in per_fold]
    return {
        "folds": per_fold,
        "summary": summary,
        "primary": {
            "metric": primary,
            "mean": summary.get(primary, (float("nan"), 0.0))[0],
            "std": summary.get(primary, (float("nan"), 0.0))[1],
            "per_fold": primary_values,
            "n_folds": folds,
        },
    }


def evaluate_fitted(pipe, split: Split, spec: TrainSpec, include_train: bool = True) -> dict:
    """Metrics on the held-out test set (and the training set, for overfitting checks)."""
    out = {}
    X_test, y_test = split.X_test, split.y_test
    if not len(X_test):
        return out
    pred = pipe.predict(X_test)
    proba = _proba_of(pipe, X_test) if spec.task != "regression" else None
    out["test"] = M.evaluate(spec.task, y_test, pred, proba, sample_weight=split.w_test)
    out["test_pred"] = np.asarray(pred)
    out["test_proba"] = None if proba is None else np.asarray(proba)
    out["test_truth"] = np.asarray(y_test)
    if include_train:
        train_pred = pipe.predict(split.X_train)
        train_proba = _proba_of(pipe, split.X_train) if spec.task != "regression" else None
        out["train"] = M.evaluate(spec.task, split.y_train, train_pred, train_proba,
                                  sample_weight=split.w_train)
        out["train_pred"] = np.asarray(train_pred)
        out["train_truth"] = np.asarray(split.y_train)
    out["overfit_gap"] = _gap(out, spec)
    return out


def _gap(out: dict, spec: TrainSpec) -> float | None:
    """How much worse the test score is than the training score, so positive = overfitting.

    The sign is what every consumer reads: the Model screen's card says "large positive =
    memorising", and both the card and the report say "worse on the test set" when it is
    positive.  For a metric where higher is better (ROC AUC, R², F1 — that is, all of
    them in practice) the overfit amount is ``train - test``, i.e. the *negative* of the
    raw difference.  Getting this the wrong way round is silent and produces a
    confidently wrong sentence, which is why it is spelled out here.
    """
    key = spec.primary_metric
    try:
        test_v = out.get("test", {}).get(key)
        train_v = out.get("train", {}).get(key)
        if test_v is None or train_v is None or not np.isfinite(test_v) or not np.isfinite(train_v):
            return None
        gap = test_v - train_v
        return float(-gap if M.higher_is_better(spec.task, key) else gap)
    except Exception:
        return None


def fit_full(pipe, split: Split, spec: TrainSpec):
    """Fit on the training split (or everything, for unsupervised work)."""
    if spec.task == "unsupervised" or not len(split.X_test):
        target = split.X_train if len(split.X_train) else split.X_test
        pipe.fit(target)
        return pipe
    if split.w_train is not None:
        try:
            pipe.fit(split.X_train, split.y_train, model__sample_weight=split.w_train)
            return pipe
        except Exception:
            # the chosen estimator takes no sample weights — say so rather than
            # silently weighting the cross-validation folds and nothing else
            split.dropped["row weights"] = "ignored: this estimator does not accept sample_weight"
    pipe.fit(split.X_train, split.y_train)
    return pipe


# --------------------------------------------------------------------------
# bundles
# --------------------------------------------------------------------------

VERSION_KEYS = ("python", "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "torch")


def library_versions() -> dict:
    import sys

    import sklearn

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    for name in ("xgboost", "lightgbm"):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except Exception:
            pass
    # A neural network is only reproducible if you know the version *and* the
    # compute setup, so record both.  The thread count matters: torch's default
    # reduction order changes with it, so the same seed can give slightly
    # different weights on a different machine.
    try:
        from .dl import runtime

        if runtime.torch_available():
            import torch

            versions["torch"] = torch.__version__
            versions["torch device"] = "cpu"
            versions["torch threads"] = str(torch.get_num_threads())
    except Exception:
        pass
    return versions


def make_bundle(
    pipe,
    spec: TrainSpec,
    split: Split,
    pre_spec: PreprocessSpec,
    model_key: str,
    model_params: dict,
    evaluation: dict,
    cv: dict | None = None,
    tuning: dict | None = None,
    source_file: str | None = None,
    cleaning_ops: list | None = None,
    notes: str = "",
    feature_importances: dict | None = None,
) -> dict:
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "task": spec.task,
        "target": spec.target,
        "features": list(split.feature_names),
        "pipeline": pipe,
        "model_key": model_key,
        "model_label": MODELS.get_model_spec(model_key).label,
        "model_params": dict(model_params or {}),
        "preprocess": pre_spec.to_dict() if isinstance(pre_spec, PreprocessSpec) else dict(pre_spec),
        "train_spec": spec.to_dict(),
        "primary_metric": spec.primary_metric,
        "metrics": {
            "test": {k: v for k, v in (evaluation.get("test") or {}).items() if not k.startswith("_")},
            "train": {k: v for k, v in (evaluation.get("train") or {}).items() if not k.startswith("_")},
        },
        "cv": None if cv is None else {"summary": cv.get("summary"), "primary": cv.get("primary"), "folds": cv.get("folds")},
        "tuning": tuning,
        "overfit_gap": evaluation.get("overfit_gap"),
        "feature_importances": feature_importances or {},
        "n_train": split.n_train,
        "n_test": split.n_test,
        "n_features_in": len(split.feature_names),
        "n_features_out": _n_output_features(pipe),
        "labels": split.labels,
        "dropped": split.dropped,
        "cleaning_ops": cleaning_ops or [],
        "source_file": source_file or "",
        "notes": notes,
        "versions": library_versions(),
    }


def _n_output_features(pipe) -> int | None:
    try:
        prep = pipe.named_steps.get("prepare")
        return len(getattr(prep, "feature_names_out_", []) or [])
    except Exception:
        return None


def save_bundle(bundle: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"{os.path.basename(path)} is not a ML Workbench model bundle "
            f"(expected format={BUNDLE_FORMAT!r}, found {type(bundle).__name__})"
        )
    return annotate_bundle(bundle)


def annotate_bundle(bundle: dict) -> dict:
    """Add non-fatal warnings: version drift, and whether the pipeline is usable."""
    warnings = []
    current = library_versions()
    saved = bundle.get("versions") or {}
    for key in VERSION_KEYS:
        if key in saved and key in current and saved[key] != current[key]:
            if key in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm"):
                warnings.append(
                    f"trained with {key} {saved[key]} but this environment has {current[key]} — "
                    "re-check the metrics before trusting a prediction"
                )
    if not hasattr(bundle.get("pipeline"), "predict") and bundle.get("task") != "unsupervised":
        warnings.append("the stored pipeline has no predict() — this bundle cannot be used for prediction")
    bundle["_warnings"] = warnings
    bundle["_versions_saved"] = saved
    return bundle


def bundle_summary(bundle: dict) -> pd.DataFrame:
    rows = [
        ("task", bundle.get("task")),
        ("target", bundle.get("target")),
        ("model", f'{bundle.get("model_label")} ({bundle.get("model_key")})'),
        ("parameters", ", ".join(f"{k}={v}" for k, v in (bundle.get("model_params") or {}).items()) or "defaults"),
        ("features in", bundle.get("n_features_in")),
        ("features after encoding", bundle.get("n_features_out")),
        ("training rows", bundle.get("n_train")),
        ("test rows", bundle.get("n_test")),
        ("primary metric", bundle.get("primary_metric")),
        ("test score", _fmt((bundle.get("metrics") or {}).get("test", {}).get(bundle.get("primary_metric")))),
        ("cross-validation", _fmt_cv(bundle)),
        ("source file", bundle.get("source_file")),
        ("saved", (bundle.get("versions") or {}).get("created")),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def _fmt(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value


def _fmt_cv(bundle):
    primary = (bundle.get("cv") or {}).get("primary")
    if not primary:
        return "not run"
    return f"{primary['mean']:.4f} ± {primary['std']:.4f} over {primary.get('n_folds')} folds"
