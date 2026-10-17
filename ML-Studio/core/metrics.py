"""Metric definitions shared by training, tuning, evaluation and prediction.

Each metric carries the scikit-learn scorer string used for cross-validation, so
the number printed on the evaluate screen and the number a grid search optimises
are guaranteed to be the same quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.abs((y_true - y_pred) / denom)
    value = float(np.nanmean(values) * 100) if np.isfinite(values).any() else float("nan")
    return min(value, 1e6)


def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denom < 1e-12, 0.0, 2 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(values) * 100)


# key -> (label, mean_squared_error-style callable, higher_is_better, sklearn scorer)
REGRESSION: dict[str, dict] = {
    "r2": {"label": "R²", "fn": skm.r2_score, "higher": True, "scorer": "r2", "format": "{:.4f}"},
    "rmse": {
        "label": "RMSE", "higher": False, "scorer": "neg_root_mean_squared_error", "format": "{:.4f}",
        "fn": lambda yt, yp: float(np.sqrt(skm.mean_squared_error(yt, yp))),
    },
    "mae": {"label": "MAE", "fn": skm.mean_absolute_error, "higher": False, "scorer": "neg_mean_absolute_error", "format": "{:.4f}"},
    "medae": {"label": "Median AE", "fn": skm.median_absolute_error, "higher": False, "scorer": "neg_median_absolute_error", "format": "{:.4f}"},
    "mse": {"label": "MSE", "fn": skm.mean_squared_error, "higher": False, "scorer": "neg_mean_squared_error", "format": "{:.4f}"},
    "mape": {"label": "MAPE %", "fn": _mape, "higher": False, "scorer": None, "format": "{:.2f}"},
    "smape": {"label": "sMAPE %", "fn": _smape, "higher": False, "scorer": None, "format": "{:.2f}"},
    "explained_variance": {"label": "Explained variance", "fn": skm.explained_variance_score, "higher": True, "scorer": "explained_variance", "format": "{:.4f}"},
    "max_error": {"label": "Max error", "fn": skm.max_error, "higher": False, "scorer": "max_error", "format": "{:.4f}"},
    "rmsle": {
        "label": "RMSLE", "higher": False, "scorer": "neg_root_mean_squared_log_error", "format": "{:.4f}",
        "fn": lambda yt, yp: float(np.sqrt(skm.mean_squared_log_error(np.clip(yt, 0, None), np.clip(yp, 0, None)))),
    },
}

CLASSIFICATION: dict[str, dict] = {
    "accuracy": {"label": "Accuracy", "fn": skm.accuracy_score, "higher": True, "scorer": "accuracy", "format": "{:.4f}"},
    "balanced_accuracy": {"label": "Balanced accuracy", "fn": skm.balanced_accuracy_score, "higher": True, "scorer": "balanced_accuracy", "format": "{:.4f}"},
    "precision_weighted": {"label": "Precision (weighted)", "fn": lambda yt, yp: skm.precision_score(yt, yp, average="weighted", zero_division=0), "higher": True, "scorer": "precision_weighted", "format": "{:.4f}"},
    "recall_weighted": {"label": "Recall (weighted)", "fn": lambda yt, yp: skm.recall_score(yt, yp, average="weighted", zero_division=0), "higher": True, "scorer": "recall_weighted", "format": "{:.4f}"},
    "f1_weighted": {"label": "F1 (weighted)", "fn": lambda yt, yp: skm.f1_score(yt, yp, average="weighted", zero_division=0), "higher": True, "scorer": "f1_weighted", "format": "{:.4f}"},
    "f1_macro": {"label": "F1 (macro)", "fn": lambda yt, yp: skm.f1_score(yt, yp, average="macro", zero_division=0), "higher": True, "scorer": "f1_macro", "format": "{:.4f}"},
    "mcc": {"label": "Matthews correlation", "fn": skm.matthews_corrcoef, "higher": True, "scorer": "matthews_corrcoef", "format": "{:.4f}"},
    "cohen_kappa": {"label": "Cohen's kappa", "fn": skm.cohen_kappa_score, "higher": True, "scorer": "cohen_kappa_score", "format": "{:.4f}"},
    "jaccard": {"label": "Jaccard", "fn": lambda yt, yp: skm.jaccard_score(yt, yp, average="weighted", zero_division=0), "higher": True, "scorer": "jaccard_weighted", "format": "{:.4f}"},
}

# metrics that need predicted probabilities
PROBA_METRICS = {
    "roc_auc": {"label": "ROC AUC", "higher": True, "scorer": "roc_auc", "format": "{:.4f}", "binary_only": False},
    "average_precision": {"label": "Average precision (PR AUC)", "higher": True, "scorer": "average_precision", "format": "{:.4f}", "binary_only": False},
    "log_loss": {"label": "Log loss", "higher": False, "scorer": "neg_log_loss", "format": "{:.4f}", "binary_only": False},
    "brier": {"label": "Brier score", "higher": False, "scorer": "neg_brier_score", "format": "{:.4f}", "binary_only": False},
}

DEFAULT_PRIMARY = {
    "regression": "r2",
    "binary": "roc_auc",
    "multiclass": "f1_weighted",
    "unsupervised": None,
}


def metrics_for_task(task: str) -> dict[str, dict]:
    if task == "regression":
        return {**REGRESSION}
    if task in ("binary", "multiclass"):
        return {**CLASSIFICATION, **PROBA_METRICS}
    return {}


def label(task: str, key: str) -> str:
    if key in PROBA_METRICS:
        return PROBA_METRICS[key]["label"]
    table = REGRESSION if task == "regression" else CLASSIFICATION
    return table[key]["label"] if key in table else key


def higher_is_better(task: str, key: str) -> bool:
    if key in PROBA_METRICS:
        return PROBA_METRICS[key]["higher"]
    table = REGRESSION if task == "regression" else CLASSIFICATION
    if key in table:
        return table[key]["higher"]
    # a neg_* scorer means greater is better by construction
    return True


def scorer_name(task: str, key: str) -> str | None:
    if key in PROBA_METRICS:
        name = PROBA_METRICS[key]["scorer"]
        return name if isinstance(name, str) else None
    if key in CLASSIFICATION:
        return CLASSIFICATION[key]["scorer"]
    if key in REGRESSION:
        return REGRESSION[key]["scorer"]
    return None


def primary_metric(task: str, y=None) -> str:
    if task == "binary":
        return "roc_auc"
    return DEFAULT_PRIMARY.get(task, "r2") or "r2"


def _needs_proba(task: str, key: str) -> bool:
    return key in PROBA_METRICS


def evaluate(task: str, y_true, y_pred, y_proba=None, sample_weight=None, labels=None) -> dict:
    """Compute every applicable metric.  Probability metrics are skipped when no
    probabilities are available, rather than inventing a value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    keys = REGRESSION if task == "regression" else CLASSIFICATION
    for key, spec in keys.items():
        try:
            if sample_weight is not None:
                try:
                    out[key] = float(spec["fn"](y_true, y_pred, sample_weight=sample_weight))
                    continue
                except TypeError:
                    pass
            out[key] = float(spec["fn"](y_true, y_pred))
        except Exception:
            out[key] = float("nan")

    if task != "regression" and y_proba is not None:
        proba = np.asarray(y_proba)
        classes = np.unique(y_true) if labels is None else np.asarray(labels)
        n_classes = len(classes)
        try:
            if n_classes == 2:
                pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
                y_bin = (y_true == classes[1]).astype(int)
                out["roc_auc"] = float(skm.roc_auc_score(y_bin, pos))
                out["average_precision"] = float(skm.average_precision_score(y_bin, pos))
                out["log_loss"] = float(skm.log_loss(y_bin, pos, labels=[0, 1]))
                out["brier"] = float(skm.brier_score_loss(y_bin, pos))
            elif proba.ndim == 2 and proba.shape[1] == n_classes:
                out["roc_auc"] = float(skm.roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
                out["average_precision"] = float(skm.average_precision_score(y_true, proba, average="weighted"))
                clipped = np.clip(proba, 1e-9, 1)
                clipped = clipped / clipped.sum(axis=1, keepdims=True)
                out["log_loss"] = float(skm.log_loss(y_true, clipped, labels=classes))
        except Exception as exc:  # a single-class fold, or a degenerate split
            for key in PROBA_METRICS:
                out.setdefault(key, float("nan"))
            out["_proba_error"] = str(exc)
    return out


def metric_frame(task: str, results: dict) -> pd.DataFrame:
    rows = []
    for key, value in results.items():
        if key.startswith("_") or not isinstance(value, (int, float)):
            continue
        rows.append(
            {
                "metric": label(task, key),
                "key": key,
                "value": value,
                "better": "higher" if higher_is_better(task, key) else "lower",
            }
        )
    for key in ("_proba_error",):
        if key in results:
            rows.append({"metric": "probability metrics unavailable", "key": key, "value": str(results[key]), "better": "-"})
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, labels=None) -> pd.DataFrame:
    lab = labels if labels is not None else sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    matrix = skm.confusion_matrix(y_true, y_pred, labels=lab)
    return pd.DataFrame(matrix, index=[f"actual {l}" for l in lab], columns=[f"predicted {l}" for l in lab])


def classification_report(y_true, y_pred, labels=None) -> pd.DataFrame:
    lab = labels if labels is not None else None
    report = skm.classification_report(y_true, y_pred, labels=lab, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(4)


def regression_residuals(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    return pd.DataFrame(
        {"actual": y_true, "predicted": y_pred, "residual": resid, "abs_residual": np.abs(resid),
         "pct_error": np.where(np.abs(y_true) > 1e-12, 100 * resid / np.abs(y_true), np.nan)}
    )


def compare_models(rows: list[dict], task: str, primary: str) -> pd.DataFrame:
    """Leaderboard frame.  Sorted best-first using the primary metric's direction."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if primary in df.columns:
        df = df.sort_values(primary, ascending=not higher_is_better(task, primary))
    return df.reset_index(drop=True)
