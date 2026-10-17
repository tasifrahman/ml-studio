"""Model interpretation: native importances, permutation importances, coefficients,
and one-dimensional dependence curves.

Everything here takes the *fitted pipeline* plus the raw feature frame, so the
column names in the output are the names the user typed, not the encoded
``Cement=high`` style names that exist only after preprocessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepared_names(pipe) -> list[str]:
    prep = getattr(pipe, "named_steps", {}).get("prepare")
    names = list(getattr(prep, "feature_names_out_", []) or [])
    if names:
        return names
    return [f"feature {i}" for i in range(getattr(prep, "n_features_in_", 0) or 0)]


def raw_feature_names(pipe) -> list[str]:
    prep = getattr(pipe, "named_steps", {}).get("prepare")
    return list(getattr(prep, "input_columns_", []) or [])


def _estimator(pipe):
    steps = getattr(pipe, "named_steps", None) or {}
    return steps.get("model", pipe)


def native_importance(pipe) -> pd.DataFrame | None:
    """Built-in importances or coefficients, when the estimator has them."""
    estimator = _estimator(pipe)
    names = prepared_names(pipe)
    values = None
    kind = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity / split importance"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        kind = "absolute coefficient"
    if values is None:
        return None
    if len(values) != len(names):
        # sklearn sometimes reports a different length; fall back to positional names
        names = [f"feature {i}" for i in range(len(values))]
    frame = pd.DataFrame({"feature": names, "importance": values})
    frame["kind"] = kind
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_frame(
    pipe, X: pd.DataFrame, y, task: str, scoring: str | None = None,
    n_repeats: int = 8, random_state: int = 0, sample_rows: int = 2000,
) -> pd.DataFrame:
    """Shuffle each original column and measure the drop in score.

    Unlike split importance this works for any model, is measured on held-out
    data when you pass held-out data, and is expressed in the metric's own units.
    """
    from sklearn.inspection import permutation_importance

    if X is None or y is None or not len(X):
        raise ValueError("permutation importance needs features and a target")
    frame = X
    target = y
    if len(frame) > sample_rows:
        idx = np.random.RandomState(random_state).choice(len(frame), sample_rows, replace=False)
        frame = frame.iloc[idx]
        target = pd.Series(y).iloc[idx]
    result = permutation_importance(
        pipe, frame, target, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=1,
    )
    out = pd.DataFrame({
        "feature": list(frame.columns),
        "perm_mean": result.importances_mean,
        "perm_std": result.importances_std,
        "perm_raw": list(result.importances),
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    out["importance"] = out["perm_mean"]
    out["kind"] = "drop in score when shuffled"
    return out


def importance_frame(
    pipe, X: pd.DataFrame, y, task: str, method: str = "auto",
    scoring: str | None = None, n_repeats: int = 8, random_state: int = 0,
) -> pd.DataFrame:
    """Importance table, choosing the most informative method available.

    ``auto`` prefers built-in importances for tree/linear models (they are free)
    and falls back to permutation for anything else.
    """
    task_is_regression = task == "regression"
    if method in ("auto", "native"):
        frame = native_importance(pipe)
        if frame is not None:
            return frame
        if method == "native":
            raise ValueError("this model has no built-in feature importances — use permutation instead")
    return permutation_importance_frame(pipe, X, y, task, scoring=scoring,
                                        n_repeats=n_repeats, random_state=random_state)


def coefficients(pipe) -> pd.DataFrame | None:
    """Signed coefficients with their prepared-feature names."""
    estimator = _estimator(pipe)
    if not hasattr(estimator, "coef_"):
        return None
    coef = np.asarray(estimator.coef_, dtype=float)
    names = prepared_names(pipe)
    if coef.ndim == 1:
        if len(coef) != len(names):
            return None
        return pd.DataFrame({"feature": names, "coefficient": coef})
    if coef.shape[1] != len(names):
        return None
    classes = list(getattr(estimator, "classes_", range(coef.shape[0])))
    if coef.shape[0] == 1:
        # a binary logistic fit stores one row of coefficients, for the positive class
        label = classes[-1] if len(classes) > 1 else (classes[0] if classes else 1)
        return pd.DataFrame({"class": label, "feature": names, "coefficient": coef[0]})
    rows = []
    for i in range(coef.shape[0]):
        cls = classes[i] if i < len(classes) else i
        for name, value in zip(names, coef[i]):
            rows.append({"class": cls, "feature": name, "coefficient": value})
    return pd.DataFrame(rows)


def intercept(pipe):
    estimator = _estimator(pipe)
    value = getattr(estimator, "intercept_", None)
    return None if value is None else np.asarray(value).tolist()


def predict_response(pipe, frame: pd.DataFrame, task: str = "regression") -> np.ndarray:
    """One number per row: the prediction, or the probability of the last class.

    Using probability-of-the-positive-class for classifiers is what makes a
    dependence curve readable; a decision-function score would be arbitrary.
    """
    if task != "regression" and hasattr(pipe, "predict_proba"):
        try:
            proba = np.asarray(pipe.predict_proba(frame), dtype=float)
            return proba[:, -1] if proba.ndim == 2 else proba.ravel()
        except Exception:
            pass
    return np.asarray(pipe.predict(frame), dtype=float)


def feature_axis(X: pd.DataFrame, feature: str, resolution: int, ranges: dict | None = None) -> np.ndarray:
    """Grid of values to sweep a feature over, clipped to the observed 2–98% range.

    Clipping matters: extrapolating a tree ensemble far outside the data it saw
    produces flat, meaningless plateaus, which would show up as a bogus optimum.
    """
    if ranges and ranges.get(feature) is not None:
        lo, hi = ranges[feature]
    else:
        series = pd.to_numeric(X[feature], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"{feature!r} has no numeric values to sweep")
        lo, hi = float(series.quantile(0.02)), float(series.quantile(0.98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(series.min()), float(series.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, max(3, int(resolution)))


def grid_prediction(
    pipe, X: pd.DataFrame, features: list[str], task: str = "regression",
    resolution: int = 40, sample_rows: int = 250, ranges: dict | None = None,
    return_rows: bool = False,
):
    """Fix one or two features on a grid, predict, average over the rest of the data.

    This is partial dependence computed directly, which keeps the units honest for
    classifiers (probability) and avoids depending on how a particular
    scikit-learn version reshapes its output array.
    """
    if not features:
        raise ValueError("no features were given")
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"not in the data: {missing}")
    sample = X.sample(min(sample_rows, len(X)), random_state=0) if len(X) > sample_rows else X.copy()
    axes = [feature_axis(X, f, resolution, ranges) for f in features]

    if len(features) == 1:
        rows = []
        for value in axes[0]:
            frame = sample.copy()
            frame[features[0]] = value
            rows.append(predict_response(pipe, frame, task))
        matrix = np.vstack(rows)
        mean = matrix.mean(axis=1)
        return axes, mean, (matrix if return_rows else None)

    surface = np.zeros((len(axes[1]), len(axes[0])))
    for i, v1 in enumerate(axes[0]):
        for j, v2 in enumerate(axes[1]):
            frame = sample.copy()
            frame[features[0]] = v1
            frame[features[1]] = v2
            surface[j, i] = float(np.mean(predict_response(pipe, frame, task)))
    return axes, surface, None


def dependence_curve(pipe, X: pd.DataFrame, feature: str, task: str = "regression",
                     grid_resolution: int = 40, sample_rows: int = 250) -> pd.DataFrame:
    """Average prediction as one feature sweeps its range, everything else held as-is."""
    axes, mean, _ = grid_prediction(pipe, X, [feature], task=task,
                                    resolution=grid_resolution, sample_rows=sample_rows)
    frame = pd.DataFrame({feature: axes[0], "prediction": mean})
    frame["effect"] = frame["prediction"] - frame["prediction"].mean()
    frame.attrs["range"] = (float(np.min(mean)), float(np.max(mean)))
    return frame


def dependence_summary(pipe, X: pd.DataFrame, features: list[str], task: str = "regression",
                       grid_resolution: int = 40, sample_rows: int = 250) -> pd.DataFrame:
    """For each feature: the value that maximises the prediction, and the plateau."""
    rows = []
    for feature in features:
        try:
            curve = dependence_curve(pipe, X, feature, task=task,
                                     grid_resolution=grid_resolution, sample_rows=sample_rows)
        except Exception as exc:
            rows.append({"feature": feature, "error": str(exc)})
            continue
        values = curve[feature].to_numpy()
        response = curve["prediction"].to_numpy()
        if not np.isfinite(response).any():
            rows.append({"feature": feature, "error": "no finite predictions"})
            continue
        best_i, worst_i = int(np.nanargmax(response)), int(np.nanargmin(response))
        span = float(np.nanmax(response) - np.nanmin(response))
        near_best = values[response >= np.nanmax(response) - 0.02 * (span or 1)]
        rows.append({
            "feature": feature,
            "maximising value": float(values[best_i]),
            "highest prediction": float(response[best_i]),
            "minimising value": float(values[worst_i]),
            "lowest prediction": float(response[worst_i]),
            "plateau (top 2%)": f"{near_best.min():.4g} – {near_best.max():.4g}" if near_best.size else "",
            "effect size": span,
            "observed range": f"{float(np.nanmin(values)):.4g} – {float(np.nanmax(values)):.4g}",
        })
    frame = pd.DataFrame(rows)
    if "effect size" in frame.columns:
        frame = frame.sort_values("effect size", ascending=False).reset_index(drop=True)
    return frame


def shap_summary(pipe, X: pd.DataFrame, sample_rows: int = 300):
    """Optional SHAP explanation, if the package happens to be installed."""
    try:
        import shap  # noqa: F401
    except Exception:
        return None
    import shap

    frame = X.sample(min(sample_rows, len(X)), random_state=0) if len(X) > sample_rows else X
    prepared = pipe.named_steps["prepare"].transform(frame)
    estimator = _estimator(pipe)
    try:
        explainer = shap.Explainer(estimator, prepared)
        values = explainer(prepared)
    except Exception:
        return None
    frame_out = pd.DataFrame(
        np.abs(np.asarray(values.values)).mean(axis=0),
        index=list(prepared.columns),
        columns=["mean |SHAP|"],
    ).sort_values("mean |SHAP|", ascending=False).reset_index(names="feature")
    return frame_out
