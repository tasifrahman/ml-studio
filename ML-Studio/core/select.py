"""Finding the best *combination of columns* to model with.

Two families live here:

* **embedded ranking** — score every column once (univariate statistics,
  permutation importance, or RFE on the encoded matrix) and rank them.
  Cheap, and usually right.
* **subset search** — actually re-fit the model on candidate subsets
  (forward, backward, exhaustive).  Expensive per candidate, but it is the only
  honest way to answer "which columns give the best score", because it measures
  the interaction between columns rather than each column alone.

Every method reports how many model fits it spent, and stops at a budget rather
than quietly running for hours.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from . import models as MODELS
from .train import Cancelled, TrainSpec, build_train_pipeline, check_cancel, cross_validate_pipeline

SEARCH_METHODS = {
    "forward": "Forward selection",
    "backward": "Backward elimination",
    "exhaustive": "Exhaustive subset search",
    "univariate": "Rank by univariate statistics",
    "importance": "Rank by model importance",
    "rfe": "Recursive feature elimination (encoded)",
}


@dataclass
class SelectSpec:
    method: str = "forward"
    metric: str = "r2"
    k: int = 8
    max_features: int | None = None
    min_delta: float = 0.0
    max_evaluations: int = 120
    cv_folds: int = 5
    score_func: str = "auto"
    n_repeats: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SelectSpec":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SelectResult:
    method: str
    metric: str
    best: dict = field(default_factory=dict)
    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_evaluated: int = 0
    elapsed: float = 0.0
    note: str = ""


def _subset_score(df, columns, spec: TrainSpec, pre_spec, model_key, model_params,
                  metric: str, folds: int, weights=None):
    """Fit + cross-validate the model on one candidate subset of columns."""
    if not columns:
        return float("nan"), float("nan")
    sub_spec = TrainSpec(**{**spec.to_dict(), "features": list(columns), "primary_metric": metric, "cv_folds": folds})
    pipe, _ = build_train_pipeline(pre_spec, model_key, model_params, spec.task)
    cv = cross_validate_pipeline(pipe, df[list(columns)], df[spec.target], sub_spec, weights=weights)
    mean, std = (cv.get("summary") or {}).get(metric, (float("nan"), float("nan")))
    return mean, std


def _better(value, incumbent, task, metric):
    if value is None or not np.isfinite(value):
        return False
    if incumbent is None or not np.isfinite(incumbent):
        return True
    return value > incumbent if M.higher_is_better(task, metric) else value < incumbent


def forward_selection(df, spec, pre_spec, model_key, model_params, sel: SelectSpec, weights=None,
                      progress=None, cancel=None) -> SelectResult:
    candidates = [c for c in (spec.features or df.columns) if c in df.columns and c != spec.target]
    max_features = sel.max_features or len(candidates)
    chosen: list[str] = []
    remaining = list(candidates)
    history = []
    evaluated = 0
    best_score = None
    started = time.time()
    while remaining and len(chosen) < max_features and evaluated < sel.max_evaluations:
        check_cancel(cancel)
        round_rows = []
        for column in remaining:
            if evaluated >= sel.max_evaluations:
                break
            subset = chosen + [column]
            if progress:
                progress(evaluated, sel.max_evaluations, f"trying {len(subset)} columns: +{column}")
            try:
                mean, std = _subset_score(df, subset, spec, pre_spec, model_key, model_params,
                                          sel.metric, sel.cv_folds, weights)
            except Cancelled:
                raise
            except Exception as exc:
                mean, std = float("nan"), float("nan")
                round_rows.append({"step": len(chosen) + 1, "added": column, "score": mean,
                                   "std": std, "columns": ", ".join(subset), "error": str(exc)})
                evaluated += 1
                continue
            round_rows.append({"step": len(chosen) + 1, "added": column, "score": mean, "std": std,
                               "columns": ", ".join(subset)})
            evaluated += 1
        history.extend(round_rows)
        scored = [r for r in round_rows if np.isfinite(r.get("score", np.nan))]
        if not scored:
            break
        winner = max(scored, key=lambda r: r["score"]) if M.higher_is_better(spec.task, sel.metric) else min(scored, key=lambda r: r["score"])
        improvement = 0.0 if best_score is None else (
            winner["score"] - best_score if M.higher_is_better(spec.task, sel.metric) else best_score - winner["score"]
        )
        if best_score is not None and improvement <= sel.min_delta:
            break
        chosen.append(winner["added"])
        remaining.remove(winner["added"])
        best_score = winner["score"]
    return _finish("forward", sel, chosen, best_score, history, evaluated, started, spec.task,
                   note="" if evaluated < sel.max_evaluations else
                   f"stopped at the budget of {sel.max_evaluations} model evaluations")


def backward_elimination(df, spec, pre_spec, model_key, model_params, sel: SelectSpec, weights=None,
                         progress=None, cancel=None) -> SelectResult:
    current = [c for c in (spec.features or df.columns) if c in df.columns and c != spec.target]
    history = []
    evaluated = 0
    started = time.time()
    floor = sel.max_features or 1
    best_score = None
    while len(current) > floor and evaluated < sel.max_evaluations:
        check_cancel(cancel)
        rows = []
        for column in list(current):
            if evaluated >= sel.max_evaluations:
                break
            subset = [c for c in current if c != column]
            if not subset:
                continue
            if progress:
                progress(evaluated, sel.max_evaluations, f"{len(current)} columns: dropping {column}")
            try:
                mean, std = _subset_score(df, subset, spec, pre_spec, model_key, model_params,
                                          sel.metric, sel.cv_folds, weights)
            except Cancelled:
                raise
            except Exception as exc:
                rows.append({"step": len(current), "removed": column, "score": float("nan"),
                             "std": float("nan"), "columns": ", ".join(subset), "error": str(exc)})
                evaluated += 1
                continue
            rows.append({"step": len(current), "removed": column, "score": mean, "std": std,
                         "columns": ", ".join(subset)})
            evaluated += 1
        history.extend(rows)
        scored = [r for r in rows if np.isfinite(r.get("score", np.nan))]
        if not scored:
            break
        winner = max(scored, key=lambda r: r["score"]) if M.higher_is_better(spec.task, sel.metric) else min(scored, key=lambda r: r["score"])
        # Only drop a column when doing so does not hurt (min_delta tolerates noise).
        if best_score is not None:
            change = winner["score"] - best_score if M.higher_is_better(spec.task, sel.metric) else best_score - winner["score"]
            if change < -abs(sel.min_delta):
                break
        best_score = winner["score"]
        current.remove(winner["removed"])
    return _finish("backward", sel, current, best_score, history, evaluated, started, spec.task,
                   note="" if evaluated < sel.max_evaluations else
                   f"stopped at the budget of {sel.max_evaluations} model evaluations")


def exhaustive_search(df, spec, pre_spec, model_key, model_params, sel: SelectSpec, weights=None,
                      progress=None, cancel=None) -> SelectResult:
    candidates = [c for c in (spec.features or df.columns) if c in df.columns and c != spec.target]
    max_features = min(sel.max_features or len(candidates), len(candidates))
    subsets = []
    for size in range(1, max_features + 1):
        subsets.extend(itertools.combinations(candidates, size))
    note = ""
    if len(subsets) > sel.max_evaluations:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(subsets), sel.max_evaluations, replace=False)
        subsets = [subsets[i] for i in sorted(idx)]
        note = (f"{len(subsets)} of {sum(1 for _ in itertools.chain.from_iterable(itertools.combinations(candidates, s) for s in range(1, max_features + 1)))} "
                f"subsets were sampled — raise the budget to try them all")
    history = []
    started = time.time()
    for i, subset in enumerate(subsets):
        check_cancel(cancel)
        if progress:
            progress(i, len(subsets), f"{len(subset)} columns: {', '.join(subset)[:70]}")
        try:
            mean, std = _subset_score(df, list(subset), spec, pre_spec, model_key, model_params,
                                      sel.metric, sel.cv_folds, weights)
        except Cancelled:
            raise
        except Exception as exc:
            history.append({"columns": ", ".join(subset), "n": len(subset), "score": float("nan"),
                            "std": float("nan"), "error": str(exc)})
            continue
        history.append({"columns": ", ".join(subset), "n": len(subset), "score": mean, "std": std})
    scored = [r for r in history if np.isfinite(r.get("score", np.nan))]
    if not scored:
        return _finish("exhaustive", sel, [], float("nan"), history, len(history), started, spec.task,
                       note=note or "nothing scored")
    winner = max(scored, key=lambda r: r["score"]) if M.higher_is_better(spec.task, sel.metric) else min(scored, key=lambda r: r["score"])
    chosen = [c.strip() for c in winner["columns"].split(",")]
    return _finish("exhaustive", sel, chosen, winner["score"], history, len(history), started, spec.task, note=note)


def rank_univariate(df, spec, sel: SelectSpec) -> SelectResult:
    """One pass of statistics: how much does each column know about the target on its own?"""
    from sklearn.feature_selection import (chi2, f_classif, f_regression, mutual_info_classif,
                                           mutual_info_regression)

    target = spec.target
    if not target or target not in df.columns:
        raise ValueError("univariate ranking needs a target column")
    frame = df[[c for c in df.columns if c != target]]
    y = df[target]
    numeric = frame.select_dtypes(include=[np.number])
    categorical = frame.drop(columns=numeric.columns)

    rows = []
    fname = {"auto": "mutual_info_regression" if spec.task == "regression" else "mutual_info_classif",
             }.get(sel.score_func, sel.score_func)
    if spec.task == "regression":
        scorers = {"f_regression": f_regression, "mutual_info_regression": mutual_info_regression}
        if fname not in scorers:
            fname = "mutual_info_regression"
    else:
        scorers = {"f_classif": f_classif, "chi2": chi2, "mutual_info_classif": mutual_info_classif}
        if fname not in scorers:
            fname = "mutual_info_classif"

    if len(numeric.columns):
        matrix = numeric.fillna(numeric.median()).to_numpy(dtype=float)
        try:
            if fname == "chi2":
                shifted = matrix - matrix.min(axis=0)
                scores, _ = chi2(shifted, y)
            else:
                scores, _ = scorers[fname](matrix, y)
            scores = np.nan_to_num(scores)
            for name, score in zip(numeric.columns, scores):
                rows.append({"feature": name, "score": float(score), "method": fname, "type": "numeric"})
        except Exception as exc:
            for name in numeric.columns:
                rows.append({"feature": name, "score": np.nan, "method": fname, "type": "numeric", "error": str(exc)})

    # categorical columns, encoded one at a time
    for name in categorical.columns:
        try:
            dummies = pd.get_dummies(categorical[name].astype(str), prefix=name)
            if fname in ("f_regression", "mutual_info_regression"):
                scores, _ = f_regression(dummies.to_numpy(dtype=float), y)
            else:
                scores, _ = scorers[fname](dummies.to_numpy(dtype=float), y)
            rows.append({"feature": name, "score": float(np.nanmax(scores)), "method": fname + " (one-hot)", "type": "categorical"})
        except Exception as exc:
            rows.append({"feature": name, "score": np.nan, "method": fname, "type": "categorical", "error": str(exc)})

    frame_out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    frame_out.insert(0, "rank", np.arange(1, len(frame_out) + 1))
    chosen = list(frame_out["feature"].head(max(1, sel.k)))
    return SelectResult(method="univariate", metric=sel.metric,
                        best={"columns": chosen, "score": float("nan")},
                        ranking=frame_out, history=pd.DataFrame(), n_evaluated=0,
                        note=f"ranked by {fname}; no model was fitted")


def rank_importance(df, spec, pre_spec, model_key, model_params, sel: SelectSpec) -> SelectResult:
    """Fit once and rank columns by the drop in score when each is shuffled."""
    from . import explain

    pipe, _ = build_train_pipeline(pre_spec, model_key, model_params, spec.task)
    X = df[[c for c in (spec.features or df.columns) if c in df.columns and c != spec.target]]
    y = df[spec.target]
    pipe.fit(X, y)
    scoring = M.scorer_name(spec.task, sel.metric) or None
    frame = explain.permutation_importance_frame(pipe, X, y, spec.task, scoring=scoring,
                                                 n_repeats=sel.n_repeats)
    chosen = list(frame["feature"].head(max(1, sel.k)))
    return SelectResult(method="importance", metric=sel.metric,
                        best={"columns": chosen, "score": float("nan")},
                        ranking=frame, history=pd.DataFrame(), n_evaluated=1,
                        note=f"permutation importance with {sel.n_repeats} shuffles per column")


def rfe_encoded(df, spec, pre_spec, model_key, model_params, sel: SelectSpec) -> SelectResult:
    """RFE on the *encoded* matrix — note that one original column can produce many."""
    from sklearn.feature_selection import RFE

    pipe, _ = build_train_pipeline(pre_spec, model_key, model_params, spec.task)
    X = df[[c for c in (spec.features or df.columns) if c in df.columns and c != spec.target]]
    y = df[spec.target]
    prep = pipe.named_steps["prepare"]
    prepared = prep.fit_transform(X, y)
    estimator = MODELS.build_model(model_key, model_params)
    n_features = max(1, min(int(sel.k), prepared.shape[1]))
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(prepared, y)
    frame = pd.DataFrame({
        "feature": list(prepared.columns),
        "rank": rfe.ranking_,
        "selected": rfe.support_,
    }).sort_values(["selected", "rank"], ascending=[False, True]).reset_index(drop=True)
    selected = list(frame.loc[frame["selected"], "feature"])
    return SelectResult(method="rfe", metric=sel.metric,
                        best={"columns": selected, "score": float("nan")},
                        ranking=frame, history=pd.DataFrame(), n_evaluated=1,
                        note=f"selected {len(selected)} of {prepared.shape[1]} encoded features "
                             f"(these are post-encoding names, not raw columns)")


def _finish(method, sel, chosen, score, history, evaluated, started, task, note=""):
    frame = pd.DataFrame(history)
    if not frame.empty and "score" in frame.columns:
        metric_label = f"mean {M.label(task, sel.metric)}"
        frame = frame.rename(columns={"score": metric_label})
        frame = frame.sort_values(metric_label, ascending=not M.higher_is_better(task, sel.metric))
        frame = frame.reset_index(drop=True)
    return SelectResult(
        method=method, metric=sel.metric,
        best={"columns": list(chosen), "score": float(score) if score is not None else float("nan"),
              "n_features": len(chosen)},
        ranking=frame, history=frame, n_evaluated=evaluated,
        elapsed=time.time() - started, note=note,
    )


def run_selection(df, spec: TrainSpec, pre_spec, model_key, model_params, sel: SelectSpec,
                  weights=None, progress=None, cancel=None) -> SelectResult:
    """Dispatch to the chosen selection method."""
    if spec.task == "unsupervised":
        raise ValueError("feature selection needs a supervised target")
    method = sel.method
    if method == "forward":
        return forward_selection(df, spec, pre_spec, model_key, model_params, sel, weights, progress, cancel)
    if method == "backward":
        return backward_elimination(df, spec, pre_spec, model_key, model_params, sel, weights, progress, cancel)
    if method == "exhaustive":
        return exhaustive_search(df, spec, pre_spec, model_key, model_params, sel, weights, progress, cancel)
    if method == "univariate":
        return rank_univariate(df, spec, sel)
    if method == "importance":
        return rank_importance(df, spec, pre_spec, model_key, model_params, sel)
    if method == "rfe":
        return rfe_encoded(df, spec, pre_spec, model_key, model_params, sel)
    raise ValueError(f"unknown selection method {method!r}")


def quick_recommendation(df, spec, pre_spec, model_key, model_params, sel: SelectSpec) -> pd.DataFrame:
    """A cheap combined view: each column's univariate rank next to nothing else.

    Deliberately not a subset search — it is the 'give me a first read' button.
    """
    result = rank_univariate(df, spec, sel)
    return result.ranking
