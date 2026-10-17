"""Hyperparameter search.

Written as an explicit candidate loop rather than a call to ``GridSearchCV`` for
two reasons that matter in a GUI: the user can watch each candidate and stop the
run, and every candidate is scored by the same cross-validation code the rest of
the workbench uses, so the number shown in the leaderboard is the number the
evaluate screen will report.

The cost is that candidates run one at a time here.  That is usually not a real
loss: the ensemble models parallelise their own fits internally, and the cheap
models have cheap candidates.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import metrics as M
from . import models as MODELS
from .train import Cancelled, TrainSpec, build_train_pipeline, check_cancel, cross_validate_pipeline


@dataclass
class TuneSpec:
    method: str = "grid"           # grid | random | halving
    n_iter: int = 30
    cv_folds: int = 5
    random_state: int = 42
    metric: str = "r2"
    max_candidates: int = 200
    param_grid: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TuneSpec":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# grid text <-> dict
# --------------------------------------------------------------------------


def _coerce_like(text: str):
    value = text.strip()
    low = value.lower()
    if low in ("none", "null", ""):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_grid_text(text: str) -> dict:
    """Parse the tuning form: one parameter per line, values comma separated.

    ::

        n_estimators: 100, 300, 600
        max_depth: None, 5, 10
    """
    grid: dict[str, list] = {}
    problems = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line and "=" not in line:
            problems.append(f"ignored {line!r} — write 'parameter: value, value'")
            continue
        name, _, values = line.replace("=", ":", 1).partition(":")
        name = name.strip()
        if not name:
            problems.append(f"ignored {line!r} — the parameter name is empty")
            continue
        items = [_coerce_like(v) for v in values.split(",") if v.strip() != ""]
        items = [v for v in items if not (v is None and len(items) > 1)]
        if not items:
            problems.append(f"ignored {name!r} — no values given")
            continue
        grid[name] = items
    return grid, problems


def grid_to_text(grid: dict) -> str:
    lines = []
    for name, values in (grid or {}).items():
        rendered = ", ".join("None" if v is None else str(v) for v in values)
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines)


def candidate_list(grid: dict, method: str, n_iter: int, random_state: int) -> list[dict]:
    """Every combination (grid) or a sampled subset (random)."""
    if not grid:
        return [{}]
    names = list(grid)
    value_lists = [list(grid[n]) for n in names]
    total = int(np.prod([len(v) for v in value_lists]))
    if method == "random" and total > n_iter:
        rng = np.random.RandomState(random_state)
        seen = set()
        candidates = []
        attempts = 0
        while len(candidates) < int(n_iter) and attempts < int(n_iter) * 40:
            attempts += 1
            pick = tuple(rng.choice(len(v)) for v in value_lists)
            if pick in seen:
                continue
            seen.add(pick)
            candidates.append({n: value_lists[i][p] for i, (n, p) in enumerate(zip(names, pick))})
        return candidates
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


# --------------------------------------------------------------------------
# the search itself
# --------------------------------------------------------------------------


def tune(
    model_key: str,
    pre_spec,
    X: pd.DataFrame,
    y,
    spec: TrainSpec,
    tune_spec: TuneSpec,
    weights=None,
    progress=None,
    cancel=None,
) -> dict:
    """Search the space and refit the best candidate on the training data."""
    candidates = candidate_list(tune_spec.param_grid, tune_spec.method, tune_spec.n_iter, tune_spec.random_state)
    if len(candidates) > tune_spec.max_candidates:
        # Keep the search bounded; a forgotten 4-parameter grid can otherwise run for days.
        rng = np.random.RandomState(tune_spec.random_state)
        idx = rng.choice(len(candidates), int(tune_spec.max_candidates), replace=False)
        candidates = [candidates[i] for i in sorted(idx)]
    total = len(candidates)
    cv_spec = TrainSpec(**{**spec.to_dict(), "cv_folds": tune_spec.cv_folds, "primary_metric": tune_spec.metric})

    rows: list[dict] = []
    started = time.time()
    best = None
    for i, params in enumerate(candidates):
        check_cancel(cancel)
        if progress:
            progress(i, total, f"candidate {i + 1}/{total}: " + (MODELS.describe_params(model_key, params) if params else "defaults"))
        try:
            pipe, _ = build_train_pipeline(pre_spec, model_key, params, spec.task)
        except Exception as exc:
            rows.append({**_flat(params), "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            cv = cross_validate_pipeline(pipe, X, y, cv_spec, weights=weights)
        except Cancelled:
            raise
        except Exception as exc:
            rows.append({**_flat(params), "error": f"{type(exc).__name__}: {exc}"})
            continue
        summary = cv["summary"]
        row = {**_flat(params)}
        if summary:
            for key, (mean, std) in summary.items():
                row[f"mean {M.label(spec.task, key)}"] = mean
                row[f"std {M.label(spec.task, key)}"] = std
        row["score"] = summary.get(tune_spec.metric, (float("nan"), 0.0))[0]
        row["score std"] = summary.get(tune_spec.metric, (float("nan"), 0.0))[1]
        row["_params"] = dict(params)
        row["_cv"] = cv
        rows.append(row)

        better = _is_better(row["score"], None if best is None else best["score"], spec.task, tune_spec.metric)
        if better:
            best = row

    elapsed = time.time() - started
    scored = [r for r in rows if "_cv" in r]
    table = _results_table(rows, spec.task, tune_spec.metric)

    result = {
        "method": tune_spec.method,
        "metric": tune_spec.metric,
        "metric_label": M.label(spec.task, tune_spec.metric),
        "n_candidates": total,
        "n_scored": len(scored),
        "elapsed": elapsed,
        "results": table,
        "failures": [r.get("error") for r in rows if r.get("error")],
    }
    if best is None:
        result["best_params"] = {}
        result["best_score"] = float("nan")
        result["best_estimator"] = None
        result["best_cv"] = None
        result["note"] = "no candidate could be scored — check the failure messages"
        return result

    result["best_params"] = best["_params"]
    result["best_score"] = best["score"]
    result["best_cv"] = best["_cv"]
    result["best_std"] = best.get("score std")
    # Refit the winner on the training split so it can be saved and evaluated.
    pipe, _ = build_train_pipeline(pre_spec, model_key, best["_params"], spec.task)
    if spec.task == "unsupervised":
        pipe.fit(X)
    else:
        try:
            pipe.fit(X, y)
        except Exception as exc:
            result["note"] = f"the best candidate could not be refit: {exc}"
            result["best_estimator"] = None
            return result
    result["best_estimator"] = pipe
    return result


def _flat(params: dict) -> dict:
    if not params:
        return {"parameters": "defaults"}
    return {f"param {k}": v for k, v in params.items()}


def _is_better(value, incumbent, task: str, metric: str) -> bool:
    if value is None or not np.isfinite(value):
        return False
    if incumbent is None or not np.isfinite(incumbent):
        return True
    return value > incumbent if M.higher_is_better(task, metric) else value < incumbent


def _results_table(rows: list[dict], task: str, metric: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    label = f"mean {M.label(task, metric)}"
    if label in frame.columns:
        frame = frame.sort_values(label, ascending=not M.higher_is_better(task, metric)).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def display_columns(frame: pd.DataFrame, task: str, metric: str, limit: int = 14) -> list[str]:
    """The columns worth putting on screen: the parameters, then the chosen metric.

    A full cross-validation table has forty columns (every metric × mean/std);
    showing all of them buries the one column the user is searching for.
    """
    if frame is None or not len(frame.columns):
        return []
    params = [c for c in frame.columns if c.startswith("param ") or c == "parameters"]
    metric_label = f"mean {M.label(task, metric)}"
    clean_label = M.label(task, metric)
    wanted = [c for c in ("rank", *params) if c in frame.columns]
    for candidate in (metric_label, clean_label, "score"):
        if candidate in frame.columns and candidate not in wanted:
            wanted.append(candidate)
    for candidate in (f"std {M.label(task, metric)}", f"± {M.label(task, metric)}"):
        if candidate in frame.columns:
            wanted.append(candidate)
    for nice in ("mean RMSE", "mean MAE", "mean Accuracy", "mean F1 (weighted)", "seconds"):
        if nice in frame.columns and nice not in wanted:
            wanted.append(nice)
    if "error" in frame.columns:
        wanted.append("error")
    return wanted[:limit] if wanted else list(frame.columns)[:limit]


def comparison_table(model_keys: list[str], pre_spec, X, y, spec: TrainSpec, weights=None,
                     progress=None, cancel=None) -> pd.DataFrame:
    """Score several models with identical folds — the leaderboard for 'compare models'."""
    rows = []
    total = len(model_keys)
    per_model_time = np.zeros(total)
    for i, key in enumerate(model_keys):
        check_cancel(cancel)
        if progress:
            progress(i, total, f"{MODELS.get_model_spec(key).label} ({i + 1}/{total})")
        started = time.time()
        row = {"model": MODELS.get_model_spec(key).label, "key": key}
        try:
            pipe, _ = build_train_pipeline(pre_spec, key, {}, spec.task)
            cv = cross_validate_pipeline(pipe, X, y, spec, weights=weights)
            for metric_key, (mean, std) in (cv.get("summary") or {}).items():
                row[M.label(spec.task, metric_key)] = mean
                row[f"± {M.label(spec.task, metric_key)}"] = std
            row["_cv"] = cv
        except Cancelled:
            raise
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        per_model_time[i] = time.time() - started
        row["seconds"] = round(per_model_time[i], 2)
        rows.append(row)
    frame = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    label = M.label(spec.task, spec.primary_metric)
    if label in frame.columns:
        frame = frame.sort_values(label, ascending=not M.higher_is_better(spec.task, spec.primary_metric))
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame.attrs["cv"] = {r["key"]: r.get("_cv") for r in rows}
    return frame
