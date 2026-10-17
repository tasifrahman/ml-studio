"""The four 'best combination' tools.

1. :func:`best_value_per_field` — for each input, the value that drives the
   prediction highest or lowest (a dependence sweep).
2. :func:`inverse_design` — given a target outcome, solve for the input values
   that reach it, inside physical bounds and any constraints you write.
3. :func:`cleaning_combo_search` — which cleaning recipe scores best.
4. :func:`feature_combo_search` — thin wrapper over :mod:`core.select`, exposed
   here so all four sit behind one screen.

The inverse-design search is the one that most easily produces nonsense, so it
refuses to extrapolate: every candidate is clipped into the observed data range,
constraints are enforced before scoring, and the result table reports how far the
best candidate sits from the data the model was trained on.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import explain, formula
from . import metrics as M
from .preprocess import PreprocessSpec
from .train import Cancelled, TrainSpec, build_train_pipeline, check_cancel, cross_validate_pipeline

OBJECTIVES = ["match a target", "maximise", "minimise"]
METHODS = ["random sampling", "grid sweep", "genetic algorithm", "differential evolution", "local search (multi-start)"]
CLEAN_METHODS = ["forward", "backward", "exhaustive", "univariate", "importance", "rfe"]


@dataclass
class DesignSpec:
    bounds: dict = field(default_factory=dict)        # feature -> [low, high]
    fixed: dict = field(default_factory=dict)         # feature -> value
    target_value: float | None = None
    objective: str = "match a target"
    method: str = "random sampling"
    n_samples: int = 2000
    levels: int = 12
    n_iter: int = 30
    population: int = 40
    constraints: list = field(default_factory=list)
    cost_weights: dict = field(default_factory=dict)
    cost_penalty: float = 0.0
    seed: int = 42
    top_n: int = 25
    allow_extrapolation: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "DesignSpec":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DesignResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)
    feasible_count: int = 0
    sampled_count: int = 0
    note: str = ""
    elapsed: float = 0.0


# --------------------------------------------------------------------------
# 1. optimum value per field
# --------------------------------------------------------------------------


def best_value_per_field(pipe, X: pd.DataFrame, task: str, features: list[str] | None = None,
                         grid_resolution: int = 40, sample_rows: int = 200) -> pd.DataFrame:
    """Sweep each feature and report the value that maximises the prediction."""
    if features is None:
        features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return explain.dependence_summary(pipe, X, list(features), task=task,
                                      grid_resolution=grid_resolution, sample_rows=sample_rows)


# --------------------------------------------------------------------------
# 2. inverse design
# --------------------------------------------------------------------------


def observed_ranges(X: pd.DataFrame, features: list[str]) -> dict:
    ranges = {}
    for name in features:
        series = pd.to_numeric(X[name], errors="coerce").dropna()
        if series.empty:
            continue
        ranges[name] = [float(series.min()), float(series.max())]
    return ranges


def _clip_bounds(bounds: dict, observed: dict, allow_extrapolation: bool) -> dict:
    out = {}
    for name, (lo, hi) in bounds.items():
        lo, hi = float(lo), float(hi)
        if lo > hi:
            lo, hi = hi, lo
        if not allow_extrapolation and name in observed:
            lo = max(lo, observed[name][0])
            hi = min(hi, observed[name][1])
            if lo > hi:  # the user's window sits entirely outside the data
                lo, hi = observed[name]
        out[name] = (lo, hi)
    return out


def _base_frame(X: pd.DataFrame, fixed: dict, n: int) -> pd.DataFrame:
    """A frame of n identical rows: median/mode for every column, so nothing arrives missing."""
    if len(X):
        base = X.sample(min(len(X), 200), random_state=0).iloc[0]
    else:
        base = pd.Series(dtype="float64")
    frame = pd.DataFrame([dict(base)] * n)
    for column in frame.columns:
        if not frame[column].isna().any():
            continue
        series = X[column] if column in X.columns else None
        if series is None:
            continue
        fill = series.median() if pd.api.types.is_numeric_dtype(series) else (series.mode().iloc[0] if len(series.mode()) else None)
        frame[column] = frame[column].fillna(fill)
    for name, value in (fixed or {}).items():
        if name in frame.columns:
            frame[name] = value
    return frame


def _loss(pred: np.ndarray, spec: DesignSpec, frame: pd.DataFrame, source_ranges: dict) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if spec.objective == "maximise":
        base = -pred
    elif spec.objective == "minimise":
        base = pred
    else:
        if spec.target_value is None:
            raise ValueError("'match a target' needs a target value")
        base = np.abs(pred - float(spec.target_value))
    # normalise so the cost penalty is comparable across objectives
    scale = np.nanstd(base)
    loss = base / (scale if scale and np.isfinite(scale) else 1.0)
    if spec.cost_weights and spec.cost_penalty:
        cost = np.zeros(len(frame))
        for name, weight in spec.cost_weights.items():
            if name in frame.columns:
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
                ref = source_ranges.get(name)
                span = (ref[1] - ref[0]) if ref and ref[1] > ref[0] else 1.0
                cost += float(weight) * np.nan_to_num(values) / span
        loss = loss + float(spec.cost_penalty) * cost
    return np.nan_to_num(loss, nan=np.inf)


def _sample_random(rng, bounds: dict, n: int) -> pd.DataFrame:
    data = {name: rng.uniform(lo, hi, size=n) for name, (lo, hi) in bounds.items()}
    return pd.DataFrame(data)


def _sample_grid(rng, bounds: dict, levels: int, cap: int) -> pd.DataFrame:
    axes = {name: np.linspace(lo, hi, max(2, int(levels))) for name, (lo, hi) in bounds.items()}
    combos = list(itertools.product(*axes.values()))
    if len(combos) > cap:
        idx = rng.choice(len(combos), cap, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    return pd.DataFrame(combos, columns=list(axes))


def _optimise_local(objective, bounds: dict, seed: int):
    from scipy.optimize import minimize

    names = list(bounds)
    x0 = np.array([(bounds[n][0] + bounds[n][1]) / 2 for n in names], dtype=float)
    scipy_bounds = [bounds[n] for n in names]
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(6):  # multi-start: a single start lands in local minima on tree models
        start = np.array([rng.uniform(lo, hi) for lo, hi in scipy_bounds])
        try:
            result = minimize(objective, start, method="Nelder-Mead", bounds=scipy_bounds,
                              options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6})
        except Exception:
            continue
        if result.success or np.isfinite(result.fun):
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        return pd.DataFrame([x0], columns=names)
    return pd.DataFrame([np.clip(best.x, [b[0] for b in scipy_bounds], [b[1] for b in scipy_bounds])], columns=names)


def _optimise_de(objective, bounds: dict, spec: DesignSpec):
    from scipy.optimize import differential_evolution

    names = list(bounds)
    result = differential_evolution(
        objective, [bounds[n] for n in names], seed=spec.seed,
        maxiter=max(5, int(spec.n_iter)), popsize=max(5, int(spec.population) // 4),
        tol=1e-7, polish=True, init="sobol",
    )
    return pd.DataFrame([result.x], columns=names)


def _optimise_genetic(evaluate, bounds: dict, spec: DesignSpec, rng, cancel=None, progress=None):
    """A small real-coded genetic algorithm: tournament selection, blend crossover,
    Gaussian mutation, elitism.  Simple on purpose — it has to be readable."""
    names = list(bounds)
    low = np.array([bounds[n][0] for n in names], dtype=float)
    high = np.array([bounds[n][1] for n in names], dtype=float)
    population = max(8, int(spec.population))
    generations = max(5, int(spec.n_iter))
    population_arr = rng.uniform(low, high, size=(population, len(names)))
    fitness = None
    history = []
    for generation in range(generations):
        check_cancel(cancel)
        if progress:
            progress(generation, generations, f"generation {generation + 1}/{generations}")
        fitness = evaluate(population_arr)
        order = np.argsort(fitness)
        best = population_arr[order[0]]
        history.append({"generation": generation + 1, "best loss": float(fitness[order[0]]),
                        "median loss": float(np.median(fitness))})
        elite = population_arr[order[: max(2, population // 8)]]
        children = [elite]
        while sum(len(c) for c in children) < population:
            parents = []
            for _ in range(2):
                contenders = rng.choice(population, size=3, replace=False)
                parents.append(population_arr[contenders[np.argmin(fitness[contenders])]])
            alpha = rng.uniform(-0.15, 1.15, size=len(names))
            child = parents[0] + alpha * (parents[1] - parents[0])
            mutate = rng.random(len(names)) < 0.25
            child[mutate] += rng.normal(0, 0.08, size=mutate.sum()) * (high - low)[mutate]
            children.append(np.clip(child, low, high).reshape(1, -1))
        population_arr = np.vstack(children)[:population]
    final = evaluate(population_arr)
    order = np.argsort(final)
    frame = pd.DataFrame([population_arr[order[0]]], columns=names)
    frame.attrs["history"] = pd.DataFrame(history)
    frame.attrs["population"] = pd.DataFrame(population_arr, columns=names)
    frame.attrs["fitness"] = final
    return frame


def inverse_design(pipe, X: pd.DataFrame, task: str, spec: DesignSpec, progress=None,
                   cancel=None) -> DesignResult:
    """Search the input space for designs that produce (or beat) a target outcome."""
    started = time.time()
    if not spec.bounds:
        raise ValueError("no design variables were given bounds")
    missing = [name for name in spec.bounds if name not in X.columns]
    if missing:
        raise ValueError(f"these design variables are not columns in the data: {missing}")

    source_ranges = observed_ranges(X, list(spec.bounds))
    bounds = _clip_bounds(spec.bounds, source_ranges, spec.allow_extrapolation)
    non_numeric = [n for n in bounds if not pd.api.types.is_numeric_dtype(X[n])]
    if non_numeric:
        raise ValueError(
            "only numeric columns can be searched — these are categorical: " + ", ".join(non_numeric)
        )

    constraints = [c for c in (spec.constraints or []) if str(c).strip()]
    rng = np.random.RandomState(spec.seed)

    def assemble(frame: pd.DataFrame) -> pd.DataFrame:
        """Merge a candidate set with the held-constant columns, in the model's own order."""
        out = _base_frame(X, spec.fixed, len(frame))
        for name in frame.columns:
            out[name] = np.asarray(frame[name], dtype=float)
        return out[X.columns]

    def predict(frame: pd.DataFrame) -> np.ndarray:
        prepared = assemble(frame)
        return explain.predict_response(pipe, prepared, task)

    def loss_of(frame: pd.DataFrame) -> np.ndarray:
        return _loss(predict(frame), spec, frame, source_ranges)

    method = spec.method
    candidates = pd.DataFrame()
    history = None

    if method == "random sampling":
        candidates = _sample_random(rng, bounds, int(spec.n_samples))
    elif method == "grid sweep":
        if len(bounds) > 6:
            raise ValueError(f"a grid over {len(bounds)} variables explodes — use random sampling instead")
        candidates = _sample_grid(rng, bounds, spec.levels, int(max(200, spec.n_samples)))
    elif method == "genetic algorithm":
        def evaluate_matrix(matrix: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(matrix, columns=list(bounds))
            return loss_of(frame)

        top = _optimise_genetic(evaluate_matrix, bounds, spec, rng, cancel, progress)
        history = top.attrs.get("history")
        population = top.attrs.get("population")
        candidates = pd.concat([top, population], ignore_index=True) if population is not None else top
    elif method in ("differential evolution", "local search (multi-start)"):
        names = list(bounds)

        def vector_loss(vector: np.ndarray, penalty: float = 0.0) -> float:
            frame = pd.DataFrame([vector], columns=names)
            if constraints:
                mask = formula.constraint_mask(assemble(frame), constraints)
                if not bool(np.all(mask)):
                    # a soft wall keeps the search moving while still refusing the design
                    return float(loss_of(frame)[0]) + 1e3 * float(np.sum(~mask))
            return float(loss_of(frame)[0])

        if method == "differential evolution":
            top = _optimise_de(lambda vector, p=0: vector_loss(vector), bounds, spec)
        else:
            top = _optimise_local(lambda vector: vector_loss(vector), bounds, spec.seed)
        candidates = top
    else:
        raise ValueError(f"unknown search method {method!r}")

    if candidates.empty:
        return DesignResult(note="the search produced no candidates", elapsed=time.time() - started)

    # Score every candidate that was produced.
    prepared = assemble(candidates)
    predictions = explain.predict_response(pipe, prepared, task)
    losses = _loss(predictions, spec, candidates, source_ranges)
    feasible = np.ones(len(candidates), dtype=bool)
    if constraints:
        feasible = formula.constraint_mask(prepared, constraints)

    table = candidates.copy()
    table["predicted"] = predictions
    table["loss"] = losses
    table["feasible"] = feasible
    if spec.target_value is not None and spec.objective == "match a target":
        table["difference from target"] = predictions - float(spec.target_value)
        table["absolute error"] = np.abs(table["difference from target"])
        denom = np.abs(float(spec.target_value)) or 1.0
        table["relative error %"] = 100 * table["absolute error"] / denom
    if spec.cost_weights and spec.cost_penalty:
        cost = np.zeros(len(candidates))
        for name, weight in spec.cost_weights.items():
            if name not in candidates.columns:
                continue
            reference = source_ranges.get(name)
            span = (reference[1] - reference[0]) if reference and reference[1] > reference[0] else 1.0
            values = pd.to_numeric(candidates[name], errors="coerce").to_numpy(dtype=float)
            cost += float(weight) * np.nan_to_num(values) / span
        table["cost"] = cost

    # Keep the shortlist informative: drop near-duplicates so 25 rows are not one design.
    ranked = table[feasible].sort_values("loss") if feasible.any() else table.sort_values("loss")
    ranked = ranked.reset_index(drop=True)
    shortlist = _dedupe(ranked, list(bounds), spec.top_n)

    note_parts = []
    if not feasible.any() and constraints:
        note_parts.append("no candidate satisfied every constraint — relax one, or widen a bound")
    if not spec.allow_extrapolation:
        note_parts.append("values were held inside the range seen in the training data")
    if method in ("differential evolution", "local search (multi-start)"):
        note_parts.append("this method returns a single optimum — use random sampling or the genetic "
                          "algorithm when you want a shortlist of alternatives")
    if spec.fixed:
        note_parts.append("held constant: " + ", ".join(f"{k}={v}" for k, v in spec.fixed.items()))

    summary = {
        "method": method,
        "objective": spec.objective,
        "candidates sampled": int(len(table)),
        "feasible candidates": int(feasible.sum()),
        "best prediction": float(ranked["predicted"].iloc[0]) if len(ranked) else float("nan"),
        "best loss": float(ranked["loss"].iloc[0]) if len(ranked) else float("nan"),
        "design variables": len(bounds),
        "seconds": round(time.time() - started, 2),
    }
    if history is not None and len(history):
        summary["generations"] = int(len(history))

    return DesignResult(table=shortlist, summary=summary,
                        feasible_count=int(feasible.sum()), sampled_count=int(len(table)),
                        note="; ".join(note_parts), elapsed=time.time() - started)


def _dedupe(frame: pd.DataFrame, columns: list[str], top_n: int, tolerance: float = 0.01) -> pd.DataFrame:
    """Keep only designs that differ meaningfully from the ones already kept.

    Rounded-exact deduplication is not enough: a genetic algorithm converges onto
    one point, so twenty-five "different" rows would all read 467.51 / 162.34 and
    the shortlist would be useless.  Two rows count as the same design when every
    variable is within `tolerance` of its observed span.
    """
    if not len(frame):
        return frame
    spans = {}
    for name in columns:
        column = pd.to_numeric(frame[name], errors="coerce")
        span = float(column.max() - column.min())
        spans[name] = span if span > 0 else 1.0
    kept: list[int] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        duplicate = False
        for j in kept:
            other = frame.iloc[j]
            same = True
            for name in columns:
                if abs(float(row[name]) - float(other[name])) > tolerance * spans[name]:
                    same = False
                    break
            if same:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
        if len(kept) >= max(1, top_n):
            break
    return frame.iloc[kept].reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. cleaning recipe search
# --------------------------------------------------------------------------


def cleaning_combo_search(df: pd.DataFrame, spec: TrainSpec, base_spec: PreprocessSpec, model_key: str,
                          model_params: dict, impute_options=("median", "mean", "knn"),
                          outlier_options=("none", "clip_iqr", "winsorize"),
                          scale_options=("standard", "none"), max_combos: int = 40,
                          weights=None, progress=None, cancel=None) -> pd.DataFrame:
    """Score each cleaning recipe with the same model and the same folds."""
    from .train import resolve_features

    features = resolve_features(df, spec)
    if not features:
        raise ValueError("no feature columns to work with")
    if spec.target and spec.target in df.columns:
        modelling = df[features + [spec.target]].dropna(subset=[spec.target])
        target = modelling[spec.target]
    else:
        modelling = df[features]
        target = None

    combos = list(itertools.product(impute_options, outlier_options, scale_options))[: max(1, int(max_combos))]
    rows = []
    started = time.time()
    for i, (impute, outlier, scale) in enumerate(combos):
        check_cancel(cancel)
        if progress:
            progress(i, len(combos), f"recipe {i + 1}/{len(combos)}: {impute} / {outlier} / {scale}")
        candidate = PreprocessSpec.from_dict({**base_spec.to_dict(),
                                              "numeric_impute": impute, "outlier": outlier, "scale": scale})
        row = {"impute": impute, "outliers": outlier, "scaling": scale}
        try:
            if target is None:
                raise ValueError("this search needs a supervised target")
            pipe, _ = build_train_pipeline(candidate, model_key, model_params, spec.task)
            cv = cross_validate_pipeline(pipe, modelling[features], target, spec, weights=weights)
            for key, (mean, std) in (cv.get("summary") or {}).items():
                row[M.label(spec.task, key)] = mean
                row[f"± {M.label(spec.task, key)}"] = std
        except Cancelled:
            raise
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    label = M.label(spec.task, spec.primary_metric)
    if label in frame.columns:
        frame = frame.sort_values(label, ascending=not M.higher_is_better(spec.task, spec.primary_metric))
    frame = frame.reset_index(drop=True)
    if not frame.empty:
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame.attrs["seconds"] = round(time.time() - started, 2)
    frame.attrs["metric"] = label
    return frame


# --------------------------------------------------------------------------
# 4. feature-combination search
# --------------------------------------------------------------------------


def feature_combo_search(df: pd.DataFrame, spec: TrainSpec, base_spec: PreprocessSpec, model_key: str,
                         model_params: dict, select_spec, weights=None, progress=None, cancel=None):
    """Delegates to :mod:`core.select`, which owns the subset-search machinery."""
    from . import select as select_module

    return select_module.run_selection(df, spec, base_spec, model_key, model_params, select_spec,
                                       weights=weights, progress=progress, cancel=cancel)


def quick_design_suggestion(X: pd.DataFrame, features: list[str] | None = None,
                            quantile: float = 0.02) -> dict:
    """Default bounds for the design form: the range each feature actually spans."""
    names = features or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    bounds = {}
    for name in names:
        series = pd.to_numeric(X[name], errors="coerce").dropna()
        if series.empty:
            continue
        lo, hi = float(series.quantile(quantile)), float(series.quantile(1 - quantile))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(series.min()), float(series.max())
        bounds[name] = [lo, hi]
    return bounds
