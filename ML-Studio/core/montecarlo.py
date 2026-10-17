"""Uncertainty propagation and reliability, without writing a simulation script.

An engineer almost always asks one of two questions near the end of a study:

* **"What is the chance of not meeting the requirement?"** — the probability that a
  predicted strength, capacity, settlement or flow falls on the wrong side of a limit.
  That is the reliability index β, and it is one line of arithmetic once you have a
  distribution of the outcome.
* **"Which uncertain input drives that chance?"** — a tornado chart, plus a rank-based
  influence measure that catches the non-linear cases a tornado misses.

The machinery is deliberately ordinary: give each input a distribution, draw from them,
push every draw through the model that has already been fitted, and read the distribution
of the result.  What makes it trustworthy rather than a toy is the details —

* **Inputs are not assumed independent unless you say so.**  Ticking "keep the observed
  correlations" imposes the rank correlation measured in the data (an Iman–Conover style
  reordering), so cement and water content move together the way they really do.
* **Physical bounds are respected.**  A truncated normal is offered for quantities that
  cannot go negative, and a triangular or PERT distribution for values a supplier quotes
  as "between X and Y, usually Z".
* **The empirical option resamples the real data**, which needs no distributional
  assumption at all and is often the most honest choice on a small dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

# --------------------------------------------------------------------------
# the catalogue of distributions
# --------------------------------------------------------------------------

#: name → (label, parameter names, what it is for)
DISTRIBUTIONS: dict[str, tuple] = {
    "normal": ("Normal (bell curve)", ("mean", "sd"),
               "the default when only a mean and a standard deviation are known"),
    "truncated_normal": ("Normal, limited to a range", ("min", "max", "mean", "sd"),
                         "use when the quantity cannot go below (or above) a value"),
    "lognormal": ("Lognormal (skewed, positive only)", ("median", "sigma (log)"),
                  "positive quantities with a long tail, such as strength or rainfall"),
    "uniform": ("Uniform (anywhere in a range)", ("min", "max"),
                "nothing is known except the limits"),
    "triangular": ("Triangular (min, usual, max)", ("min", "mode", "max"),
                   "the classic three-point estimate"),
    "pert": ("PERT (smoothly weighted min, usual, max)", ("min", "mode", "max"),
             "like the triangular but smoother; the usual choice for expert estimates"),
    "beta": ("Beta on a range", ("min", "max", "mean", "sd"),
             "a flexible bounded shape matched to a mean and a spread"),
    "empirical": ("Resample the observed values", (),
                  "no distribution assumed at all — draws from the data itself"),
    "constant": ("Fixed (no uncertainty)", ("value",),
                 "a factor held at one value for this study"),
}

#: Distribution families that describe a single continuous quantity.
_SAMPLEABLE = tuple(DISTRIBUTIONS)


def distribution_label(name: str) -> str:
    return DISTRIBUTIONS.get(name, (name, (), ""))[0]


def distribution_help(name: str) -> str:
    return DISTRIBUTIONS.get(name, (name, (), ""))[2]


def distribution_params(name: str) -> tuple:
    return DISTRIBUTIONS.get(name, (name, (), ""))[1]


# --------------------------------------------------------------------------
# describing one uncertain input
# --------------------------------------------------------------------------

@dataclass
class InputSpec:
    """One uncertain input: what it is, how it varies, and how sure we are of that."""

    column: str
    distribution: str = "normal"
    params: dict = field(default_factory=dict)
    note: str = ""

    def label(self) -> str:
        if not self.params:
            return distribution_label(self.distribution)
        bits = ", ".join(f"{k} = {_short(v)}" for k, v in self.params.items())
        return f"{distribution_label(self.distribution)} ({bits})"

    def quantile(self, q: float) -> float:
        """The q-quantile of this input — used to build the tornado chart."""
        if self.distribution == "empirical":
            values = self.params.get("values")
            values = np.asarray(values if values is not None else [], dtype=float)
            return float(np.quantile(values, q)) if len(values) else float("nan")
        if self.distribution == "constant":
            # A fixed *category* ("held at water curing") has no quantiles, exactly like a
            # triangular: there is no order to move along.
            try:
                return float(self.params.get("value", 0.0))
            except (TypeError, ValueError):
                return None
        if self.distribution == "uniform":
            low, high = float(self.params["min"]), float(self.params["max"])
            return low + q * (high - low)
        if self.distribution in ("triangular", "pert"):
            return None  # not needed; the tornado draws rather than inverts
        if self.distribution == "truncated_normal":
            low, high = float(self.params["min"]), float(self.params["max"])
            mean, sd = float(self.params["mean"]), max(float(self.params["sd"]), 1e-12)
            dist = sps.truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
            return float(dist.ppf(q))
        if self.distribution == "lognormal":
            return float(sps.lognorm.ppf(q, s=max(float(self.params["sigma (log)"]), 1e-12),
                                         scale=float(self.params["median"])))
        if self.distribution == "beta":
            mean, sd = float(self.params["mean"]), max(float(self.params["sd"]), 1e-12)
            low, high = float(self.params["min"]), float(self.params["max"])
            span = high - low
            if span <= 0:
                return low
            scaled_mean = (mean - low) / span
            variance = (sd / span) ** 2
            factor = scaled_mean * (1 - scaled_mean) / variance - 1
            alpha = max(scaled_mean * factor, 1e-6)
            beta = max((1 - scaled_mean) * factor, 1e-6)
            return float(low + span * sps.beta.ppf(q, alpha, beta))
        mean, sd = float(self.params["mean"]), max(float(self.params["sd"]), 1e-12)
        return float(sps.norm.ppf(q, loc=mean, scale=sd))

    def draw(self, n: int, rng) -> np.ndarray:
        n = int(max(1, n))
        if self.distribution == "empirical":
            values = np.asarray(self.params.get("values", []), dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return np.full(n, float("nan"))
            return rng.choice(values, size=n, replace=True)
        if self.distribution == "constant":
            value = self.params.get("value", 0.0)
            try:
                return np.full(n, float(value), dtype=float)
            except (TypeError, ValueError):
                # A held *category*: the same value on every row, as text.  With `features`
                # including a categorical column, the pipeline is what turns it into
                # numbers, and doing that here would be guessing at its encoding.
                return np.full(n, value, dtype=object)
        if self.distribution == "normal":
            return rng.normal(float(self.params["mean"]), max(abs(float(self.params["sd"])), 0.0), n)
        if self.distribution == "truncated_normal":
            low, high = float(self.params["min"]), float(self.params["max"])
            mean, sd = float(self.params["mean"]), max(abs(float(self.params["sd"])), 1e-12)
            if high <= low:
                return np.full(n, mean)
            return sps.truncnorm.rvs((low - mean) / sd, (high - mean) / sd,
                                     loc=mean, scale=sd, size=n, random_state=rng)
        if self.distribution == "lognormal":
            return sps.lognorm.rvs(s=max(abs(float(self.params["sigma (log)"])), 1e-9),
                                   scale=max(float(self.params["median"]), 1e-12),
                                   size=n, random_state=rng)
        if self.distribution == "uniform":
            return rng.uniform(float(self.params["min"]), float(self.params["max"]), n)
        if self.distribution == "triangular":
            low, mode, high = (float(self.params[k]) for k in ("min", "mode", "max"))
            if not low <= mode <= high:
                mode = min(max(mode, low), high)
            return rng.triangular(low, mode, high, n)
        if self.distribution == "pert":
            low, mode, high = (float(self.params[k]) for k in ("min", "mode", "max"))
            if high <= low:
                return np.full(n, low)
            mode = min(max(mode, low), high)
            span = high - low
            alpha = 1 + 4 * (mode - low) / span
            beta = 1 + 4 * (high - mode) / span
            return low + span * rng.beta(alpha, beta, n)
        if self.distribution == "beta":
            low, high = float(self.params["min"]), float(self.params["max"])
            mean, sd = float(self.params["mean"]), max(abs(float(self.params["sd"])), 1e-12)
            span = high - low
            if span <= 0:
                return np.full(n, low)
            scaled_mean = min(max((mean - low) / span, 1e-6), 1 - 1e-6)
            variance = min(max((sd / span) ** 2, 1e-9), scaled_mean * (1 - scaled_mean) * 0.999)
            factor = scaled_mean * (1 - scaled_mean) / variance - 1
            alpha = max(scaled_mean * factor, 1e-6)
            beta = max((1 - scaled_mean) * factor, 1e-6)
            return low + span * rng.beta(alpha, beta, n)
        raise ValueError(f"unknown distribution {self.distribution!r}")


def _short(value) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
            return f"{value:.3g}"
        return f"{value:,.4g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        return f"{len(value)} values"
    return str(value)


def default_params(series: pd.Series, distribution: str) -> dict:
    """Method-of-moments parameters from the observed data — the honest starting point.

    The values are only a *starting point* and every screen shows them for editing.  A
    normal fitted to a strictly positive quantity, for instance, will put some of its mass
    below zero, which is why ``truncated_normal`` and ``lognormal`` exist.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if distribution == "constant":
        # A column of names (a curing method, a supplier) has no mean; its typical value is
        # the most common one, and a fixed input has to be *some* value.
        if len(values) == 0 or numeric.notna().mean() < 0.9:
            modes = series.dropna().astype(str).mode()
            return {"value": str(modes.iloc[0])} if len(modes) else {}
        return {"value": float(np.mean(values))}
    if len(values) == 0:
        return {}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    low, high = float(values.min()), float(values.max())
    if distribution == "normal":
        return {"mean": mean, "sd": sd}
    if distribution == "truncated_normal":
        return {"min": low, "max": high, "mean": mean, "sd": sd}
    if distribution == "lognormal":
        if mean <= 0:
            return {"median": max(mean, 1e-9), "sigma (log)": 0.1}
        sigma = math.sqrt(math.log(1 + (sd / mean) ** 2)) if mean else 0.1
        median = mean / math.exp(sigma ** 2 / 2)
        return {"median": median, "sigma (log)": max(sigma, 1e-6)}
    if distribution == "uniform":
        return {"min": low, "max": high}
    if distribution in ("triangular", "pert"):
        return {"min": low, "mode": float(np.median(values)), "max": high}
    if distribution == "beta":
        return {"min": low, "max": high, "mean": mean, "sd": sd}
    if distribution == "empirical":
        return {"values": values}
    if distribution == "constant":
        return {"value": mean}
    return {}


def from_series(series: pd.Series, distribution: str = "normal") -> InputSpec:
    return InputSpec(column=str(series.name), distribution=distribution,
                     params=default_params(series, distribution),
                     note="fitted to the observed values in the dataset")


def choose_distribution(series: pd.Series) -> str:
    """A sensible default per column: positive and skewed → lognormal, else truncated normal.

    A column of names gets ``constant`` instead — there is no curve to fit to "steam", and
    a simulation that pretended otherwise would be inventing categories.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if is_categorical(series):
        return "constant"
    values = numeric.dropna().to_numpy(dtype=float)
    if len(values) < 8:
        return "truncated_normal" if len(values) else "constant"
    positive = bool(values.min() > 0)
    try:
        skew = float(sps.skew(values))
    except Exception:  # noqa: BLE001
        skew = 0.0
    if positive and skew > 1.0:
        return "lognormal"
    return "truncated_normal"


def is_categorical(series) -> bool:
    """True for a column of names rather than numbers.

    Used to decide whether an input *can* be sampled from a distribution.  A curing method
    or a supplier cannot: the honest treatment is to hold it at one value and let the user
    choose which, which is why ``constant`` accepts text at all.
    """
    try:
        if pd.api.types.is_numeric_dtype(series):
            return False
        numeric = pd.to_numeric(pd.Series(series), errors="coerce")
        return bool(numeric.notna().mean() < 0.9)
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------
# drawing the inputs
# --------------------------------------------------------------------------

def correlation_of(frame: pd.DataFrame, columns, method: str = "spearman") -> np.ndarray:
    """The observed rank correlation between the chosen columns."""
    columns = [c for c in columns if c in frame.columns]
    if len(columns) < 2:
        return np.eye(max(1, len(columns)))
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").corr(method=method).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Make a correlation matrix usable for a Cholesky factor without changing it much."""
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 1e-8, None)
    repaired = vectors @ np.diag(values) @ vectors.T
    diagonal = np.sqrt(np.clip(np.diag(repaired), 1e-12, None))
    repaired = repaired / np.outer(diagonal, diagonal)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def sample_inputs(specs: list[InputSpec], n: int, seed: int = 42,
                  correlation: np.ndarray | None = None) -> pd.DataFrame:
    """Draw ``n`` samples of every input, optionally with the observed correlations.

    The correlated case uses a Gaussian copula with a rank reordering: a reference normal
    sample carries the correlation, its ranks decide the order, and the values drawn from
    each marginal are placed in that order.  The marginals are therefore exactly the ones
    specified — the correlation is imposed without distorting any single input.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))
    columns = {spec.column: spec.draw(n, rng) for spec in specs}
    frame = pd.DataFrame(columns)
    # A held *category* is a column of names, and a rank reordering has nothing to sort.
    # It is left out of the copula and keeps its own values, which is what "held constant"
    # means; the numeric inputs are still correlated among themselves.
    orderable = [c for c in frame.columns
                 if pd.api.types.is_numeric_dtype(frame[c])]
    if correlation is not None and len(orderable) > 1:
        matrix = np.asarray(correlation, dtype=float)
        if matrix.shape != (len(frame.columns), len(frame.columns)):
            return frame
        positions = [frame.columns.get_loc(c) for c in orderable]
        block = _nearest_positive_definite(matrix[np.ix_(positions, positions)])
        try:
            factor = np.linalg.cholesky(block)
        except np.linalg.LinAlgError:
            return frame
        reference = rng.standard_normal((n, len(orderable))) @ factor.T
        for j, column in enumerate(orderable):
            ranks = np.argsort(np.argsort(reference[:, j]))
            values = np.sort(frame[column].to_numpy(dtype=float))
            frame[column] = values[ranks]
    return frame


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

@dataclass
class MonteCarloSpec:
    """Everything one simulation needs, so a run can be repeated exactly."""

    inputs: list[InputSpec]
    n_samples: int = 2000
    seed: int = 42
    keep_correlations: bool = False
    fixed: dict = field(default_factory=dict)
    fill: str = "median"          # median | mode | mean | zero
    threshold: float | None = None
    direction: str = "above"      # "above" → P(Y > threshold); "below" → P(Y < threshold)

    def to_dict(self) -> dict:
        return {
            "inputs": [{"column": s.column, "distribution": s.distribution,
                        "params": {k: (list(v) if isinstance(v, np.ndarray) else v)
                                   for k, v in s.params.items()}, "note": s.note}
                       for s in self.inputs],
            "n_samples": self.n_samples, "seed": self.seed,
            "keep_correlations": self.keep_correlations, "fixed": dict(self.fixed),
            "fill": self.fill, "threshold": self.threshold, "direction": self.direction,
        }


@dataclass
class MonteCarloResult:
    """The outcome distribution, plus every way of asking about it."""

    values: np.ndarray
    frame: pd.DataFrame                 # the sampled inputs, for reference
    mode: str = "regression"            # regression | probability
    threshold: float | None = None
    direction: str = "above"
    summary: dict = field(default_factory=dict)
    percentiles: pd.DataFrame | None = None
    probability: float | None = None
    reliability_index: float | None = None
    failure_probability: float | None = None
    tornado: pd.DataFrame | None = None
    influence: pd.DataFrame | None = None
    exceedance: pd.DataFrame | None = None
    histogram: tuple | None = None
    notes: list = field(default_factory=list)

    def sentence(self) -> str:
        """One plain sentence a report can quote."""
        if self.mode == "probability":
            mean = self.summary.get("mean", float("nan"))
            return (f"The probability of a positive outcome has a mean of {mean:.3f} across "
                    f"{len(self.values):,} simulations, ranging from "
                    f"{self.summary.get('P5', float('nan')):.3f} to "
                    f"{self.summary.get('P95', float('nan')):.3f} (5th to 95th percentile).")
        if self.threshold is None or self.probability is None:
            return (f"The outcome averages {self.summary.get('mean', float('nan')):,.4g} with a "
                    f"standard deviation of {self.summary.get('sd', float('nan')):,.4g} over "
                    f"{len(self.values):,} simulations.")
        word = "above" if self.direction == "above" else "below"
        tail = ""
        if self.reliability_index is not None and math.isfinite(self.reliability_index):
            if self.reliability_index < 0:
                tail = (f".  The reliability index is β = {self.reliability_index:,.2f}, which "
                        f"is negative — the mean outcome is already on the wrong side of the "
                        f"limit")
            else:
                tail = (f".  The reliability index is β = {self.reliability_index:,.2f}, a "
                        f"probability of failure of "
                        f"{self.failure_probability:,.3g}")
        return (f"{self.probability * 100:,.2f}% of the {len(self.values):,} simulations fell "
                f"{word} {self.threshold:,.6g}{tail}.")


def _fill_values(frame: pd.DataFrame, features, fill: str, fixed: dict) -> dict:
    """The value given to every feature that is not itself being varied."""
    out: dict = {}
    for column in features:
        if column in fixed:
            out[column] = fixed[column]
            continue
        if column not in frame.columns:
            out[column] = 0.0 if fill == "zero" else 0.0
            continue
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.9 and not pd.api.types.is_object_dtype(series):
            if fill == "mean":
                out[column] = float(numeric.mean())
            elif fill == "zero":
                out[column] = 0.0
            else:
                out[column] = float(numeric.median())
        else:
            modes = series.dropna().astype(str).mode()
            out[column] = str(modes.iloc[0]) if len(modes) else ""
    return out


def build_frame(sampled: pd.DataFrame, features, base: dict) -> pd.DataFrame:
    """The full feature table the model expects: sampled columns, everything else held."""
    out = pd.DataFrame(index=sampled.index)
    for column in features:
        if column in sampled.columns:
            out[column] = sampled[column].to_numpy()
        else:
            out[column] = base.get(column, 0.0)
    return out


def positive_class_index(pipe) -> int:
    """Which column of ``predict_proba`` is "the event happened"."""
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        return -1
    classes = list(classes)
    return len(classes) - 1 if len(classes) >= 2 else 0


def _predict_chunked(pipe, matrix: pd.DataFrame, task: str, progress=None,
                     cancel=None, chunk: int = 5000) -> np.ndarray:
    """Predict in chunks so progress is reported and Cancel is honoured."""
    n = len(matrix)
    pieces = []
    position = 0
    while position < n:
        if cancel is not None and cancel.is_set():
            from .train import Cancelled
            raise Cancelled()
        if progress is not None and position == 0:
            progress(0, max(1, n), "simulating")
        block = matrix.iloc[position:position + chunk]
        if task == "classification":
            probabilities = pipe.predict_proba(block)
            pieces.append(np.asarray(probabilities)[:, positive_class_index(pipe)])
        else:
            pieces.append(np.asarray(pipe.predict(block), dtype=float))
        position += chunk
        if progress is not None:
            progress(min(position, n), max(1, n), f"simulated {min(position, n):,} of {n:,}")
    if not pieces:
        return np.array([], dtype=float)
    return np.concatenate(pieces)


def _percentile_table(values: np.ndarray) -> pd.DataFrame:
    points = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    return pd.DataFrame({
        "percentile": [f"P{p:g}" for p in points],
        "% below or equal": points,
        "value": [float(np.percentile(values, p)) for p in points],
    })


def reliability(values: np.ndarray, threshold: float, direction: str = "above") -> dict:
    """P(Y beyond the threshold), the reliability index β and the failure probability.

    β is the standard reliability index from structural code work, arranged so that a
    **larger β is always safer** and the failure probability is Φ(−β):

    * ``direction="below"`` — the requirement is that Y stays *above* the limit (a strength,
      a capacity).  Safe means the mean is well above the limit, so β = (mean − limit)/σ.
    * ``direction="above"`` — the requirement is that Y stays *below* the limit (a
      settlement, a deflection, a concentration).  β = (limit − mean)/σ.

    A negative β therefore means the mean outcome is already on the wrong side of the
    limit, which is the case worth flagging loudest.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"note": "not enough simulated values"}
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if direction == "above":
        probability = float((values >= threshold).mean())
        margin = threshold - mean
    else:
        probability = float((values <= threshold).mean())
        margin = mean - threshold
    beta = margin / sd if sd > 0 else float("inf")
    failure = float(sps.norm.cdf(-beta)) if math.isfinite(beta) else 0.0
    return {
        "threshold": float(threshold), "direction": direction,
        "probability": probability,
        "reliability index": beta,
        "failure probability": failure,
        "probability simulated": probability,
        "margin": margin, "mean": mean, "sd": sd,
        "note": "β uses the simulated mean and standard deviation; it is exact for a normal "
                "outcome and a good approximation otherwise.  β < 0 means the mean outcome "
                "is on the wrong side of the limit",
    }


def run(pipe, frame: pd.DataFrame, spec: MonteCarloSpec, features, task: str = "regression",
        progress=None, cancel=None) -> MonteCarloResult:
    """Draw the inputs, push them through the fitted model, and describe the outcome."""
    features = [c for c in features]
    specs = [s for s in spec.inputs if s.column in features]
    if not specs:
        raise ValueError("none of the chosen inputs are columns the model uses")

    correlation = None
    if spec.keep_correlations and len(specs) > 1:
        correlation = correlation_of(frame, [s.column for s in specs])

    sampled = sample_inputs(specs, spec.n_samples, spec.seed, correlation)
    base = _fill_values(frame, features, spec.fill, spec.fixed)
    matrix = build_frame(sampled, features, base)
    values = _predict_chunked(pipe, matrix, task, progress, cancel)
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    result = MonteCarloResult(values=values, frame=sampled,
                              mode="probability" if task == "classification" else "regression",
                              threshold=spec.threshold, direction=spec.direction)
    if len(finite) == 0:
        result.notes.append("every simulated prediction came back as a missing value")
        return result

    result.summary = {
        "n": int(len(finite)), "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "coefficient of variation %": (float(finite.std(ddof=1) / finite.mean() * 100)
                                       if len(finite) > 1 and finite.mean() else float("nan")),
        "minimum": float(finite.min()), "maximum": float(finite.max()),
        "P5": float(np.percentile(finite, 5)), "P50": float(np.percentile(finite, 50)),
        "P95": float(np.percentile(finite, 95)),
        "share below zero %": float((finite < 0).mean() * 100) if result.mode == "regression" else float("nan"),
    }
    result.percentiles = _percentile_table(finite)
    result.histogram = (finite, spec.threshold)

    if spec.threshold is not None:
        info = reliability(finite, spec.threshold, spec.direction)
        result.probability = info.get("probability")
        result.reliability_index = info.get("reliability index")
        result.failure_probability = info.get("failure probability")
        result.summary["probability %"] = (result.probability * 100
                                           if result.probability is not None else float("nan"))
        result.summary["reliability index β"] = result.reliability_index
        result.summary["failure probability Φ(−β)"] = result.failure_probability
        if (result.probability is not None and result.failure_probability is not None
                and abs(result.probability - result.failure_probability) > 0.15):
            result.notes.append(
                "the simulated probability and the normal-based Φ(−β) differ noticeably, so "
                "the outcome is not close to normally distributed — quote the simulated value")
    else:
        result.notes.append("give a limit value to get the probability of exceeding it")

    result.influence = influence(specs, sampled, finite)
    return result


def influence(specs, sampled: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Rank-correlation between each uncertain input and the outcome.

    A cheap global sensitivity measure: it answers "which input does the answer move with?",
    including for curved relationships, and it is honest about its own limit — it only sees
    *monotone* influence, so a factor that matters only through an interaction can hide.
    """
    values = np.asarray(values, dtype=float)
    rows = []
    for spec_ in specs:
        if spec_.column not in sampled.columns:
            continue
        if not pd.api.types.is_numeric_dtype(sampled[spec_.column]):
            rows.append({"input": spec_.column, "rank correlation": float("nan"),
                         "strength": "held at a fixed value — no ranking to do",
                         "p": float("nan")})
            continue
        column = sampled[spec_.column].to_numpy(dtype=float)
        if np.allclose(column, column[0]):
            rows.append({"input": spec_.column, "rank correlation": 0.0,
                         "strength": "none — held constant", "p": float("nan")})
            continue
        try:
            result = sps.spearmanr(column, values)
            rho = float(np.asarray(getattr(result, "statistic", result[0])).ravel()[0])
            p = float(np.asarray(getattr(result, "pvalue", result[1])).ravel()[0])
        except Exception:  # noqa: BLE001
            continue
        magnitude = abs(rho)
        strength = ("negligible" if magnitude < 0.1 else "weak" if magnitude < 0.3
                    else "moderate" if magnitude < 0.5 else "strong")
        rows.append({"input": spec_.column, "rank correlation": rho, "p": p,
                     "strength": strength,
                     "direction": "higher input → higher result" if rho > 0
                                  else "higher input → lower result"})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    return table.reindex(table["rank correlation"].abs().sort_values(ascending=False).index
                         ).reset_index(drop=True)


def tornado(pipe, frame: pd.DataFrame, spec: MonteCarloSpec, features, task: str = "regression",
            low_q: float = 0.1, high_q: float = 0.9, seed: int | None = None) -> pd.DataFrame:
    """How far the answer moves when one input moves, everything else held still.

    One input at a time is pushed from its low quantile to its high quantile (the 10th to
    the 90th percentile by default) while the rest sit at their base values.  The swing is
    what the bar chart shows, and sorting by it is what makes the chart a priority list.
    """
    features = list(features)
    base = _fill_values(frame, features, spec.fill, spec.fixed)
    specs = [s for s in spec.inputs if s.column in features]
    if not specs:
        return pd.DataFrame()

    def predict_one(overrides: dict) -> float:
        row = dict(base)
        row.update(overrides)
        matrix = pd.DataFrame([row])[features]
        values = _predict_chunked(pipe, matrix, task)
        return float(values[0]) if len(values) else float("nan")

    # Shared draws, so the comparison is like for like rather than one noisy sample.
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    reference = {}
    skipped: list[str] = []
    for spec_ in specs:
        drawn = spec_.draw(600, rng)
        numeric = pd.to_numeric(pd.Series(drawn), errors="coerce").to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric) < 2 or np.allclose(numeric, numeric[0]):
            # A fixed value, or a name that has no order — either way there is no pair of
            # quantiles to move between, so it cannot be a tornado bar.
            skipped.append(spec_.column)
            continue
        reference[spec_.column] = (float(np.percentile(numeric, low_q * 100)),
                                   float(np.percentile(numeric, high_q * 100)),
                                   float(np.median(numeric)))
    specs = [s for s in specs if s.column in reference]
    if not specs:
        return pd.DataFrame()
    baseline = predict_one({})
    rows = []
    for spec_ in specs:
        low, high, middle = reference[spec_.column]
        at_low = predict_one({spec_.column: low})
        at_high = predict_one({spec_.column: high})
        rows.append({
            "input": spec_.column,
            "low (P%g)" % (low_q * 100): low,
            "high (P%g)" % (high_q * 100): high,
            "at low": at_low, "at high": at_high,
            "swing": abs(at_high - at_low),
            "baseline": baseline,
        })
    table = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    total = table["swing"].sum()
    table["% of total swing"] = table["swing"] / total * 100 if total else float("nan")
    # Which inputs could not be put on the chart, so the screen can say so rather than
    # let a missing bar look like "this input does not matter".
    table.attrs["skipped"] = skipped
    return table


def exceedance_curve(values, points: int = 60) -> pd.DataFrame:
    """The probability of exceeding each of a series of limits — the design-check curve."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return pd.DataFrame()
    lows, highs = float(values.min()), float(values.max())
    if highs == lows:
        return pd.DataFrame()
    limits = np.linspace(lows, highs, int(max(5, points)))
    above = np.array([(values >= limit).mean() for limit in limits])
    return pd.DataFrame({"limit": limits, "probability above %": above * 100,
                         "probability below %": (1 - above) * 100})


def safety_factors(values, demand: float) -> dict:
    """The distribution of ``resistance / demand`` — the civil-engineering view of margin.

    Given a capacity (a strength, a bearing pressure, a flow) and a demand (a load, a
    settlement limit), this reports the distribution of the safety factor and the chance it
    drops below a chosen target.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if demand == 0 or len(values) < 2:
        return {"note": "a demand of zero (or too few values) has no safety factor"}
    factors = values / float(demand)
    return {
        "demand": float(demand),
        "mean safety factor": float(factors.mean()),
        "sd": float(factors.std(ddof=1)),
        "P5": float(np.percentile(factors, 5)),
        "P50": float(np.percentile(factors, 50)),
        "P95": float(np.percentile(factors, 95)),
        "probability below 1.0 %": float((factors < 1.0).mean() * 100),
        "probability below 1.5 %": float((factors < 1.5).mean() * 100),
        "factors": factors,
    }
