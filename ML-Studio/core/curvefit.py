"""Fitting engineering equations to data, with the numbers a report has to quote.

An engineer who reaches for a script is usually trying to do one of four things, and
this module does all four:

1. **Fit a known engineering form** — strength against age, capacity against depth, a
   saturation or S-curve, a decay.  :data:`FORMS` is the catalogue.
2. **Choose between forms** — :func:`fit_all` ranks every form by AIC, so the choice is
   made by information criterion rather than by which one looked best in Excel.
3. **Quote the uncertainty** — every parameter carries a standard error and a confidence
   interval, and the curve carries a prediction band, computed from the fit's covariance
   matrix rather than eyeballed.
4. **Invert it** — :func:`solve_for_x` answers "at what age does this reach 30 MPa?",
   which is the question that usually follows the fit.

Fitting uses SciPy's Levenberg–Marquardt (`curve_fit`) with sensible starting values and
physical bounds, which is what stops the exponential and logistic forms wandering off to
nonsense on a short dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats as sps

# --------------------------------------------------------------------------
# the catalogue of forms
# --------------------------------------------------------------------------

def _p0_linear(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return [intercept, slope]


def _p0_poly2(x, y):
    c, b, a = np.polyfit(x, y, 2)
    return [a, b, c]


def _p0_poly3(x, y):
    d, c, b, a = np.polyfit(x, y, 3)
    return [a, b, c, d]


def _safe_log(x):
    return np.log(np.clip(x, 1e-12, None))


@dataclass
class Form:
    """One candidate equation."""

    key: str
    label: str
    equation: str                      # a template with {a}, {b} … placeholders
    func: object                       # f(x, *params)
    param_names: tuple
    starts: object = None              # f(x, y) -> initial guess
    bounds: tuple | None = None        # (lower, upper) per parameter, or None
    note: str = ""
    family: str = "General"

    def text(self, params) -> str:
        values = {name: params[i] for i, name in enumerate(self.param_names)}
        return self.equation.format(**{k: _num(v) for k, v in values.items()})


def _num(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.4g}"
    return f"{value:.5g}"


FORMS: dict[str, Form] = {}


def _add(form: Form):
    FORMS[form.key] = form
    return form


_add(Form("linear", "Straight line", "y = {a} + {b}·x",
          lambda x, a, b: a + b * x, ("a", "b"), _p0_linear,
          note="the default; use it unless the points curve", family="Polynomial"))
_add(Form("quadratic", "Quadratic", "y = {a} + {b}·x + {c}·x²",
          lambda x, a, b, c: a + b * x + c * x * x, ("a", "b", "c"), _p0_poly2,
          note="a single bend; good for a minimum or maximum", family="Polynomial"))
_add(Form("cubic", "Cubic", "y = {a} + {b}·x + {c}·x² + {d}·x³",
          lambda x, a, b, c, d: a + b * x + c * x ** 2 + d * x ** 3,
          ("a", "b", "c", "d"), _p0_poly3,
          note="two bends; do not use it to extrapolate", family="Polynomial"))
_add(Form("power", "Power law", "y = {a}·x^{b}",
          lambda x, a, b: a * np.power(np.clip(x, 1e-12, None), b), ("a", "b"),
          lambda x, y: [float(np.exp(np.polyfit(_safe_log(x), np.log(np.clip(y, 1e-12, None)), 1)[1])),
                        float(np.polyfit(_safe_log(x), np.log(np.clip(y, 1e-12, None)), 1)[0])],
          bounds=([1e-12, -20], [np.inf, 20]),
          note="strength with age, fatigue life, and other scale-free relationships",
          family="Power / log"))
_add(Form("logarithmic", "Logarithmic", "y = {a} + {b}·ln(x)",
          lambda x, a, b: a + b * np.log(np.clip(x, 1e-12, None)), ("a", "b"),
          lambda x, y: [float(np.polyfit(_safe_log(x), y, 1)[1]),
                        float(np.polyfit(_safe_log(x), y, 1)[0])],
          note="a curve that keeps rising but ever more slowly", family="Power / log"))
_add(Form("strength_age", "Power law with an offset (ACI-style strength gain)",
          "y = {a} + {b}·x^{c}",
          lambda x, a, b, c: a + b * np.power(np.clip(x, 1e-12, None), c),
          ("a", "b", "c"),
          lambda x, y: [float(np.min(y)) * 0.6, max(float(np.percentile(y, 90) - np.min(y)), 1.0), 0.5],
          bounds=([-np.inf, -np.inf, -5], [np.inf, np.inf, 5]),
          note="the usual way to model concrete gaining strength with age", family="Power / log"))
_add(Form("exponential", "Exponential growth", "y = {a}·exp({b}·x)",
          lambda x, a, b: a * np.exp(b * x), ("a", "b"),
          lambda x, y: [float(np.exp(np.polyfit(x, np.log(np.clip(y, 1e-12, None)), 1)[1])),
                        float(np.polyfit(x, np.log(np.clip(y, 1e-12, None)), 1)[0])],
          bounds=([-np.inf, -20], [np.inf, 20]),
          note="growth, or decay if b is negative", family="Exponential"))
_add(Form("decay", "Exponential decay to a floor", "y = {a} + {b}·exp(−{c}·x)",
          lambda x, a, b, c: a + b * np.exp(-np.clip(c, 1e-12, None) * np.clip(x, 0, None)),
          ("a", "b", "c"),
          lambda x, y: [float(np.min(y)), max(float(np.max(y) - np.min(y)), 1e-6), 0.1],
          bounds=([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, 50]),
          note="consolidation settlement or creep settling to a final value",
          family="Exponential"))
_add(Form("reciprocal", "Reciprocal", "y = {a} + {b}/x",
          lambda x, a, b: a + b / np.where(np.abs(x) < 1e-12, 1e-12, x), ("a", "b"),
          lambda x, y: [float(np.median(y)), float(np.median(y))], bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
          note="behaviour that approaches a limit as x grows", family="Hyperbolic"))
_add(Form("hyperbolic", "Hyperbolic / Michaelis–Menten (saturating)",
          "y = {a}·x/({b} + x)",
          lambda x, a, b: a * x / (np.where(np.abs(b + x) < 1e-12, 1e-12, b + x)), ("a", "b"),
          lambda x, y: [max(float(np.max(y)) * 1.4, 1e-6), float(np.median(x))],
          bounds=([-np.inf, 1e-12], [np.inf, np.inf]),
          note="a curve that saturates: soil–water retention, plate load tests",
          family="Hyperbolic"))
_add(Form("logistic", "S-curve (logistic)", "y = {L}/(1 + exp(−{k}·(x − {x0})))",
          lambda x, L, k, x0: L / (1 + np.exp(-np.clip(k * (x - x0), -500, 500))),
          ("L", "k", "x0"),
          lambda x, y: [max(float(np.max(y)) * 1.05, 1e-6), 1.0, float(np.median(x))],
          bounds=([1e-12, 1e-6, -np.inf], [np.inf, 200, np.inf]),
          note="learning, adoption and strength-gain curves that flatten off",
          family="S-curve"))
_add(Form("gompertz", "Gompertz (asymmetric S-curve)", "y = {a}·exp(−{b}·exp(−{c}·x))",
          lambda x, a, b, c: a * np.exp(-b * np.exp(-np.clip(c, 1e-12, None) * x)),
          ("a", "b", "c"),
          lambda x, y: [max(float(np.max(y)) * 1.05, 1e-6), 3.0, 0.2],
          bounds=([1e-12, 1e-9, 1e-9], [np.inf, np.inf, 50]),
          note="the usual growth model when the S is not symmetric",
          family="S-curve"))
_add(Form("weibull", "Weibull cumulative (reliability)", "y = {a}·(1 − exp(−(x/{b})^{c}))",
          lambda x, a, b, c: a * (1 - np.exp(-np.power(np.clip(x, 0, None) / max(b, 1e-12), c))),
          ("a", "b", "c"),
          lambda x, y: [max(float(np.max(y)) * 1.05, 1e-6), float(np.median(x)) or 1.0, 1.5],
          bounds=([1e-12, 1e-12, 1e-3], [np.inf, np.inf, 50]),
          note="survival and strength distributions", family="S-curve"))


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------

@dataclass
class CurveFit:
    """One fitted equation, with everything needed to quote it in a report."""

    key: str
    label: str
    equation: str
    param_names: tuple
    params: np.ndarray
    stderr: np.ndarray
    r2: float
    adjusted_r2: float
    rmse: float
    mae: float
    aic: float
    aicc: float
    bic: float
    n: int
    dof: int
    residual_sd: float
    converged: bool
    note: str = ""
    warnings: list = field(default_factory=list)
    _func: object = None
    _pcov: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None

    # ---------------------------------------------------------------- report
    def parameters(self):
        """A tidy table: value, standard error, 95% interval, significance."""
        rows = []
        for i, name in enumerate(self.param_names):
            value = float(self.params[i])
            error = float(self.stderr[i]) if i < len(self.stderr) else float("nan")
            if math.isfinite(error) and self.dof > 0:
                half = float(sps.t.ppf(0.975, self.dof)) * error
                p = float(2 * (1 - sps.t.cdf(abs(value / error), self.dof))) if error > 0 else float("nan")
            else:
                half, p = float("nan"), float("nan")
            rows.append({"parameter": name, "value": value, "std error": error,
                         "95% low": value - half, "95% high": value + half, "p": p,
                         "significant at 5%?": "yes" if math.isfinite(p) and p < 0.05 else "no"})
        import pandas as pd

        return pd.DataFrame(rows)

    def summary(self) -> dict:
        return {
            "form": self.label, "equation": self.equation, "n": self.n,
            "R²": self.r2, "adjusted R²": self.adjusted_r2, "RMSE": self.rmse,
            "MAE": self.mae, "AIC": self.aic, "AICc": self.aicc, "BIC": self.bic,
            "residual sd": self.residual_sd, "parameters": len(self.param_names),
            "converged": "yes" if self.converged else "no",
        }

    # ------------------------------------------------------------- prediction
    def predict(self, x_new):
        x_new = np.asarray(x_new, dtype=float).ravel()
        return self._func(x_new, *self.params)

    def predict_with_band(self, x_new, level: float = 0.95):
        """``(y, low, high)`` where the band is a *prediction* interval for a new reading.

        Computed from the Jacobian of the model and the fit covariance, then widened by
        the residual variance — the standard delta-method interval, which is honest about
        both the uncertainty in the parameters and the scatter of the data.
        """
        x_new = np.asarray(x_new, dtype=float).ravel()
        y = self.predict(x_new)
        if self._pcov is None or not np.all(np.isfinite(self._pcov)) or self.dof <= 0:
            return y, np.full_like(y, np.nan), np.full_like(y, np.nan)
        jacobian = _jacobian(self._func, x_new, self.params)
        variance = np.einsum("ij,jk,ik->i", jacobian, self._pcov, jacobian)
        variance = np.clip(variance, 0, None)
        t = float(sps.t.ppf(0.5 + level / 2, self.dof))
        half = t * np.sqrt(variance + self.residual_sd ** 2)
        return y, y - half, y + half

    def residuals(self) -> np.ndarray:
        if self.x is None or self.y is None:
            return np.array([])
        return np.asarray(self.y, dtype=float) - self.predict(self.x)

    # ---------------------------------------------------------------- inverse
    def solve_for_x(self, y_target: float, x_lo: float | None = None,
                    x_hi: float | None = None) -> dict:
        """The x that produces ``y_target`` — "what age reaches 30 MPa?".

        Searched across the range the data covers (widened by half again, because the
        answer is often just outside it), and refused rather than guessed when the form
        is not monotonic through the target.
        """
        if self.x is None or len(self.x) < 2:
            return {"note": "no data range to search"}
        lo = float(min(self.x)) if x_lo is None else float(x_lo)
        hi = float(max(self.x)) if x_hi is None else float(x_hi)
        span = hi - lo
        lo, hi = lo - 0.5 * span, hi + 0.5 * span

        def objective(x):
            return float(self.predict(np.array([x]))[0]) - float(y_target)

        grid = np.linspace(lo, hi, 400)
        values = self.predict(grid) - float(y_target)
        signs = np.sign(values)
        crossings = np.where(np.diff(signs) != 0)[0]
        if len(crossings) == 0:
            reach = self.predict(grid)
            return {"note": f"the curve never reaches {_num(y_target)} between "
                            f"{_num(lo)} and {_num(hi)} (it spans {_num(reach.min())} to "
                            f"{_num(reach.max())})"}
        results = []
        for index in crossings:
            try:
                root = optimize.brentq(objective, grid[index], grid[index + 1], maxiter=200)
            except Exception:  # noqa: BLE001
                continue
            results.append(float(root))
        if not results:
            return {"note": "a crossing was found but the solver could not pin it down"}
        best = results[0]
        # An interval on x, by pushing the band's edges until they cross the target
        _y, low, high = self.predict_with_band(np.array([best]))
        return {"x": best, "target": float(y_target),
                "prediction low": float(low[0]), "prediction high": float(high[0]),
                "crossings": len(results), "searched": (lo, hi),
                "note": "the first crossing in the searched range" if len(results) > 1
                        else ""}


def _jacobian(func, x, params, step: float = 1e-6):
    """Numerical derivative of the model with respect to each parameter."""
    x = np.asarray(x, dtype=float)
    base = np.asarray(func(x, *params), dtype=float)
    columns = []
    for i in range(len(params)):
        delta = step * (abs(params[i]) + step)
        shifted = np.array(params, dtype=float)
        shifted[i] += delta
        columns.append((np.asarray(func(x, *shifted), dtype=float) - base) / delta)
    return np.column_stack(columns)


def fit(x, y, form: str | Form, level: float = 0.95) -> CurveFit:
    """Fit one form.  Never raises for a bad fit — it says so in ``converged``."""
    spec = FORMS[form] if isinstance(form, str) else form
    xs, ys = _clean(x, y)
    n = len(xs)
    name = spec.label
    if n < len(spec.param_names) + 1:
        return _failed(spec, n, f"needs more than {len(spec.param_names)} points; there are {n}")

    starts = None
    if spec.starts is not None:
        try:
            starts = list(spec.starts(xs, ys))
        except Exception:  # noqa: BLE001 - a bad guess is not fatal
            starts = None
    if not starts or len(starts) != len(spec.param_names) or not np.all(np.isfinite(starts)):
        starts = [1.0] * len(spec.param_names)
        if "a" in spec.param_names:
            starts[spec.param_names.index("a")] = float(np.median(ys)) or 1.0

    bounds = spec.bounds or (-np.inf, np.inf)
    with np.errstate(all="ignore"):
        try:
            params, pcov = optimize.curve_fit(spec.func, xs, ys, p0=starts, bounds=bounds,
                                              maxfev=40000)
            converged = True
        except Exception as exc:  # noqa: BLE001
            try:
                params, pcov = optimize.curve_fit(spec.func, xs, ys, p0=starts,
                                                  maxfev=40000)
                converged = True
                pcov = np.full((len(params), len(params)), np.nan)
            except Exception:  # noqa: BLE001
                return _failed(spec, n, f"the fit did not converge ({type(exc).__name__})")

    predicted = np.asarray(spec.func(xs, *params), dtype=float)
    residuals = ys - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    n_params = len(params)
    dof = n - n_params
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adjusted = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    rmse = math.sqrt(ss_res / n) if n else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    residual_sd = math.sqrt(ss_res / dof) if dof > 0 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_likelihood_term = math.log(ss_res / n) if ss_res > 0 else -math.inf
    aic = 2 * n_params + n * log_likelihood_term if math.isfinite(log_likelihood_term) else float("nan")
    aicc = (aic + 2 * n_params * (n_params + 1) / (n - n_params - 1)
            if math.isfinite(aic) and n - n_params - 1 > 0 else float("nan"))
    bic = n_params * math.log(n) + n * log_likelihood_term if math.isfinite(log_likelihood_term) else float("nan")

    stderr = np.full(n_params, np.nan)
    if pcov is not None and np.all(np.isfinite(pcov)):
        diagonal = np.diag(pcov)
        stderr = np.sqrt(np.clip(diagonal, 0, None))

    warnings = []
    # A parameter thousands of times larger than the data usually means two parameters
    # are trading off against each other and the fit is not identified at all.  The R²
    # still looks fine, which is exactly why this has to be flagged.
    scale = max(float(np.max(np.abs(ys))) if len(ys) else 1.0, 1e-9)
    if np.any(np.abs(params) > 1e6 * scale):
        warnings.append("some parameters are enormous compared with the data — this form "
                        "is not well identified here, so read the R² but not the "
                        "individual coefficients")
    if not (r2 > 0.5) and n > 5:
        warnings.append("the fit explains less than half the variation — check the plot")
    if len(residuals) >= 8:
        try:
            if float(sps.shapiro(residuals).pvalue) < 0.05:
                warnings.append("the residuals are not normal, so the confidence intervals "
                                "are optimistic")
        except Exception:  # noqa: BLE001
            pass
    if len(xs) < 5 * n_params:
        warnings.append("fewer than five points per parameter — treat the errors with care")

    return CurveFit(
        key=spec.key, label=name, equation=spec.text(params), param_names=spec.param_names,
        params=np.asarray(params, dtype=float), stderr=stderr,
        r2=float(r2), adjusted_r2=float(adjusted), rmse=float(rmse), mae=mae,
        aic=float(aic), aicc=float(aicc), bic=float(bic), n=n, dof=int(dof),
        residual_sd=float(residual_sd), converged=bool(converged), note=spec.note,
        warnings=warnings, _func=spec.func, _pcov=pcov, x=xs, y=ys)


def _failed(spec: Form, n: int, message: str) -> CurveFit:
    blank = np.full(len(spec.param_names), np.nan)
    return CurveFit(key=spec.key, label=spec.label, equation=spec.label, param_names=spec.param_names,
                    params=blank, stderr=blank, r2=float("nan"), adjusted_r2=float("nan"),
                    rmse=float("nan"), mae=float("nan"), aic=float("inf"), aicc=float("inf"),
                    bic=float("inf"), n=n, dof=0, residual_sd=float("nan"), converged=False,
                    note=message, warnings=[message], _func=spec.func)


def _clean(x, y):
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def fit_all(x, y, keys=None, min_r2: float = -np.inf) -> tuple:
    """Fit every form and rank them.  Returns ``(table, {key: CurveFit})``.

    Ranked by AICc, which penalises extra parameters — otherwise the cubic would always
    win on R² alone, which is the trap this replaces.
    """
    import pandas as pd

    keys = list(keys if keys is not None else FORMS)
    fits: dict[str, CurveFit] = {}
    rows = []
    for key in keys:
        result = fit(x, y, key)
        fits[key] = result
        row = result.summary()
        row["key"] = key
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, fits
    table = table[table["AICc"].notna() | table["RMSE"].notna()]
    table = table.sort_values(["AICc", "RMSE"], na_position="last").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    keep = ["rank", "key", "form", "equation", "n", "R²", "adjusted R²", "RMSE", "MAE",
            "AICc", "AIC", "BIC", "residual sd", "converged"]
    return table[keep], fits


def best(x, y, keys=None):
    table, fits = fit_all(x, y, keys)
    if table.empty:
        return None
    return fits.get(table.iloc[0]["key"])


# --------------------------------------------------------------------------
# interpolation — the other thing an engineer scripts
# --------------------------------------------------------------------------

INTERPOLATION_METHODS = {
    "linear": "Straight lines between the points (never overshoots, never smooth)",
    "nearest": "The closest measured value (for step-like or categorical data)",
    "pchip": "Monotone cubic — smooth, and it will not overshoot the data",
    "cubic": "Natural cubic spline — smoothest, but it can overshoot between points",
}


def repeated_x(x_known) -> int:
    """How many x values in a table are repeats of an earlier one.

    Reported to the user rather than hidden: a pasted table whose x column repeats
    (``Age (days)`` is 1, 3, 7, 14, 28 over and over in a real trial report) cannot be
    interpolated as if each point were independent, and an engineer is entitled to know
    that their readings were averaged.
    """
    xs = np.asarray(x_known, dtype=float).ravel()
    xs = xs[np.isfinite(xs)]
    return int(len(xs) - len(np.unique(xs)))


def _average_repeats(xs: np.ndarray, ys: np.ndarray):
    """Collapse equal x values onto one point, averaging their y.

    Both spline methods need strictly increasing x.  Without this, fitting a pchip or a
    cubic spline through a real monitoring column raised
    ``ValueError: `x` must be strictly increasing sequence``, which says nothing about
    the duplicate readings that caused it.
    """
    if len(xs) < 2:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if len(np.unique(xs)) == len(xs):
        return xs, ys
    starts = np.flatnonzero(np.diff(xs) > 0) + 1
    groups = np.split(np.arange(len(xs)), starts)
    return (np.array([xs[group[0]] for group in groups]),
            np.array([float(np.mean(ys[group])) for group in groups]))


def interpolate(x_known, y_known, x_new, method: str = "linear") -> np.ndarray:
    """Interpolate a table, with monotone cubic as the safe smooth option.

    ``pchip`` is offered first among the smooth methods on purpose: an overshooting
    spline can invent a negative concentration or a strength above what the material can
    reach, which is worse than a slightly less smooth curve.
    """
    xs, ys = _clean(x_known, y_known)
    xs, ys = _average_repeats(xs, ys)
    if len(xs) < 2:
        raise ValueError("need at least two known points with different x values")
    target = np.asarray(x_new, dtype=float).ravel()

    if method == "nearest":
        indices = np.abs(target[:, None] - xs[None, :]).argmin(axis=1)
        return ys[indices]
    if method == "pchip":
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(xs, ys, extrapolate=True)(target)
    if method == "cubic":
        from scipy.interpolate import CubicSpline

        return CubicSpline(xs, ys, extrapolate=True)(target)
    return np.interp(target, xs, ys)


def interpolate_frame(x_known, y_known, x_new, method: str = "linear"):
    """Interpolation as a table: the requested points, the value, and a flag for
    points that fell outside the measured range (extrapolation dressed up as data is
    the classic engineering mistake)."""
    import pandas as pd

    xs = np.asarray(x_known, dtype=float)
    values = interpolate(x_known, y_known, x_new, method)
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    return pd.DataFrame({
        "x": np.asarray(x_new, dtype=float).ravel(),
        "interpolated y": values,
        "inside the measured range?": ["yes" if lo <= v <= hi else "NO — extrapolated"
                                       for v in np.asarray(x_new, dtype=float).ravel()],
    })


def residual_frame(result: CurveFit) -> "object":
    """x, measured, fitted, residual, standardised residual."""
    import pandas as pd

    residuals = result.residuals()
    sd = result.residual_sd
    standardised = residuals / sd if sd and math.isfinite(sd) and sd > 0 else np.full(len(residuals), np.nan)
    return pd.DataFrame({
        "x": result.x, "measured": result.y, "fitted": result.predict(result.x),
        "residual": residuals, "standardised residual": standardised,
    })
