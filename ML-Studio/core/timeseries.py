"""Time series, trends and quality-control charts.

Monitoring data is the other half of civil engineering: settlement readings, piezometer
levels, traffic counts, cube-crushing results, rainfall.  The questions are always the
same few, so they are answered directly here rather than through a script:

* **Is it moving?** — :func:`trend`, with an ordinary-least-squares slope *and* the
  rank-based Mann–Kendall test and Sen's slope, because monitoring data is rarely normal.
  The slope is reported per year, with a confidence interval, so a report can say
  "settling at 2.1 mm/year (95% CI 1.4 to 2.8)".
* **Is it in control?** — :func:`control_chart`, which builds the chart an engineer would
  build (individuals/moving-range, X-bar/R, CUSUM, EWMA) and then applies Nelson's rules.
* **What is the pattern?** — :func:`seasonal_decompose`, :func:`profile_by_period` and
  :func:`autocorrelation`.
* **Make it into features** — :func:`add_lags`, so lags, differences and rolling means can
  be fed to the model on the Features screen.

The control-chart constants are the standard ones from Montgomery's *Introduction to
Statistical Quality Control*; they are tabulated here and cited rather than derived.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

#: Control-chart constants by subgroup size (Montgomery, *Introduction to Statistical
#: Quality Control*, Appendix VI).
CHART_CONSTANTS = {
    2: {"d2": 1.128, "A2": 1.880, "D3": 0.0, "D4": 3.267},
    3: {"d2": 1.693, "A2": 1.023, "D3": 0.0, "D4": 2.574},
    4: {"d2": 2.059, "A2": 0.729, "D3": 0.0, "D4": 2.282},
    5: {"d2": 2.326, "A2": 0.577, "D3": 0.0, "D4": 2.114},
    6: {"d2": 2.534, "A2": 0.483, "D3": 0.0, "D4": 2.004},
    7: {"d2": 2.704, "A2": 0.419, "D3": 0.076, "D4": 1.924},
    8: {"d2": 2.847, "A2": 0.373, "D3": 0.136, "D4": 1.864},
    9: {"d2": 2.970, "A2": 0.337, "D3": 0.184, "D4": 1.816},
    10: {"d2": 3.078, "A2": 0.308, "D3": 0.223, "D4": 1.777},
}

CHART_KINDS = {
    "individuals": "Individuals & moving range (one reading at a time)",
    "xbar_r": "X-bar & R (average and range of small subgroups)",
    "cusum": "CUSUM (catches a small sustained shift)",
    "ewma": "EWMA (smoothed; catches slow drift)",
}

#: Friendly names → the pandas offset alias they mean.  Pandas 3 renamed the anchored
#: offsets ('M' → 'ME', 'Q' → 'QE', 'Y' → 'YE'), so they are spelled out here: the old
#: single-letter aliases raise rather than warn.
RESAMPLE_RULES = {
    "hour": ("h", "hourly"),
    "day": ("D", "daily"),
    "week": ("W", "weekly"),
    "month": ("ME", "monthly"),
    "quarter": ("QE", "quarterly"),
    "year": ("YE", "yearly"),
}

_AGGREGATIONS = ["mean", "median", "min", "max", "sum", "std", "count", "first", "last"]


# --------------------------------------------------------------------------
# getting the data into shape
# --------------------------------------------------------------------------

def to_datetime(series, dayfirst: bool = False) -> pd.Series:
    """Parse a date/time column, coercing anything unparseable to NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")


def ordered(df: pd.DataFrame, time_col: str, value_cols=None) -> pd.DataFrame:
    """A frame sorted by time with the date column actually parsed.

    Duplicate timestamps are kept (and reported) rather than silently averaged — two
    readings at one time is a data problem the user needs to see.
    """
    if time_col not in df.columns:
        raise ValueError(f"there is no column called {time_col!r}")
    out = df.copy()
    out[time_col] = to_datetime(out[time_col])
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if value_cols:
        keep = [time_col] + [c for c in value_cols if c in out.columns]
        out = out[keep]
    return out


def numeric_columns(frame: pd.DataFrame, time_col: str | None = None) -> list[str]:
    return [c for c in frame.columns
            if c != time_col and pd.api.types.is_numeric_dtype(frame[c])]


def time_axis(frame: pd.DataFrame, time_col: str) -> tuple:
    """``(years_since_start, label)`` — an x axis a regression can be run against."""
    times = to_datetime(frame[time_col])
    origin = times.min()
    days = (times - origin).dt.total_seconds() / 86400.0
    return days.to_numpy(dtype=float) / 365.25, origin


def gaps(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """The spacing between readings — a flaky logger shows up here first."""
    times = to_datetime(frame[time_col]).dropna().sort_values()
    if len(times) < 3:
        return pd.DataFrame()
    diffs = times.diff().dropna()
    return pd.DataFrame({
        "gap": diffs.values,
        "days": diffs.dt.total_seconds().to_numpy() / 86400.0,
    })


# --------------------------------------------------------------------------
# resampling and rolling statistics
# --------------------------------------------------------------------------

def resample(frame: pd.DataFrame, time_col: str, rule: str = "month",
             aggregation: str = "mean", value_cols=None) -> pd.DataFrame:
    """Average (or sum, or count) readings onto a regular calendar."""
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    cols = value_cols or numeric_columns(frame, time_col)
    work = ordered(frame, time_col, cols)
    if work.empty:
        return pd.DataFrame()
    offset = RESAMPLE_RULES.get(rule, (rule, rule))[0]
    work = work.set_index(time_col)
    grouped = work.resample(offset)
    out = grouped.agg(aggregation) if aggregation != "count" else grouped.count()
    out = out.reset_index()
    out.insert(1, "readings", grouped.size().to_numpy())
    return out


def rolling(frame: pd.DataFrame, time_col: str, window: int = 3, value_col: str | None = None,
            stats: tuple = ("mean", "std")) -> pd.DataFrame:
    """A moving average (and spread) — the smoothed line to plot over the raw points."""
    cols = [value_col] if value_col else numeric_columns(frame, time_col)
    cols = [c for c in cols if c]
    if not cols:
        return pd.DataFrame()
    work = ordered(frame, time_col, cols)
    window = max(2, int(window))
    out = pd.DataFrame({time_col: work[time_col]})
    series = work[cols].astype(float)
    for column in cols:
        for stat in stats:
            out[f"{column} · {stat}"] = getattr(series[column].rolling(window, min_periods=2), stat)()
    out["readings in window"] = series[cols[0]].rolling(window, min_periods=2).count()
    return out


def add_lags(frame: pd.DataFrame, time_col: str, value_cols, lags=(1,),
             differences: bool = True, rolling_windows=(3,), percent_change: bool = False
             ) -> pd.DataFrame:
    """Build lag, difference and rolling-mean columns ready to use as model features.

    The frame is sorted by time first, which matters: a lag on unsorted rows is
    meaningless and is a classic silent bug in hand-written preprocessing.
    """
    cols = [c for c in (value_cols or []) if c in frame.columns]
    if not cols:
        raise ValueError("choose at least one column to lag")
    work = ordered(frame, time_col, cols)
    out = work.copy()
    made = []
    for column in cols:
        for lag in lags:
            lag = int(lag)
            if lag > 0:
                name = f"{column} (lag {lag})"
                out[name] = out[column].shift(lag)
            elif lag < 0:
                name = f"{column} (lead {abs(lag)})"
                out[name] = out[column].shift(lag)
            else:
                continue
            made.append(name)
        for window in rolling_windows:
            window = int(window)
            if window < 2:
                continue
            name = f"{column} (mean of last {window})"
            out[name] = out[column].rolling(window, min_periods=2).mean()
            made.append(name)
        if differences:
            name = f"{column} (change per step)"
            out[name] = out[column].diff()
            made.append(name)
        if percent_change:
            name = f"{column} (percent change)"
            out[name] = out[column].pct_change() * 100
            made.append(name)
    return out, made


# --------------------------------------------------------------------------
# trend
# --------------------------------------------------------------------------

def mann_kendall(values, alpha: float = 0.05) -> dict:
    """The rank-based trend test — the one to quote for monitoring data.

    Returns Kendall's S, the variance with the tie correction, the normal approximation
    Z, the two-sided p-value, and the Kendall tau.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return {"n": n, "note": "Mann–Kendall needs at least four readings"}
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(x[i + 1:] - x[i])))
    _, counts = np.unique(x, return_counts=True)
    tie_term = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        return {"n": n, "S": s, "note": "every reading is identical"}
    if s > 0:
        z = (s - 1) / math.sqrt(variance)
    elif s < 0:
        z = (s + 1) / math.sqrt(variance)
    else:
        z = 0.0
    p = float(2 * (1 - sps.norm.cdf(abs(z))))
    tau = s / (0.5 * n * (n - 1))
    direction = "increasing" if s > 0 else ("decreasing" if s < 0 else "flat")
    if p >= alpha:
        direction = "no detectable trend"
    return {"n": n, "S": s, "Z": z, "p": p, "tau": tau, "trend": direction,
            "significant at 5%?": "yes" if p < alpha else "no"}


def sens_slope(x, y, level: float = 0.95) -> dict:
    """Sen's (Theil) slope: the median of all pairwise slopes.

    Preferred over least squares for monitoring data because one bad reading — a sensor
    spike — cannot drag it far.  The confidence interval comes from SciPy.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 4:
        return {"note": "Sen's slope needs at least four points"}
    try:
        slope, intercept, low, high = sps.theilslopes(ys, xs, level)
    except Exception as exc:  # noqa: BLE001
        return {"note": f"Sen's slope could not be computed: {exc}"}
    return {"slope": float(slope), "intercept": float(intercept),
            "ci low": float(low), "ci high": float(high), "level": level}


def trend(frame: pd.DataFrame, time_col: str, value_col: str, level: float = 0.95) -> dict:
    """Everything a monitoring report needs about one series' direction of travel."""
    work = ordered(frame, time_col, [value_col])
    work = work.dropna(subset=[value_col])
    if len(work) < 4:
        return {"column": value_col, "note": "need at least four readings with a valid date"}
    years, origin = time_axis(work, time_col)
    values = work[value_col].to_numpy(dtype=float)

    ols = sps.linregress(years, values)
    predicted = ols.intercept + ols.slope * years
    residuals = values - predicted
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((values - values.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    mk = mann_kendall(values)
    sen = sens_slope(years, values)
    span_years = float(years.max() - years.min())
    change = float(ols.slope * span_years)

    out = {
        "column": value_col,
        "first reading": float(values[0]), "last reading": float(values[-1]),
        "n": int(len(values)),
        "span (years)": span_years,
        "slope per year": float(ols.slope),
        "slope 95% low": float(ols.slope - sps.t.ppf(0.5 + level / 2, len(values) - 2) * ols.stderr),
        "slope 95% high": float(ols.slope + sps.t.ppf(0.5 + level / 2, len(values) - 2) * ols.stderr),
        "slope std error": float(ols.stderr),
        "p (linear)": float(ols.pvalue),
        "R²": float(r2),
        "total change": change,
        "relative change %": float(change / values[0] * 100) if values[0] else float("nan"),
        "mean": float(values.mean()),
        "Mann–Kendall Z": mk.get("Z", float("nan")),
        "Mann–Kendall p": mk.get("p", float("nan")),
        "Kendall tau": mk.get("tau", float("nan")),
        "Sen slope per year": sen.get("slope", float("nan")),
        "Sen slope 95% low": sen.get("ci low", float("nan")),
        "Sen slope 95% high": sen.get("ci high", float("nan")),
        "verdict": mk.get("trend", "—"),
        "residuals normal?": "yes" if _normal(residuals) else "no",
    }
    out["sentence"] = _trend_sentence(out)
    return out


def _normal(values) -> bool:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return True
    try:
        return float(sps.shapiro(x).pvalue) >= 0.05
    except Exception:  # noqa: BLE001
        return True


def _trend_sentence(info: dict) -> str:
    """A sentence that could go straight into a report."""
    slope = info["slope per year"]
    low, high = info["slope 95% low"], info["slope 95% high"]
    p = info["p (linear)"]
    up_down = "rising" if slope > 0 else "falling"
    if not math.isfinite(p):
        return "not enough data to judge a trend"
    if p >= 0.05:
        return (f"No significant trend over {info['n']} readings: the best-fit slope is "
                f"{slope:,.4g} per year (p = {p:.3f}).")
    return (f"{info['column']} is {up_down} by {abs(slope):,.4g} per year "
            f"(95% CI {min(abs(low), abs(high)):,.4g} to {max(abs(low), abs(high)):,.4g}, "
            f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}), a total change of "
            f"{info['total change']:,.4g} over the record.")


# --------------------------------------------------------------------------
# seasonality
# --------------------------------------------------------------------------

def profile_by_period(frame: pd.DataFrame, time_col: str, value_col: str,
                      period: str = "month") -> pd.DataFrame:
    """Average by month / weekday / hour — the seasonal signature in one table."""
    work = ordered(frame, time_col, [value_col]).dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame()
    times = work[time_col]
    if period == "month":
        keys = times.dt.month
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                 "Nov", "Dec"]
        key_names = {i + 1: n for i, n in enumerate(names)}
    elif period == "weekday":
        keys = times.dt.dayofweek
        key_names = dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    elif period == "hour":
        keys = times.dt.hour
        key_names = {h: f"{h:02d}:00" for h in range(24)}
    elif period == "quarter":
        keys = times.dt.quarter
        key_names = {q: f"Q{q}" for q in range(1, 5)}
    elif period == "year":
        keys = times.dt.year
        key_names = {y: str(y) for y in sorted(keys.unique())}
    else:
        raise ValueError(f"unknown period {period!r}")
    grouped = work.groupby(keys)[value_col]
    table = pd.DataFrame({"mean": grouped.mean(), "median": grouped.median(),
                          "std": grouped.std(), "min": grouped.min(),
                          "max": grouped.max(), "readings": grouped.count()})
    table.index = [key_names.get(k, str(k)) for k in table.index]
    table.index.name = period
    return table.reset_index()


def autocorrelation(values, nlags: int = 20) -> pd.DataFrame:
    """ACF and PACF — how far back today's reading reaches.

    PACF is computed with the Durbin–Levinson recursion rather than pulled from
    statsmodels, which is not installed.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return pd.DataFrame()
    nlags = int(min(nlags, n - 2))
    centred = x - x.mean()
    denominator = float((centred ** 2).sum())
    if denominator == 0:
        return pd.DataFrame()
    acf = [1.0]
    for lag in range(1, nlags + 1):
        acf.append(float((centred[lag:] * centred[:-lag]).sum() / denominator))

    pacf = [1.0]
    phi = np.zeros((nlags + 1, nlags + 1))
    for k in range(1, nlags + 1):
        if k == 1:
            phi[1, 1] = acf[1]
        else:
            numerator = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
            denominator_k = 1 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
            phi[k, k] = numerator / denominator_k if denominator_k else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf.append(float(phi[k, k]))

    confidence = 1.96 / math.sqrt(n)
    return pd.DataFrame({
        "lag": list(range(nlags + 1)),
        "acf": acf, "pacf": pacf,
        "outside the 95% band?": ["yes" if abs(v) > confidence else "no" for v in acf],
        "band": [confidence] * (nlags + 1),
    })


def seasonal_decompose(frame: pd.DataFrame, time_col: str, value_col: str, period: int
                       ) -> dict:
    """Trend, seasonal and residual components, plus Hyndman's strength measures.

    A centred moving average for the trend; the seasonal component is the average of the
    detrended values at each phase, recentred to sum to zero.  This is the textbook
    additive decomposition and is exactly what the STL-lite a script would do.
    """
    work = ordered(frame, time_col, [value_col]).dropna(subset=[value_col]).reset_index(drop=True)
    period = int(period)
    if period < 2 or len(work) < 2 * period:
        return {"note": f"need at least {2 * period} readings for a period of {period}"}
    values = work[value_col].to_numpy(dtype=float)
    series = pd.Series(values, index=work[time_col])

    trend = series.rolling(period, center=True, min_periods=period).mean()
    detrended = series - trend
    phase = np.arange(len(series)) % period
    seasonal_means = pd.Series(detrended.to_numpy()).groupby(phase).mean()
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal = np.array([seasonal_means[i % period] for i in range(len(series))])
    seasonal = np.where(np.isfinite(series.to_numpy()), seasonal, np.nan)
    residual = series.to_numpy() - trend.to_numpy() - seasonal

    def strength(component):
        mask = np.isfinite(component) & np.isfinite(residual)
        if mask.sum() < 3:
            return float("nan")
        remainder = residual[mask]
        total = component[mask] + remainder
        if total.var() == 0:
            return float("nan")
        return float(max(0.0, 1 - remainder.var() / total.var()))

    return {
        "period": period, "n": len(series),
        "trend": trend.to_numpy(), "seasonal": seasonal, "residual": residual,
        "observed": series.to_numpy(), "times": work[time_col].to_numpy(),
        "seasonal strength": strength(seasonal),
        "trend strength": strength(trend.to_numpy()),
        "seasonal amplitude": float(np.nanmax(seasonal) - np.nanmin(seasonal)),
        "note": ("strength is 1 − Var(residual)/Var(component + residual); above 0.6 means "
                 "the component is worth reporting"),
    }


# --------------------------------------------------------------------------
# control charts
# --------------------------------------------------------------------------

#: Nelson's rules, in the wording an SPC reference would use.
RULES = {
    1: "one point beyond 3σ — a large, sudden change",
    2: "nine points in a row on the same side of the centre line — a shift",
    3: "six points in a row all rising or all falling — a drift",
    4: "fourteen points in a row alternating up and down — over-control",
    5: "two of three consecutive points beyond 2σ on the same side",
    6: "four of five consecutive points beyond 1σ on the same side",
    7: "fifteen points in a row within 1σ of the centre — too little variation",
    8: "eight points in a row with none within 1σ of the centre",
}


#: How many consecutive points each rule looks at.  Used to report a rule once per
#: episode rather than once per overlapping window, which is what makes the output
#: readable: a single step change should be one finding, not thirty.
RULE_SPANS = {1: 1, 2: 9, 3: 6, 4: 14, 5: 3, 6: 5, 7: 15, 8: 8}


def _run_starts(flags: list) -> list[int]:
    """The first window of each maximal run of True."""
    return [i for i, flag in enumerate(flags) if flag and (i == 0 or not flags[i - 1])]


def nelson_violations(values, centre: float, sigma: float) -> list[dict]:
    """Apply Nelson's eight rules.  One entry per episode, not per window."""
    x = np.asarray(values, dtype=float)
    hits: list[dict] = []
    n = len(x)
    if sigma <= 0 or n == 0:
        return hits
    z = (x - centre) / sigma

    def record(rule, window_start, detail):
        index = min(window_start + RULE_SPANS[rule] - 1, n - 1)
        hits.append({"rule": rule, "point": int(index + 1), "value": float(x[index]),
                     "rule description": RULES[rule], "what was seen": detail})

    # 1 — a single point beyond 3σ
    for i in range(n):
        if abs(z[i]) > 3:
            record(1, i, f"{z[i]:+.2f}σ")

    def flags_for(span, predicate):
        return [bool(predicate(z[i:i + span])) for i in range(max(0, n - span + 1))]

    # 2 — nine on one side
    for i in _run_starts(flags_for(9, lambda w: np.all(w > 0) or np.all(w < 0))):
        record(2, i, f"9 in a row {'above' if z[i] > 0 else 'below'} the centre")

    # 3 — six rising or six falling
    def monotone(w):
        diffs = np.diff(w)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    for i in _run_starts(flags_for(6, monotone)):
        record(3, i, "6 in a row " + ("rising" if z[i + 1] > z[i] else "falling"))

    # 4 — fourteen alternating
    def alternating(w):
        diffs = np.diff(w)
        return bool(len(diffs) > 1 and np.all(diffs[:-1] * diffs[1:] < 0))

    for i in _run_starts(flags_for(14, alternating)):
        record(4, i, "14 alternating up and down")

    # 5 — two of three beyond 2σ, same side
    for side, word in ((1, "high"), (-1, "low")):
        for i in _run_starts(flags_for(3, lambda w, s=side: int((w * s > 2).sum()) >= 2)):
            record(5, i, f"2 of 3 beyond 2σ on the {word} side")

    # 6 — four of five beyond 1σ, same side
    for side, word in ((1, "high"), (-1, "low")):
        for i in _run_starts(flags_for(5, lambda w, s=side: int((w * s > 1).sum()) >= 4)):
            record(6, i, f"4 of 5 beyond 1σ on the {word} side")

    # 7 — fifteen within 1σ
    for i in _run_starts(flags_for(15, lambda w: bool(np.all(np.abs(w) < 1)))):
        record(7, i, "15 in a row within 1σ")

    # 8 — eight with none within 1σ
    for i in _run_starts(flags_for(8, lambda w: bool(np.all(np.abs(w) > 1)))):
        record(8, i, "8 in a row outside 1σ")

    hits.sort(key=lambda hit: (hit["point"], hit["rule"]))
    return hits


def control_chart(frame: pd.DataFrame, time_col: str, value_col: str,
                  chart: str = "individuals", subgroup: int = 5, sigma_limits: float = 3.0,
                  ewma_lambda: float = 0.2, cusum_k: float = 0.5, cusum_h: float = 5.0
                  ) -> dict:
    """Build a control chart and check it against Nelson's rules.

    Returns the centre line, the limits, the estimated σ, the plotting table and every
    rule violation with the point it happened at.
    """
    work = ordered(frame, time_col, [value_col]).dropna(subset=[value_col]).reset_index(drop=True)
    if len(work) < 5:
        return {"note": "a control chart needs at least five readings"}
    values = work[value_col].to_numpy(dtype=float)
    times = work[time_col]
    subgroup = int(max(2, min(subgroup, 10)))

    if chart == "xbar_r":
        n_full = (len(values) // subgroup) * subgroup
        if n_full < subgroup * 2:
            return {"note": f"need at least {subgroup * 2} readings for subgroups of {subgroup}"}
        blocked = values[:n_full].reshape(-1, subgroup)
        group_means = blocked.mean(axis=1)
        group_ranges = blocked.max(axis=1) - blocked.min(axis=1)
        constants = CHART_CONSTANTS[subgroup]
        grand = float(group_means.mean())
        r_bar = float(group_ranges.mean())
        sigma = r_bar / constants["d2"]
        x_limits = (grand - constants["A2"] * r_bar, grand + constants["A2"] * r_bar)
        r_limits = (constants["D3"] * r_bar, constants["D4"] * r_bar)
        points = pd.DataFrame({
            "subgroup": np.arange(1, len(group_means) + 1),
            "mean": group_means, "range": group_ranges,
            "lower": x_limits[0], "upper": x_limits[1],
            "centre": grand,
            "range lower": r_limits[0], "range upper": r_limits[1],
            "range centre": r_bar,
        })
        violations = nelson_violations(group_means, grand, (x_limits[1] - grand) / sigma_limits)
        return {"chart": "xbar_r", "subgroup": subgroup, "centre": grand, "sigma": sigma,
                "lower": x_limits[0], "upper": x_limits[1],
                "range centre": r_bar, "range lower": r_limits[0], "range upper": r_limits[1],
                "limits multiples": sigma_limits, "points": points, "raw": work,
                "violations": violations, "n": len(group_means),
                "basis": f"R̄/d₂ with d₂ = {CHART_CONSTANTS[subgroup]['d2']} for n = {subgroup}"}

    if chart == "cusum":
        # The CUSUM target has to be the *in-control* mean, not the mean of everything —
        # otherwise the calm readings before a step sit below the inflated grand mean and
        # the chart fires early on the wrong side.  The first quarter of the record is the
        # reference period, and σ comes from the mean moving range, which a single shift
        # barely moves.
        reference = max(5, len(values) // 4)
        target = float(np.median(values[:reference]))
        moving = np.abs(np.diff(values))
        sigma = float(moving.mean() / CHART_CONSTANTS[2]["d2"]) if len(moving) else 0.0
        sigma = sigma or float(values.std(ddof=1)) or 1.0
        # The two arms of a tabular CUSUM are not symmetric in the offset: the upper arm
        # subtracts the slack k and the lower arm adds it.  Using one series for both
        # biases the chart downward and it alarms on every in-control record.
        z = (values - target) / sigma
        up_dev = z - cusum_k
        low_dev = z + cusum_k
        high = np.zeros(len(values))
        low = np.zeros(len(values))
        for i in range(1, len(values)):
            high[i] = max(0.0, high[i - 1] + up_dev[i])
            low[i] = min(0.0, low[i - 1] + low_dev[i])
        points = pd.DataFrame({"index": np.arange(1, len(values) + 1), "value": values,
                               "cusum high": high, "cusum low": low,
                               "high limit": cusum_h, "low limit": -cusum_h,
                               "centre": 0.0})
        # The statistic is *not* reset at a signal.  A sustained shift therefore crosses
        # the limit once and stays across it, so the crossing marks the change point —
        # exactly the answer wanted.  Resetting would instead produce one alarm every few
        # points for as long as the shift lasted, which buries the one useful number.
        hits = []
        above = [bool(v > cusum_h) for v in high]
        below = [bool(v < -cusum_h) for v in low]
        for i in _run_starts(above):
            hits.append({"rule": "CUSUM", "point": int(i + 1), "value": float(values[i]),
                         "rule description": f"the upper CUSUM first passed h = {cusum_h}",
                         "what was seen": f"a sustained rise starting around reading {i + 1}"})
        for i in _run_starts(below):
            hits.append({"rule": "CUSUM", "point": int(i + 1), "value": float(values[i]),
                         "rule description": f"the lower CUSUM first passed −h = −{cusum_h}",
                         "what was seen": f"a sustained fall starting around reading {i + 1}"})
        hits.sort(key=lambda hit: hit["point"])
        return {"chart": "cusum", "centre": target, "sigma": sigma, "lower": -cusum_h,
                "upper": cusum_h, "points": points, "raw": work, "violations": hits,
                "n": len(values), "k": cusum_k, "h": cusum_h,
                "basis": f"tabular CUSUM on the standardised deviation, k = {cusum_k}, "
                         f"h = {cusum_h}, target from the first {reference} readings"}

    if chart == "ewma":
        lam = float(min(max(ewma_lambda, 0.01), 1.0))
        target = float(values.mean())
        sigma = float(values.std(ddof=1)) or 1.0
        smoothed = np.zeros(len(values))
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = lam * values[i] + (1 - lam) * smoothed[i - 1]
        widths = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * np.arange(1, len(values) + 1))))
        lower = target - sigma_limits * widths
        upper = target + sigma_limits * widths
        points = pd.DataFrame({"index": np.arange(1, len(values) + 1), "value": values,
                               "ewma": smoothed, "centre": target,
                               "lower": lower, "upper": upper})
        # An EWMA is checked against its own widening limits.  Nelson's run rules are
        # *not* applied: the smoothed series is deliberately autocorrelated, so run rules
        # would fire on the smoothing itself rather than on the process.
        hits = []
        for i in range(len(values)):
            if smoothed[i] > upper[i] or smoothed[i] < lower[i]:
                hits.append({"rule": "EWMA", "point": int(i + 1), "value": float(values[i]),
                             "rule description": "the smoothed value passed its control limit",
                             "what was seen": f"EWMA {smoothed[i]:.4g} against "
                                              f"{lower[i]:.4g} to {upper[i]:.4g}"})
        return {"chart": "ewma", "centre": target, "sigma": sigma,
                "lower": float(np.mean(lower)), "upper": float(np.mean(upper)),
                "points": points, "raw": work, "violations": hits, "n": len(values),
                "lambda": lam, "basis": f"EWMA, λ = {lam}, ±{sigma_limits}σ time-varying limits"}

    # individuals & moving range
    mean = float(values.mean())
    moving_ranges = np.abs(np.diff(values))
    mr_bar = float(moving_ranges.mean()) if len(moving_ranges) else 0.0
    sigma = mr_bar / CHART_CONSTANTS[2]["d2"] if mr_bar else float(values.std(ddof=1))
    sigma = sigma or 1.0
    lower, upper = mean - sigma_limits * sigma, mean + sigma_limits * sigma
    mr_upper = CHART_CONSTANTS[2]["D4"] * mr_bar
    points = pd.DataFrame({
        "index": np.arange(1, len(values) + 1), "value": values,
        "centre": mean, "lower": lower, "upper": upper,
    })
    return {"chart": "individuals", "centre": mean, "sigma": sigma, "lower": lower,
            "upper": upper, "points": points, "raw": work,
            "violations": nelson_violations(values, mean, sigma),
            "n": len(values), "moving range mean": mr_bar,
            "moving range upper": mr_upper,
            "basis": f"σ from the mean moving range ÷ d₂ = {CHART_CONSTANTS[2]['d2']}"
                     if mr_bar else "σ from the sample standard deviation"}


def process_capability(frame: pd.DataFrame, time_col: str, value_col: str,
                       lower_spec: float | None = None, upper_spec: float | None = None
                       ) -> dict:
    """Cp, Cpk and the expected defect rate against a specification.

    This is the number a concrete supplier or a precast yard is actually asked for, and
    it is three lines of arithmetic that nobody should have to look up twice.
    """
    work = ordered(frame, time_col, [value_col]).dropna(subset=[value_col])
    values = work[value_col].to_numpy(dtype=float)
    if len(values) < 5:
        return {"note": "need at least five readings"}
    mean = float(values.mean())
    sigma = float(values.std(ddof=1)) or 1.0
    out = {"n": len(values), "mean": mean, "sigma": sigma,
           "lower spec": lower_spec, "upper spec": upper_spec}
    if lower_spec is None and upper_spec is None:
        out["note"] = "give at least one specification limit"
        return out
    if lower_spec is not None and upper_spec is not None:
        out["Cp"] = (upper_spec - lower_spec) / (6 * sigma)
    if lower_spec is not None:
        out["Cpl"] = (mean - lower_spec) / (3 * sigma)
    if upper_spec is not None:
        out["Cpu"] = (upper_spec - mean) / (3 * sigma)
    candidates = [v for v in (out.get("Cpl"), out.get("Cpu")) if v is not None]
    if candidates:
        out["Cpk"] = min(candidates)
    # Expected fraction outside the limits, from the normal distribution
    below = float(sps.norm.cdf((lower_spec - mean) / sigma)) if lower_spec is not None else 0.0
    above = float(sps.norm.sf((upper_spec - mean) / sigma)) if upper_spec is not None else 0.0
    out["expected below spec %"] = below * 100
    out["expected above spec %"] = above * 100
    out["expected outside spec %"] = (below + above) * 100
    cpk = out.get("Cpk")
    if cpk is not None:
        out["verdict"] = ("capable (Cpk ≥ 1.33)" if cpk >= 1.33
                          else "marginal (1.00 ≤ Cpk < 1.33)" if cpk >= 1.0
                          else "not capable (Cpk < 1.00)")
    return out
