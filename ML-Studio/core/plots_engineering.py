"""Figures for the engineering screens.

These live apart from :mod:`core.plots` on purpose.  Everything in the plot registry is
driven by a :class:`~core.plots.PlotContext` — the loaded frame, the fitted model, the
evaluation tables — and is therefore reusable from any screen.  These figures are not:
each one draws a *result* that a single screen computed (a fitted curve, a control chart,
a simulation histogram), so they take that result directly.

Sharing the styling is still worth it, so the palette, the fonts and the render lock all
come from :mod:`core.plots` and the charts sit next to the registered ones without looking
like they came from a different program.

Every function returns a matplotlib figure and raises :class:`core.plots.PlotError` with a
sentence the screen can show verbatim when it cannot draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plots as P

__all__ = [
    "draw", "curve_fit_plot", "residual_plot", "prediction_band_plot", "fit_ranking_plot",
    "control_chart_plot", "trend_plot", "acf_plot", "decomposition_plot", "profile_plot",
    "doe_pareto_plot", "doe_main_effects_plot", "design_plot",
    "simulation_histogram", "tornado_plot", "exceedance_plot", "influence_plot",
    "distribution_review_plot", "group_comparison_plot", "correlation_plot",
]


def draw(function, *args, **kwargs):
    """Call a figure function while holding the shared matplotlib lock."""
    with P.RENDER_LOCK:
        if not P.STYLE_APPLIED:
            P.apply_style()
        return function(*args, **kwargs)


def _finish(fig, note: str = ""):
    if note:
        fig.text(0.995, 0.005, note, ha="right", va="bottom", fontsize=7.2, color="#6b7280")
    return fig


def _usable(fit) -> bool:
    """A fit is drawable when it converged onto finite parameters."""
    if fit is None:
        return False
    if not getattr(fit, "converged", False):
        return False
    params = getattr(fit, "params", None)
    return params is not None and np.all(np.isfinite(np.asarray(params, dtype=float)))


def _fit_note(fit) -> str:
    return str(getattr(fit, "note", "") or getattr(fit, "problem", "") or "")


# --------------------------------------------------------------------------
# curve fitting
# --------------------------------------------------------------------------

def curve_fit_plot(fit, x=None, y=None, xlabel: str = "x", ylabel: str = "y",
                   log_x: bool = False, log_y: bool = False, targets=None,
                   show_points: bool = True) -> "object":
    """The fitted curve over the data, with its confidence band and any solved points."""
    if fit is None:
        raise P.PlotError("nothing has been fitted yet")
    fig = P.figure((7.2, 4.6))
    ax = fig.add_subplot(111)

    x_values = np.asarray(getattr(fit, "x", x) if x is None else x, dtype=float)
    y_values = np.asarray(getattr(fit, "y", y) if y is None else y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values) == 0:
        raise P.PlotError("there is no data to plot")

    if show_points:
        ax.scatter(x_values, y_values, s=18, alpha=0.55, edgecolor="none",
                   color="#2f6fb3", label="observed", zorder=3)

    if not _usable(fit):
        ax.set_title("no curve could be fitted")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return _finish(fig, _fit_note(fit))

    if log_x and x_values.min() > 0:
        grid = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), 250)
    else:
        pad = 0.03 * (x_values.max() - x_values.min() or 1)
        grid = np.linspace(x_values.min() - pad, x_values.max() + pad, 250)
    try:
        curve = np.asarray(fit.predict(grid), dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise P.PlotError(f"the fitted curve could not be drawn: {exc}") from exc
    if log_y:
        curve = np.where(curve > 0, curve, np.nan)

    ax.plot(grid, curve, color="#c0392b", linewidth=2.0, label=fit.label, zorder=4)
    try:
        _y, lower, upper = fit.predict_with_band(grid)
        ax.fill_between(grid, lower, upper, color="#c0392b", alpha=0.12,
                        label="95% band for new readings", zorder=2)
    except Exception:  # noqa: BLE001
        pass

    if targets:
        for target in np.atleast_1d(targets):
            try:
                solution = fit.solve_for_x(float(target))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(solution, dict) or "x" not in solution:
                continue
            position = float(solution["x"])
            if np.isfinite(position):
                ax.axhline(float(target), color="#7f8c8d", linewidth=0.8, linestyle=":")
                ax.axvline(position, color="#16a085", linewidth=1.2, linestyle="--")
                ax.annotate(f"x = {position:,.4g}\nfor y = {float(target):,.4g}",
                            xy=(position, float(target)), xytext=(8, 10),
                            textcoords="offset points", fontsize=8, color="#0e6655",
                            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#16a085", lw=0.7))

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{fit.label}  ·  R² = {fit.r2:,.4f}")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    note = "" if np.isfinite(fit.aicc) else "AICc not defined"
    return _finish(fig, note)


def residual_plot(fit) -> "object":
    """Residuals against the fitted value — the standard check for a wrong shape."""
    if not _usable(fit):
        raise P.PlotError("fit a curve first, and make sure it converged")
    fig, axes = P.plt.subplots(1, 2, figsize=(9.4, 3.8))
    fitted = np.asarray(fit.predict(fit.x), dtype=float)
    residuals = np.asarray(fit.y, dtype=float) - fitted

    axes[0].axhline(0, color="#7f8c8d", linewidth=0.9)
    axes[0].scatter(fitted, residuals, s=16, alpha=0.6, color="#2f6fb3", edgecolor="none")
    axes[0].set_xlabel("fitted value")
    axes[0].set_ylabel("residual (observed − fitted)")
    axes[0].set_title("Residuals against the fit")

    axes[1].scatter(np.asarray(fit.x, dtype=float), residuals, s=16, alpha=0.6,
                    color="#2f6fb3", edgecolor="none")
    axes[1].axhline(0, color="#7f8c8d", linewidth=0.9)
    axes[1].set_xlabel("input value")
    axes[1].set_title("Residuals against the input")

    if len(residuals) >= 3:
        for ax in axes:
            spread = residuals.std(ddof=1)
            ax.axhline(2 * spread, color="#c0392b", linewidth=0.8, linestyle="--")
            ax.axhline(-2 * spread, color="#c0392b", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    skew = float(pd.Series(residuals).skew())
    return _finish(fig, f"residual sd {residuals.std(ddof=1):,.4g} · skew {skew:+.2f}")


def prediction_band_plot(fit, xlabel: str = "x", ylabel: str = "y") -> "object":
    """The fitted curve with the band drawn as two series, so the numbers can be read off."""
    if not _usable(fit):
        raise P.PlotError("fit a curve first, and make sure it converged")
    x_values = np.asarray(fit.x, dtype=float)
    grid = np.linspace(x_values.min(), x_values.max(), 200)
    _y, lower, upper = fit.predict_with_band(grid)
    fig = P.figure((7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.fill_between(grid, lower, upper, color="#2f6fb3", alpha=0.18)
    ax.plot(grid, fit.predict(grid), color="#1f3864", linewidth=2.0)
    ax.scatter(x_values, np.asarray(fit.y, dtype=float), s=16, alpha=0.6, color="#c0392b",
               edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Fitted curve with the 95% band")
    fig.tight_layout()
    return _finish(fig)


def fit_ranking_plot(table: pd.DataFrame, metric: str = "AICc", limit: int = 12) -> "object":
    """The candidate equations ranked, so the runner-up is visible rather than hidden."""
    if table is None or not len(table):
        raise P.PlotError("fit some forms first")
    if metric not in table.columns:
        metric = [c for c in table.columns if c.lower().startswith("aic")] or [table.columns[0]]
        metric = metric[0]
    frame = table.dropna(subset=[metric]).sort_values(metric).head(int(limit))
    frame = frame.iloc[::-1]
    fig = P.figure((7.4, max(2.6, 0.34 * len(frame) + 1.4)))
    ax = fig.add_subplot(111)
    positions = np.arange(len(frame))
    best = frame[metric].min()
    colours = ["#1f3864" if abs(v - best) < 1e-9 else "#8fa8c8" for v in frame[metric]]
    ax.barh(positions, frame[metric], color=colours, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels([str(v) for v in frame["form"]], fontsize=8)
    ax.set_xlabel(metric + ("  (lower is better)" if metric.lower().startswith("aic") else ""))
    ax.set_title(f"Equations ranked by {metric}")
    for y, value in zip(positions, frame[metric]):
        ax.text(value, y, f" {value:,.1f}", va="center", fontsize=7.5, color="#374151")
    ax.margins(x=0.14)
    fig.tight_layout()
    return _finish(fig)


# --------------------------------------------------------------------------
# time series and control charts
# --------------------------------------------------------------------------

def control_chart_plot(result, date_column: str = "date", value_column: str = "value") -> "object":
    """The control chart itself: the series, its centre line, its limits and every signal.

    Individual charts are drawn the way a QC engineer reads them, with the zone boundaries
    at ±1σ, ±2σ and ±3σ marked faintly, and each violation annotated with the rule number
    so the picture can be matched to the table underneath it.
    """
    if not result:
        raise P.PlotError("run a control chart first")
    points = result.get("points")
    if points is None or not len(points):
        raise P.PlotError("the chart has no points to draw")

    chart = result.get("chart", "individuals")
    fig = P.figure((9.6, 4.2))
    ax = fig.add_subplot(111)

    # The x̄–R chart plots subgroup means rather than readings, and calls its position
    # column "subgroup"; every other chart plots "index" against "value".
    if chart == "xbar_r":
        index = np.asarray(points["subgroup"], dtype=float)
        series = np.asarray(points["mean"], dtype=float)
    else:
        index = np.asarray(points["index"], dtype=float)
        series = (np.asarray(points["value"], dtype=float) if "value" in points.columns
                  else np.asarray(points.iloc[:, 1], dtype=float))

    if chart in ("individuals", "xbar_r"):
        values = series
        centre = float(result["centre"])
        sigma = float(result.get("sigma") or 0.0)
        ax.plot(index, values, marker="o", markersize=3.6, linewidth=1.0, color="#1f3864",
                label="subgroup mean" if chart == "xbar_r" else "reading")
        ax.axhline(centre, color="#16a085", linewidth=1.4, label=f"centre {centre:,.4g}")
        for multiple, style, label in ((1, ":", "±1σ"), (2, "--", "±2σ"), (3, "-", "±3σ")):
            if sigma <= 0:
                continue
            ax.axhline(centre + multiple * sigma, color="#c0392b" if multiple == 3 else "#b0b7c3",
                       linewidth=1.3 if multiple == 3 else 0.9, linestyle=style,
                       label=label if multiple == 3 else None)
            ax.axhline(centre - multiple * sigma, color="#c0392b" if multiple == 3 else "#b0b7c3",
                       linewidth=1.3 if multiple == 3 else 0.9, linestyle=style)
    elif chart == "cusum":
        ax.plot(index, points["cusum high"], linewidth=1.4, color="#c0392b", label="upper CUSUM")
        ax.plot(index, points["cusum low"], linewidth=1.4, color="#2f6fb3", label="lower CUSUM")
        limit = float(result.get("h", 5.0))
        ax.axhline(limit, color="#c0392b", linewidth=1.0, linestyle="--", label=f"+h = {limit:g}")
        ax.axhline(-limit, color="#2f6fb3", linewidth=1.0, linestyle="--", label=f"−h = {limit:g}")
        ax.axhline(0, color="#7f8c8d", linewidth=0.8)
    else:   # ewma
        values = np.asarray(points["value"], dtype=float)
        smoothed = np.asarray(points.get("ewma", pd.Series(values)), dtype=float)
        ax.plot(index, values, linewidth=0.8, color="#b0b7c3", label="reading")
        ax.plot(index, smoothed, linewidth=1.8, color="#1f3864", label="EWMA")
        for key, label in (("upper", "control limits"), ("lower", None)):
            if key in points.columns:
                ax.plot(index, np.asarray(points[key], dtype=float), linewidth=1.0,
                        linestyle="--", color="#c0392b", label=label)
        centre = float(result["centre"])
        ax.axhline(centre, color="#16a085", linewidth=1.2, label=f"target {centre:,.4g}")

    violations = result.get("violations") or []
    if violations:
        x_hits = [v["point"] for v in violations]
        if chart in ("individuals", "xbar_r", "ewma"):
            y_hits = [series[int(p) - 1] if 0 < int(p) <= len(series) else np.nan for p in x_hits]
        else:
            high = np.asarray(points["cusum high"], dtype=float)
            low = np.asarray(points["cusum low"], dtype=float)
            y_hits = []
            for violation in violations:
                position = int(violation["point"]) - 1
                if not 0 <= position < len(high):
                    y_hits.append(np.nan)
                elif "lower" in str(violation.get("rule description", "")).lower():
                    y_hits.append(low[position])
                else:
                    y_hits.append(high[position])
        ax.scatter(x_hits, y_hits, s=64, facecolors="none", edgecolors="#e67e22", linewidths=1.8,
                   zorder=5, label=f"{len(violations)} signal(s)")
        for violation, x_hit, y_hit in zip(violations, x_hits, y_hits):
            if np.isfinite(y_hit):
                ax.annotate(str(violation.get("rule", "!")), xy=(x_hit, y_hit), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=7.5, color="#a04000")

    ax.set_xlabel("reading")
    ax.set_ylabel("cusum statistic" if chart == "cusum" else value_column)
    ax.set_title(result.get("title", f"Control chart — {chart}"))
    ax.legend(loc="best", fontsize=7.5, ncol=2, frameon=True)
    fig.tight_layout()
    return _finish(fig, result.get("basis", "")[:110])


def trend_plot(dates, values, trend_info: dict | None = None, ylabel: str = "value") -> "object":
    """The series with its fitted straight line and a Theil–Sen slope where one was asked for.

    The lines are drawn through the mean of the data — ``mean + slope × (x − mean x)`` is the
    ordinary least-squares line — so the picture needs only the slopes, which is all the
    trend table carries.
    """
    dates = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    values = np.asarray(list(values), dtype=float)
    order = np.argsort(dates.to_numpy())
    dates, values = dates.iloc[order], values[order]
    years = (dates - dates.iloc[0]).dt.total_seconds().to_numpy() / (365.25 * 24 * 3600)
    mean_value = float(np.nanmean(values))
    mean_year = float(np.nanmean(years))

    fig = P.figure((9.4, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(dates, values, linewidth=1.1, color="#2f6fb3", label="observed")
    if trend_info:
        slope = trend_info.get("slope per year")
        if slope is not None and np.isfinite(slope):
            ax.plot(dates, mean_value + slope * (years - mean_year), color="#c0392b",
                    linewidth=1.8, label=f"straight-line fit ({slope:+,.4g}/yr)")
        sen = trend_info.get("Sen slope per year")
        if sen is not None and np.isfinite(sen):
            ax.plot(dates, mean_value + sen * (years - mean_year), color="#16a085",
                    linewidth=1.4, linestyle="--", label=f"Theil–Sen ({sen:+,.4g}/yr)")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("date")
    ax.set_title("Trend")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig)


def acf_plot(frame: pd.DataFrame, title: str = "Autocorrelation") -> "object":
    """The ACF and PACF bar charts side by side — what the AR order is chosen from."""
    if frame is None or not len(frame):
        raise P.PlotError("compute the autocorrelation first")
    columns = [c for c in ("acf", "pacf") if c in frame.columns]
    if not columns:
        raise P.PlotError("the autocorrelation table has no values")
    band = float(frame["band"].iloc[0]) if "band" in frame.columns else float("nan")
    fig, axes = P.plt.subplots(1, len(columns), figsize=(4.7 * len(columns), 3.6), squeeze=False)
    lags = np.asarray(frame["lag"], dtype=float)
    for ax, column in zip(axes[0], columns):
        values = np.asarray(frame[column], dtype=float)
        ax.bar(lags, values, width=0.7, color="#2f6fb3", alpha=0.85)
        ax.axhline(0, color="#7f8c8d", linewidth=0.8)
        if np.isfinite(band):
            ax.axhline(band, color="#c0392b", linewidth=0.9, linestyle="--",
                       label=f"±{band:.3f} band" if column == "acf" else None)
            ax.axhline(-band, color="#c0392b", linewidth=0.9, linestyle="--")
        ax.set_title(column.upper(), fontsize=10)
        ax.set_xlabel("lag")
        ax.set_ylabel("correlation")
        ax.set_ylim(-1.05, 1.05)
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig)


def decomposition_plot(result: dict) -> "object":
    """Observed, trend, seasonal and remainder stacked — the classic decomposition picture."""
    if not result:
        raise P.PlotError("decompose the series first")
    observed = np.asarray(result["observed"], dtype=float)
    trend = np.asarray(result["trend"], dtype=float)
    seasonal = np.asarray(result["seasonal"], dtype=float)
    residual = np.asarray(result["residual"], dtype=float)
    x = np.arange(len(observed))
    fig, axes = P.plt.subplots(4, 1, figsize=(9.4, 7.4), sharex=True)
    for ax, series, label, colour in (
            (axes[0], observed, "observed", "#1f3864"),
            (axes[1], trend, "trend", "#c0392b"),
            (axes[2], seasonal, "seasonal", "#16a085"),
            (axes[3], residual, "remainder", "#7f8c8d")):
        ax.plot(x, series, linewidth=1.1, color=colour)
        ax.set_ylabel(label)
    axes[0].set_title(result.get("note", "Seasonal decomposition"))
    axes[3].set_xlabel("position in the series")
    fig.tight_layout()
    return _finish(fig, f"seasonal strength {result.get('seasonal strength', float('nan')):.3f}")


def profile_plot(table: pd.DataFrame, period_column: str | None = None) -> "object":
    """The average value in each season — which month is wet, which is dry.

    The period column is called whatever the period is (``month``, ``weekday``…), so it is
    found by elimination rather than guessed.
    """
    if table is None or not len(table):
        raise P.PlotError("profile the series first")
    statistics = {"mean", "median", "std", "min", "max", "readings", "count", "n"}
    if period_column is None:
        candidates = [c for c in table.columns if str(c) not in statistics]
        if not candidates:
            raise P.PlotError("the profile table has no period column")
        period_column = candidates[0]
    value_column = "mean" if "mean" in table.columns else next(
        (c for c in table.columns if c != period_column), None)
    if value_column is None:
        raise P.PlotError("the profile table has no values")
    fig = P.figure((8.2, 3.8))
    ax = fig.add_subplot(111)
    labels = [str(v) for v in table[period_column]]
    positions = np.arange(len(table))
    ax.bar(positions, table[value_column], color="#2f6fb3", alpha=0.9)
    count_column = "readings" if "readings" in table.columns else (
        "count" if "count" in table.columns else None)
    if count_column:
        top = float(pd.to_numeric(table[value_column], errors="coerce").max())
        for position, count in zip(positions, table[count_column]):
            ax.text(position, 0, f"n={int(count)}", ha="center", va="bottom", fontsize=6.6,
                    rotation=90, color="#374151")
    overall = float(pd.to_numeric(table[value_column], errors="coerce").mean())
    ax.axhline(overall, color="#c0392b", linewidth=1.2, linestyle="--",
               label=f"average {overall:,.4g}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean {value_column}")
    ax.set_title(f"Average by {period_column}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig)


# --------------------------------------------------------------------------
# design of experiments
# --------------------------------------------------------------------------

def doe_pareto_plot(table: pd.DataFrame, contribution: str = "% contribution") -> "object":
    """The Pareto of term contributions — the one chart that says what mattered."""
    if table is None or not len(table):
        raise P.PlotError("analyse the experiment first")
    frame = table.copy()
    if contribution not in frame.columns:
        candidates = [c for c in frame.columns if "%" in str(c) or "sum of squares" in str(c)]
        if not candidates:
            raise P.PlotError("the analysis table has no contribution column")
        contribution = candidates[0]
    frame = frame.dropna(subset=[contribution]).sort_values(contribution)
    fig = P.figure((7.8, max(2.6, 0.36 * len(frame) + 1.5)))
    ax = fig.add_subplot(111)
    positions = np.arange(len(frame))
    total = float(frame[contribution].sum()) or 1.0
    running = np.cumsum(frame[contribution].to_numpy(dtype=float)[::-1])[::-1] / total * 100
    colours = ["#1f3864" if i >= len(frame) - 3 else "#8fa8c8" for i in range(len(frame))]
    ax.barh(positions, frame[contribution], color=colours, height=0.66)
    for y, value in zip(positions, frame[contribution]):
        ax.text(value, y, f" {value:,.1f}%", va="center", fontsize=7.5, color="#374151")
    ax.set_yticks(positions)
    ax.set_yticklabels([str(v) for v in frame["term"]], fontsize=8.5)
    ax.set_xlabel("% of the explained variation")
    ax.set_title("Which terms mattered")
    ax.margins(x=0.16)
    twist = ax.twiny()
    twist.plot(running, positions, color="#c0392b", marker="o", markersize=3.4, linewidth=1.1)
    twist.set_xlim(0, 105)
    twist.set_xlabel("cumulative %", color="#c0392b", fontsize=8)
    twist.tick_params(axis="x", labelsize=7.5, colors="#c0392b")
    fig.tight_layout()
    return _finish(fig)


def doe_main_effects_plot(table: pd.DataFrame) -> "object":
    """The average response at each level of each factor, one line per factor."""
    if table is None or not len(table):
        raise P.PlotError("compute the main effects first")
    level_columns = [c for c in table.columns if str(c).startswith("mean at")]
    if not level_columns:
        raise P.PlotError("the main-effects table has no level columns")
    fig = P.figure((7.8, 4.2))
    ax = fig.add_subplot(111)
    palette = P.plt.get_cmap("tab10")
    for i, record in enumerate(table.itertuples(index=False)):
        row = dict(zip(table.columns, record))
        values = [row[c] for c in level_columns if row[c] == row[c]]
        labels = [c.replace("mean at", "").strip() for c in level_columns
                  if row[c] == row[c]]
        ax.plot(range(len(values)), values, marker="o", linewidth=1.8,
                color=palette(i % 10), label=str(row["factor"]))
        for position, value in enumerate(values):
            ax.annotate(f"{value:,.4g}", (position, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=palette(i % 10))
    ax.set_xticks(range(len(level_columns)))
    ax.set_xticklabels([c.replace("mean at", "coded").strip() for c in level_columns])
    ax.set_xlabel("factor level (coded)")
    ax.set_ylabel("average response")
    ax.set_title("Main effects — the steeper the line, the more that factor matters")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig)


def design_plot(design, factors=None) -> "object":
    """Where the runs sit in the factor space — the picture that shows a design is balanced."""
    if design is None:
        raise P.PlotError("build a design first")
    coded = design.coded
    if coded is None or coded.shape[1] < 2:
        raise P.PlotError("a design scatter needs at least two factors")
    columns = list(coded.columns)[:4]
    frame = coded[columns].to_numpy(dtype=float)
    if len(columns) == 2:
        fig = P.figure((5.4, 4.8))
        ax = fig.add_subplot(111)
        ax.scatter(frame[:, 0], frame[:, 1], s=46, color="#2f6fb3", alpha=0.85,
                   edgecolor="#1f3864")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        axes = [ax]
    else:
        count = len(columns) * (len(columns) - 1) // 2
        fig, grid = P.plt.subplots(1, count, figsize=(3.5 * count, 3.6), squeeze=False)
        axes = []
        position = 0
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                ax = grid[0][position]
                position += 1
                axes.append(ax)
                ax.scatter(frame[:, i], frame[:, j], s=34, color="#2f6fb3", alpha=0.8,
                           edgecolor="#1f3864")
                ax.set_xlabel(columns[i], fontsize=9)
                ax.set_ylabel(columns[j], fontsize=9)
    for ax in axes:
        ax.axhline(0, color="#d5d8dd", linewidth=0.8)
        ax.axvline(0, color="#d5d8dd", linewidth=0.8)
    fig.suptitle(f"{design.kind} — {len(design)} runs in coded units")
    fig.tight_layout()
    return _finish(fig)


# --------------------------------------------------------------------------
# uncertainty and reliability
# --------------------------------------------------------------------------

def simulation_histogram(values, threshold=None, direction: str = "above",
                         xlabel: str = "predicted outcome") -> "object":
    """The simulated outcome distribution, with the part that fails shaded in."""
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise P.PlotError("there are no simulated values to draw")
    fig = P.figure((8.6, 4.4))
    ax = fig.add_subplot(111)
    counts, edges, patches = ax.hist(values, bins=min(60, max(12, int(np.sqrt(len(values)) * 1.6))),
                                     color="#8fa8c8", edgecolor="white", linewidth=0.4)
    if threshold is not None:
        failing = (values >= threshold) if direction == "above" else (values <= threshold)
        for patch, left, right in zip(patches, edges[:-1], edges[1:]):
            middle = (left + right) / 2
            if (middle >= threshold) if direction == "above" else (middle <= threshold):
                patch.set_facecolor("#c0392b")
                patch.set_alpha(0.75)
        share = failing.mean() * 100
        ax.axvline(threshold, color="#7b241c", linewidth=1.8)
        ax.annotate(f"limit {threshold:,.4g}\n{share:,.2f}% of simulations on the failing side",
                    xy=(threshold, counts.max() * 0.88), xytext=(12 if direction == "above" else -12, 0),
                    textcoords="offset points", fontsize=8.5, color="#7b241c",
                    ha="left" if direction == "above" else "right",
                    bbox=dict(boxstyle="round,pad=0.34", fc="#fdedec", ec="#c0392b", lw=0.8))
    mean = float(values.mean())
    ax.axvline(mean, color="#16a085", linewidth=1.4, linestyle="--", label=f"mean {mean:,.4g}")
    for label, value, colour in (("P5", np.percentile(values, 5), "#7f8c8d"),
                                 ("P50", np.percentile(values, 50), "#7f8c8d"),
                                 ("P95", np.percentile(values, 95), "#7f8c8d")):
        ax.axvline(value, color=colour, linewidth=0.8, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of simulations")
    ax.set_title(f"Simulated outcome · {len(values):,} runs")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig, "dotted lines: P5, P50, P95")


def tornado_plot(table: pd.DataFrame, baseline: float | None = None) -> "object":
    """One bar per uncertain input, longest first — the priority list."""
    if table is None or not len(table):
        raise P.PlotError("run a sensitivity sweep first")
    frame = table.dropna(subset=["swing"]).sort_values("swing")
    fig = P.figure((8.2, max(2.6, 0.42 * len(frame) + 1.6)))
    ax = fig.add_subplot(111)
    positions = np.arange(len(frame))
    swing = frame["swing"].to_numpy(dtype=float)
    colours = ["#1f3864" if i >= len(frame) - 3 else "#8fa8c8" for i in range(len(frame))]
    ax.barh(positions, swing, color=colours, height=0.62)
    for y, value in zip(positions, swing):
        ax.text(value, y, f" {value:,.4g}", va="center", fontsize=7.8, color="#374151")
    ax.set_yticks(positions)
    ax.set_yticklabels([str(v) for v in frame["input"]], fontsize=8.5)
    if baseline is not None and np.isfinite(baseline):
        ax.set_xlabel(f"change in the outcome, shown from 0  (baseline {baseline:,.4g})")
    else:
        ax.set_xlabel("change in the outcome")
    ax.set_title("What the answer is sensitive to")
    ax.margins(x=0.18)
    fig.tight_layout()
    return _finish(fig)


def exceedance_plot(frame: pd.DataFrame, xlabel: str = "limit value") -> "object":
    """Probability of exceeding each limit — the curve a design check is read from."""
    if frame is None or not len(frame):
        raise P.PlotError("build an exceedance curve first")
    fig = P.figure((8.0, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(frame["limit"], frame["probability above %"], color="#c0392b", linewidth=1.9,
            label="probability the outcome is ABOVE the limit")
    if "probability below %" in frame.columns:
        ax.plot(frame["limit"], frame["probability below %"], color="#2f6fb3", linewidth=1.9,
                linestyle="--", label="probability it is BELOW the limit")
    ax.axhline(50, color="#d5d8dd", linewidth=0.9)
    for level, colour in ((5, "#16a085"), (95, "#16a085")):
        ax.axhline(level, color=colour, linewidth=0.9, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability, %")
    ax.set_title("Probability of exceeding a limit")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    return _finish(fig)


def influence_plot(table: pd.DataFrame) -> "object":
    """Rank correlation between each input and the outcome, signed."""
    if table is None or not len(table):
        raise P.PlotError("run a simulation first")
    frame = table.dropna(subset=["rank correlation"]).copy()
    frame = frame.reindex(frame["rank correlation"].abs().sort_values().index)
    fig = P.figure((8.0, max(2.4, 0.42 * len(frame) + 1.5)))
    ax = fig.add_subplot(111)
    positions = np.arange(len(frame))
    values = frame["rank correlation"].to_numpy(dtype=float)
    ax.barh(positions, values, color=["#c0392b" if v < 0 else "#2f6fb3" for v in values],
            height=0.6)
    ax.axvline(0, color="#7f8c8d", linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels([str(v) for v in frame["input"]], fontsize=8.5)
    ax.set_xlabel("rank correlation with the outcome")
    ax.set_title("Influence of each input")
    ax.set_xlim(-1.05, 1.05)
    fig.tight_layout()
    return _finish(fig, "bar on the right: raising the input raises the answer")


def distribution_review_plot(sampled: pd.DataFrame, specs=None, limit: int = 4) -> "object":
    """What was actually drawn for each input, so a wrong distribution is obvious."""
    if sampled is None or not len(sampled):
        raise P.PlotError("run a simulation first")
    columns = list(sampled.columns)[:int(limit)]
    if not columns:
        raise P.PlotError("there are no sampled inputs")
    fig, axes = P.plt.subplots(1, len(columns), figsize=(3.3 * len(columns), 3.2), squeeze=False)
    lookup = {s.column: s for s in (specs or [])}
    for ax, column in zip(axes[0], columns):
        values = pd.to_numeric(sampled[column], errors="coerce").dropna().to_numpy(dtype=float)
        spec = lookup.get(column)
        title = column
        if spec is not None:
            title = f"{column}\n{spec.distribution.replace('_', ' ')}"
        ax.set_title(title, fontsize=8.5)
        ax.set_yticks([])
        if len(values) < 2 or np.allclose(values, values[0]):
            # A held value has no distribution to draw.  A flat line of bars would look like
            # a mistake; saying "held at 12.5" is the truth.
            held = values[0] if len(values) else (
                str(pd.Series(sampled[column]).dropna().iloc[0])
                if len(pd.Series(sampled[column]).dropna()) else "nothing")
            ax.hist([], bins=1, color="#c8d2e0")
            ax.text(0.5, 0.5, f"held at\n{held}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9.5, color="#33475f")
            continue
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))), color="#8fa8c8",
                edgecolor="white", linewidth=0.3)
    fig.suptitle("The inputs as they were actually drawn", fontsize=10)
    fig.tight_layout()
    return _finish(fig)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def group_comparison_plot(frame: pd.DataFrame, group_column: str, value_column: str,
                          result: dict | None = None) -> "object":
    """Box plots of each group with its mean and standard error marked on top."""
    if frame is None or group_column not in frame.columns or value_column not in frame.columns:
        raise P.PlotError("choose a grouping column and a numeric column")
    data = frame[[group_column, value_column]].dropna()
    if not len(data):
        raise P.PlotError("those two columns have no complete rows in common")
    groups = [g for g, block in data.groupby(group_column, dropna=True)
              if len(pd.to_numeric(block[value_column], errors="coerce").dropna())]
    if not groups:
        raise P.PlotError("the grouping column has no usable groups")
    series = [pd.to_numeric(data.loc[data[group_column] == g, value_column],
                            errors="coerce").dropna().to_numpy(dtype=float) for g in groups]

    fig = P.figure((max(6.4, 1.25 * len(groups) + 3.2), 4.4))
    ax = fig.add_subplot(111)
    ax.boxplot(series, tick_labels=[str(g) for g in groups], patch_artist=True,
               boxprops=dict(facecolor="#dce6f5", edgecolor="#1f3864"),
               medianprops=dict(color="#c0392b", linewidth=1.6),
               whiskerprops=dict(color="#1f3864"), capprops=dict(color="#1f3864"),
               flierprops=dict(marker="o", markersize=3, markerfacecolor="#8fa8c8",
                               markeredgecolor="none"))
    for position, values in enumerate(series, 1):
        if len(values) < 2:
            continue
        mean = values.mean()
        error = values.std(ddof=1) / np.sqrt(len(values))
        ax.errorbar(position, mean, yerr=error, color="#c0392b", capsize=5, linewidth=1.4,
                    marker="D", markersize=4.5, label="mean ± standard error" if position == 1 else None)
        ax.annotate(f"n={len(values)}", (position, ax.get_ylim()[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7, color="#6b7280")
    ax.set_xlabel(group_column)
    ax.set_ylabel(value_column)
    title = "Groups compared"
    if result and result.get("test"):
        title = f"{result['test']} — p = {result.get('p', float('nan')):,.4g}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig)


def correlation_plot(matrix: pd.DataFrame, method: str = "spearman") -> "object":
    """A readable correlation heat map for the statistics screen."""
    if matrix is None or not len(matrix):
        raise P.PlotError("compute the correlation table first")
    values = matrix.to_numpy(dtype=float)
    fig = P.figure((max(5.4, 0.52 * len(matrix) + 2.6), max(4.6, 0.52 * len(matrix) + 2.0)))
    ax = fig.add_subplot(111)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(i) for i in matrix.index], fontsize=8)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.6 else "#374151")
    ax.set_title(f"{method.title()} correlation")
    fig.colorbar(image, ax=ax, shrink=0.82, label="coefficient")
    fig.tight_layout()
    return _finish(fig)
