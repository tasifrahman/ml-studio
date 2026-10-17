"""Every graph the workbench can draw.

Renderers are plain functions returning a matplotlib ``Figure``.  They know
nothing about Qt, which means the exact same code path produces the picture you
see on screen and the picture embedded in the HTML report — if a plot works in
the GUI, it works in the report, and vice versa.

A :class:`PlotContext` carries whatever is currently available (a frame, a
fitted pipeline, metrics).  Each plot declares its requirements as a predicate,
so the GUI can grey out the ones that cannot be drawn yet instead of failing
when the user clicks.
"""
from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass, field

import matplotlib
import numpy as np
import pandas as pd

# The backend is deliberately left alone: the Qt GUI attaches these figures to
# its own canvas, and the report writer only ever calls savefig().  Forcing a
# backend here would break whichever of the two imported this module second.
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_LIMIT = 4000
STYLE_APPLIED = False

# pyplot's figure registry is process-wide state, and the report writer renders
# its figures on a worker thread while the GUI can be drawing its own.  Every
# path that creates or serialises a figure takes this lock, which costs nothing
# measurable and removes a real race.
RENDER_LOCK = threading.RLock()


class PlotError(Exception):
    """Raised when a plot cannot be drawn; the GUI shows the message verbatim."""


@dataclass
class PlotContext:
    df: pd.DataFrame | None = None
    target: str | None = None
    task: str | None = None
    model: object | None = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    evaluation: dict | None = None
    importance: pd.DataFrame | None = None
    clusters: np.ndarray | None = None
    labels: list = field(default_factory=list)
    preprocessor: object | None = None
    model_key: str | None = None
    prepared: object | None = None   # the encoded numeric matrix, when available


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def apply_style():
    global STYLE_APPLIED
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.autolayout": False,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })
    STYLE_APPLIED = True


def figure(figsize=(7.0, 4.6)):
    if not STYLE_APPLIED:
        apply_style()
    return plt.figure(figsize=figsize)


def _frame(ctx: PlotContext) -> pd.DataFrame:
    if ctx.df is None or not len(ctx.df):
        raise PlotError("no data is loaded yet")
    return ctx.df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def _need(df: pd.DataFrame, column: str, want: str = "any") -> str:
    if not column or column not in df.columns:
        raise PlotError(f"column {column!r} is not in the data")
    if want == "numeric" and not pd.api.types.is_numeric_dtype(df[column]):
        raise PlotError(f"{column!r} is not numeric")
    if want == "categorical" and pd.api.types.is_numeric_dtype(df[column]):
        raise PlotError(f"{column!r} is numeric, not categorical")
    return column


def _opt(opts, name, default=None):
    value = (opts or {}).get(name)
    if value is None or value == "":
        return default
    return value


def _num(opts, name, default, minimum=None, maximum=None):
    try:
        value = float(_opt(opts, name, default))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _flag(opts, name, default=False):
    value = _opt(opts, name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


def _first(opts, names, df, want="numeric", default=None):
    for name in names:
        value = _opt(opts, name)
        if value and value in df.columns:
            if want == "numeric" and not pd.api.types.is_numeric_dtype(df[value]):
                continue
            return value
    pool = numeric_columns(df) if want == "numeric" else categorical_columns(df)
    if default and default in pool:
        return default
    if not pool:
        raise PlotError(f"this plot needs at least one {want} column")
    return pool[0]


def _sample(df: pd.DataFrame, limit: int = SAMPLE_LIMIT, seed: int = 0) -> pd.DataFrame:
    return df.sample(limit, random_state=seed) if len(df) > limit else df


def _numeric_matrix(ctx: PlotContext, min_columns: int = 2) -> np.ndarray:
    """A numeric matrix for PCA / clustering / t-SNE.

    Prefers the encoded matrix the pipeline produced; falls back to whatever
    numeric columns the raw frame has, median-imputing so a single missing cell
    cannot stop a projection.  Categorical columns are excluded rather than
    silently one-hot encoded here — that would disagree with the model's own
    preprocessing.
    """
    if ctx.prepared is not None and len(ctx.prepared):
        matrix = np.asarray(ctx.prepared, dtype=float)
        if matrix.ndim == 2 and matrix.shape[1] < min_columns:
            raise PlotError(
                f"this plot needs at least {min_columns} numeric features, and this model "
                f"was fitted on {matrix.shape[1]}"
            )
        return np.nan_to_num(matrix)
    if ctx.X is None or not len(ctx.X):
        raise PlotError("prepare or fit a model first — there is no feature matrix yet")
    frame = ctx.X.select_dtypes(include=[np.number])
    if frame.shape[1] < min_columns:
        raise PlotError(
            f"this plot needs at least {min_columns} numeric features; "
            "categorical columns are not included here"
        )
    matrix = frame.to_numpy(dtype=float)
    if np.isnan(matrix).any():
        medians = np.nanmedian(matrix, axis=0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(medians, inds[1])
    return matrix


def _target_numeric(ctx: PlotContext, df: pd.DataFrame, opts) -> str:
    col = _opt(opts, "target", ctx.target)
    if not col or col not in df.columns:
        raise PlotError("choose a target column that exists in the data")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise PlotError(f"target {col!r} is not numeric — train a model first, or pick another column")
    return col


def _evaluation(ctx: PlotContext) -> dict:
    if not ctx.evaluation or "test" not in ctx.evaluation:
        raise PlotError("train a model first — this plot shows how the fitted model behaves")
    return ctx.evaluation


def _proba(ctx: PlotContext):
    ev = _evaluation(ctx)
    proba = ev.get("test_proba")
    truth = ev.get("test_truth")
    if proba is None or truth is None:
        raise PlotError("this model does not expose probabilities (try a model with predict_proba)")
    return np.asarray(proba), np.asarray(truth)


# --------------------------------------------------------------------------
# univariate
# --------------------------------------------------------------------------


def _hist(ctx, opts):
    df = _frame(ctx)
    col = _need(df, _first(opts, ["column", "x"], df), "numeric")
    bins = int(_num(opts, "bins", 30, 3, 200))
    fig = figure()
    ax = fig.add_subplot(111)
    hue = _opt(opts, "hue")
    if hue and hue in df.columns and df[hue].nunique() <= 10:
        sns.histplot(data=df, x=col, hue=hue, bins=bins, kde=_flag(opts, "kde"), multiple="stack", ax=ax)
    else:
        sns.histplot(data=df, x=col, bins=bins, kde=_flag(opts, "kde"), ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    return fig


def _kde(ctx, opts):
    df = _frame(ctx)
    col = _need(df, _first(opts, ["column", "x"], df), "numeric")
    fig = figure()
    ax = fig.add_subplot(111)
    hue = _opt(opts, "hue")
    if hue and hue in df.columns and df[hue].nunique() <= 10:
        sns.kdeplot(data=df, x=col, hue=hue, fill=_flag(opts, "fill", True), common_norm=False, ax=ax)
    else:
        sns.kdeplot(data=df, x=col, fill=_flag(opts, "fill", True), ax=ax)
    ax.set_title(f"Kernel density of {col}")
    return fig


def _box(ctx, opts):
    df = _frame(ctx)
    y = _need(df, _first(opts, ["column", "y"], df), "numeric")
    x = _opt(opts, "group_by") or _opt(opts, "x")
    fig = figure()
    ax = fig.add_subplot(111)
    if x and x in df.columns and x != y and df[x].nunique() <= 30:
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        if _flag(opts, "points", True):
            sns.stripplot(data=_sample(df, 1500), x=x, y=y, color="0.25", size=2, alpha=0.5, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        sns.boxplot(y=df[y], ax=ax)
    ax.set_title(f"Box plot of {y}")
    return fig


def _violin(ctx, opts):
    df = _frame(ctx)
    y = _need(df, _first(opts, ["column", "y"], df), "numeric")
    x = _opt(opts, "group_by") or _opt(opts, "x")
    fig = figure()
    ax = fig.add_subplot(111)
    if x and x in df.columns and x != y and df[x].nunique() <= 30:
        sns.violinplot(data=df, x=x, y=y, inner="quartile", cut=0, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        sns.violinplot(y=df[y], inner="quartile", cut=0, ax=ax)
    ax.set_title(f"Violin plot of {y}")
    return fig


def _strip(ctx, opts):
    df = _frame(ctx)
    y = _need(df, _first(opts, ["column", "y"], df), "numeric")
    x = _opt(opts, "group_by") or _opt(opts, "x")
    fig = figure()
    ax = fig.add_subplot(111)
    if x and x in df.columns and x != y:
        sns.stripplot(data=_sample(df, 1500), x=x, y=y, jitter=True, alpha=0.6, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        sns.stripplot(y=df[y], jitter=True, alpha=0.6, ax=ax)
    ax.set_title(f"Strip plot of {y}")
    return fig


def _count(ctx, opts):
    df = _frame(ctx)
    col = _opt(opts, "column") or _opt(opts, "x")
    col = _need(df, col or _first(opts, [], df, "categorical"), "any")
    top = int(_num(opts, "top", 15, 1, 100))
    order = df[col].astype(str).value_counts().head(top).index
    fig = figure()
    ax = fig.add_subplot(111)
    sns.countplot(data=df, y=col, order=order, ax=ax)
    ax.set_title(f"Counts of {col}" + (f" (top {top})" if df[col].nunique() > top else ""))
    return fig


def _bar(ctx, opts):
    df = _frame(ctx)
    cat = _opt(opts, "column") or _opt(opts, "x")
    if not cat:
        pool = categorical_columns(df)
        if pool:
            cat = pool[0]
        elif ctx.target:
            cat = ctx.target
    cat = _need(df, cat, "any")
    value = _opt(opts, "value")
    agg = _opt(opts, "aggfunc", "mean")
    top = int(_num(opts, "top", 15, 1, 100))
    fig = figure()
    ax = fig.add_subplot(111)
    if value and value in df.columns and value != cat:
        grouped = df.groupby(cat, observed=True)[value]
        series = {"mean": grouped.mean, "median": grouped.median, "sum": grouped.sum, "count": grouped.count}[agg]()
        series = series.sort_values(ascending=False).head(top)
        sns.barplot(x=series.values, y=series.index.astype(str), ax=ax)
        ax.set_title(f"{agg.title()} of {value} by {cat}")
        ax.set_xlabel(f"{agg} of {value}")
    else:
        series = df[cat].astype(str).value_counts().head(top)
        sns.barplot(x=series.values, y=series.index, ax=ax)
        ax.set_title(f"Frequency of {cat}")
    return fig


def _pie(ctx, opts):
    df = _frame(ctx)
    col = _opt(opts, "column") or _opt(opts, "x")
    if not col or col not in df.columns:
        pool = categorical_columns(df)
        if pool:
            col = pool[0]
        elif ctx.target:
            col = ctx.target
    col = _need(df, col, "any")
    top = int(_num(opts, "top", 8, 2, 20))
    counts = df[col].astype(str).value_counts()
    if len(counts) > top:
        head = counts.head(top)
        head[f"other ({len(counts) - top})"] = counts.iloc[top:].sum()
        counts = head
    fig = figure((6.2, 5.0))
    ax = fig.add_subplot(111)
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title(f"Share of {col}")
    return fig


def _ecdf(ctx, opts):
    df = _frame(ctx)
    col = _need(df, _first(opts, ["column", "x"], df), "numeric")
    fig = figure()
    ax = fig.add_subplot(111)
    sns.ecdfplot(data=df, x=col, ax=ax)
    ax.set_title(f"Cumulative distribution of {col}")
    return fig


def _qq(ctx, opts):
    from scipy import stats

    df = _frame(ctx)
    col = _need(df, _first(opts, ["column", "x"], df), "numeric")
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    fig = figure()
    ax = fig.add_subplot(111)
    stats.probplot(values, dist=_opt(opts, "distribution", "norm"), plot=ax)
    ax.set_title(f"Q-Q plot of {col}")
    fig.axes[0].get_lines()[0].set_markerfacecolor("#3b6ea5")
    fig.axes[0].get_lines()[0].set_markeredgecolor("#3b6ea5")
    fig.axes[0].get_lines()[0].set_markersize(3.5)
    fig.axes[0].get_lines()[1].set_color("#c0392b")
    return fig


def _target_dist(ctx, opts):
    df = _frame(ctx)
    col = _target_numeric(ctx, df, opts)
    fig = figure((9.0, 3.6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.histplot(df[col].dropna(), bins=int(_num(opts, "bins", 30, 3, 200)), kde=True, ax=ax1)
    ax1.set_title(f"Distribution of the target: {col}")
    sns.boxplot(x=df[col].dropna(), ax=ax2)
    ax2.set_title(f"Box plot of {col}")
    return fig


def _class_balance(ctx, opts):
    df = _frame(ctx)
    col = _opt(opts, "target", ctx.target)
    if not col or col not in df.columns:
        raise PlotError("choose a target column")
    counts = df[col].astype(str).value_counts()
    fig = figure()
    ax = fig.add_subplot(111)
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value}\n{100 * value / counts.sum():.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Class balance of {col}")
    ax.set_ylim(0, counts.max() * 1.18)
    return fig


def _missing_bar(ctx, opts):
    df = _frame(ctx)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig = figure()
    ax = fig.add_subplot(111)
    if missing.empty:
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.axis("off")
        return fig
    share = 100 * missing / max(len(df), 1)
    sns.barplot(x=share.values, y=share.index, ax=ax)
    ax.set_title("Missing values by column")
    ax.set_xlabel("% missing")
    return fig


def _missing_matrix(ctx, opts):
    df = _frame(ctx)
    matrix = df.isna().to_numpy()
    if not matrix.any():
        fig = figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.axis("off")
        return fig
    fig = figure((8.4, 4.4))
    ax = fig.add_subplot(111)
    ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1.6)
    ax.set_yticks(range(df.shape[1]))
    ax.set_yticklabels(df.columns, fontsize=7)
    ax.set_xlabel("row")
    ax.set_title("Where the gaps are (dark = missing)")
    return fig


# --------------------------------------------------------------------------
# bivariate
# --------------------------------------------------------------------------


def _scatter(ctx, opts, with_fit=False):
    df = _sample(_frame(ctx), int(_num(opts, "max_points", SAMPLE_LIMIT, 100, 200000)))
    x = _need(df, _first(opts, ["x", "column"], df), "numeric")
    y = _opt(opts, "y", ctx.target)
    if not y or y not in df.columns:
        pool = [c for c in numeric_columns(df) if c != x]
        if not pool:
            raise PlotError("this plot needs two numeric columns")
        y = pool[0]
    _need(df, y, "numeric")
    hue = _opt(opts, "hue")
    size = _opt(opts, "size")
    fig = figure()
    ax = fig.add_subplot(111)
    if size and size in df.columns and pd.api.types.is_numeric_dtype(df[size]):
        sizes = df[size]
        sizes = 20 + 160 * (sizes - sizes.min()) / ((sizes.max() - sizes.min()) or 1)
    else:
        sizes = None
    alpha = _num(opts, "alpha", 0.65, 0.05, 1.0)
    sns.scatterplot(data=df, x=x, y=y, hue=hue if hue in (df.columns if hue else []) else None,
                    size=sizes, alpha=alpha, ax=ax, legend=bool(hue))
    if with_fit:
        try:
            sns.regplot(data=df, x=x, y=y, scatter=False, order=int(_num(opts, "order", 1, 1, 3)),
                        color="#c0392b", ax=ax)
        except Exception:
            pass
    ax.set_title(f"{y} versus {x}")
    return fig


def _scatter_basic(ctx, opts):
    return _scatter(ctx, opts, with_fit=False)


def _scatter_reg(ctx, opts):
    return _scatter(ctx, opts, with_fit=True)


def _bubble(ctx, opts):
    return _scatter(ctx, opts, with_fit=False)


def _hexbin(ctx, opts):
    df = _sample(_frame(ctx), int(_num(opts, "max_points", SAMPLE_LIMIT, 100, 200000)))
    x = _need(df, _first(opts, ["x", "column"], df), "numeric")
    y = _opt(opts, "y", ctx.target)
    if not y or y not in df.columns:
        y = [c for c in numeric_columns(df) if c != x][0]
    fig = figure()
    ax = fig.add_subplot(111)
    hb = ax.hexbin(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"),
                   gridsize=int(_num(opts, "gridsize", 30, 5, 100)), cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Density of {y} against {x}")
    return fig


def _line(ctx, opts):
    df = _frame(ctx)
    x = _opt(opts, "x")
    if not x or x not in df.columns:
        raise PlotError("choose an x column")
    y = _opt(opts, "y", ctx.target)
    if not y or y not in df.columns:
        pool = [c for c in numeric_columns(df) if c != x]
        y = pool[0] if pool else None
    agg = _opt(opts, "aggfunc", None)
    fig = figure()
    ax = fig.add_subplot(111)
    if agg and y:
        grouped = df.groupby(x, observed=True)[y].agg(agg).sort_index()
        ax.plot(grouped.index, grouped.values, marker="o", ms=3)
        ax.set_ylabel(f"{agg}({y})")
    elif y:
        ordered = df[[x, y]].sort_values(x)
        ax.plot(ordered[x], ordered[y], lw=1.2)
        ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.set_title(f"{y or 'count'} versus {x}")
    return fig


def _area(ctx, opts):
    df = _frame(ctx)
    x = _opt(opts, "x")
    y = _opt(opts, "y", ctx.target)
    if not x or x not in df.columns or not y or y not in df.columns:
        raise PlotError("choose both an x and a numeric y column")
    ordered = df[[x, y]].dropna().sort_values(x)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.fill_between(ordered[x], ordered[y], alpha=0.45)
    ax.plot(ordered[x], ordered[y], lw=1.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Area chart of {y} over {x}")
    return fig


def _pairplot(ctx, opts):
    df = _frame(ctx)
    max_cols = int(_num(opts, "max_columns", 5, 2, 8))
    chosen = []
    for name in ("columns", "x", "y"):
        value = _opt(opts, name)
        if value:
            chosen.extend([v.strip() for v in str(value).split(",") if v.strip() in df.columns])
    if len(chosen) < 2:
        pool = numeric_columns(df)
        if ctx.target and ctx.target in pool:
            pool = [c for c in pool if c != ctx.target] + [ctx.target]
        chosen = pool[:max_cols]
    chosen = list(dict.fromkeys(chosen))[:max_cols]
    if len(chosen) < 2:
        raise PlotError("this plot needs at least two numeric columns")
    data = _sample(df[chosen + ([ctx.target] if ctx.target in df.columns and ctx.target not in chosen else [])],
                   int(_num(opts, "max_points", 1200, 100, 20000)))
    hue = _opt(opts, "hue") or (ctx.target if ctx.target in chosen else None)
    if hue and (df[hue].nunique() > 10 or not pd.api.types.is_numeric_dtype(df[hue])):
        grid = sns.pairplot(data, vars=chosen, corner=True, plot_kws={"s": 12, "alpha": 0.55})
    else:
        grid = sns.pairplot(data, vars=chosen, hue=hue, corner=True, plot_kws={"s": 12, "alpha": 0.55})
    grid.figure.set_size_inches(min(2.1 * len(chosen), 12), min(2.1 * len(chosen), 12))
    grid.figure.suptitle("Pairwise relationships", y=1.02)
    return grid.figure


def _jointplot(ctx, opts):
    df = _sample(_frame(ctx), int(_num(opts, "max_points", 3000, 100, 50000)))
    x = _need(df, _first(opts, ["x", "column"], df), "numeric")
    y = _opt(opts, "y", ctx.target)
    if not y or y not in df.columns:
        y = [c for c in numeric_columns(df) if c != x][0]
    kind = _opt(opts, "kind", "scatter")
    grid = sns.jointplot(data=df, x=x, y=y, kind=kind if kind in ("scatter", "hex", "kde", "reg", "hist") else "scatter",
                         height=6.0, joint_kws={"s": 14, "alpha": 0.55} if kind == "scatter" else {})
    grid.figure.suptitle(f"{y} versus {x}", y=1.0)
    return grid.figure


def _corr_heatmap(ctx, opts):
    df = _frame(ctx)
    cols = None
    if _opt(opts, "columns"):
        cols = [c.strip() for c in str(_opt(opts, "columns")).split(",") if c.strip() in df.columns]
    cols = cols or numeric_columns(df)
    if len(cols) < 2:
        raise PlotError("at least two numeric columns are needed for a correlation matrix")
    method = _opt(opts, "method", "pearson")
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    fig = figure((max(5.0, 0.55 * len(cols) + 2.5), max(4.2, 0.5 * len(cols) + 1.8)))
    ax = fig.add_subplot(111)
    mask = None
    if _flag(opts, "triangle", True):
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=len(cols) <= 14, fmt=".2f", cmap=_opt(opts, "cmap", "vlag"),
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"{method.title()} correlation")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def _clustermap(ctx, opts):
    df = _frame(ctx)
    cols = numeric_columns(df)
    if len(cols) < 2:
        raise PlotError("at least two numeric columns are needed")
    max_cols = int(_num(opts, "max_columns", 12, 2, 40))
    cols = cols[:max_cols]
    method = _opt(opts, "method", "pearson")
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    grid = sns.clustermap(corr, cmap=_opt(opts, "cmap", "vlag"), center=0, figsize=(8.2, 8.2),
                          annot=len(cols) <= 14, fmt=".2f", annot_kws={"size": 6.5},
                          cbar_pos=(0.02, 0.82, 0.03, 0.15))
    grid.figure.suptitle("Clustered correlation matrix", y=0.98)
    return grid.figure


def _pivot_heatmap(ctx, opts):
    df = _frame(ctx)
    cats = categorical_columns(df)
    if len(cats) < 2:
        raise PlotError("this plot needs two categorical columns")
    index = _opt(opts, "index")
    column = _opt(opts, "column")
    # whatever the form sent, the axes have to be categorical and different
    if index not in cats:
        index = cats[0]
    if column not in cats or column == index:
        column = next((c for c in cats if c != index), None)
    if not column:
        raise PlotError("this plot needs two different categorical columns")
    value = _opt(opts, "value", ctx.target)
    agg = _opt(opts, "aggfunc", "mean")
    # averaging needs numbers; a text column (a class label, say) can only be counted
    numeric_value = (value in df.columns and value not in (index, column)
                     and pd.api.types.is_numeric_dtype(df[value])
                     and agg not in ("size", "count"))
    if numeric_value:
        table = df.pivot_table(index=index, columns=column, values=value, aggfunc=agg, observed=True)
        title = f"{agg} of {value}: {index} × {column}"
    else:
        table = df.pivot_table(index=index, columns=column, aggfunc="size", observed=True)
        title = f"counts: {index} × {column}"
    max_cells = int(_num(opts, "max_cells", 40, 3, 200))
    table = table.iloc[:max_cells, :max_cells]
    fig = figure((max(5.5, 0.45 * table.shape[1] + 2.6), max(4.2, 0.34 * table.shape[0] + 1.8)))
    ax = fig.add_subplot(111)
    sns.heatmap(table, annot=table.size <= 200, fmt=".2g", cmap=_opt(opts, "cmap", "rocket_r"),
                linewidths=0.5, ax=ax)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return fig


# --------------------------------------------------------------------------
# model diagnostics
# --------------------------------------------------------------------------


def _importance_bar(ctx, opts):
    if ctx.importance is None or not len(ctx.importance):
        raise PlotError("no feature importances available — train a model first")
    frame = ctx.importance.copy()
    value_col = "importance" if "importance" in frame.columns else frame.columns[-1]
    top = int(_num(opts, "top", 20, 1, 200))
    frame = frame.sort_values(value_col, ascending=False).head(top).iloc[::-1]
    fig = figure((7.2, max(3.0, 0.28 * len(frame) + 1.4)))
    ax = fig.add_subplot(111)
    sns.barplot(x=frame[value_col], y=frame[frame.columns[0]].astype(str), ax=ax,
                color=_opt(opts, "color", "#3b6ea5"))
    ax.set_title(f"Top {len(frame)} features by importance")
    ax.set_xlabel(value_col)
    return fig


def _permutation_box(ctx, opts):
    if ctx.importance is None or not len(ctx.importance):
        raise PlotError("no permutation importances available")
    frame = ctx.importance.copy()
    if "perm_mean" not in frame.columns:
        return _importance_bar(ctx, opts)
    top = int(_num(opts, "top", 15, 1, 100))
    frame = frame.sort_values("perm_mean", ascending=False).head(top).iloc[::-1]
    std = frame["perm_std"] if "perm_std" in frame.columns else pd.Series(0.0, index=frame.index)
    fig = figure((7.2, max(3.0, 0.3 * len(frame) + 1.4)))
    ax = fig.add_subplot(111)
    ax.barh(frame.iloc[:, 0].astype(str), frame["perm_mean"], xerr=std, color="#3b6ea5",
            error_kw={"ecolor": "#7f8c8d", "lw": 1})
    ax.axvline(0, color="0.35", lw=1)
    ax.set_title(f"Permutation importance (top {len(frame)})")
    ax.set_xlabel("drop in score when shuffled")
    return fig


def _partial_dependence(ctx, opts):
    if ctx.model is None or ctx.X is None:
        raise PlotError("train a model first")
    from . import explain

    X = ctx.X
    feature = _opt(opts, "feature")
    if not feature or feature not in X.columns:
        pool = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not pool:
            raise PlotError("partial dependence needs a numeric feature")
        feature = pool[0]
    second = _opt(opts, "feature2")
    if second and second not in X.columns:
        second = None
    grid = int(_num(opts, "grid_resolution", 40, 10, 200))
    ice = _flag(opts, "ice", False)
    note = "probability of the highest class" if ctx.task in ("binary", "multiclass") else "prediction"
    try:
        axes, response, matrix = explain.grid_prediction(
            ctx.model, X, [feature] if not second else [feature, second],
            task=ctx.task or "regression", resolution=grid, sample_rows=int(_num(opts, "sample_rows", 250, 20, 5000)),
            return_rows=ice,
        )
    except Exception as exc:
        raise PlotError(f"partial dependence failed: {exc}") from exc

    if not second:
        fig = figure((8.0, 4.4))
        ax = fig.add_subplot(111)
        if ice and matrix is not None:
            for row in matrix.T[:60]:
                ax.plot(axes[0], row, color="0.75", lw=0.6, alpha=0.5)
        ax.plot(axes[0], response, lw=2.4, color="#3b6ea5")
        ax.set_xlabel(feature)
        ax.set_ylabel(f"mean {note}")
        ax.set_title(f"Partial dependence of {note} on {feature}")
        ax.grid(alpha=0.3)
        return fig

    fig = figure((8.6, 4.6))
    ax = fig.add_subplot(111)
    mesh = ax.contourf(axes[0], axes[1], response, levels=18, cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=f"mean {note}")
    ax.set_xlabel(feature)
    ax.set_ylabel(second)
    ax.set_title(f"Joint effect of {feature} and {second}")
    return fig


def _learning_curve(ctx, opts):
    if ctx.model is None or ctx.X is None or ctx.y is None:
        raise PlotError("train a model first")
    from sklearn.model_selection import learning_curve

    scoring = _opt(opts, "scoring", "r2" if ctx.task == "regression" else "accuracy")
    cv = int(_num(opts, "cv", 5, 2, 10))
    points = int(_num(opts, "points", 6, 3, 12))
    try:
        sizes, train_scores, test_scores = learning_curve(
            ctx.model, ctx.X, ctx.y, cv=cv, scoring=scoring, n_jobs=-1,
            train_sizes=np.linspace(0.15, 1.0, points), random_state=0,
        )
    except Exception as exc:
        raise PlotError(f"learning curve failed: {exc}") from exc
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(sizes, train_scores.mean(1), "o-", label="training", color="#2e7d5b")
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1), train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#2e7d5b")
    ax.plot(sizes, test_scores.mean(1), "o-", label="cross-validation", color="#3b6ea5")
    ax.fill_between(sizes, test_scores.mean(1) - test_scores.std(1), test_scores.mean(1) + test_scores.std(1), alpha=0.15, color="#3b6ea5")
    ax.set_xlabel("training rows")
    ax.set_ylabel(scoring)
    ax.set_title("Learning curve")
    ax.legend(loc="best")
    return fig


def _training_curve(ctx, opts):
    """Loss per epoch, with the epoch the best weights came from marked.

    This is the diagnostic a neural network needs and no other model in the
    registry has: it shows whether training was still improving when it stopped,
    and whether the validation loss started climbing (overfitting) before that.
    """
    estimator = _model_step(ctx)
    history = _training_history(ctx)
    if not history:
        raise PlotError("this model has no training history — only the neural networks record one")

    epochs = list(history.get("epoch") or [])
    train = list(history.get("train_loss") or [])
    valid = list(history.get("val_loss") or [])
    if not epochs or not train:
        raise PlotError("the training history is empty")

    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train, "-", lw=1.8, color="#2e7d5b", label="training loss")
    paired = [(e, v) for e, v in zip(epochs, valid) if v is not None]
    if paired:
        ax.plot([e for e, _ in paired], [v for _, v in paired], "-", lw=1.8,
                color="#3b6ea5", label="validation loss")

    best = getattr(estimator, "best_epoch_", None)
    if best:
        ax.axvline(int(best), ls="--", lw=1.2, color="#c1440e", label=f"best epoch ({int(best)})")

    if _opt(opts, "log", False):
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    stopped = getattr(estimator, "stopped_reason_", "")
    ax.set_title("Training curve" + (f" — {stopped}" if stopped else ""), fontsize=10)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _reconstruction_error_plot(ctx, opts):
    """Histogram of per-row reconstruction error from an autoencoder.

    Rows far to the right of the bulk are the ones the network could not rebuild —
    the usual reading of this as an anomaly score.
    """
    estimator = _model_step(ctx)
    if not callable(getattr(estimator, "reconstruction_error", None)):
        raise PlotError("this model does not report a reconstruction error")
    matrix = ctx.prepared
    if matrix is None or not len(matrix):
        raise PlotError("the prepared feature matrix is not available")

    errors = np.asarray(estimator.reconstruction_error(matrix), dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if not errors.size:
        raise PlotError("no finite reconstruction errors")

    fig = figure()
    ax = fig.add_subplot(111)
    ax.hist(errors, bins=int(_num(opts, "bins", 40, 5, 200)), color="#3b6ea5", alpha=0.85)
    threshold = float(np.quantile(errors, float(_num(opts, "quantile", 0.99, 0.5, 1.0))))
    ax.axvline(threshold, ls="--", lw=1.4, color="#c1440e",
               label=f"{float(_num(opts, 'quantile', 0.99, 0.5, 1.0)):.0%} quantile")
    if _opt(opts, "log", True):
        ax.set_yscale("log")
    ax.set_xlabel("squared reconstruction error per row")
    ax.set_ylabel("rows")
    ax.set_title("Autoencoder reconstruction error", fontsize=10)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _auto_validation_param(ctx):
    """Pick a parameter to sweep, and a few values, from the model registry."""
    from . import models as MODELS

    if ctx.model_key:
        try:
            for name, values in MODELS.param_default_grid(ctx.model_key).items():
                if len(values) >= 2:
                    return name, values
        except KeyError:
            pass
    estimator = getattr(ctx.model, "named_steps", {}).get("model")
    if estimator is not None:
        for name, values in estimator.get_params().items():
            if name.endswith("_") or not isinstance(values, (int, float)):
                continue
            if name in ("n_estimators", "max_iter", "alpha", "C", "n_neighbors", "max_depth", "min_samples_leaf"):
                base = values if isinstance(values, int) and values > 3 else 10
                return name, [max(1, base // 2), base, base * 2]
    return None, []


def _validation_curve(ctx, opts):
    if ctx.model is None or ctx.X is None or ctx.y is None:
        raise PlotError("train a model first")
    from sklearn.model_selection import validation_curve

    param = _opt(opts, "param")
    values = _opt(opts, "values")
    if not param or not values:
        auto_param, auto_values = _auto_validation_param(ctx)
        if not auto_param:
            raise PlotError("choose a parameter and its values (comma separated)")
        param = param or auto_param
        values = values or ", ".join(str(v) for v in auto_values)
    raw = [v.strip() for v in str(values).split(",") if v.strip()]
    grid = []
    for item in raw:
        try:
            grid.append(float(item) if "." in item or "e" in item.lower() else int(item))
        except ValueError:
            grid.append(item)
    scoring = _opt(opts, "scoring", "r2" if ctx.task == "regression" else "accuracy")
    cv = int(_num(opts, "cv", 5, 2, 10))
    try:
        train_scores, test_scores = validation_curve(
            ctx.model, ctx.X, ctx.y, param_name=f"model__{param}", param_range=grid,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
    except Exception as exc:
        raise PlotError(f"validation curve failed: {exc}") from exc
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(grid, test_scores.mean(1), "o-", label="cross-validation", color="#3b6ea5")
    ax.fill_between(range(len(grid)), test_scores.mean(1) - test_scores.std(1), test_scores.mean(1) + test_scores.std(1),
                    alpha=0.15, color="#3b6ea5")
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([str(g) for g in grid], rotation=30, ha="right")
    ax.set_xlabel(param)
    ax.set_ylabel(scoring)
    ax.set_title(f"Validation curve for {param}")
    ax.legend(loc="best")
    return fig


def _residuals(ctx, opts):
    ev = _evaluation(ctx)
    truth = np.asarray(ev["test_truth"], dtype=float)
    pred = np.asarray(ev["test_pred"], dtype=float)
    resid = truth - pred
    fig = figure((9.0, 4.0))
    ax1 = fig.add_subplot(121)
    ax1.scatter(pred, resid, s=14, alpha=0.55, color="#3b6ea5")
    ax1.axhline(0, color="#c0392b", lw=1.4)
    ax1.set_xlabel("predicted")
    ax1.set_ylabel("residual (actual − predicted)")
    ax1.set_title("Residuals versus fitted")
    ax2 = fig.add_subplot(122)
    sns.histplot(resid, bins=int(_num(opts, "bins", 25, 5, 120)), kde=True, ax=ax2, color="#2e7d5b")
    ax2.set_title("Residual distribution")
    ax2.set_xlabel("residual")
    return fig


def _actual_vs_predicted(ctx, opts):
    ev = _evaluation(ctx)
    truth = np.asarray(ev["test_truth"], dtype=float)
    pred = np.asarray(ev["test_pred"], dtype=float)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.scatter(truth, pred, s=16, alpha=0.6, color="#3b6ea5", label="test rows")
    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    pad = 0.03 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c0392b", lw=1.5, label="perfect prediction")
    from sklearn.metrics import r2_score

    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(f"Actual versus predicted (R² = {r2_score(truth, pred):.3f})")
    ax.legend(loc="best")
    return fig


def _confusion(ctx, opts):
    if not _is_classification(ctx):
        raise PlotError("a confusion matrix needs a classification model — train one, or use the residual plots")
    ev = _evaluation(ctx)
    truth = np.asarray(ev["test_truth"])
    pred = np.asarray(ev["test_pred"])
    labels = ctx.labels or sorted(set(truth.tolist()) | set(pred.tolist()))
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(truth, pred, labels=labels)
    normalize = _opt(opts, "normalize")
    if normalize in ("true", "pred", "all"):
        with np.errstate(invalid="ignore", divide="ignore"):
            if normalize == "true":
                matrix = matrix / matrix.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                matrix = matrix / matrix.sum(axis=0, keepdims=True)
            else:
                matrix = matrix / matrix.sum()
        matrix = np.nan_to_num(matrix)
    fig = figure((max(4.6, 0.6 * len(labels) + 3.0), max(4.0, 0.5 * len(labels) + 2.6)))
    ax = fig.add_subplot(111)
    sns.heatmap(matrix, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=len(labels) > 3, linewidths=0.5, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix" + (f" (normalised by {normalize})" if normalize else " (test set)"))
    return fig


def _roc_curve(ctx, opts):
    proba, truth = _proba(ctx)
    from sklearn.metrics import auc, roc_curve

    fig = figure()
    ax = fig.add_subplot(111)
    labels = ctx.labels or sorted(set(truth.tolist()))
    if len(labels) == 2 or proba.shape[1] == 2:
        binary = np.asarray(truth) == labels[-1]
        fpr, tpr, _ = roc_curve(binary, proba[:, 1])
        ax.plot(fpr, tpr, lw=2, color="#3b6ea5", label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        from sklearn.preprocessing import label_binarize

        binarised = label_binarize(np.asarray(truth), classes=labels)
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(binarised[:, i], proba[:, i])
            ax.plot(fpr, tpr, lw=1.6, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _pr_curve(ctx, opts):
    proba, truth = _proba(ctx)
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig = figure()
    ax = fig.add_subplot(111)
    labels = ctx.labels or sorted(set(truth.tolist()))
    if len(labels) == 2 or proba.shape[1] == 2:
        binary = np.asarray(truth) == labels[-1]
        precision, recall, _ = precision_recall_curve(binary, proba[:, 1])
        ax.plot(recall, precision, lw=2, color="#2e7d5b",
                label=f"AP = {average_precision_score(binary, proba[:, 1]):.3f}")
    else:
        from sklearn.preprocessing import label_binarize

        binarised = label_binarize(np.asarray(truth), classes=labels)
        for i, label in enumerate(labels):
            precision, recall, _ = precision_recall_curve(binarised[:, i], proba[:, i])
            ax.plot(recall, precision, lw=1.6, label=str(label))
    base = float(np.mean(np.asarray(truth) == labels[-1])) if len(labels) == 2 else None
    if base is not None:
        ax.axhline(base, ls="--", color="0.55", lw=1, label=f"chance ({base:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–recall curve (test set)")
    ax.legend(loc="best", fontsize=8)
    return fig


def _calibration(ctx, opts):
    proba, truth = _proba(ctx)
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    labels = ctx.labels or sorted(set(truth.tolist()))
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1, label="perfectly calibrated")
    if proba.shape[1] == 2:
        clipped = np.clip(proba[:, 1], 1e-6, 1 - 1e-6)
        frac, mean_pred = calibration_curve((np.asarray(truth) == labels[-1]).astype(int), clipped,
                                            n_bins=int(_num(opts, "bins", 10, 3, 30)))
        ax.plot(mean_pred, frac, "o-", color="#3b6ea5",
                label=f"Brier = {brier_score_loss((np.asarray(truth) == labels[-1]).astype(int), clipped):.3f}")
    else:
        for i, label in enumerate(labels):
            clipped = np.clip(proba[:, i], 1e-6, 1 - 1e-6)
            frac, mean_pred = calibration_curve((np.asarray(truth) == label).astype(int), clipped,
                                                n_bins=int(_num(opts, "bins", 10, 3, 30)))
            ax.plot(mean_pred, frac, "o-", lw=1.4, label=str(label))
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(loc="best", fontsize=8)
    return fig


def _cv_folds(ctx, opts):
    if not ctx.evaluation or "cv" not in ctx.evaluation:
        raise PlotError("no cross-validation results — tick 'run cross-validation' and retrain")
    cv = ctx.evaluation["cv"]
    folds = cv.get("folds") or []
    if not folds:
        raise PlotError("cross-validation produced no folds")
    metric = _opt(opts, "metric", (cv.get("primary") or {}).get("metric", "r2"))
    values = [f.get(metric, np.nan) for f in folds]
    fig = figure()
    ax = fig.add_subplot(111)
    bars = ax.bar([f"fold {i + 1}" for i in range(len(values))], values,
                  color="#3b6ea5", edgecolor="white")
    mean = np.nanmean(values)
    ax.axhline(mean, color="#c0392b", lw=1.4, label=f"mean {mean:.4f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=7.5)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per cross-validation fold")
    ax.legend(loc="best", fontsize=8)
    return fig


# --------------------------------------------------------------------------
# unsupervised
# --------------------------------------------------------------------------


def _cluster_series(ctx, opts):
    if ctx.clusters is None or ctx.X is None:
        raise PlotError("fit a clustering model first")
    return np.asarray(ctx.clusters), ctx.X


def _pca_scatter(ctx, opts):
    X, _ = _embedding(ctx, opts)
    label_values = ctx.clusters if ctx.clusters is not None else None
    fig = figure()
    ax = fig.add_subplot(111)
    if label_values is not None and len(label_values) == len(X):
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=np.asarray(label_values).astype(str), palette="tab10",
                        s=26, alpha=0.8, ax=ax, legend="full")
    else:
        target = _opt(opts, "color")
        if target and ctx.df is not None and target in ctx.df.columns and len(ctx.df) == len(X):
            ax.scatter(X[:, 0], X[:, 1], c=pd.to_numeric(ctx.df[target], errors="coerce"), cmap="viridis", s=26, alpha=0.85)
            fig.colorbar(ax.collections[0], ax=ax, label=target)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=26, alpha=0.8, color="#3b6ea5")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(_opt(opts, "title", "Projection of the data"))
    return fig


def _embedding(ctx, opts):
    from sklearn.decomposition import PCA

    matrix = _numeric_matrix(ctx, min_columns=2)
    n = int(_num(opts, "max_points", 5000, 100, 50000))
    if len(matrix) > n:
        idx = np.random.RandomState(0).choice(len(matrix), n, replace=False)
        matrix = matrix[idx]
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(matrix), None


def _tsne_scatter(ctx, opts):
    from sklearn.manifold import TSNE

    matrix = _numeric_matrix(ctx, min_columns=2)
    n = int(_num(opts, "max_points", 1500, 100, 20000))
    if len(matrix) > n:
        idx = np.random.RandomState(0).choice(len(matrix), n, replace=False)
        matrix = matrix[idx]
    perplexity = min(_num(opts, "perplexity", 30, 5, 100), max(5, len(matrix) // 4))
    embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                     random_state=0).fit_transform(matrix)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], s=22, alpha=0.8, color="#3b6ea5")
    ax.set_title(f"t-SNE embedding ({len(matrix)} rows)")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    return fig


def _elbow(ctx, opts):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix = _numeric_matrix(ctx, min_columns=1)
    lo = int(_num(opts, "min_k", 2, 2, 50))
    hi = int(_num(opts, "max_k", 10, 3, 60))
    hi = max(hi, lo + 1)
    inertias, silhouettes, ks = [], [], list(range(lo, hi + 1))
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matrix)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(matrix, model.labels_) if k > 1 and len(set(model.labels_)) > 1 else np.nan)
    fig = figure((9.0, 4.0))
    ax1 = fig.add_subplot(121)
    ax1.plot(ks, inertias, "o-", color="#3b6ea5")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia (within-cluster SSE)")
    ax1.set_title("Elbow plot")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(122)
    ax2.plot(ks, silhouettes, "o-", color="#2e7d5b")
    best = ks[int(np.nanargmax(silhouettes))] if np.isfinite(silhouettes).any() else None
    if best:
        ax2.axvline(best, color="#c0392b", ls="--", lw=1.2, label=f"best k = {best}")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("k")
    ax2.set_ylabel("silhouette score")
    ax2.set_title("Silhouette by k")
    ax2.grid(alpha=0.3)
    return fig


def _silhouette(ctx, opts):
    if ctx.clusters is None or ctx.X is None:
        raise PlotError("fit a clustering model first")
    from sklearn.metrics import silhouette_samples, silhouette_score

    labels = np.asarray(ctx.clusters)
    matrix = _numeric_matrix(ctx, min_columns=1)
    if len(matrix) != len(labels):
        raise PlotError("the cluster labels and the feature matrix have different lengths")
    unique = [l for l in np.unique(labels) if l != -1]
    if len(unique) < 2:
        raise PlotError("silhouette needs at least two clusters")
    samples = silhouette_samples(matrix, labels)
    fig = figure()
    ax = fig.add_subplot(111)
    y = 10
    for label in unique:
        values = np.sort(samples[labels == label])
        ax.fill_betweenx(np.arange(y, y + len(values)), 0, values, alpha=0.75)
        ax.text(-0.05, y + len(values) / 2, str(label), fontsize=8, va="center")
        y += len(values) + 10
    ax.axvline(samples.mean(), color="#c0392b", ls="--", lw=1.3,
               label=f"mean = {silhouette_score(matrix, labels):.3f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_ylabel("rows grouped by cluster")
    ax.set_yticks([])
    ax.set_title("Silhouette by cluster")
    ax.legend(fontsize=8, loc="best")
    return fig


def _dendrogram(ctx, opts):
    from scipy.cluster.hierarchy import dendrogram, linkage

    matrix = _numeric_matrix(ctx, min_columns=1)
    n = int(_num(opts, "max_points", 300, 10, 5000))
    if len(matrix) > n:
        matrix = matrix[np.random.RandomState(0).choice(len(matrix), n, replace=False)]
    method = _opt(opts, "linkage", "ward")
    links = linkage(matrix, method=method if method in ("ward", "complete", "average", "single") else "ward")
    fig = figure((9.4, 4.4))
    ax = fig.add_subplot(111)
    dendrogram(links, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(f"Hierarchical clustering ({method} linkage)")
    ax.set_xlabel("rows")
    ax.set_ylabel("merge distance")
    return fig


def _cluster_sizes(ctx, opts):
    if ctx.clusters is None:
        raise PlotError("fit a clustering model first")
    labels = pd.Series(np.asarray(ctx.clusters)).astype(str)
    counts = labels.value_counts().sort_index()
    fig = figure()
    ax = fig.add_subplot(111)
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_title("Rows per cluster")
    ax.set_xlabel("cluster")
    ax.set_ylim(0, counts.max() * 1.12)
    return fig


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass
class PlotSpec:
    key: str
    label: str
    family: str
    render: object
    needs: object
    params: list = field(default_factory=list)
    doc: str = ""


def _has_df(ctx):
    return ctx.df is not None and len(ctx.df) > 0


def _has_numeric(ctx):
    return _has_df(ctx) and bool(numeric_columns(ctx.df))


def _has_two_numeric(ctx):
    return _has_numeric(ctx) and len(numeric_columns(ctx.df)) >= 2


def _projection_width(ctx: PlotContext) -> int:
    """How many columns there are to project, by the same rule ``_numeric_matrix`` uses.

    The prepared (encoded) matrix wins over the raw frame, so a model fitted on a single
    feature projects a single column even when the frame has ten.
    """
    if ctx.prepared is not None and len(ctx.prepared):
        matrix = np.asarray(ctx.prepared)
        return int(matrix.shape[1]) if matrix.ndim == 2 else 0
    if ctx.X is None or not len(ctx.X):
        return 0
    return int(ctx.X.select_dtypes(include=[np.number]).shape[1])


def _has_projection(ctx):
    """PCA / t-SNE need two numeric columns to draw *two* components.

    Without this the plot is listed as available and then dies inside scikit-learn with
    ``n_components=2 must be between 0 and min(n_samples, n_features)=1`` — which is what
    the "draw every available graph" check saw on a frame whose best feature combination
    was one column.
    """
    return ctx.X is not None and _projection_width(ctx) >= 2


def _has_categorical(ctx):
    return _has_df(ctx) and bool(categorical_columns(ctx.df))


def _has_two_categorical(ctx):
    return _has_df(ctx) and len(categorical_columns(ctx.df)) >= 2


def _has_target(ctx):
    return _has_df(ctx) and ctx.target in ctx.df.columns


def _has_numeric_target(ctx):
    return _has_target(ctx) and pd.api.types.is_numeric_dtype(ctx.df[ctx.target])


def _has_model(ctx):
    return ctx.model is not None and ctx.evaluation is not None


def _model_step(ctx):
    """The fitted estimator inside the pipeline — the object that owns ``history_``."""
    model = ctx.model
    if model is None:
        return None
    steps = getattr(model, "named_steps", None)
    if steps is None:
        return model
    try:
        return steps["model"]
    except Exception:
        return model


def _is_classification(ctx):
    return ctx.task in ("binary", "multiclass")


def _has_model_classification(ctx):
    return _has_model(ctx) and _is_classification(ctx)


def _has_model_regression(ctx):
    return _has_model(ctx) and ctx.task == "regression"


def _has_proba(ctx):
    return _has_model(ctx) and ctx.evaluation.get("test_proba") is not None


def _has_importance(ctx):
    return ctx.importance is not None and len(ctx.importance) > 0


def _training_history(ctx):
    """The loss history a neural network records while it trains, if any."""
    history = getattr(_model_step(ctx), "history_", None)
    return history if isinstance(history, dict) else None


def _has_training_history(ctx):
    history = _training_history(ctx)
    return bool(history and history.get("epoch"))


def _has_reconstruction_error(ctx):
    return callable(getattr(_model_step(ctx), "reconstruction_error", None))


def _has_tunable_param(ctx):
    """A validation curve needs one numeric hyperparameter to sweep."""
    if not _has_model(ctx):
        return False
    try:
        param, values = _auto_validation_param(ctx)
    except Exception:
        return False
    return bool(param and values)


def _has_clusters(ctx):
    return ctx.clusters is not None and ctx.X is not None


def _P(name, label, kind="choice", choices=None, default=None, optional=True, help=""):
    return {"name": name, "label": label, "kind": kind, "choices": choices or [],
            "default": default, "optional": optional, "help": help}


_COLUMN = _P("column", "column", "column")
_X = _P("x", "x axis", "column")
_Y = _P("y", "y axis", "column")

PLOTS: list[PlotSpec] = [
    PlotSpec("hist", "Histogram", "Univariate", _hist, _has_numeric,
             [_COLUMN, _P("bins", "bins", "int", default=30), _P("kde", "overlay kernel density", "bool", default=False),
              _P("hue", "split by", "column")]),
    PlotSpec("kde", "Density plot", "Univariate", _kde, _has_numeric,
             [_COLUMN, _P("fill", "fill under curve", "bool", default=True), _P("hue", "split by", "column")]),
    PlotSpec("box", "Box plot", "Univariate", _box, _has_numeric,
             [_COLUMN, _P("group_by", "group by", "column"), _P("points", "show points", "bool", default=True)]),
    PlotSpec("violin", "Violin plot", "Univariate", _violin, _has_numeric,
             [_COLUMN, _P("group_by", "group by", "column")]),
    PlotSpec("strip", "Strip plot", "Univariate", _strip, _has_numeric,
             [_COLUMN, _P("group_by", "group by", "column")]),
    PlotSpec("ecdf", "Cumulative distribution", "Univariate", _ecdf, _has_numeric, [_COLUMN]),
    PlotSpec("qq", "Q-Q plot", "Univariate", _qq, _has_numeric,
             [_COLUMN, _P("distribution", "reference distribution", "choice",
                          choices=["norm", "uniform", "expon", "logistic"], default="norm")]),
    PlotSpec("count", "Count plot", "Categorical", _count, _has_df,
             [_P("column", "column", "column"), _P("top", "show top N", "int", default=15)]),
    PlotSpec("bar", "Bar chart", "Categorical", _bar, _has_df,
             [_P("column", "category", "column"), _P("value", "value column", "column"),
              _P("aggfunc", "aggregate", "choice", choices=["mean", "median", "sum", "count"], default="mean"),
              _P("top", "show top N", "int", default=15)]),
    PlotSpec("pie", "Pie chart", "Categorical", _pie, _has_df,
             [_P("column", "category", "column"), _P("top", "show top N", "int", default=8)]),
    PlotSpec("target_dist", "Target distribution", "Target", _target_dist, _has_numeric_target,
             [_P("target", "target", "column"), _P("bins", "bins", "int", default=30)]),
    PlotSpec("class_balance", "Class balance", "Target", _class_balance, _has_df, [_P("target", "target", "column")]),
    PlotSpec("missing_bar", "Missing values by column", "Missing data", _missing_bar, _has_df, []),
    PlotSpec("missing_matrix", "Missing value map", "Missing data", _missing_matrix, _has_df, []),
    PlotSpec("scatter", "Scatter plot", "Relationship", _scatter_basic, _has_two_numeric,
             [_X, _Y, _P("hue", "colour by", "column"), _P("size", "size by", "column"),
              _P("max_points", "max points", "int", default=SAMPLE_LIMIT), _P("alpha", "opacity", "float", default=0.65)]),
    PlotSpec("scatter_reg", "Scatter with trend line", "Relationship", _scatter_reg, _has_two_numeric,
             [_X, _Y, _P("order", "polynomial order", "choice", choices=[1, 2, 3], default=1),
              _P("max_points", "max points", "int", default=SAMPLE_LIMIT)]),
    PlotSpec("bubble", "Bubble chart", "Relationship", _bubble, _has_two_numeric,
             [_X, _Y, _P("size", "size by", "column"), _P("hue", "colour by", "column"),
              _P("max_points", "max points", "int", default=SAMPLE_LIMIT)]),
    PlotSpec("hexbin", "Density hexbin", "Relationship", _hexbin, _has_two_numeric,
             [_X, _Y, _P("gridsize", "bins per axis", "int", default=30)]),
    PlotSpec("line", "Line chart", "Relationship", _line, _has_df,
             [_X, _Y, _P("aggfunc", "aggregate", "choice", choices=["mean", "median", "sum", "min", "max"], default=None)]),
    PlotSpec("area", "Area chart", "Relationship", _area, _has_df, [_X, _Y]),
    PlotSpec("pairplot", "Pair plot", "Relationship", _pairplot, _has_two_numeric,
             [_P("columns", "columns (comma separated)", "columns"),
              _P("max_columns", "max columns", "int", default=5), _P("hue", "hue", "column"),
              _P("max_points", "max points", "int", default=1200)]),
    PlotSpec("jointplot", "Joint plot", "Relationship", _jointplot, _has_two_numeric,
             [_X, _Y, _P("kind", "style", "choice", choices=["scatter", "hex", "kde", "reg", "hist"], default="scatter")]),
    PlotSpec("corr_heatmap", "Correlation heatmap", "Relationship", _corr_heatmap, _has_two_numeric,
             [_P("method", "method", "choice", choices=["pearson", "spearman", "kendall"], default="pearson"),
              _P("triangle", "lower triangle only", "bool", default=True),
              _P("columns", "columns (comma separated)", "columns"),
              _P("cmap", "colour map", "choice", choices=["vlag", "coolwarm", "RdBu_r", "viridis"], default="vlag")]),
    PlotSpec("clustermap", "Clustered correlation map", "Relationship", _clustermap, _has_two_numeric,
             [_P("method", "method", "choice", choices=["pearson", "spearman"], default="pearson"),
              _P("max_columns", "max columns", "int", default=12)]),
    PlotSpec("pivot_heatmap", "Pivot table heatmap", "Relationship", _pivot_heatmap, _has_two_categorical,
             [_P("index", "rows (category)", "column"), _P("column", "columns (category)", "column"),
              _P("value", "values", "column"),
              _P("aggfunc", "aggregate", "choice", choices=["mean", "median", "sum", "count", "max", "min"], default="mean")]),
    PlotSpec("feature_importance", "Feature importance", "Model", _importance_bar, _has_importance,
             [_P("top", "show top N", "int", default=20)]),
    PlotSpec("permutation_box", "Permutation importance", "Model", _permutation_box, _has_importance,
             [_P("top", "show top N", "int", default=15)]),
    PlotSpec("partial_dependence", "Partial dependence", "Model", _partial_dependence, _has_model,
             [_P("feature", "feature", "column"), _P("feature2", "second feature (2-D)", "column"),
              _P("grid_resolution", "grid resolution", "int", default=40), _P("ice", "overlay ICE lines", "bool", default=False)]),
    PlotSpec("learning_curve", "Learning curve", "Model", _learning_curve, _has_model,
             [_P("scoring", "score", "choice", choices=["r2", "neg_root_mean_squared_error", "accuracy", "f1_weighted", "roc_auc"], default=None),
              _P("cv", "folds", "int", default=5), _P("points", "points", "int", default=6)]),
    PlotSpec("validation_curve", "Validation curve", "Model", _validation_curve, _has_tunable_param,
             [_P("param", "parameter name", "text", default=None), _P("values", "values (comma separated)", "text", default=None),
              _P("scoring", "score", "choice", choices=["r2", "accuracy", "f1_weighted"], default=None), _P("cv", "folds", "int", default=5)]),
    PlotSpec("training_curve", "Training curve (loss per epoch)", "Model", _training_curve, _has_training_history,
             [_P("log", "log scale", "bool", default=False)]),
    PlotSpec("residuals", "Residual plots", "Model", _residuals, _has_model_regression, [_P("bins", "bins", "int", default=25)]),
    PlotSpec("actual_vs_predicted", "Actual vs predicted", "Model", _actual_vs_predicted, _has_model_regression, []),
    PlotSpec("confusion", "Confusion matrix", "Model", _confusion, _has_model_classification,
             [_P("normalize", "normalise", "choice", choices=["", "true", "pred", "all"], default="")]),
    PlotSpec("roc_curve", "ROC curve", "Model", _roc_curve, _has_proba, []),
    PlotSpec("pr_curve", "Precision–recall curve", "Model", _pr_curve, _has_proba, []),
    PlotSpec("calibration", "Calibration curve", "Model", _calibration, _has_proba,
             [_P("bins", "bins", "int", default=10)]),
    PlotSpec("cv_folds", "Score per CV fold", "Model", _cv_folds,
             lambda ctx: ctx.evaluation is not None and (ctx.evaluation or {}).get("cv"), [_P("metric", "metric", "text")]),
    PlotSpec("pca_scatter", "PCA projection", "Unsupervised", _pca_scatter, _has_projection,
             [_P("max_points", "max points", "int", default=5000), _P("color", "colour by column", "column")]),
    PlotSpec("tsne_scatter", "t-SNE embedding", "Unsupervised", _tsne_scatter, _has_projection,
             [_P("max_points", "max points", "int", default=1500), _P("perplexity", "perplexity", "int", default=30)]),
    PlotSpec("elbow", "Elbow & silhouette", "Unsupervised", _elbow, lambda ctx: ctx.X is not None,
             [_P("min_k", "smallest k", "int", default=2), _P("max_k", "largest k", "int", default=10)]),
    PlotSpec("silhouette", "Silhouette plot", "Unsupervised", _silhouette, _has_clusters, []),
    PlotSpec("dendrogram", "Dendrogram", "Unsupervised", _dendrogram, lambda ctx: ctx.X is not None,
             [_P("linkage", "linkage", "choice", choices=["ward", "complete", "average", "single"], default="ward"),
              _P("max_points", "max points", "int", default=300)]),
    PlotSpec("cluster_sizes", "Rows per cluster", "Unsupervised", _cluster_sizes, _has_clusters, []),
    PlotSpec("reconstruction_error", "Reconstruction error (anomalies)", "Unsupervised",
             _reconstruction_error_plot, _has_reconstruction_error,
             [_P("bins", "bins", "int", default=40),
              _P("quantile", "threshold quantile", "float", default=0.99),
              _P("log", "log row count", "bool", default=True)]),
]

PLOT_INDEX = {p.key: p for p in PLOTS}
PLOT_FAMILIES = list(dict.fromkeys(p.family for p in PLOTS))


def available_plots(ctx: PlotContext) -> list[PlotSpec]:
    """Plots whose requirements the current context satisfies."""
    out = []
    for spec in PLOTS:
        try:
            if spec.needs(ctx):
                out.append(spec)
        except Exception:
            continue
    return out


def plot_requirement_hint(spec: PlotSpec) -> str:
    hints = {
        _has_df: "load some data",
        _has_numeric: "load data with a numeric column",
        _has_two_numeric: "load data with two numeric columns",
        _has_categorical: "load data with a categorical column",
        _has_two_categorical: "load data with two categorical columns",
        _has_target: "choose a target column",
        _has_numeric_target: "choose a numeric target column (use 'Class balance' for classes)",
        _has_model: "train a model",
        _has_model_classification: "train a classification model",
        _has_model_regression: "train a regression model",
        _has_proba: "train a model that can output probabilities",
        _has_importance: "train a model and compute importances",
        _has_clusters: "fit a clustering model",
        _has_tunable_param: "train a model that has a numeric hyperparameter to sweep",
    }
    return hints.get(spec.needs, "this plot is not available yet")


def render(kind: str, ctx: PlotContext, opts: dict | None = None):
    """Draw one plot and return its figure, raising PlotError with a usable message."""
    spec = PLOT_INDEX.get(kind)
    if spec is None:
        raise PlotError(f"unknown plot {kind!r}")
    with RENDER_LOCK:
        if not STYLE_APPLIED:
            apply_style()
        try:
            fig = spec.render(ctx, opts or {})
        except PlotError:
            raise
        except Exception as exc:
            raise PlotError(f"{spec.label} could not be drawn: {type(exc).__name__}: {exc}") from exc
        if not hasattr(fig, "savefig"):
            raise PlotError(f"{spec.label} did not return a figure")
        return fig


def save_figure(fig, path: str, dpi: int = 150) -> str:
    with RENDER_LOCK:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def figure_png_bytes(fig, dpi: int = 120) -> bytes:
    with RENDER_LOCK:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        return buffer.getvalue()


def figure_data_uri(fig, dpi: int = 120) -> str:
    return "data:image/png;base64," + base64.b64encode(figure_png_bytes(fig, dpi)).decode("ascii")


def default_options(spec: PlotSpec, ctx: PlotContext) -> dict:
    """Fill in sensible defaults for a plot's parameters given the context."""
    opts = {}
    df = ctx.df
    for param in spec.params:
        name = param["name"]
        default = param["default"]
        if param["kind"] == "column" and df is not None:
            if name in ("target",):
                default = ctx.target
            elif name in ("x", "column"):
                pool = numeric_columns(df)
                default = (ctx.target if ctx.target in pool and name == "x" else (pool[0] if pool else None))
            elif name == "y":
                pool = [c for c in numeric_columns(df) if c != opts.get("x")]
                default = (ctx.target if ctx.target in pool else (pool[0] if pool else None))
            elif name == "value":
                # a heatmap can only average numbers; fall back to the first numeric column
                pool = numeric_columns(df)
                default = (ctx.target if ctx.target in pool else (pool[0] if pool else None))
        if isinstance(default, float) and param["kind"] == "int":
            default = int(default)
        opts[name] = default
    return opts
