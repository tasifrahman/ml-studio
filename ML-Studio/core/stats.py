"""Statistics a researcher would otherwise write a script for.

Two layers:

* **Test functions** — one per test, each returning a ``dict`` and, where a table makes
  sense, a :class:`pandas.DataFrame`.  Nothing is invented: every p-value comes from
  SciPy.
* **A chooser** — :func:`compare_groups`, :func:`analyse_pair` and :func:`test_guide`.
  This is the part that matters.  Picking the wrong test is the single most common error
  in an engineering report, so the workbench reads the data first (how many groups, are
  they normal, do they have equal variance, are the two columns paired?) and then runs
  the test that is actually valid, saying in plain words why it chose it.

Everything is defensive about small samples: a test that cannot be trusted on the data
at hand is reported as such rather than printed with a straight face.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _array(values) -> np.ndarray:
    """A float array with the missing values removed."""
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    out = np.asarray(values, dtype=float).ravel()
    return out[np.isfinite(out)]


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
        return f"{value:.3e}"
    return f"{value:.{digits}g}"


def _p_text(p: float) -> str:
    if not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def significant(p: float, alpha: float = 0.05) -> bool:
    return math.isfinite(p) and p < alpha


def _t_crit(df_: float, level: float = 0.95) -> float:
    return float(sps.t.ppf(0.5 + level / 2, max(df_, 1e-9)))


def ci_mean(sample, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a mean, using the t distribution (correct for small n)."""
    x = _array(sample)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    half = _t_crit(n - 1, level) * se
    return (mean - half, mean + half)


def describe(sample, level: float = 0.95) -> dict:
    """Mean, spread, confidence interval and shape — with the caveats stated."""
    x = _array(sample)
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    low, high = ci_mean(x, level)
    q1, median, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": sd / math.sqrt(n) if n > 1 else float("nan"),
        f"ci{int(level * 100)} low": low,
        f"ci{int(level * 100)} high": high,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "cv %": (sd / mean * 100) if mean else float("nan"),
        "skew": float(sps.skew(x, bias=False)) if n > 2 else float("nan"),
        "kurtosis": float(sps.kurtosis(x, bias=False)) if n > 3 else float("nan"),
    }


def describe_frame(frame: pd.DataFrame, columns=None, level: float = 0.95) -> pd.DataFrame:
    columns = list(columns if columns is not None else frame.columns)
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        row = {"column": column}
        row.update(describe(frame[column], level))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# assumption checks
# --------------------------------------------------------------------------

def normality(sample, level: float = 0.05) -> pd.DataFrame:
    """Shapiro–Wilk, D'Agostino and Anderson–Darling, with a plain verdict.

    Shapiro–Wilk is the one to trust below about 5,000 points; above that its p-value
    becomes hypersensitive, so the verdict falls back to D'Agostino.  A tiny departure
    from normality is rarely worth abandoning a t-test over, and the notes say so.
    """
    x = _array(sample)
    n = len(x)
    rows: list[dict] = []
    if n < 3:
        return pd.DataFrame([{"test": "—", "statistic": float("nan"), "p": float("nan"),
                              "note": f"only {n} value(s); normality cannot be checked"}])
    if n <= 5000:
        s, p = sps.shapiro(x)
        rows.append({"test": "Shapiro–Wilk", "statistic": float(s), "p": float(p),
                     "note": "the usual choice for n up to a few thousand"})
    try:
        s, p = sps.normaltest(x)
        rows.append({"test": "D'Agostino K²", "statistic": float(s), "p": float(p),
                     "note": "needs n ≥ 8; robust to outliers"})
    except Exception:  # noqa: BLE001 - tiny samples
        pass
    try:
        try:
            # SciPy 1.17+ wants the p-value method named.  "interpolate" reads the classic
            # published tables, which is exactly what an engineer would look up by hand.
            result = sps.anderson(x, dist="norm", method="interpolate")
            rows.append({"test": "Anderson–Darling", "statistic": float(result.statistic),
                         "p": float(result.pvalue),
                         "note": "p-value interpolated from the standard tables"})
        except TypeError:  # older SciPy: critical values only, no p-value
            result = sps.anderson(x, dist="norm")
            index = min(range(len(result.significance_level)),
                        key=lambda i: abs(result.significance_level[i] - 5.0))
            critical = float(result.critical_values[index])
            rows.append({
                "test": "Anderson–Darling", "statistic": float(result.statistic),
                "p": float("nan"),
                "note": f"{'exceeds' if result.statistic > critical else 'below'} the 5% "
                        f"critical value {_fmt(critical)}",
            })
    except Exception:  # noqa: BLE001
        pass
    frame = pd.DataFrame(rows)
    frame["normal at 5%?"] = [
        "—" if not math.isfinite(v) else ("yes" if v >= level else "no")
        for v in frame["p"]
    ]
    return frame


def is_normal(sample, level: float = 0.05, min_n: int = 8) -> bool:
    """A single yes/no for the chooser.  Small samples are treated as normal.

    With fewer than eight points no normality test has any power, and falling back to
    a rank test on six values throws away more than it protects, so the t-test is used.
    """
    x = _array(sample)
    if len(x) < min_n:
        return True
    try:
        return float(sps.shapiro(x).pvalue) >= level
    except Exception:  # noqa: BLE001
        return True


def variance_test(groups: dict, level: float = 0.05) -> dict:
    """Levene (robust, the default) and Bartlett (classical) tests for equal variance."""
    arrays = [_array(v) for v in groups.values() if len(_array(v)) >= 2]
    if len(arrays) < 2:
        return {"equal variance?": True, "note": "fewer than two groups with enough values"}
    out: dict = {}
    try:
        s, p = sps.levene(*arrays, center="median")
        out["Levene statistic"] = float(s)
        out["Levene p"] = float(p)
    except Exception:  # noqa: BLE001
        pass
    try:
        s, p = sps.bartlett(*arrays)
        out["Bartlett statistic"] = float(s)
        out["Bartlett p"] = float(p)
    except Exception:  # noqa: BLE001
        pass
    levene_p = out.get("Levene p", float("nan"))
    out["equal variance?"] = not significant(levene_p, level)
    out["note"] = ("Levene is the one used for the decision — it does not assume normality"
                   if math.isfinite(levene_p) else "could not be tested")
    return out


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------

def cohens_d(a, b) -> float:
    x, y = _array(a), _array(b)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = math.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def hedges_g(a, b) -> float:
    d = cohens_d(a, b)
    n1, n2 = len(_array(a)), len(_array(b))
    if not math.isfinite(d) or n1 + n2 < 4:
        return float("nan")
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * correction)


def rank_biserial(a, b) -> float:
    """Effect size for a Mann–Whitney comparison: −1 to +1."""
    x, y = _array(a), _array(b)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    try:
        u = float(sps.mannwhitneyu(x, y, alternative="two-sided").statistic)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float(2 * u / (len(x) * len(y)) - 1)


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    chi2 = float(sps.chi2_contingency(table)[0])
    n = table.sum()
    smallest = min(table.shape) - 1
    if n == 0 or smallest <= 0:
        return float("nan")
    return float(math.sqrt(chi2 / (n * smallest)))


def eta_squared(groups: dict) -> float:
    arrays = [_array(v) for v in groups.values() if len(_array(v)) > 0]
    if len(arrays) < 2:
        return float("nan")
    grand = np.concatenate(arrays)
    ss_total = float(((grand - grand.mean()) ** 2).sum())
    ss_between = sum(len(a) * (a.mean() - grand.mean()) ** 2 for a in arrays)
    return float(ss_between / ss_total) if ss_total else float("nan")


def omega_squared(groups: dict) -> float:
    arrays = [_array(v) for v in groups.values() if len(_array(v)) > 0]
    k = len(arrays)
    if k < 2:
        return float("nan")
    n_total = sum(len(a) for a in arrays)
    grand = np.concatenate(arrays)
    ss_total = float(((grand - grand.mean()) ** 2).sum())
    ss_between = sum(len(a) * (a.mean() - grand.mean()) ** 2 for a in arrays)
    ss_within = ss_total - ss_between
    df_within = n_total - k
    if ss_total == 0 or df_within <= 0:
        return float("nan")
    ms_within = ss_within / df_within
    return float((ss_between - (k - 1) * ms_within) / (ss_total + ms_within))


def interpret_effect(value: float, kind: str = "d") -> str:
    """Cohen's rough bands, stated as words so a report can quote them."""
    if value is None or not math.isfinite(value):
        return "—"
    magnitude = abs(value)
    if kind == "d":
        bands = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large"))
    elif kind == "v":
        bands = ((0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large"))
    elif kind == "eta":
        bands = ((0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large"))
    elif kind == "r":
        bands = ((0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large"))
    else:
        return "—"
    for edge, word in bands:
        if magnitude < edge:
            return word
    return "large"


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------

def _test_row(name: str, statistic: float, p: float, note: str = "", **extra) -> dict:
    row = {"test": name, "statistic": float(statistic), "p": float(p),
           "significant at 5%?": "yes" if significant(p) else "no", "note": note}
    row.update(extra)
    return row


def one_sample(sample, mu: float = 0.0) -> pd.DataFrame:
    """Is the mean different from a target value?"""
    x = _array(sample)
    if len(x) < 2:
        return pd.DataFrame([{"test": "—", "note": "need at least two values"}])
    rows = []
    s, p = sps.ttest_1samp(x, mu)
    low, high = ci_mean(x)
    rows.append(_test_row("One-sample t-test", s, p,
                          f"mean {_fmt(x.mean())} vs target {_fmt(mu)}; "
                          f"95% CI {_fmt(low)} to {_fmt(high)}",
                          mean=float(x.mean()), target=float(mu),
                          cohens_d=float((x.mean() - mu) / x.std(ddof=1)) if x.std(ddof=1) else float("nan")))
    if len(x) >= 6:
        try:
            s, p = sps.wilcoxon(x - mu)
            rows.append(_test_row("Wilcoxon signed-rank", s, p,
                                  "does not assume normality; compares the median to the target"))
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(rows)


def two_sample(a, b, equal_var: bool = False, labels: tuple[str, str] = ("A", "B")) -> pd.DataFrame:
    """Two independent groups: t-test (Welch by default) and Mann–Whitney side by side."""
    x, y = _array(a), _array(b)
    rows = []
    if len(x) < 2 or len(y) < 2:
        return pd.DataFrame([{"test": "—", "note": "each group needs at least two values"}])
    s, p = sps.ttest_ind(x, y, equal_var=equal_var)
    rows.append(_test_row(
        "Student t-test" if equal_var else "Welch t-test", s, p,
        f"{labels[0]} mean {_fmt(x.mean())} vs {labels[1]} mean {_fmt(y.mean())}; "
        f"difference {_fmt(x.mean() - y.mean())}",
        mean_a=float(x.mean()), mean_b=float(y.mean()),
        difference=float(x.mean() - y.mean()),
        ci_low=float((x.mean() - y.mean()) - _t_crit(len(x) + len(y) - 2) *
                     math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))) if not equal_var else float("nan"),
        ci_high=float((x.mean() - y.mean()) + _t_crit(len(x) + len(y) - 2) *
                      math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))) if not equal_var else float("nan"),
        cohens_d=cohens_d(x, y), hedges_g=hedges_g(x, y)))
    try:
        s, p = sps.mannwhitneyu(x, y, alternative="two-sided")
        rows.append(_test_row("Mann–Whitney U", s, p,
                              f"compares the medians ({_fmt(np.median(x))} vs {_fmt(np.median(y))}); "
                              "no normality assumption",
                              rank_biserial=rank_biserial(x, y)))
    except Exception:  # noqa: BLE001
        pass
    try:
        s, p = sps.levene(x, y, center="median")
        rows.append(_test_row("Levene (equal variance)", s, p,
                              "if this is significant, read the Welch row, not Student's"))
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(rows)


def paired(a, b, labels: tuple[str, str] = ("A", "B")) -> pd.DataFrame:
    """Two measurements on the same rows: paired t-test and Wilcoxon signed-rank."""
    x, y = _array(a), _array(b)
    rows = []
    n = min(len(x), len(y))
    if n < 3:
        return pd.DataFrame([{"test": "—", "note": "need at least three pairs"}])
    x, y = x[:n], y[:n]
    diff = x - y
    s, p = sps.ttest_rel(x, y)
    low, high = ci_mean(diff)
    rows.append(_test_row("Paired t-test", s, p,
                          f"mean change {_fmt(diff.mean())} "
                          f"(95% CI {_fmt(low)} to {_fmt(high)}); n = {n} pairs",
                          mean_change=float(diff.mean()), ci_low=low, ci_high=high,
                          cohens_d=float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else float("nan")))
    try:
        s, p = sps.wilcoxon(x, y)
        rows.append(_test_row("Wilcoxon signed-rank", s, p,
                              "the paired test that does not assume normality"))
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(rows)


def anova(groups: dict) -> pd.DataFrame:
    """One-way ANOVA, plus Welch's version when the variances are unequal."""
    arrays = {str(k): _array(v) for k, v in groups.items() if len(_array(v)) > 0}
    rows = []
    if len(arrays) < 2:
        return pd.DataFrame([{"test": "—", "note": "need at least two groups with values"}])
    samples = list(arrays.values())
    if all(len(a) >= 2 for a in samples):
        s, p = sps.f_oneway(*samples)
        rows.append(_test_row("One-way ANOVA", s, p,
                              f"{len(samples)} groups; η² = {_fmt(eta_squared(arrays))} "
                              f"({interpret_effect(eta_squared(arrays), 'eta')} effect)",
                              eta_squared=eta_squared(arrays),
                              omega_squared=omega_squared(arrays)))
    welch = welch_anova(arrays)
    if welch is not None:
        rows.append(_test_row("Welch ANOVA", welch["F"], welch["p"],
                              "does not assume equal variances" if not welch["unequal"] else "",
                              df1=welch["df1"], df2=welch["df2"]))
    try:
        s, p = sps.kruskal(*samples)
        rows.append(_test_row("Kruskal–Wallis", s, p,
                              "the rank-based alternative when the data are not normal"))
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(rows)


def welch_anova(groups: dict) -> dict | None:
    """Welch's ANOVA, computed directly — SciPy has no one-way version of it."""
    arrays = [_array(v) for v in groups.values() if len(_array(v)) >= 2]
    k = len(arrays)
    if k < 2:
        return None
    n = np.array([len(a) for a in arrays], dtype=float)
    var = np.array([a.var(ddof=1) for a in arrays], dtype=float)
    if np.any(var <= 0):
        return None
    w = n / var
    total_w = w.sum()
    means = np.array([a.mean() for a in arrays])
    grand = float((w * means).sum() / total_w)
    a_part = float((w * (means - grand) ** 2).sum())
    b_part = float((((1 - w / total_w) ** 2) / (n - 1)).sum())
    lam = 3 * b_part / (k * k - 1)
    f = (a_part / (k - 1)) / (1 + 2 * lam * (k - 2) / 3)
    df2 = 1 / lam if lam else float("nan")
    p = float(sps.f.sf(f, k - 1, df2)) if math.isfinite(df2) and df2 > 0 else float("nan")
    return {"F": f, "p": p, "df1": k - 1, "df2": df2, "unequal": True}


def tukey(groups: dict) -> pd.DataFrame:
    """Tukey HSD: every pair of groups, honest about multiple comparisons."""
    arrays = {str(k): _array(v) for k, v in groups.items() if len(_array(v)) >= 2}
    if len(arrays) < 2:
        return pd.DataFrame([{"pair": "—", "note": "need at least two groups with two values each"}])
    try:
        result = sps.tukey_hsd(*arrays.values())
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([{"pair": "—", "note": f"Tukey HSD unavailable: {exc}"}])
    names = list(arrays)
    try:
        low, high = result.confidence_interval(confidence_level=0.95)
    except Exception:  # noqa: BLE001
        low = high = None
    rows = []
    for i in range(len(names)):
        for j in range(len(names)):
            if j <= i:
                continue
            rows.append({
                "pair": f"{names[i]} vs {names[j]}",
                "mean difference": float(np.mean(arrays[names[i]]) - np.mean(arrays[names[j]])),
                "95% low": float(low[i, j]) if low is not None else float("nan"),
                "95% high": float(high[i, j]) if high is not None else float("nan"),
                "p": float(result.pvalue[i, j]),
                "significant at 5%?": "yes" if significant(float(result.pvalue[i, j])) else "no",
            })
    return pd.DataFrame(rows)


def dunn(groups: dict) -> pd.DataFrame:
    """Dunn's post-hoc test after Kruskal–Wallis, with a Bonferroni correction.

    Implemented directly, because SciPy does not ship it and `scikit-posthocs` is not
    installed.  The tie correction follows the standard formulation.
    """
    arrays = {str(k): _array(v) for k, v in groups.items() if len(_array(v)) >= 2}
    if len(arrays) < 2:
        return pd.DataFrame([{"pair": "—", "note": "need at least two groups"}])
    names = list(arrays)
    all_values = np.concatenate([arrays[k] for k in names])
    n_total = len(all_values)
    ranks = sps.rankdata(all_values)
    mean_ranks = {}
    start = 0
    for name in names:
        size = len(arrays[name])
        mean_ranks[name] = float(ranks[start:start + size].mean())
        start += size
    _, counts = np.unique(all_values, return_counts=True)
    tie = float((counts ** 3 - counts).sum())
    variance = n_total * (n_total + 1) / 12 - tie / (12 * (n_total - 1))
    comparisons = len(names) * (len(names) - 1) / 2
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            se = math.sqrt(variance * (1 / len(arrays[a]) + 1 / len(arrays[b])))
            z = (mean_ranks[a] - mean_ranks[b]) / se if se else float("nan")
            p = float(2 * (1 - sps.norm.cdf(abs(z)))) if math.isfinite(z) else float("nan")
            p_adj = min(1.0, p * comparisons) if math.isfinite(p) else float("nan")
            rows.append({"pair": f"{a} vs {b}", "mean rank difference":
                         mean_ranks[a] - mean_ranks[b], "z": z, "p": p,
                         "p (Bonferroni)": p_adj,
                         "significant at 5%?": "yes" if significant(p_adj) else "no"})
    return pd.DataFrame(rows)


def mann_whitney(a, b) -> pd.DataFrame:
    x, y = _array(a), _array(b)
    if len(x) == 0 or len(y) == 0:
        return pd.DataFrame([{"test": "—", "note": "both groups need values"}])
    s, p = sps.mannwhitneyu(x, y, alternative="two-sided")
    return pd.DataFrame([_test_row("Mann–Whitney U", s, p,
                                   "compares the medians; no normality assumption",
                                   rank_biserial=rank_biserial(x, y),
                                   median_a=float(np.median(x)), median_b=float(np.median(y)))])


def wilcoxon(a, b) -> pd.DataFrame:
    x, y = _array(a), _array(b)
    n = min(len(x), len(y))
    if n < 6:
        return pd.DataFrame([{"test": "—", "note": "need at least six pairs"}])
    s, p = sps.wilcoxon(x[:n], y[:n])
    return pd.DataFrame([_test_row("Wilcoxon signed-rank", s, p, f"{n} pairs")])


def kruskal(groups: dict) -> pd.DataFrame:
    arrays = [_array(v) for v in groups.values() if len(_array(v)) > 0]
    if len(arrays) < 2:
        return pd.DataFrame([{"test": "—", "note": "need at least two groups"}])
    s, p = sps.kruskal(*arrays)
    return pd.DataFrame([_test_row("Kruskal–Wallis", s, p,
                                   "rank-based; use Dunn's test to see which groups differ")])


def chi_square(table: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Association between two categorical columns, with Cramér's V as effect size."""
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        return pd.DataFrame([{"test": "—", "note": "a contingency table needs at least 2×2 cells"}])
    if values.sum() == 0:
        return pd.DataFrame([{"test": "—", "note": "the table is empty"}])
    chi2, p, dof, expected = sps.chi2_contingency(values)
    small = (expected < 5).mean()
    note = "all expected counts ≥ 5, so the approximation is sound"
    if small > 0.2:
        note = (f"{small:.0%} of expected counts are below 5 — read Fisher's exact test "
                "for a 2×2 table, or pool sparse categories")
    rows = [_test_row("Chi-square test of independence", chi2, p, note,
                      dof=int(dof), cramers_v=cramers_v(values),
                      effect=interpret_effect(cramers_v(values), "v"))]
    if values.shape == (2, 2):
        try:
            odds, p_fisher = sps.fisher_exact(values)
            rows.append(_test_row("Fisher's exact test", float(odds), float(p_fisher),
                                  "exact p-value; the right choice for a small 2×2 table",
                                  odds_ratio=float(odds)))
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(rows)


def correlation_p(a, b, method: str = "pearson") -> dict:
    """A correlation with its p-value, sample size and confidence interval."""
    x, y = _array(a), _array(b)
    n = min(len(x), len(y))
    if n < 3:
        return {"n": n, "note": "need at least three pairs"}
    x, y = x[:n], y[:n]
    if method == "spearman":
        result = sps.spearmanr(x, y)
    elif method == "kendall":
        result = sps.kendalltau(x, y)
    else:
        result = sps.pearsonr(x, y)
    r = float(np.asarray(getattr(result, "statistic", result[0])).ravel()[0])
    p = float(np.asarray(getattr(result, "pvalue", result[1])).ravel()[0])
    low = high = float("nan")
    if method == "pearson" and n > 3:
        # Fisher z interval — the standard way to put a range on a correlation
        z = math.atanh(max(min(r, 0.999999), -0.999999))
        se = 1 / math.sqrt(n - 3)
        low = math.tanh(z - 1.96 * se)
        high = math.tanh(z + 1.96 * se)
    elif method == "spearman" and n > 3:
        z = math.atanh(max(min(r, 0.999999), -0.999999))
        se = 1.03 / math.sqrt(n - 3)
        low = math.tanh(z - 1.96 * se)
        high = math.tanh(z + 1.96 * se)
    return {"method": method, "n": n, "r": r, "p": p, "ci low": low, "ci high": high,
            "strength": interpret_effect(r, "r"),
            "direction": "positive" if r > 0 else ("negative" if r < 0 else "none"),
            "significant at 5%?": "yes" if significant(p) else "no"}


def correlation_table(frame: pd.DataFrame, target: str, method: str = "pearson",
                      columns=None) -> pd.DataFrame:
    """Correlate one column against many, sorted by strength of relationship."""
    if target not in frame.columns:
        return pd.DataFrame()
    columns = [c for c in (columns if columns is not None else frame.columns)
               if c != target and c in frame.columns]
    rows = []
    for column in columns:
        try:
            pair = frame[[target, column]].dropna()
            if len(pair) < 3:
                continue
            info = correlation_p(pair[target], pair[column], method)
            rows.append({"column": column, "n": info["n"], "r": info["r"], "p": info["p"],
                         "95% low": info["ci low"], "95% high": info["ci high"],
                         "strength": info["strength"],
                         "significant at 5%?": info["significant at 5%?"]})
        except Exception:  # noqa: BLE001 - a non-numeric column is simply skipped
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def linear_fit(x, y) -> dict:
    """Ordinary least squares with slope confidence interval and residual diagnostics."""
    xs, ys = _array(x), _array(y)
    n = min(len(xs), len(ys))
    if n < 3:
        return {"n": n, "note": "need at least three pairs"}
    xs, ys = xs[:n], ys[:n]
    result = sps.linregress(xs, ys)
    predicted = result.intercept + result.slope * xs
    residuals = ys - predicted
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    dof = n - 2
    se = math.sqrt(ss_res / dof) if dof > 0 else float("nan")
    return {
        "n": n, "slope": float(result.slope), "intercept": float(result.intercept),
        "slope standard error": float(result.stderr),
        "slope 95% low": float(result.slope - _t_crit(dof) * result.stderr),
        "slope 95% high": float(result.slope + _t_crit(dof) * result.stderr),
        "r": float(result.rvalue), "r2": float(r2),
        "p": float(result.pvalue), "residual sd": se,
        "equation": f"y = {_fmt(result.intercept)} + {_fmt(result.slope)}·x",
        "residuals normal?": "yes" if is_normal(residuals) else "no",
        "significant at 5%?": "yes" if significant(result.pvalue) else "no",
    }


def grubbs(sample, alpha: float = 0.05) -> pd.DataFrame:
    """Grubbs' test for a single outlier — the one an engineer would script by hand."""
    x = _array(sample)
    n = len(x)
    if n < 7:
        return pd.DataFrame([{"test": "Grubbs", "note": "needs at least seven values"}])
    mean, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        return pd.DataFrame([{"test": "Grubbs", "note": "every value is identical"}])
    deviations = np.abs(x - mean)
    index = int(deviations.argmax())
    g = float(deviations[index] / sd)
    t = float(sps.t.ppf(1 - alpha / (2 * n), n - 2))
    critical = (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))
    return pd.DataFrame([{
        "test": "Grubbs (single outlier)", "G": g, "critical value": critical,
        "outlier?": "yes" if g > critical else "no",
        "value": float(x[index]), "row position": index,
        "note": f"the most extreme value is {_fmt(x[index])}, {_fmt(g)} standard deviations out",
    }])


# --------------------------------------------------------------------------
# confidence intervals and sample size
# --------------------------------------------------------------------------

def proportion_ci(successes: int, total: int, level: float = 0.95) -> dict:
    """Wilson interval — the one to use, since the textbook interval fails near 0 and 1."""
    if total <= 0:
        return {"n": total, "note": "no trials"}
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    p = successes / total
    z = float(sps.norm.ppf(0.5 + level / 2))
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"n": total, "successes": successes, "proportion": p,
            "percent": p * 100, f"ci{int(level * 100)} low": centre - half,
            f"ci{int(level * 100)} high": centre + half, "method": "Wilson"}


def sample_size_mean(effect_size: float, power: float = 0.80, alpha: float = 0.05,
                     two_sided: bool = True) -> dict:
    """How many samples to detect a given standardised difference (Cohen's d)."""
    if effect_size <= 0:
        return {"note": "the effect size must be positive"}
    z_alpha = float(sps.norm.ppf(1 - alpha / (2 if two_sided else 1)))
    z_beta = float(sps.norm.ppf(power))
    n = (z_alpha + z_beta) ** 2 / (effect_size ** 2)
    return {"effect size (d)": effect_size, "power": power, "alpha": alpha,
            "n per group": math.ceil(n), "note": "add 10–20% for dropouts or failed specimens"}


def sample_size_proportion(p1: float, p2: float, power: float = 0.80,
                           alpha: float = 0.05) -> dict:
    """How many per group to detect a difference between two proportions."""
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return {"note": "both proportions must be between 0 and 1"}
    if p1 == p2:
        return {"note": "the two proportions are identical, so no sample size exists"}
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_beta = float(sps.norm.ppf(power))
    average = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * average * (1 - average))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numerator / ((p1 - p2) ** 2)
    return {"proportion 1": p1, "proportion 2": p2, "power": power, "alpha": alpha,
            "n per group": math.ceil(n)}


def power_for_mean(effect_size: float, n_per_group: int, alpha: float = 0.05) -> dict:
    """The power you actually have with the sample you have."""
    if effect_size <= 0 or n_per_group < 2:
        return {"note": "give a positive effect size and at least two samples per group"}
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    noncentral = effect_size * math.sqrt(n_per_group / 2)
    power = float(1 - sps.norm.cdf(z_alpha - noncentral) + sps.norm.cdf(-z_alpha - noncentral))
    return {"effect size (d)": effect_size, "n per group": n_per_group,
            "alpha": alpha, "power": power,
            "verdict": "adequate (≥ 0.80)" if power >= 0.80 else "underpowered"}


# --------------------------------------------------------------------------
# the chooser — this is what makes the screen usable without a statistician
# --------------------------------------------------------------------------

def test_guide() -> pd.DataFrame:
    """A decision table: the question, the test, and when it is the right one."""
    return pd.DataFrame([
        {"your question": "Is this sample's mean different from a target?",
         "test to use": "One-sample t-test",
         "how many groups": "1 sample",
         "assumes": "values roughly normal (or n ≥ 30 by the central limit theorem)"},
        {"your question": "Do two groups differ?",
         "test to use": "Welch t-test",
         "how many groups": "2, independent",
         "assumes": "roughly normal; does not need equal variances"},
        {"your question": "Do two groups differ, without assuming normality?",
         "test to use": "Mann–Whitney U",
         "how many groups": "2, independent",
         "assumes": "nothing about the shape — the safe default for small samples"},
        {"your question": "Did the same specimens change between two tests?",
         "test to use": "Paired t-test / Wilcoxon signed-rank",
         "how many groups": "2, the same rows twice",
         "assumes": "each row is a pair — before/after on the same item"},
        {"your question": "Do three or more groups differ?",
         "test to use": "One-way ANOVA + Tukey HSD",
         "how many groups": "3 or more, independent",
         "assumes": "roughly normal with similar spread in every group"},
        {"your question": "Do three or more groups differ, without assuming normality?",
         "test to use": "Kruskal–Wallis + Dunn's test",
         "how many groups": "3 or more, independent",
         "assumes": "nothing about the shape"},
        {"your question": "Are two categories associated (e.g. supplier vs PASS/FAIL)?",
         "test to use": "Chi-square; Fisher's exact for a small 2×2",
         "how many groups": "2 categorical columns",
         "assumes": "expected counts of 5 or more per cell, unless Fisher's is used"},
        {"your question": "Do two measurements move together?",
         "test to use": "Pearson r (Spearman rho if not linear)",
         "how many groups": "2 numeric columns",
         "assumes": "Pearson needs a straight-line relationship and no strong outliers"},
        {"your question": "Is one measurement out of line with the rest?",
         "test to use": "Grubbs' test",
         "how many groups": "1 sample",
         "assumes": "one suspect value; run it again after removing a confirmed outlier"},
        {"your question": "How many specimens do I need?",
         "test to use": "Sample-size calculator",
         "how many groups": "—",
         "assumes": "an effect size you are willing to call important"},
    ])


def _column_kind(series: pd.Series) -> str:
    """numeric / datetime / text — the crude split the chooser needs."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    converted = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() and converted.notna().mean() > 0.9:
        return "numeric"
    return "text"


def compare_groups(frame: pd.DataFrame, value_col: str, group_col: str,
                   max_groups: int = 12, alpha: float = 0.05) -> dict:
    """Pick and run the right group-comparison test for two columns.

    Returns a dict with ``headline`` (one plain sentence), ``decision`` (why that test),
    ``result`` (the test table), ``groups`` (per-group summary), ``assumptions`` and — for
    three or more groups — ``posthoc``.  The screen shows all of it, so the reasoning is
    visible rather than buried in a script nobody can check.
    """
    if value_col not in frame.columns or group_col not in frame.columns:
        raise ValueError("both columns must exist")
    data = frame[[value_col, group_col]].dropna()
    if data.empty:
        raise ValueError("no rows left after dropping missing values")

    value_kind = _column_kind(data[value_col])
    if value_kind == "text":
        return {"headline": f"“{value_col}” is not a number, so there is nothing to average.",
                "decision": "Pick a numeric measurement column for the value.",
                "result": pd.DataFrame(), "groups": pd.DataFrame(), "assumptions": pd.DataFrame()}

    data = data.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])

    counts = data[group_col].astype(str).value_counts()
    kept = list(counts[counts >= 2].index)[:max_groups]
    dropped = [str(k) for k in counts.index if str(k) not in kept]
    data["_group"] = data[group_col].astype(str)
    data = data[data["_group"].isin(kept)]
    groups = {name: data.loc[data["_group"] == name, value_col].to_numpy() for name in kept}

    summary = pd.DataFrame([dict({"group": name}, **describe(values)) for name, values in groups.items()])

    if len(groups) < 2:
        return {"headline": "Only one group has two or more values, so no comparison is possible.",
                "decision": "A test needs at least two groups with a few values each.",
                "result": pd.DataFrame(), "groups": summary, "assumptions": pd.DataFrame()}

    normal = {name: is_normal(values) for name, values in groups.items()}
    var = variance_test(groups)
    equal_var = bool(var.get("equal variance?", True))
    all_normal = all(normal.values())

    assumption_rows = [{"group": name, "n": len(values), "normal at 5%?": "yes" if normal[name] else "no"}
                       for name, values in groups.items()]
    assumptions = pd.DataFrame(assumption_rows)

    if len(groups) == 2:
        (a_name, a), (b_name, b) = list(groups.items())
        if all_normal and equal_var:
            chosen, why = "Student t-test", ("both groups look normal and their variances match "
                                             "(Levene p ≥ 0.05)")
        elif all_normal:
            chosen, why = "Welch t-test", ("both groups look normal but their variances differ, "
                                           "so the unequal-variance version is used")
        else:
            chosen, why = "Mann–Whitney U", ("at least one group is not normal, so a rank test "
                                             "is the honest choice")
        if chosen == "Mann–Whitney U":
            result = mann_whitney(a, b)
        else:
            result = two_sample(a, b, equal_var=(chosen == "Student t-test"),
                                labels=(a_name, b_name))
            result = result.iloc[[0]].reset_index(drop=True)
        result.loc[:, "test"] = chosen
        p = float(result.iloc[0]["p"])
        effect = result.iloc[0].get("cohens_d", result.iloc[0].get("rank_biserial"))
        direction = "higher" if np.mean(a) > np.mean(b) else "lower"
        headline = (f"{a_name} is {_fmt(abs(np.mean(a) - np.mean(b)))} {direction} than {b_name} "
                    f"({_fmt(np.mean(a))} vs {_fmt(np.mean(b))}); {_p_text(p)}")
        if not significant(p, alpha):
            headline = (f"No significant difference between {a_name} and {b_name} "
                        f"({_p_text(p)}); the means are {_fmt(np.mean(a))} and {_fmt(np.mean(b))}")
        return {"headline": headline, "decision": f"{chosen} — {why}.",
                "result": result, "groups": summary, "assumptions": assumptions,
                "variance": var, "effect": effect,
                "effect size": interpret_effect(effect) if effect is not None else "—"}

    if all_normal and equal_var:
        chosen, why = "One-way ANOVA + Tukey HSD", ("every group looks normal with a similar "
                                                    "spread, which is exactly what ANOVA wants")
    elif all_normal:
        chosen, why = "Welch ANOVA", ("groups are normal but their spreads differ, so the "
                                      "unequal-variance version is used")
    else:
        chosen, why = "Kruskal–Wallis + Dunn's test", ("at least one group is not normal, so a "
                                                       "rank-based test is used")
    result = anova(groups) if chosen != "Kruskal–Wallis + Dunn's test" else kruskal(groups)
    row = result.iloc[0]
    p = float(row["p"])
    strongest = max(groups, key=lambda name: np.mean(groups[name]))
    weakest = min(groups, key=lambda name: np.mean(groups[name]))
    headline = (f"{len(groups)} groups differ overall ({_p_text(p)}); "
                f"{strongest} is highest ({_fmt(np.mean(groups[strongest]))}), "
                f"{weakest} lowest ({_fmt(np.mean(groups[weakest]))})")
    if not significant(p, alpha):
        headline = f"No significant difference across the {len(groups)} groups ({_p_text(p)})"
    posthoc = tukey(groups) if chosen.startswith("One-way") else dunn(groups)
    if dropped:
        headline += f" · {len(dropped)} group(s) with fewer than two values were left out"
    return {"headline": headline, "decision": f"{chosen} — {why}.",
            "result": result, "groups": summary, "assumptions": assumptions,
            "posthoc": posthoc, "variance": var,
            "effect": eta_squared(groups) if chosen != "Kruskal–Wallis + Dunn's test" else None,
            "effect size": interpret_effect(eta_squared(groups), "eta")}


def analyse_pair(frame: pd.DataFrame, col_a: str, col_b: str, paired_rows: bool = False,
                 method: str = "pearson") -> dict:
    """Look at two columns and do the right thing, whatever their types are.

    numeric + numeric → correlation and a straight-line fit
    numeric + text    → group comparison (t-test / ANOVA / their rank equivalents)
    text + text       → contingency table and chi-square
    A datetime column becomes the numeric one, so "strength vs age" just works.
    """
    if col_a not in frame.columns or col_b not in frame.columns:
        raise ValueError("both columns must exist")
    kind_a, kind_b = _column_kind(frame[col_a]), _column_kind(frame[col_b])

    def numeric_version(series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.astype("int64") / 8.64e13  # days, so the numbers are readable
        return pd.to_numeric(series, errors="coerce")

    if kind_a == "numeric" and kind_b == "numeric":
        data = frame[[col_a, col_b]].copy()
        data[col_a], data[col_b] = numeric_version(data[col_a]), numeric_version(data[col_b])
        data = data.dropna()
        if len(data) < 3:
            raise ValueError("fewer than three complete rows")
        info = correlation_p(data[col_a], data[col_b], method)
        fit = linear_fit(data[col_a], data[col_b])
        row = {k: v for k, v in info.items()}
        row.update({k: v for k, v in fit.items() if k != "n"})
        headline = (f"{method.title()} correlation r = {_fmt(info['r'])} "
                    f"({info['strength']}, {info['direction']}); {_p_text(info['p'])} over "
                    f"{info['n']} rows")
        decision = (f"{method.title()} correlation answers 'do they move together?'; "
                    "the straight-line fit answers 'by how much per unit?'.")
        out = {
            "kind": "numeric-numeric", "headline": headline, "decision": decision,
            "result": pd.DataFrame([row]), "notes": pd.DataFrame([fit]),
            "x": col_a, "y": col_b, "frame": data,
        }
        out["paired"] = bool(paired_rows)
        if paired_rows:
            change = paired(data[col_a], data[col_b], labels=(col_a, col_b))
            out["paired_result"] = change
            p_change = float(change.iloc[0]["p"])
            out["headline"] += (f" · the paired test on the same rows gives "
                                f"{_p_text(p_change)}")
        return out

    if kind_a == "text" and kind_b == "text":
        table = pd.crosstab(frame[col_a].astype(str), frame[col_b].astype(str))
        result = chi_square(table)
        p = float(result.iloc[0].get("p", float("nan")))
        headline = (f"{col_a} and {col_b} are {'associated' if significant(p) else 'not detectably associated'} "
                    f"({_p_text(p)}, Cramér's V = {_fmt(result.iloc[0].get('cramers_v'))})")
        return {"kind": "text-text", "headline": headline,
                "decision": "Chi-square tests whether the two categories are independent; "
                            "the table below shows where the counts sit.",
                "result": result, "contingency": table}

    # one numeric, one group
    if kind_a == "text":
        value_col, group_col = col_b, col_a
    else:
        value_col, group_col = col_a, col_b
    data = frame[[value_col, group_col]].copy()
    if pd.api.types.is_datetime64_any_dtype(data[value_col]):
        data[value_col] = numeric_version(data[value_col])
    info = compare_groups(data, value_col, group_col)
    info["kind"] = "numeric-text"
    info["paired"] = bool(paired_rows)
    if paired_rows:
        # The same rows measured twice: the paired test is both more powerful and the
        # one the question actually asks, so it is shown alongside the group test.
        change = paired(data[value_col], data[group_col])
        info["paired_result"] = change
    return info
