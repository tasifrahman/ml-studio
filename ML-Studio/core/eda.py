"""Exploratory data analysis: profiling, column typing, correlation, task detection.

Nothing here mutates the frame it is handed; every function returns a fresh
object so the GUI can call them freely while the user clicks around.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A column counts as numeric once at least this share of its non-null values
# parse as numbers (after stripping currency symbols and thousands separators).
NUMERIC_COERCION_THRESHOLD = 0.9

_UNIQUE_SAMPLE = 5


# --------------------------------------------------------------------------
# column typing
# --------------------------------------------------------------------------


def _looks_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    sample = sample.str.replace(r"[,\s]", "", regex=True).str.strip()
    if sample.empty:
        return False
    import re

    ok = sample.str.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    return bool(ok.mean() >= NUMERIC_COERCION_THRESHOLD)


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    hits = 0
    for value in sample:
        if any(ch in value for ch in ("-", "/", ":")):
            try:
                pd.Timestamp(value)
                hits += 1
            except (ValueError, TypeError):
                pass
    return hits >= max(3, int(0.6 * len(sample)))


@dataclass
class ColumnInfo:
    name: str
    kind: str            # numeric | categorical | datetime | boolean | text
    dtype: str
    n_missing: int
    n_unique: int
    is_constant: bool
    samples: list = field(default_factory=list)


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Map each column to one of numeric/categorical/datetime/boolean/text."""
    kinds = {}
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            kinds[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[col] = "datetime"
        elif _looks_numeric(s):
            kinds[col] = "numeric"
        elif _is_datetime_like(s):
            kinds[col] = "datetime"
        else:
            nun = s.nunique(dropna=True)
            kinds[col] = "categorical" if nun <= max(50, 0.2 * n) else "text"
    return kinds


def numeric_columns(df, kinds=None) -> list[str]:
    kinds = kinds or column_kinds(df)
    return [c for c, k in kinds.items() if k == "numeric"]


def categorical_columns(df, kinds=None) -> list[str]:
    kinds = kinds or column_kinds(df)
    return [c for c, k in kinds.items() if k in ("categorical", "boolean")]


def datetime_columns(df, kinds=None) -> list[str]:
    kinds = kinds or column_kinds(df)
    return [c for c, k in kinds.items() if k == "datetime"]


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missingness, cardinality, a few sample values."""
    kinds = column_kinds(df)
    n = max(len(df), 1)
    rows = []
    for col in df.columns:
        s = df[col]
        nun = int(s.nunique(dropna=True))
        samples = [str(v) for v in s.dropna().unique()[:_UNIQUE_SAMPLE]]
        rows.append(
            {
                "column": col,
                "type": kinds[col],
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_%": round(100 * s.isna().sum() / n, 2),
                "unique": nun,
                "constant": nun <= 1,
                "sample values": ", ".join(samples)[:80],
            }
        )
    return pd.DataFrame(rows)


def frame_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate rows": int(df.duplicated().sum()),
        "cells missing": int(df.isna().sum().sum()),
        "memory MB": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    out = pd.DataFrame(
        {
            "column": counts.index,
            "missing": counts.values,
            "missing_%": (100 * counts.values / max(len(df), 1)).round(2),
            "dtype": [str(df[c].dtype) for c in counts.index],
        }
    )
    return out


def numeric_describe(df: pd.DataFrame, kinds=None) -> pd.DataFrame:
    cols = numeric_columns(df, kinds)
    if not cols:
        return pd.DataFrame()
    desc = df[cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["skew"] = df[cols].apply(pd.to_numeric, errors="coerce").skew()
    desc["kurtosis"] = df[cols].apply(pd.to_numeric, errors="coerce").kurtosis()
    desc["zeros"] = (df[cols].apply(pd.to_numeric, errors="coerce") == 0).sum()
    return desc.round(4)


def categorical_describe(df: pd.DataFrame, kinds=None, top: int = 10) -> pd.DataFrame:
    cols = categorical_columns(df, kinds)
    rows = []
    for col in cols:
        vc = df[col].value_counts(dropna=False).head(top)
        for value, count in vc.items():
            rows.append({"column": col, "value": value, "count": int(count), "%": round(100 * count / max(len(df), 1), 2)})
    return pd.DataFrame(rows)


def correlation(df: pd.DataFrame, columns=None, method: str = "pearson", round_to: int = 3) -> pd.DataFrame:
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    return num.corr(method=method).round(round_to)


def top_correlations(corr: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Long-form view of the strongest off-diagonal correlations."""
    if corr.empty:
        return pd.DataFrame()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val):
                pairs.append({"feature A": a, "feature B": b, "correlation": val, "|r|": abs(val)})
    out = pd.DataFrame(pairs).sort_values("|r|", ascending=False).head(limit)
    return out.drop(columns="|r|").reset_index(drop=True)


# --------------------------------------------------------------------------
# target / task detection
# --------------------------------------------------------------------------


def detect_task(y: pd.Series, max_classes: int = 30) -> str:
    """Return 'regression', 'binary', 'multiclass' or 'ambiguous'."""
    if y is None:
        return "none"
    s = y.dropna()
    if s.empty:
        return "ambiguous"
    if pd.api.types.is_bool_dtype(s):
        return "binary" if s.nunique() == 2 else "multiclass"
    if not pd.api.types.is_numeric_dtype(s):
        if _looks_numeric(s):
            return detect_task(pd.to_numeric(s, errors="coerce"), max_classes)
        nun = s.nunique()
        if nun == 2:
            return "binary"
        return "multiclass" if nun <= max_classes else "ambiguous"
    nun = s.nunique()
    if nun <= 2:
        return "binary"
    # A small set of whole numbers reads as labels (0/1/2 ratings, counts of
    # classes).  This is only a suggestion — the GUI always lets the user
    # override it, because a genuine measurement can look the same way.
    if nun <= 5 and _integer_coded(s):
        return "multiclass"
    return "regression"


def _integer_coded(s: pd.Series) -> bool:
    """True when an integer column is probably a class label, not a measurement.

    A column of 0/1/2 with only a handful of distinct values and no spread
    between them reads as labels; age or strength does not.
    """
    uniq = np.sort(s.dropna().unique())
    if len(uniq) > 10:
        return False
    return bool(np.all(np.equal(np.mod(uniq, 1), 0)))


def suggested_target(df: pd.DataFrame) -> str | None:
    """Guess the target: prefer a name containing 'target'/'label'/'y'/'output'/'price'/'strength'."""
    import re

    keywords = ("target", "label", "output", "result", "price", "strength", "class", "y", "sales", "score")
    lower = {c: str(c).lower() for c in df.columns}
    for kw in keywords:
        for col, low in lower.items():
            if re.search(rf"(^|[_\W]){kw}($|[_\W])", low) or low == kw:
                return col
    # else the last column
    return df.columns[-1] if len(df.columns) else None


def class_balance(y: pd.Series) -> pd.DataFrame:
    vc = y.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "class": vc.index.astype(str),
            "count": vc.values,
            "share_%": (100 * vc.values / max(len(y), 1)).round(2),
        }
    )


# --------------------------------------------------------------------------
# leakage / sanity checks
# --------------------------------------------------------------------------


def suspicious_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Flag columns that will leak the answer or carry no information."""
    flags = []
    kinds = column_kinds(df)
    for col in df.columns:
        if col == target:
            continue
        if df[col].nunique(dropna=True) <= 1:
            flags.append(f"{col}: constant (no information)")
        elif col == "Unnamed: 0" or str(col).lower() in ("index", "id", "unnamed: 0"):
            flags.append(f"{col}: looks like a row index")
        elif kinds[col] == "text":
            flags.append(f"{col}: high-cardinality text ({df[col].nunique()} unique) — needs encoding or dropping")
    if target and target in df.columns:
        if target in numeric_columns(df) and df[target].nunique() > 1:
            corr = df[numeric_columns(df)].apply(pd.to_numeric, errors="coerce").corr()[target].drop(target, errors="ignore")
            for col, r in corr.items():
                if pd.notna(r) and abs(r) > 0.98:
                    flags.append(f"{col}: |r|={abs(r):.3f} with the target — possible leakage")
    return flags
