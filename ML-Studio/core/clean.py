"""Structural cleaning: rows, columns, dtypes, duplicates, filters, recoding.

Everything in this module is *stateless* — it transforms a frame without
learning anything from it.  That is deliberate: imputation, outlier treatment,
encoding and scaling all learn parameters from data, so they live in
:mod:`core.preprocess` inside the scikit-learn pipeline instead.  Mixing the
two would leak test statistics into training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------
# operator catalogue (drives both the engine and the GUI form builder)
# --------------------------------------------------------------------------

FILTER_TESTS = {
    "eq": "equals",
    "ne": "does not equal",
    "gt": "greater than",
    "ge": "greater than or equal",
    "lt": "less than",
    "le": "less than or equal",
    "between": "between (inclusive)",
    "in": "is one of",
    "not_in": "is none of",
    "contains": "text contains",
    "starts_with": "text starts with",
    "ends_with": "text ends with",
    "regex": "matches regex",
    "is_null": "is missing",
    "not_null": "is not missing",
}

CAST_CHOICES = ["numeric", "float", "integer", "string", "category", "boolean", "datetime"]


@dataclass
class OpSpec:
    """Declarative description of one operation; the GUI builds its widgets from this."""

    op: str
    label: str
    params: list = field(default_factory=list)
    doc: str = ""
    order: int = 0


P = lambda name, label, kind="text", default=None, choices=None, optional=False, help="": {  # noqa: E731
    "name": name,
    "label": label,
    "kind": kind,               # text | number | bool | choice | columns | column
    "default": default,
    "choices": choices or [],
    "optional": optional,
    "help": help,
}

OP_SPECS: list[OpSpec] = [
    OpSpec("filter_rows", "Filter rows", [P("conditions", "conditions", "conditions")],
           "Keep only rows matching one or more conditions, combined with AND/OR."),
    OpSpec("drop_columns", "Drop columns", [P("columns", "columns", "columns")],
           "Remove the listed columns."),
    OpSpec("keep_columns", "Keep only these columns", [P("columns", "columns", "columns")],
           "Discard everything not listed — handy for removing leakage."),
    OpSpec("rename_column", "Rename column", [P("column", "column", "column"), P("new_name", "new name")],
           "Rename a single column."),
    OpSpec("cast_dtype", "Change type", [P("column", "column", "column"),
                                         P("dtype", "new type", "choice", "numeric", CAST_CHOICES)],
           "Coerce a column to another type; unparseable values become missing."),
    OpSpec("drop_duplicates", "Drop duplicate rows", [P("subset", "compare only these columns", "columns", None, None, True),
                                                      P("keep", "keep", "choice", "first", ["first", "last", "none"])],
           "Remove repeated rows, optionally comparing only some columns."),
    OpSpec("drop_missing_rows", "Drop rows with missing values",
           [P("columns", "consider only these columns", "columns", None, None, True),
            P("how", "drop when", "choice", "any", ["any", "all"]),
            P("min_non_null", "or: require at least N non-null values", "number", None, None, True)],
           "Row-wise deletion of incomplete records."),
    OpSpec("replace_value", "Replace values", [P("column", "column", "column"),
                                               P("old", "replace this"),
                                               P("new", "with this")],
           "Exact-value substitution, applied to the whole column including missing markers."),
    OpSpec("map_values", "Bulk replace", [P("column", "column", "column"), P("mapping", "old:new pairs", "mapping")],
           "Several substitutions at once, e.g. 'N/A:unknown; -:0'."),
    OpSpec("clip", "Clamp / winsorise", [P("column", "column", "column"),
                                         P("low", "lower bound", "number", None, None, True),
                                         P("high", "upper bound", "number", None, None, True)],
           "Force values into a fixed range."),
    OpSpec("sort", "Sort by column(s)", [P("by", "columns", "columns"),
                                         P("ascending", "ascending", "bool", True)],
           "Reorder rows.  Affects rolling features and the train/test split order."),
    OpSpec("derive_column", "Derive a new column", [P("name", "new column name"),
                                                    P("formula", "formula")],
           "Build a column from the others with the same arithmetic engine the features "
           "screen uses — Water / Cement, lag(settlement, 1), rolling_mean(rain, 7)."),
    OpSpec("reset_index", "Reset the index", [], "Drop the old index and renumber from zero."),
]


def op_spec(op: str) -> OpSpec | None:
    return next((s for s in OP_SPECS if s.op == op), None)


# --------------------------------------------------------------------------
# filter engine
# --------------------------------------------------------------------------


def _series_for(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"no column named {column!r}")
    return df[column]


def _as_number(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_test(df: pd.DataFrame, cond: dict) -> pd.Series:
    """Evaluate one condition dict -> boolean Series aligned to df."""
    col = cond.get("column")
    test = cond.get("test", "eq")
    value = cond.get("value")
    value2 = cond.get("value2")

    if test in ("is_null", "not_null"):
        if col in (None, "", "*"):
            mask = df.isna().any(axis=1) if test == "is_null" else df.notna().all(axis=1)
        else:
            mask = _series_for(df, col).isna() if test == "is_null" else _series_for(df, col).notna()
        return mask.fillna(False)

    s = _series_for(df, col)

    if test in ("gt", "ge", "lt", "le", "between"):
        num = pd.to_numeric(s, errors="coerce")
        if num.isna().all():
            # fall back to string comparison so dates and versions still work
            num = s
        a, b = _as_number(value), _as_number(value2)
        if test == "between":
            if a is None or b is None:
                raise ValueError("'between' needs both bounds")
            lo, hi = min(a, b), max(a, b)
            return ((num >= lo) & (num <= hi)).fillna(False)
        if a is None:
            raise ValueError(f"'{test}' needs a value")
        mask = {"gt": num > a, "ge": num >= a, "lt": num < a, "le": num <= a}[test]
        return pd.Series(mask, index=df.index).fillna(False)

    text = s.astype(str)
    if test == "eq":
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce") == _as_number(value)).fillna(False)
        return text.str.strip().str.lower() == str(value).strip().lower()
    if test == "ne":
        if pd.api.types.is_numeric_dtype(s):
            return (pd.to_numeric(s, errors="coerce") != _as_number(value)).fillna(True)
        return text.str.strip().str.lower() != str(value).strip().lower()
    if test in ("in", "not_in"):
        items = value
        if isinstance(items, str):
            items = [v.strip() for v in re.split(r"[,;|]", items) if v.strip()]
        items = [str(v).strip() for v in (items or [])]
        hit = text.str.strip().str.lower().isin([v.lower() for v in items])
        return ~hit if test == "not_in" else hit
    if test == "contains":
        return text.str.contains(re.escape(str(value)), case=False, na=False, regex=True)
    if test == "starts_with":
        return text.str.lower().str.startswith(str(value).lower(), na=False)
    if test == "ends_with":
        return text.str.lower().str.endswith(str(value).lower(), na=False)
    if test == "regex":
        try:
            return text.str.contains(str(value), case=False, na=False, regex=True)
        except re.error as exc:
            raise ValueError(f"bad regex {value!r}: {exc}") from exc
    raise ValueError(f"unknown test {test!r}")


def apply_filters(df: pd.DataFrame, conditions: list[dict], logic: str = "and", negate: bool = False) -> pd.Series:
    mask = None
    for cond in conditions:
        if not cond.get("column") and cond.get("test") not in ("is_null", "not_null"):
            continue
        m = apply_test(df, cond)
        if mask is None:
            mask = m
        elif logic == "or":
            mask = mask | m
        else:
            mask = mask & m
    if mask is None:
        mask = pd.Series(True, index=df.index)
    return ~mask if negate else mask


# --------------------------------------------------------------------------
# dtype coercion
# --------------------------------------------------------------------------


def coerce_column(s: pd.Series, dtype: str) -> pd.Series:
    dtype = (dtype or "").lower()
    if dtype in ("numeric", "float", "float64", "number"):
        if pd.api.types.is_bool_dtype(s):
            return s.astype("float64")
        cleaned = (
            s.astype(str)
            .str.strip()
            .str.replace(r"^[\$€£৳]\s*", "", regex=True)
            .str.replace(",", "", regex=False)
            .replace({"": None, "nan": None, "NaN": None, "None": None, "-": None, "N/A": None, "na": None})
        )
        return pd.to_numeric(cleaned, errors="coerce")
    if dtype in ("integer", "int", "int64"):
        return pd.to_numeric(s, errors="coerce").round().astype("Int64")
    if dtype in ("string", "str", "object", "text"):
        return s.astype("string")
    if dtype in ("category", "categorical"):
        return s.astype("category")
    if dtype in ("boolean", "bool"):
        if pd.api.types.is_numeric_dtype(s):
            out = pd.Series(pd.NA, index=s.index, dtype="boolean")
            out[s == 1] = True
            out[s == 0] = False
            return out
        truthy = {"true", "yes", "y", "1", "t", "on"}
        falsy = {"false", "no", "n", "0", "f", "off"}
        low = s.astype(str).str.strip().str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out[low.isin(truthy)] = True
        out[low.isin(falsy)] = False
        return out
    if dtype in ("datetime", "date", "datetime64"):
        return pd.to_datetime(s, errors="coerce")
    raise ValueError(f"unsupported dtype {dtype!r}")


def _parse_mapping(mapping) -> dict:
    """'N/A:unknown; -:0' -> {'N/A': 'unknown', '-': '0'} (strings, see caller)."""
    if isinstance(mapping, dict):
        return mapping
    out = {}
    if not mapping:
        return out
    for chunk in re.split(r"[;\n]", str(mapping)):
        if not chunk.strip():
            continue
        if ":" not in chunk:
            raise ValueError(f"mapping entry {chunk!r} is missing a colon")
        old, new = chunk.split(":", 1)
        out[old.strip()] = new.strip()
    return out


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def apply_ops(df: pd.DataFrame, ops: list[dict], keep_log: bool = True):
    """Run a list of operations in order.  Returns (frame, log_lines)."""
    out = df.copy()
    log: list[str] = []
    for i, raw in enumerate(ops or []):
        op = raw.get("op")
        spec = op_spec(op)
        if spec is None:
            log.append(f"{i + 1}. SKIPPED unknown operation {op!r}")
            continue
        before = out.shape
        try:
            out = _apply_one(out, raw)
        except Exception as exc:
            raise RuntimeError(f"operation {i + 1} ({spec.label}) failed: {exc}") from exc
        desc = describe_op(raw)
        if keep_log:
            log.append(f"{i + 1}. {desc}  ({before[0]}x{before[1]} -> {out.shape[0]}x{out.shape[1]})")
    return out, log


def _apply_one(out: pd.DataFrame, raw: dict) -> pd.DataFrame:
    op = raw["op"]

    if op == "filter_rows":
        conditions = raw.get("conditions") or []
        if isinstance(conditions, dict):
            conditions = [conditions]
        conditions = [c for c in conditions if c]
        if not conditions:
            return out
        mask = apply_filters(out, conditions, raw.get("logic", "and"), bool(raw.get("negate", False)))
        return out.loc[mask].copy()

    if op == "drop_columns":
        cols = [c for c in _as_list(raw.get("columns")) if c in out.columns]
        return out.drop(columns=cols)

    if op == "keep_columns":
        cols = [c for c in _as_list(raw.get("columns")) if c in out.columns]
        if not cols:
            raise ValueError("no valid columns listed")
        return out[cols].copy()

    if op == "rename_column":
        col, new = raw.get("column"), raw.get("new_name")
        if not col or not new:
            raise ValueError("needs both a column and a new name")
        return out.rename(columns={col: new})

    if op == "cast_dtype":
        col = raw.get("column")
        if col not in out.columns:
            raise ValueError(f"no column {col!r}")
        out = out.copy()
        out[col] = coerce_column(out[col], raw.get("dtype"))
        return out

    if op == "drop_duplicates":
        subset = [c for c in _as_list(raw.get("subset")) if c in out.columns] or None
        keep = raw.get("keep", "first")
        return out.drop_duplicates(subset=subset, keep=False if keep == "none" else keep)

    if op == "drop_missing_rows":
        cols = [c for c in _as_list(raw.get("columns")) if c in out.columns] or None
        min_non_null = raw.get("min_non_null")
        if min_non_null not in (None, ""):
            n = int(float(min_non_null))
            return out.loc[out.notna().sum(axis=1) >= n]
        return out.dropna(subset=cols, how=raw.get("how", "any"))

    if op == "replace_value":
        col = raw.get("column")
        if col not in out.columns:
            raise ValueError(f"no column {col!r}")
        old, new = raw.get("old"), raw.get("new")
        out = out.copy()
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            o = _as_number(old, old)
            n = _as_number(new, new)
            out[col] = s.replace(o, n)
        else:
            out[col] = s.replace({str(old): ("" if new is None else str(new))})
        return out

    if op == "map_values":
        col = raw.get("column")
        if col not in out.columns:
            raise ValueError(f"no column {col!r}")
        mapping = _parse_mapping(raw.get("mapping"))
        out = out.copy()
        if pd.api.types.is_numeric_dtype(out[col]):
            mapping = {
                (_as_number(k, k) if _as_number(k) is not None else k): (_as_number(v, v) if _as_number(v) is not None else v)
                for k, v in mapping.items()
            }
        out[col] = out[col].replace(mapping)
        return out

    if op == "clip":
        col = raw.get("column")
        if col not in out.columns:
            raise ValueError(f"no column {col!r}")
        low, high = _as_number(raw.get("low")), _as_number(raw.get("high"))
        out = out.copy()
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=low, upper=high)
        return out

    if op == "sort":
        by = [c for c in _as_list(raw.get("by")) if c in out.columns]
        if not by:
            raise ValueError("no valid columns listed")
        return out.sort_values(by=by, ascending=bool(raw.get("ascending", True))).reset_index(drop=True)

    if op == "derive_column":
        from . import formula as formula_mod

        name = str(raw.get("name") or "").strip()
        expression = str(raw.get("formula") or "").strip()
        if not name:
            raise ValueError("the new column needs a name")
        if not expression:
            raise ValueError(f"column {name!r} needs a formula")
        problem = formula_mod.check_expression(expression, out.columns)
        if problem:
            raise ValueError(problem)
        values = formula_mod.evaluate(expression, out)
        out = out.copy()
        # A column built from a whole-series helper (lag, rolling_mean) is a Series; a
        # plain arithmetic expression can come back as a scalar, so both are handled.
        out[name] = values if isinstance(values, pd.Series) else pd.Series(
            np.asarray(values).ravel(), index=out.index)
        return out

    if op == "reset_index":
        return out.reset_index(drop=True)

    raise ValueError(f"unknown operation {op!r}")


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,;]", value) if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def describe_op(raw: dict) -> str:
    op = raw.get("op")
    if op == "filter_rows":
        conds = raw.get("conditions") or []
        joiner = " OR " if raw.get("logic") == "or" else " AND "
        bits = [f"{c.get('column')} {FILTER_TESTS.get(c.get('test'), c.get('test'))} {c.get('value', '')}".strip() for c in conds]
        text = joiner.join(bits)
        return f"filter_rows({'NOT ' if raw.get('negate') else ''}{text})"
    if op in ("drop_columns", "keep_columns", "sort", "drop_duplicates", "drop_missing_rows"):
        bits = ", ".join(f"{k}={v}" for k, v in raw.items() if k != "op" and v not in (None, "", []))
        return f"{op}({bits})"
    bits = ", ".join(f"{k}={v}" for k, v in raw.items() if k != "op" and v not in (None, "", []))
    return f"{op}({bits})"


# --------------------------------------------------------------------------
# convenience used by the GUI summary panel
# --------------------------------------------------------------------------


def changed_columns(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return {
        "dropped": [c for c in before.columns if c not in after.columns],
        "added": [c for c in after.columns if c not in before.columns],
        "rows": len(after) - len(before),
    }


def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non-null": [int(df[c].notna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
