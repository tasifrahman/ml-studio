"""A small, allowlisted arithmetic evaluator for column formulas and constraints.

Used for engineered features (``Water / Cement``) and for design constraints
(``Cement + Water <= 780``).  Expressions are checked against a character and
name allowlist before being evaluated with builtins removed.  This is a guard
against typos and accidental function calls in a config file — it is not a
security boundary, and nothing here should ever evaluate text from a stranger.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SAFE_FUNCS = {
    "log": np.log, "log1p": np.log1p, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
    "square": np.square, "sign": np.sign, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "floor": np.floor, "ceil": np.ceil, "round": np.round, "clip": np.clip,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
    "pi": np.pi, "e": np.e,
}


# --------------------------------------------------------------------------
# time-series and shape helpers
#
# These are what let the Features screen build a lag or a moving average without a script.
# They take a whole column and return a whole column, which is why they are kept apart
# from the element-wise NumPy functions above.
# --------------------------------------------------------------------------

def _as_series(value) -> pd.Series:
    return value if isinstance(value, pd.Series) else pd.Series([value])


def lag(series, steps: int = 1) -> pd.Series:
    """The value this many rows earlier — a lagged feature (rows must already be in order)."""
    return _as_series(series).shift(int(steps))


def lead(series, steps: int = 1) -> pd.Series:
    """The value this many rows later.  Useful for building a target, unsafe as a feature."""
    return _as_series(series).shift(-int(steps))


def difference(series, steps: int = 1) -> pd.Series:
    """The change from one row to the one ``steps`` before it."""
    return _as_series(series).diff(int(steps))


def pct_change(series, steps: int = 1) -> pd.Series:
    """The change as a percentage of the earlier value."""
    return _as_series(series).pct_change(int(steps)) * 100


def rolling_mean(series, window: int) -> pd.Series:
    """Moving average over the last ``window`` rows, including short windows at the start."""
    return _as_series(series).rolling(int(window), min_periods=1).mean()


def rolling_std(series, window: int) -> pd.Series:
    return _as_series(series).rolling(int(window), min_periods=2).std()


def rolling_sum(series, window: int) -> pd.Series:
    return _as_series(series).rolling(int(window), min_periods=1).sum()


def rolling_min(series, window: int) -> pd.Series:
    return _as_series(series).rolling(int(window), min_periods=1).min()


def rolling_max(series, window: int) -> pd.Series:
    return _as_series(series).rolling(int(window), min_periods=1).max()


def cumulative_sum(series) -> pd.Series:
    return _as_series(series).cumsum()


def rank(series) -> pd.Series:
    """Percentile rank from 0 to 100 — robust to outliers, unlike the raw value."""
    return _as_series(series).rank(pct=True) * 100


def zscore(series) -> pd.Series:
    """Standardised to mean 0 and standard deviation 1."""
    values = _as_series(series)
    spread = values.std(ddof=1)
    return (values - values.mean()) / spread if spread else values * 0


def normalise(series) -> pd.Series:
    """Scaled to the 0–1 range."""
    values = _as_series(series)
    low, high = values.min(), values.max()
    return (values - low) / (high - low) if high > low else values * 0


SAFE_FUNCS.update({
    "lag": lag, "lead": lead, "diff": difference, "pct_change": pct_change,
    "rolling_mean": rolling_mean, "rolling_std": rolling_std, "rolling_sum": rolling_sum,
    "rolling_min": rolling_min, "rolling_max": rolling_max,
    "cumulative_sum": cumulative_sum, "rank": rank, "zscore": zscore, "normalise": normalise,
})

ALLOWED_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "_ .+-*/()**,'\"")
ALLOWED_WORDS = {"and", "or", "not", "True", "False"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
#: A column name written as a quoted string: "Blast furnace slag", 'Age (days)'.
_QUOTED = re.compile(r'"([^"]*)"|\'([^\']*)\'')


def quote(name) -> str:
    """How to write a column name inside a formula, whatever characters it contains.

    A plain identifier is left bare so ordinary formulas stay readable; anything with a
    space, a bracket or a unit in it is quoted.
    """
    text = str(name)
    if text.isidentifier() and not text.startswith("__"):
        return text
    return '"' + text.replace('"', "") + '"'


def _bind_quoted_names(expr: str, columns):
    """Rewrite quoted column names as safe placeholders.

    The evaluator is an ``eval`` over a scope keyed by column name, and a column is not
    allowed to *be* a Python identifier — "Blast furnace slag" and "Age (days)" are
    ordinary headings in a real test report.  Rather than teach the tokeniser about
    headings, each quoted name is swapped for a placeholder that is a valid identifier and
    bound to that column in the scope.  The placeholder cannot collide with a real column,
    because the counter is advanced until it does not.

    Returns ``(rewritten expression, {placeholder: column name})``.
    """
    taken = {str(c) for c in columns}
    bindings: dict[str, str] = {}
    counter = 0

    def replace(match: "re.Match") -> str:
        nonlocal counter
        name = match.group(1) if match.group(1) is not None else match.group(2)
        counter += 1
        placeholder = f"__col{counter}__"
        while placeholder in taken:
            counter += 1
            placeholder = f"__col{counter}__"
        bindings[placeholder] = name
        return placeholder

    return _QUOTED.sub(replace, expr), bindings


def check_expression(expr: str, columns) -> str | None:
    """Return a human-readable problem with the expression, or None if it looks usable."""
    if not expr or not expr.strip():
        return "the formula is empty"
    illegal = sorted(set(expr) - ALLOWED_CHARS)
    if illegal:
        return f"illegal character(s): {', '.join(repr(c) for c in illegal)}"
    known = set(map(str, columns))
    rewritten, quoted = _bind_quoted_names(expr, columns)
    missing = sorted({name for name in quoted.values() if name not in known})
    if missing:
        return f"no such column(s): {', '.join(repr(n) for n in missing)}"
    words = set(_WORD.findall(rewritten))
    unknown = words - known - set(SAFE_FUNCS) - ALLOWED_WORDS - set(quoted)
    if unknown:
        return f"unknown name(s): {', '.join(sorted(unknown))}"
    # Every name is allowed, but the expression can still be malformed — a stray bracket
    # used to surface later as a raw SyntaxError from inside eval(), which told the user
    # nothing.  Compile it here so the message names the problem.
    try:
        compile(rewritten, "<formula>", "eval")
    except SyntaxError as exc:
        return f"the formula does not parse: {exc.msg}"
    return None


def evaluate(expr: str, frame: pd.DataFrame, extra: dict | None = None):
    """Evaluate an allowlisted expression against the columns of ``frame``."""
    rewritten, quoted = _bind_quoted_names(expr, frame.columns)
    scope = {str(col): frame[col] for col in frame.columns}
    scope.update(SAFE_FUNCS)
    for placeholder, name in quoted.items():
        if name not in frame.columns:
            raise ValueError(f"no such column: {name!r}")
        scope[placeholder] = frame[name]
    if extra:
        scope.update(extra)
    return eval(rewritten, {"__builtins__": {}}, scope)  # noqa: S307 — guarded by check_expression


def as_series(value, index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index, dtype="float64")


CONSTRAINT_PATTERN = re.compile(r"^\s*(.+?)\s*(<=|>=|<|>|==)\s*(-?[\d.eE+]+)\s*$")


def parse_constraint(text: str) -> dict | None:
    """'Cement + Water <= 780' -> {"expr": ..., "op": "<=", "value": 780.0}."""
    match = CONSTRAINT_PATTERN.match(text or "")
    if not match:
        return None
    expr, op, value = match.group(1), match.group(2), match.group(3)
    try:
        return {"expr": expr.strip(), "op": op, "value": float(value), "text": text.strip()}
    except ValueError:
        return None


def constraint_mask(frame: pd.DataFrame, constraints: list[str]) -> np.ndarray:
    """Boolean array: True where every constraint is satisfied.

    An unparseable constraint is refused loudly rather than silently ignored —
    a design that quietly violates a physical limit is worse than no answer.
    """
    mask = np.ones(len(frame), dtype=bool)
    for text in constraints or []:
        if not text or not str(text).strip():
            continue
        parsed = parse_constraint(str(text))
        if parsed is None:
            raise ValueError(f"cannot read the constraint {text!r} — write it like 'Cement + Water <= 780'")
        problem = check_expression(parsed["expr"], frame.columns)
        if problem:
            raise ValueError(f"constraint {text!r}: {problem}")
        left = np.asarray(evaluate(parsed["expr"], frame), dtype=float)
        target = parsed["value"]
        compare = {
            "<=": left <= target, ">=": left >= target, "<": left < target,
            ">": left > target, "==": np.isclose(left, target),
        }[parsed["op"]]
        mask &= np.nan_to_num(compare).astype(bool)
    return mask
