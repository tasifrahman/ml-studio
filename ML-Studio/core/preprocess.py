"""The learned half of cleaning: imputation, encoding, scaling, feature engineering.

This is the part that makes a saved model reusable.  Everything that computes a
statistic from data (a median, a category list, a standard deviation) is fitted
once on the training split and stored inside the pipeline, so new data goes
through exactly the same arithmetic as the data the model was measured on.

Implemented directly in pandas rather than with ``ColumnTransformer`` so that
column names survive the trip — every plot and importance chart in the GUI
depends on knowing which feature is which.  ``handle_unknown`` semantics are
copied from scikit-learn: unseen categories become all-zeros, because a model
must never see a category it was not trained on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import eda
from .formula import check_expression, evaluate, as_series

RARE = "__rare__"
MISSING = "__missing__"

NUMERIC_IMPUTE = ["median", "mean", "zero", "constant", "ffill", "knn", "iterative", "none"]
CATEGORICAL_IMPUTE = ["mode", "constant", "none"]
ENCODERS = ["onehot", "ordinal", "label", "target", "none"]
SCALERS = ["none", "standard", "minmax", "robust", "maxabs"]
OUTLIER_MODES = ["none", "clip_iqr", "clip_zscore", "winsorize"]


@dataclass
class PreprocessSpec:
    """User-facing description of how to prepare features."""

    numeric_impute: str = "median"
    numeric_constant: float = 0.0
    categorical_impute: str = "mode"
    categorical_constant: str = "missing"
    encoder: str = "onehot"
    max_categories: int = 30
    scale: str = "standard"
    outlier: str = "none"
    outlier_columns: list = field(default_factory=list)
    drop_constant: bool = True
    cast_numeric_like: bool = True
    engineer: list = field(default_factory=list)   # [{"name": "wcr", "formula": "Water/Cement"}]
    expand_datetime: bool = False
    target_encode_smoothing: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PreprocessSpec":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class FormulaFeature(TransformerMixin, BaseEstimator):
    """Adds columns from user-supplied arithmetic, e.g. ``Water / Cement``.

    Expressions are checked against an allowlist before evaluation — see
    :mod:`core.formula`.  A formula that does not check out is recorded in
    ``problems_`` and skipped, so one typo cannot stop a whole run.
    """

    def __init__(self, formulas=None):
        self.formulas = formulas or []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.valid_ = []
        problems = []
        for item in self.formulas:
            name = (item.get("name") or "").strip()
            expr = (item.get("formula") or "").strip()
            if not name or not expr:
                continue
            problem = check_expression(expr, X.columns)
            if problem:
                problems.append(f"{name}: {problem}")
                continue
            try:
                evaluate(expr, X, {})
            except Exception as exc:
                problems.append(f"{name}: {exc}")
                continue
            self.valid_.append({"name": name, "formula": expr})
        self.problems_ = problems
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        added = {}
        for item in getattr(self, "valid_", []):
            try:
                value = evaluate(item["formula"], X, added)
            except Exception:
                continue
            if isinstance(value, pd.Series):
                added[item["name"]] = pd.to_numeric(value, errors="coerce")
            else:
                added[item["name"]] = as_series(value, X.index)
        for name, series in added.items():
            X[name] = series
        self.feature_names_out_ = list(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "feature_names_out_", list(self.feature_names_in_)), dtype=object)


class WorkbenchPreprocessor(TransformerMixin, BaseEstimator):
    """Fit-on-train transformer producing a fully numeric DataFrame."""

    def __init__(self, spec: PreprocessSpec | dict | None = None, task: str = "regression"):
        self.spec = spec
        self.task = task

    # -- plumbing ---------------------------------------------------------
    def _spec(self) -> PreprocessSpec:
        if isinstance(self.spec, PreprocessSpec):
            return self.spec
        return PreprocessSpec.from_dict(self.spec)

    def get_params(self, deep=True):
        return {"spec": self.spec, "task": self.task}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # -- fit --------------------------------------------------------------
    def fit(self, X, y=None):
        spec = self._spec()
        X = pd.DataFrame(X).copy()

        self.formulas_ = FormulaFeature(spec.engineer).fit(X)
        self.feature_problems_ = list(self.formulas_.problems_)
        X = self.formulas_.transform(X)

        kinds = eda.column_kinds(X)
        self.numeric_cols_ = [c for c in X.columns if kinds[c] == "numeric"]
        self.categorical_cols_ = [c for c in X.columns if kinds[c] in ("categorical", "boolean")]
        self.datetime_cols_ = [c for c in X.columns if kinds[c] == "datetime"]

        # columns the user asked to treat as numeric but that arrived as text
        if spec.cast_numeric_like:
            for col in self.categorical_cols_:
                if col in self.numeric_cols_ or not eda._looks_numeric(X[col]):
                    continue
                X[col] = pd.to_numeric(X[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
                self.numeric_cols_.append(col)
                self.categorical_cols_.remove(col)

        if spec.drop_constant:
            keep = []
            for col in X.columns:
                if X[col].nunique(dropna=True) <= 1 and col in self.numeric_cols_ + self.categorical_cols_:
                    self.feature_problems_.append(f"{col}: constant in the training split, dropped")
                    continue
                keep.append(col)
            self.numeric_cols_ = [c for c in self.numeric_cols_ if c in keep]
            self.categorical_cols_ = [c for c in self.categorical_cols_ if c in keep]
            self.datetime_cols_ = [c for c in self.datetime_cols_ if c in keep]

        self.input_columns_ = list(X.columns)
        self._fit_numeric(X, y, spec)
        self._fit_categorical(X, y, spec)
        self._fit_datetime(X, spec)
        self._fit_outliers(X, spec)
        self._fit_scaler(X, spec)
        self.n_features_in_ = len(self.input_columns_)
        self.feature_names_out_ = list(self._column_names_after_encode())
        return self

    # -- numeric block ----------------------------------------------------
    def _fit_numeric(self, X, y, spec):
        num = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce") if self.numeric_cols_ else pd.DataFrame(index=X.index)
        self.numeric_fill_: dict = {}
        mode = spec.numeric_impute
        if mode == "constant":
            self.numeric_fill_ = {c: float(spec.numeric_constant) for c in self.numeric_cols_}
        elif mode == "mean":
            self.numeric_fill_ = {c: float(num[c].mean()) if num[c].notna().any() else 0.0 for c in self.numeric_cols_}
        elif mode == "median":
            self.numeric_fill_ = {c: float(num[c].median()) if num[c].notna().any() else 0.0 for c in self.numeric_cols_}
        elif mode == "zero":
            self.numeric_fill_ = {c: 0.0 for c in self.numeric_cols_}
        # 'ffill' needs no fitted state; 'knn'/'iterative'/'none' handled below

        self.knn_imputer_ = None
        if mode in ("knn", "iterative") and self.numeric_cols_:
            # A column that is *entirely* missing leaves both imputers with
            # nothing to work from, so it is pinned to a constant first.
            degenerate = {c: 0.0 for c in self.numeric_cols_ if not num[c].notna().any()}
            if degenerate:
                self.numeric_fill_.update(degenerate)
                num = num.fillna(degenerate)
            if mode == "knn":
                from sklearn.impute import KNNImputer

                self.knn_imputer_ = KNNImputer(n_neighbors=5).fit(num)
            else:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                self.knn_imputer_ = IterativeImputer(random_state=0, max_iter=10).fit(num)

    # -- categorical block ------------------------------------------------
    def _fit_categorical(self, X, y, spec):
        self.categories_: dict[str, list] = {}
        self.rare_categories_: dict[str, list] = {}
        self.cat_fill_: dict = {}
        for col in self.categorical_cols_:
            s = X[col].astype("object")
            if spec.categorical_impute == "mode":
                mode = s.mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else MISSING
            elif spec.categorical_impute == "constant":
                fill = spec.categorical_constant
            else:
                fill = None
            self.cat_fill_[col] = fill

            values = s.dropna().astype(str).value_counts()
            if spec.max_categories and spec.max_categories > 0 and len(values) > spec.max_categories:
                top = list(values.head(spec.max_categories).index)
                rare = [v for v in values.index if v not in top]
            else:
                top, rare = list(values.index), []
            self.categories_[col] = top
            self.rare_categories_[col] = rare

        # target encoding
        self.target_maps_: dict[str, dict] = {}
        self.target_prior_ = None
        if spec.encoder == "target" and self.categorical_cols_:
            if y is None:
                raise ValueError("target encoding needs a target column")
            y = pd.Series(y).reset_index(drop=True)
            self.target_prior_ = float(pd.to_numeric(y, errors="coerce").mean())
            frame = X[self.categorical_cols_].reset_index(drop=True).astype(str)
            for col in self.categorical_cols_:
                grouped = pd.to_numeric(y, errors="coerce").groupby(frame[col])
                stats = grouped.agg(["mean", "count"])
                k = float(spec.target_encode_smoothing)
                smoothed = (stats["mean"] * stats["count"] + self.target_prior_ * k) / (stats["count"] + k)
                self.target_maps_[col] = smoothed.to_dict()

    # -- datetime ---------------------------------------------------------
    def _fit_datetime(self, X, spec):
        self.datetime_expanded_ = bool(spec.expand_datetime) and bool(self.datetime_cols_)
        if self.datetime_expanded_:
            self.datetime_parts_ = ["year", "month", "day", "weekday", "dayofyear", "hour", "minute"]

    # -- outliers ---------------------------------------------------------
    def _fit_outliers(self, X, spec):
        self.outlier_bounds_: dict = {}
        if spec.outlier == "none" or not self.numeric_cols_:
            return
        cols = [c for c in (spec.outlier_columns or self.numeric_cols_) if c in self.numeric_cols_]
        for col in cols:
            s = pd.to_numeric(X[col], errors="coerce")
            if spec.outlier == "clip_iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                self.outlier_bounds_[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
            elif spec.outlier == "clip_zscore":
                mu, sd = s.mean(), s.std()
                if sd and not np.isnan(sd):
                    self.outlier_bounds_[col] = (float(mu - 3 * sd), float(mu + 3 * sd))
            elif spec.outlier == "winsorize":
                self.outlier_bounds_[col] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

    # -- scaler -----------------------------------------------------------
    def _fit_scaler(self, X, spec):
        self.scaler_stats_: dict = {}
        if spec.scale == "none":
            return
        frame = self._numeric_frame(X)
        for col in frame.columns:
            s = frame[col]
            if spec.scale == "standard":
                self.scaler_stats_[col] = ("standard", float(s.mean()), float(s.std()) if s.std() else 1.0)
            elif spec.scale == "minmax":
                lo, hi = float(s.min()), float(s.max())
                self.scaler_stats_[col] = ("minmax", lo, (hi - lo) or 1.0)
            elif spec.scale == "robust":
                med = float(s.median())
                iqr = float(s.quantile(0.75) - s.quantile(0.25))
                self.scaler_stats_[col] = ("robust", med, iqr or 1.0)
            elif spec.scale == "maxabs":
                m = float(np.abs(s).max())
                self.scaler_stats_[col] = ("maxabs", 0.0, m or 1.0)

    def _numeric_frame(self, X) -> pd.DataFrame:
        """Numeric block after imputation + outliers, before encoding/scaling."""
        spec = self._spec()
        cols = self.numeric_cols_
        if not cols:
            return pd.DataFrame(index=X.index)
        num = X[cols].apply(pd.to_numeric, errors="coerce").copy()
        if spec.numeric_impute == "ffill":
            num = num.ffill().bfill()
        elif self.knn_imputer_ is not None:
            if self.numeric_fill_:
                num = num.fillna(self.numeric_fill_)
            num = pd.DataFrame(self.knn_imputer_.transform(num), columns=cols, index=X.index)
        elif self.numeric_fill_:
            num = num.fillna(self.numeric_fill_)
        for col, (lo, hi) in getattr(self, "outlier_bounds_", {}).items():
            if col in num.columns:
                num[col] = num[col].clip(lo, hi)
        return num

    def _column_names_after_encode(self) -> list[str]:
        spec = self._spec()
        names = list(self.numeric_cols_)
        if getattr(self, "datetime_expanded_", False):
            for col in self.datetime_cols_:
                names.extend(f"{col}_{part}" for part in self.datetime_parts_)
        for col in self.categorical_cols_:
            if spec.encoder == "onehot":
                for cat in self.categories_.get(col, []):
                    names.append(f"{col}={cat}")
                if self.rare_categories_.get(col):
                    names.append(f"{col}={RARE}")
            else:
                names.append(col)
        return names

    # -- transform --------------------------------------------------------
    def transform(self, X) -> pd.DataFrame:
        spec = self._spec()
        X = pd.DataFrame(X).copy()
        X = self.formulas_.transform(X)

        missing = [c for c in self.input_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"these columns the model was trained on are absent: {missing}")

        blocks = [self._numeric_frame(X)]

        if getattr(self, "datetime_expanded_", False):
            dt_block = {}
            for col in self.datetime_cols_:
                s = pd.to_datetime(X[col], errors="coerce")
                dt_block[f"{col}_year"] = s.dt.year.astype("float64")
                dt_block[f"{col}_month"] = s.dt.month.astype("float64")
                dt_block[f"{col}_day"] = s.dt.day.astype("float64")
                dt_block[f"{col}_weekday"] = s.dt.weekday.astype("float64")
                dt_block[f"{col}_dayofyear"] = s.dt.dayofyear.astype("float64")
                dt_block[f"{col}_hour"] = s.dt.hour.astype("float64")
                dt_block[f"{col}_minute"] = s.dt.minute.astype("float64")
            blocks.append(pd.DataFrame(dt_block, index=X.index))

        cat_block = self._encode_categorical(X, spec)
        if not cat_block.empty:
            blocks.append(cat_block)

        out = pd.concat([b for b in blocks if not b.empty], axis=1) if any(not b.empty for b in blocks) else pd.DataFrame(index=X.index)

        # scaler
        for col, (kind, centre, spread) in getattr(self, "scaler_stats_", {}).items():
            if col in out.columns and spread:
                out[col] = (out[col] - centre) / spread
        out = out.reindex(columns=self.feature_names_out_, fill_value=0.0)
        return out.astype("float64").fillna(0.0)

    def _encode_categorical(self, X, spec) -> pd.DataFrame:
        if not self.categorical_cols_:
            return pd.DataFrame(index=X.index)
        data = {}
        for col in self.categorical_cols_:
            raw = X[col]
            fill = getattr(self, "cat_fill_", {}).get(col)
            as_str = raw.astype("object")
            if fill is not None:
                as_str = as_str.where(as_str.notna(), other=fill)
            as_str = as_str.astype(str)

            if spec.encoder == "target":
                mapping = self.target_maps_.get(col, {})
                values = as_str.map(mapping)
                data[col] = pd.to_numeric(values, errors="coerce").fillna(self.target_prior_ or 0.0)
                continue

            known = set(self.categories_.get(col, []))
            rare = set(self.rare_categories_.get(col, []))
            if spec.encoder == "ordinal":
                order = {cat: i for i, cat in enumerate(self.categories_.get(col, []))}
                fallback = len(order) + (1 if rare else 0)
                data[col] = np.where(
                    as_str.isin(known), as_str.map(order).astype("float64"), float(fallback if rare else -1)
                )
            elif spec.encoder == "label":
                order = {cat: i for i, cat in enumerate(self.categories_.get(col, []))}
                unknown_base = len(order)
                codes = as_str.map(order)
                extra = as_str[~as_str.isin(known)]
                if len(extra):
                    extras = {v: unknown_base + i for i, v in enumerate(sorted(extra.unique()))}
                    codes = codes.fillna(extra.map(extras))
                data[col] = pd.to_numeric(codes, errors="coerce").fillna(-1.0)
            else:  # onehot
                for cat in self.categories_.get(col, []):
                    data[f"{col}={cat}"] = (as_str == cat).astype("float64")
                if rare:
                    data[f"{col}={RARE}"] = as_str.isin(rare).astype("float64")
        return pd.DataFrame(data, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def build_pipeline(pre_spec: PreprocessSpec | dict, model, task: str = "regression"):
    """Wrap preprocessing + estimator into one pickleable sklearn Pipeline."""
    from sklearn.pipeline import Pipeline

    return Pipeline([("prepare", WorkbenchPreprocessor(pre_spec, task=task)), ("model", model)])


def describe_spec(spec: PreprocessSpec) -> list[str]:
    """Plain-language summary of a preprocessing recipe, for the UI log."""
    lines = [
        f"numeric missing -> {spec.numeric_impute}"
        + (f" ({spec.numeric_constant})" if spec.numeric_impute == "constant" else ""),
        f"categorical missing -> {spec.categorical_impute}",
        f"encoding -> {spec.encoder}" + (f" (max {spec.max_categories} categories)" if spec.encoder == "onehot" else ""),
        f"scaling -> {spec.scale}",
        f"outliers -> {spec.outlier}",
    ]
    if spec.engineer:
        lines.append("engineered: " + ", ".join(f"{i.get('name')} = {i.get('formula')}" for i in spec.engineer))
    return lines
