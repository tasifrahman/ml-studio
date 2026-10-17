"""The analysis screens: the four 'best combination' tools, clustering, plots, and
the one-click run.

The four optimisation tools answer four different questions and are deliberately
kept apart on the same screen:

1. **Best value per field** — for each input, the value that drives the
   prediction highest (a dependence sweep).
2. **Inverse design** — given a target outcome, which input values reach it,
   inside physical bounds and your own constraints.
3. **Best cleaning recipe** — which imputation/outlier/scaling combination scores
   best with the same model and the same folds.
4. **Best feature combination** — which subset of columns scores best.

The one-click screen calls exactly the same jobs as the buttons do, so a run it
produces can always be reproduced by hand afterwards.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import eda, metrics as M, models as MODELS, optimize, plots as plots_mod
from core import select as select_mod
from core import train as train_mod
from core import tuning as tuning_mod
from core.preprocess import PreprocessSpec

from .. import theme
from ..widgets import (
    DataFrameView, FieldForm, KeyValueView, MetricCards, PlotCanvas, StepPage,
    fields_from_model_params, fields_from_plot, hint, scrollable, section_title,
)
from .model_steps import default_plot_requests, describe_result, metric_choices, model_choices


# --------------------------------------------------------------------------
# small table helpers
# --------------------------------------------------------------------------

def fill_table(table: QTableWidget, rows: list, columns: int):
    table.setRowCount(0)
    for row in rows:
        index = table.rowCount()
        table.insertRow(index)
        for i in range(columns):
            value = row[i] if i < len(row) else ""
            table.setItem(index, i, QTableWidgetItem("" if value is None else str(value)))


def read_pairs(table: QTableWidget) -> list[tuple]:
    out = []
    for row in range(table.rowCount()):
        values = []
        for column in range(table.columnCount()):
            item = table.item(row, column)
            values.append(item.text().strip() if item else "")
        if values and values[0]:
            out.append(tuple(values))
    return out


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

def best_value_job(progress, cancel, pipe, X, task, features):
    train_mod.check_cancel(cancel)
    progress(0, len(features) or 1, "sweeping each feature")
    frame = optimize.best_value_per_field(pipe, X, task, features=features)
    progress(len(features) or 1, len(features) or 1, "done")
    return frame


def design_job(progress, cancel, pipe, X, task, spec_dict):
    spec = optimize.DesignSpec.from_dict(spec_dict)
    return optimize.inverse_design(pipe, X, task, spec, progress=progress, cancel=cancel)


def cleaning_combo_job(progress, cancel, df, spec_dict, pre_dict, model_key, model_params, options):
    spec = train_mod.TrainSpec.from_dict(spec_dict)
    pre_spec = PreprocessSpec.from_dict(pre_dict)
    frame = optimize.cleaning_combo_search(
        df, spec, pre_spec, model_key, model_params,
        impute_options=options.get("impute") or ("median",),
        outlier_options=options.get("outlier") or ("none",),
        scale_options=options.get("scale") or ("standard",),
        max_combos=int(options.get("max_combos") or 40),
        progress=progress, cancel=cancel,
    )
    return {"frame": frame, "seconds": frame.attrs.get("seconds"),
            "metric": frame.attrs.get("metric")}


def selection_job(progress, cancel, df, spec_dict, pre_dict, model_key, model_params, sel_dict):
    spec = train_mod.TrainSpec.from_dict(spec_dict)
    pre_spec = PreprocessSpec.from_dict(pre_dict)
    sel = select_mod.SelectSpec.from_dict(sel_dict)
    return select_mod.run_selection(df, spec, pre_spec, model_key, model_params, sel,
                                    progress=progress, cancel=cancel)


def cluster_job(progress, cancel, df, features, pre_dict, model_key, model_params):
    pre_spec = PreprocessSpec.from_dict(pre_dict)
    spec = train_mod.TrainSpec(target=None, features=list(features), task="unsupervised")
    progress(0, 3, "splitting")
    split = train_mod.make_split(df, spec)
    train_mod.check_cancel(cancel)
    label = MODELS.get_model_spec(model_key).label
    progress(1, 3, f"fitting {label}")
    pipe, _ = train_mod.build_train_pipeline(pre_spec, model_key, model_params, "unsupervised")
    pipe.fit(split.X_train)
    train_mod.check_cancel(cancel)
    progress(2, 3, "projecting and scoring")
    prepared = pipe.named_steps["prepare"].transform(split.X_train)
    model = pipe.named_steps["model"]
    labels = getattr(model, "labels_", None)
    # A deep-clustering model assigns its labels in a *latent* space, so the
    # silhouette has to be measured there as well.  Scoring the raw prepared
    # matrix would be judging a different clustering from the one produced.
    embedding = getattr(model, "embedding_", None)
    scored_on = prepared if embedding is None else np.asarray(embedding)
    silhouette = None
    if labels is not None and len(set(np.asarray(labels).tolist())) > 1:
        try:
            from sklearn.metrics import silhouette_score

            silhouette = float(silhouette_score(scored_on, labels))
        except Exception:  # noqa: BLE001 - DBSCAN can produce a single cluster
            silhouette = None
    explained = None
    ratio = getattr(model, "explained_variance_ratio_", None)
    if ratio is not None:
        explained = pd.DataFrame({
            "component": [f"PC{i + 1}" for i in range(len(ratio))],
            "explained variance %": np.asarray(ratio, dtype=float) * 100,
            "cumulative %": np.cumsum(np.asarray(ratio, dtype=float)) * 100,
        })
    sizes = None
    if labels is not None:
        counts = pd.Series(np.asarray(labels)).value_counts().sort_index()
        sizes = pd.DataFrame({"cluster": counts.index, "rows": counts.values})
    return {"split": split, "pipeline": pipe, "prepared": prepared, "labels": labels,
            "silhouette": silhouette, "explained": explained, "sizes": sizes,
            "embedding": embedding, "scored_on": "latent code" if embedding is not None else "prepared features",
            "model_key": model_key, "model_params": dict(model_params)}


def auto_job(progress, cancel, payload):
    """The whole workflow in one pass, with honest progress at every stage."""
    from core import export_script
    from core import report as report_mod

    df = payload["df"]
    spec = train_mod.TrainSpec.from_dict(payload["spec"])
    pre_spec = PreprocessSpec.from_dict(payload["pre"])
    keys = [k for k in payload["model_keys"] if k in MODELS.MODEL_INDEX]
    out_dir = payload["out_dir"]
    total = 10
    notes: list[str] = []

    progress(0, total, "splitting the data")
    split = train_mod.make_split(df, spec)
    train_mod.check_cancel(cancel)

    progress(1, total, f"cross-validating {len(keys)} candidate models")
    leaderboard = tuning_mod.comparison_table(keys, pre_spec, split.X_train, split.y_train, spec,
                                             weights=split.w_train, progress=None, cancel=cancel)
    usable = leaderboard
    if "error" in leaderboard.columns:
        usable = leaderboard[leaderboard["error"].isna()]
    if len(usable):
        best_key = str(usable.iloc[0]["key"])
    elif keys:
        best_key = keys[0]
        notes.append("no candidate cross-validated cleanly — falling back to the first model")
    else:
        raise ValueError("no models were selected to try")

    best_params: dict = {}
    tuning_result = None
    if payload.get("tune"):
        progress(2, total, f"tuning {MODELS.get_model_spec(best_key).label}")
        grid = MODELS.suggested_grid(best_key)
        combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        method = "random" if combos > 30 else "grid"
        tune_spec = tuning_mod.TuneSpec(method=method, n_iter=min(30, max(5, combos)),
                                        cv_folds=spec.cv_folds,
                                        metric=spec.primary_metric, param_grid=grid,
                                        max_candidates=40)
        tuning_result = tuning_mod.tune(best_key, pre_spec, split.X_train, split.y_train, spec,
                                        tune_spec, weights=split.w_train, progress=None,
                                        cancel=cancel)
        best_params = dict(tuning_result.get("best_params") or {})
        if tuning_result.get("note"):
            notes.append(str(tuning_result["note"]))

    progress(3, total, f"fitting {MODELS.get_model_spec(best_key).label}")
    pipe, _ = train_mod.build_train_pipeline(pre_spec, best_key, best_params, spec.task)
    train_mod.fit_full(pipe, split, spec)
    evaluation = train_mod.evaluate_fitted(pipe, split, spec)
    train_mod.check_cancel(cancel)

    progress(4, total, "cross-validating the winner and reading importances")
    cv = train_mod.cross_validate_pipeline(pipe, split.X_train, split.y_train, spec,
                                           weights=split.w_train)
    try:
        importance = explain.importance_frame(pipe, split.X_train, split.y_train, spec.task)
    except Exception as exc:  # noqa: BLE001
        importance = None
        notes.append(f"importances unavailable: {exc}")
    prepared = None
    try:
        prepared = pipe.named_steps["prepare"].transform(split.X_train)
    except Exception:  # noqa: BLE001
        pass

    progress(5, total, "sweeping every feature for its best value")
    try:
        best_values = optimize.best_value_per_field(pipe, split.X_train, spec.task)
    except Exception as exc:  # noqa: BLE001
        best_values = None
        notes.append(f"the per-field sweep failed: {exc}")

    progress(6, total, "packing the model bundle")
    importances = {}
    if importance is not None and len(importance) and "feature" in importance.columns:
        column = "importance" if "importance" in importance.columns else importance.columns[1]
        importances = dict(zip(importance["feature"].astype(str), importance[column].astype(float)))
    bundle = train_mod.make_bundle(pipe, spec, split, pre_spec, best_key, best_params, evaluation,
                                   cv=cv, tuning=tuning_result, source_file=payload.get("source_file"),
                                   cleaning_ops=payload.get("cleaning_ops") or [],
                                   notes=payload.get("notes") or "", feature_importances=importances)

    context = plots_mod.PlotContext(
        df=df, target=spec.target, task=spec.task, model=pipe, X=split.X_train, y=split.y_train,
        evaluation={**evaluation, "cv": cv}, importance=importance, labels=split.labels,
        preprocessor=pre_spec, model_key=best_key, prepared=prepared,
    )
    requests = default_plot_requests(context, spec.task, spec.target)
    extra_sections = {"Model comparison": leaderboard}
    tuning_rows = tuning_result.get("results") if tuning_result else None
    if tuning_rows is not None and len(tuning_rows):
        extra_sections["Hyperparameter search"] = tuning_rows
    if best_values is not None and len(best_values):
        extra_sections["Best value per field"] = best_values

    paths: dict = {}
    os.makedirs(out_dir, exist_ok=True)
    if payload.get("write_bundle", True):
        progress(7, total, "saving the model bundle")
        paths["bundle"] = train_mod.save_bundle(bundle, os.path.join(out_dir, "model.joblib"))
    if payload.get("write_report", True):
        progress(8, total, f"drawing {len(requests)} figures into the report")
        html, warnings = report_mod.build_report(
            bundle, context, requests, extra_sections=extra_sections,
            title=payload.get("title") or "Machine learning report",
            progress=lambda i, n, m: progress(8, total, f"figure {i}/{n}: {m}"),
        )
        path = os.path.join(out_dir, "report.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        paths["report"] = path
        notes.extend(warnings)
    if payload.get("write_script", True):
        progress(9, total, "writing the reproduction script")
        paths["script"] = export_script.write_script(
            os.path.join(out_dir, "reproduce_model.py"), model_key=best_key,
            model_params=best_params, train_spec=spec.to_dict(),
            preprocess_spec=pre_spec.to_dict(), cleaning_ops=payload.get("cleaning_ops") or [],
            source_file=payload.get("source_file") or "",
            source_format=payload.get("source_format"), source_table=payload.get("source_table"),
            notes=payload.get("notes") or "", title=f"Reproduce {MODELS.get_model_spec(best_key).label}",
        )

    progress(total, total, "finished")
    return {"split": split, "pipeline": pipe, "evaluation": evaluation, "cv": cv,
            "importance": importance, "prepared": prepared, "model_key": best_key,
            "model_params": best_params, "spec": spec, "leaderboard": leaderboard,
            "tuning": tuning_result, "best_values": best_values, "bundle": bundle,
            "paths": paths, "notes": notes}


# --------------------------------------------------------------------------
# 10. optimise
# --------------------------------------------------------------------------

class OptimizePage(StepPage):
    title = "10 · Best combinations"
    requires = "model"
    subtitle = ("Four different questions: which value of each field drives the prediction, "
                "which inputs reach a target outcome, which cleaning recipe scores best, and "
                "which subset of columns scores best.")

    def build(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_field_tab(), "Best value per field")
        self.tabs.addTab(self._build_design_tab(), "Inverse design")
        self.tabs.addTab(self._build_recipe_tab(), "Best cleaning recipe")
        self.tabs.addTab(self._build_selection_tab(), "Best feature combination")
        self.body.addWidget(self.tabs, 1)

    # -------------------------------------------------------- 1. per-field
    def _build_field_tab(self) -> QWidget:
        self.field_form = FieldForm([{
            "name": "features", "label": "features to sweep", "kind": "columns", "default": [],
            "choices": [], "optional": True, "help": "blank sweeps every numeric column",
        }])
        self.field_run = QPushButton("Sweep the features")
        self.field_run.clicked.connect(self._run_field)
        self.field_cards = MetricCards()
        self.field_table = DataFrameView(max_rows=200)
        self.field_notes = QLabel()
        self.field_notes.setWordWrap(True)
        theme.tag(self.field_notes, "muted")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("Each feature is swept across the range the data actually spans, with "
                              "everything else held as it is; the table reports the value that "
                              "maximises the prediction and how large the effect is."))
        layout.addWidget(self.field_form)
        layout.addWidget(self.field_run, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.field_cards)
        layout.addWidget(self.field_notes)
        layout.addWidget(self.field_table, 1)
        return page

    def _run_field(self):
        if not self.require_model():
            return
        features = self.field_form.values().get("features") or []
        X = self.state.split.X_train if self.state.split else None
        if X is None or not len(X):
            self.warn("No training rows", "Train a model first.")
            return
        if not features:
            features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if not features:
            self.warn("No numeric features", "There is nothing numeric to sweep.")
            return
        self.field_run.setEnabled(False)
        self.run_task(best_value_job, "Sweeping features", on_done=self._field_done,
                      on_cancelled=lambda: self.field_run.setEnabled(True),
                      pipe=self.state.pipeline, X=X, task=self.state.train_spec.task,
                      features=features)

    def _field_done(self, frame):
        self.field_run.setEnabled(True)
        self.field_table.set_frame(frame)
        if len(frame) and "effect size" in frame.columns:
            top = frame.iloc[0]
            self.field_cards.set_metrics([
                ("strongest feature", top.get("feature"), f"effect {top.get('effect size'):,.4f}"),
                ("its best value", top.get("maximising value"),
                 f"prediction {top.get('highest prediction'):,.4f}"),
                ("features swept", len(frame), ""),
            ])
        self.field_notes.setText("The 'plateau' column shows the whole range that is within 2% of the "
                                 "best prediction — usually a more useful operating window than a "
                                 "single value.")
        self.state.log_line(f"swept {len(frame)} feature(s) for their best value")

    # --------------------------------------------------------- 2. inverse design
    def _build_design_tab(self) -> QWidget:
        self.design_objective = QComboBox()
        for name in optimize.OBJECTIVES:
            self.design_objective.addItem(name, name)
        self.design_target = FieldForm([
            {"name": "target_value", "label": "target outcome", "kind": "float", "default": 0.0,
             "choices": [], "optional": True, "help": "used by 'match a target'"},
            {"name": "method", "label": "search method", "kind": "choice",
             "default": optimize.METHODS[0], "choices": list(optimize.METHODS), "optional": False,
             "help": ""},
            {"name": "n_samples", "label": "random samples", "kind": "int", "default": 2000,
             "choices": [], "optional": False, "help": ""},
            {"name": "levels", "label": "levels per axis (grid)", "kind": "int", "default": 12,
             "choices": [], "optional": False, "help": ""},
            {"name": "n_iter", "label": "generations / iterations", "kind": "int", "default": 30,
             "choices": [], "optional": False, "help": ""},
            {"name": "population", "label": "population (genetic)", "kind": "int", "default": 40,
             "choices": [], "optional": False, "help": ""},
            {"name": "top_n", "label": "shortlist size", "kind": "int", "default": 25,
             "choices": [], "optional": False, "help": ""},
            {"name": "seed", "label": "random seed", "kind": "int", "default": 42, "choices": [],
             "optional": False, "help": ""},
            {"name": "allow_extrapolation", "label": "allow values outside the observed range",
             "kind": "bool", "default": False, "choices": [], "optional": False,
             "help": "off by default: a tree ensemble has no idea what happens out there"},
        ])
        self.bounds_table = QTableWidget(0, 3)
        self.bounds_table.setHorizontalHeaderLabels(["design variable", "lowest", "highest"])
        self.bounds_table.horizontalHeader().setStretchLastSection(True)
        self.fixed_table = QTableWidget(0, 2)
        self.fixed_table.setHorizontalHeaderLabels(["held constant", "value"])
        self.fixed_table.horizontalHeader().setStretchLastSection(True)

        fill_bounds = QPushButton("Fill the bounds from the data")
        fill_bounds.clicked.connect(self._fill_bounds)
        add_bound = QPushButton("Add a variable")
        add_bound.clicked.connect(lambda: self.bounds_table.insertRow(self.bounds_table.rowCount()))
        remove_bound = QPushButton("Remove selected variable")
        remove_bound.clicked.connect(lambda: self._remove_row(self.bounds_table))
        add_fixed = QPushButton("Add a constant")
        add_fixed.clicked.connect(lambda: self.fixed_table.insertRow(self.fixed_table.rowCount()))
        remove_fixed = QPushButton("Remove selected constant")
        remove_fixed.clicked.connect(lambda: self._remove_row(self.fixed_table))

        self.constraints_edit = QPlainTextEdit()
        self.constraints_edit.setMaximumHeight(64)
        self.constraints_edit.setPlaceholderText("one per line, e.g.  Cement + Water <= 780")
        self.cost_edit = QLineEdit()
        self.cost_edit.setPlaceholderText("feature:weight; other:weight")
        self.cost_penalty_form = FieldForm([
            {"name": "cost_penalty", "label": "cost penalty strength", "kind": "float", "default": 0.0,
             "choices": [], "optional": True,
             "help": "pushes the search towards cheaper designs; 0 disables it"},
        ])

        self.design_run = QPushButton("Search for designs")
        theme.tag(self.design_run, "primary")
        self.design_run.clicked.connect(self._run_design)
        self.design_cards = MetricCards()
        self.design_note = QLabel()
        self.design_note.setWordWrap(True)
        theme.tag(self.design_note, "warning")
        self.design_table = DataFrameView(max_rows=100)

        bounds_box = QGroupBox("Design variables")
        bounds_layout = QVBoxLayout(bounds_box)
        bounds_layout.addWidget(self.bounds_table)
        bounds_buttons = QHBoxLayout()
        for button in (fill_bounds, add_bound, remove_bound):
            bounds_buttons.addWidget(button)
        bounds_buttons.addStretch(1)
        bounds_layout.addLayout(bounds_buttons)

        fixed_box = QGroupBox("Held constant")
        fixed_layout = QVBoxLayout(fixed_box)
        fixed_layout.addWidget(self.fixed_table)
        fixed_buttons = QHBoxLayout()
        fixed_buttons.addWidget(add_fixed)
        fixed_buttons.addWidget(remove_fixed)
        fixed_buttons.addStretch(1)
        fixed_layout.addLayout(fixed_buttons)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("objective:"))
        left_layout.addWidget(self.design_objective)
        left_layout.addWidget(self.design_target)
        left_layout.addWidget(bounds_box)
        left_layout.addWidget(fixed_box)
        left_layout.addWidget(QLabel("constraints (one per line):"))
        left_layout.addWidget(self.constraints_edit)
        left_layout.addWidget(QLabel("cost weights:"))
        left_layout.addWidget(self.cost_edit)
        left_layout.addWidget(self.cost_penalty_form)
        left_layout.addWidget(self.design_run)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.design_cards)
        right_layout.addWidget(self.design_note)
        right_layout.addWidget(self.design_table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        return splitter

    def _remove_row(self, table: QTableWidget):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _fill_bounds(self, quiet: bool = False):
        split = self.state.split
        X = split.X_train if split is not None else self.state.df
        if X is None or not len(X):
            if not quiet:
                self.warn("No data", "Load data and train a model first.")
            return
        bounds = optimize.quick_design_suggestion(X)
        rows = [[name, f"{lo:.4f}", f"{hi:.4f}"] for name, (lo, hi) in bounds.items()]
        fill_table(self.bounds_table, rows, 3)
        if not quiet:
            self.state.log_line(f"filled {len(rows)} design variables from the 2–98% range of the data")

    def _run_design(self):
        if not self.require_model():
            return
        X = self.state.split.X_train if self.state.split else None
        if X is None:
            self.warn("No training rows", "Train a model first.")
            return
        bounds = {}
        for name, low, high in read_pairs(self.bounds_table):
            try:
                bounds[name] = [float(low), float(high)]
            except ValueError:
                self.warn("Bad bound", f"{name}: low and high must be numbers.")
                return
        if not bounds:
            self.warn("No design variables", "Add at least one variable with bounds.")
            return
        fixed = {}
        for name, value in read_pairs(self.fixed_table):
            fixed[name] = float(value) if _is_number(value) else value
        values = self.design_target.values()
        objective = self.design_objective.currentData()
        target_value = values.get("target_value")
        if objective == "match a target" and target_value in (None, 0.0):
            if QMessageBox.question(self, "Target value",
                                    "The target outcome is 0. Use it anyway?") != QMessageBox.StandardButton.Yes:
                return
        cost_weights = {}
        for chunk in (self.cost_edit.text() or "").split(";"):
            if ":" in chunk:
                name, _, weight = chunk.partition(":")
                if _is_number(weight):
                    cost_weights[name.strip()] = float(weight)
        constraints = [line.strip() for line in self.constraints_edit.toPlainText().splitlines() if line.strip()]
        spec = {
            "bounds": bounds, "fixed": fixed, "target_value": target_value, "objective": objective,
            "method": values.get("method"), "n_samples": values.get("n_samples"),
            "levels": values.get("levels"), "n_iter": values.get("n_iter"),
            "population": values.get("population"), "top_n": values.get("top_n"),
            "seed": values.get("seed"), "allow_extrapolation": values.get("allow_extrapolation"),
            "constraints": constraints, "cost_weights": cost_weights,
            "cost_penalty": values.get("cost_penalty") or 0.0,
        }
        self.design_run.setEnabled(False)
        self.state.log_line(f"inverse design: {objective} over {len(bounds)} variable(s), "
                            f"method {spec['method']}")
        self.run_task(design_job, "Searching for designs", on_done=self._design_done,
                      on_cancelled=lambda: self.design_run.setEnabled(True),
                      pipe=self.state.pipeline, X=X, task=self.state.train_spec.task,
                      spec_dict=spec)

    def _design_done(self, result):
        self.design_run.setEnabled(True)
        self.state.design = result
        self.design_table.set_frame(result.table)
        summary = result.summary or {}
        self.design_cards.set_metrics([
            ("best prediction", summary.get("best prediction"), summary.get("objective", "")),
            ("feasible", f"{result.feasible_count} / {result.sampled_count}", "candidates"),
            ("design variables", summary.get("design variables"), summary.get("method", "")),
            ("seconds", summary.get("seconds"), ""),
        ])
        self.design_note.setText(result.note or "")
        self.state.log_line(f"inverse design finished: best prediction "
                            f"{summary.get('best prediction')}")

    # ------------------------------------------------------- 3. best recipe
    def _build_recipe_tab(self) -> QWidget:
        self.recipe_form = FieldForm([
            {"name": "impute", "label": "imputation options", "kind": "columns",
             "default": ["median", "mean", "knn"], "choices": [], "optional": False, "help": ""},
            {"name": "outlier", "label": "outlier options", "kind": "columns",
             "default": ["none", "clip_iqr", "winsorize"], "choices": [], "optional": False, "help": ""},
            {"name": "scale", "label": "scaling options", "kind": "columns",
             "default": ["standard", "none"], "choices": [], "optional": False, "help": ""},
            {"name": "max_combos", "label": "combination budget", "kind": "int", "default": 40,
             "choices": [], "optional": False, "help": ""},
        ])
        self.recipe_model = QLabel()
        self.recipe_model.setWordWrap(True)
        self.recipe_run = QPushButton("Search the recipes")
        theme.tag(self.recipe_run, "primary")
        self.recipe_run.clicked.connect(self._run_recipe)
        adopt = QPushButton("Adopt the best recipe on the Features screen")
        adopt.clicked.connect(self._adopt_recipe)
        self.recipe_cards = MetricCards()
        self.recipe_table = DataFrameView(max_rows=100)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("Each recipe is scored with the same model and the same folds as the "
                              "Model screen, so the comparison is apples to apples.  The recipe is "
                              "imputation + outlier treatment + scaling; encoding is held fixed."))
        layout.addWidget(self.recipe_form)
        layout.addWidget(self.recipe_model)
        row = QHBoxLayout()
        row.addWidget(self.recipe_run)
        row.addWidget(adopt)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.recipe_cards)
        layout.addWidget(self.recipe_table, 1)
        return page

    def _run_recipe(self):
        if not self.require_data():
            return
        state = self.state
        if not state.model_key:
            self.warn("No model", "Pick and train a model on the Model screen first — the search "
                                  "needs one algorithm to compare recipes with.")
            return
        if state.train_spec.task == "unsupervised":
            self.warn("Not available", "This search needs a supervised target.")
            return
        values = self.recipe_form.values()
        spec = state.train_spec
        spec.features = spec.features or [c for c in state.df.columns if c != spec.target]
        options = {"impute": values.get("impute"), "outlier": values.get("outlier"),
                   "scale": values.get("scale"), "max_combos": values.get("max_combos")}
        self.recipe_run.setEnabled(False)
        self.run_task(cleaning_combo_job, "Searching cleaning recipes", on_done=self._recipe_done,
                      on_cancelled=lambda: self.recipe_run.setEnabled(True),
                      df=state.df, spec_dict=spec.to_dict(), pre_dict=state.pre_spec.to_dict(),
                      model_key=state.model_key, model_params=state.model_params, options=options)

    def _recipe_done(self, result):
        self.recipe_run.setEnabled(True)
        frame = result["frame"]
        self.recipe_table.set_frame(frame)
        metric = result.get("metric")
        best = frame.iloc[0] if len(frame) else None
        self.recipe_cards.set_metrics([
            (metric or "score", None if best is None else best.get(metric), "best recipe"),
            ("recipes tried", len(frame), f"{result.get('seconds')} s"),
            ("best imputation", None if best is None else best.get("impute"), ""),
            ("best outliers", None if best is None else best.get("outliers"),
             None if best is None else f"scaling {best.get('scaling')}"),
        ])
        self.state.log_line(f"scored {len(frame)} cleaning recipes; best {metric} = "
                            f"{None if best is None else best.get(metric)}")

    def _adopt_recipe(self):
        frame = self.recipe_table.current_frame()
        if frame is None or not len(frame):
            self.warn("Nothing to adopt", "Run the search first.")
            return
        best = frame.iloc[0]
        spec = PreprocessSpec.from_dict({
            **self.state.pre_spec.to_dict(),
            "numeric_impute": best.get("impute") or self.state.pre_spec.numeric_impute,
            "outlier": best.get("outliers") or self.state.pre_spec.outlier,
            "scale": best.get("scaling") or self.state.pre_spec.scale,
        })
        self.state.adopt_preprocess(spec, "adopted the best cleaning recipe from the search")

    # ---------------------------------------------------- 4. best feature set
    def _build_selection_tab(self) -> QWidget:
        self.select_form = FieldForm([
            {"name": "method", "label": "method", "kind": "choice", "default": "forward",
             "choices": list(select_mod.SEARCH_METHODS), "optional": False,
             "help": "forward/backward/exhaustive actually refit the model per candidate"},
            {"name": "metric", "label": "optimise", "kind": "choice", "default": "r2",
             "choices": [], "optional": False, "help": ""},
            {"name": "k", "label": "keep this many (rankings)", "kind": "int", "default": 8,
             "choices": [], "optional": False, "help": ""},
            {"name": "max_features", "label": "largest subset to consider", "kind": "int",
             "default": 0, "choices": [], "optional": True, "help": "0 = no limit"},
            {"name": "max_evaluations", "label": "model-fit budget", "kind": "int", "default": 60,
             "choices": [], "optional": False, "help": ""},
            {"name": "cv_folds", "label": "folds", "kind": "int", "default": 5, "choices": [],
             "optional": False, "help": ""},
            {"name": "score_func", "label": "univariate score", "kind": "choice", "default": "auto",
             "choices": ["auto", "f_regression", "mutual_info_regression", "f_classif", "chi2",
                         "mutual_info_classif"], "optional": False, "help": ""},
            {"name": "n_repeats", "label": "shuffles per column (importance)", "kind": "int",
             "default": 5, "choices": [], "optional": False, "help": ""},
        ])
        self.select_run = QPushButton("Search feature combinations")
        theme.tag(self.select_run, "primary")
        self.select_run.clicked.connect(self._run_selection)
        adopt = QPushButton("Use the winning columns on the Model screen")
        adopt.clicked.connect(self._adopt_selection)
        self.select_cards = MetricCards()
        self.select_note = QLabel()
        self.select_note.setWordWrap(True)
        theme.tag(self.select_note, "muted")
        self.select_table = DataFrameView(max_rows=300)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("Forward, backward and exhaustive selection refit the model on each "
                              "candidate subset, which is the only honest way to measure how columns "
                              "interact.  Univariate and importance rankings fit nothing extra, and "
                              "RFE works on the encoded matrix."))
        layout.addWidget(self.select_form)
        row = QHBoxLayout()
        row.addWidget(self.select_run)
        row.addWidget(adopt)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.select_cards)
        layout.addWidget(self.select_note)
        layout.addWidget(self.select_table, 1)
        return page

    def _run_selection(self):
        if not self.require_data():
            return
        state = self.state
        if not state.model_key:
            self.warn("No model", "Choose a model on the Model screen first.")
            return
        if state.train_spec.task == "unsupervised":
            self.warn("Not available", "Feature selection needs a supervised target.")
            return
        values = self.select_form.values()
        sel = {
            "method": values.get("method"), "metric": values.get("metric"),
            "k": int(values.get("k") or 8),
            "max_features": int(values.get("max_features") or 0) or None,
            "max_evaluations": int(values.get("max_evaluations") or 60),
            "cv_folds": int(values.get("cv_folds") or 5),
            "score_func": values.get("score_func") or "auto",
            "n_repeats": int(values.get("n_repeats") or 5),
        }
        spec = state.train_spec
        spec.features = spec.features or [c for c in state.df.columns if c != spec.target]
        self.select_run.setEnabled(False)
        self.run_task(selection_job, "Searching feature combinations",
                      on_done=self._selection_done,
                      on_cancelled=lambda: self.select_run.setEnabled(True),
                      df=state.df, spec_dict=spec.to_dict(), pre_dict=state.pre_spec.to_dict(),
                      model_key=state.model_key, model_params=state.model_params, sel_dict=sel)

    def _selection_done(self, result):
        self.select_run.setEnabled(True)
        self.state.selection = result
        frame = result.ranking if result.ranking is not None and len(result.ranking) else result.history
        self.select_table.set_frame(frame)
        columns = (result.best or {}).get("columns") or []
        self.select_cards.set_metrics([
            ("method", result.method, f"{result.n_evaluated} model fit(s)"),
            ("columns chosen", len(columns), ""),
            ("score", (result.best or {}).get("score"), result.metric),
            ("seconds", round(result.elapsed, 1), ""),
        ])
        note = result.note or ""
        self.select_note.setText(("chosen: " + ", ".join(map(str, columns)) + "  ·  " + note).strip(" ·"))
        self.state.log_line(f"{result.method} selection picked {len(columns)} column(s): "
                            f"{', '.join(map(str, columns))[:120]}")

    def _adopt_selection(self):
        result = self.state.selection
        if result is None or not (result.best or {}).get("columns"):
            self.warn("Nothing to adopt", "Run a search first.")
            return
        columns = list(result.best["columns"])
        if result.method == "rfe":
            if QMessageBox.question(self, "Encoded names",
                                    "RFE selects names from the *encoded* matrix, not the original "
                                    "columns. Use them anyway?") != QMessageBox.StandardButton.Yes:
                return
        self.state.adopt_features(columns, f"adopted {len(columns)} columns from {result.method} selection")

    # --------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        if token != getattr(self, "_token", None):
            self._token = token
            X = self.state.split.X_train if self.state.split else self.state.df
            columns = [str(c) for c in (X.columns if X is not None else [])]
            for form in (self.field_form, self.design_target, self.select_form):
                form.set_columns(columns)
            if X is not None and self.bounds_table.rowCount() == 0:
                self._fill_bounds(quiet=True)

        task = self.state.train_spec.task or "regression"
        keys = [key for _, key in metric_choices(task)] if task != "unsupervised" else ["r2"]
        metric_field = next((f for f in self.select_form.fields if f["name"] == "metric"), None)
        if metric_field is not None and metric_field["choices"] != keys:
            self.select_form.update_field("metric", choices=keys)
        if self.select_form.values().get("metric") not in keys:
            self.select_form.set_values({"metric": keys[0] if keys else "r2"})

        self.recipe_model.setText(
            f"Recipes will be scored with: {MODELS.get_model_spec(self.state.model_key).label}"
            if self.state.model_key else
            "Recipes need a model — choose one on the Model screen.")


def _is_number(text) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# 11. cluster
# --------------------------------------------------------------------------

class ClusterPage(StepPage):
    title = "11 · Cluster & project"
    requires = "data"
    subtitle = ("Unsupervised work: group rows that behave alike, or compress many columns into "
                "a few axes you can plot.")

    def build(self):
        self.features_form = FieldForm([{
            "name": "features", "label": "columns to use", "kind": "columns", "default": [],
            "choices": [], "optional": True, "help": "blank uses every column",
        }])
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.model_notes = QLabel()
        self.model_notes.setWordWrap(True)
        theme.tag(self.model_notes, "muted")
        self.param_form = FieldForm()
        self.fit_button = QPushButton("Fit")
        theme.tag(self.fit_button, "primary")
        self.fit_button.clicked.connect(self._fit)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.features_form)
        left_layout.addWidget(QLabel("algorithm:"))
        left_layout.addWidget(self.model_combo)
        left_layout.addWidget(self.model_notes)
        left_layout.addWidget(self.param_form)
        left_layout.addWidget(self.fit_button)
        left_layout.addStretch(1)

        self.cards = MetricCards()
        self.canvas = PlotCanvas()
        self.table = DataFrameView(max_rows=200)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cards)
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(section_title("Cluster sizes / explained variance"))
        right_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

        self._model_changed()

    def _model_changed(self):
        key = self.model_combo.currentData()
        if not key:
            return
        spec = MODELS.get_model_spec(key)
        self.param_form.set_fields(fields_from_model_params(spec.params))
        self.model_notes.setText(spec.notes or "")

    def _fit(self):
        if not self.require_data():
            return
        key = self.model_combo.currentData()
        if not key:
            return
        state = self.state
        features = self.features_form.values().get("features") or []
        if not features:
            features = [c for c in state.df.columns if c != state.train_spec.target]
        features = [c for c in features if c in state.df.columns]
        if not features:
            self.warn("No columns", "Choose at least one column.")
            return
        params = {k: v for k, v in self.param_form.values().items() if v is not None}
        self.fit_button.setEnabled(False)
        self.run_task(cluster_job, f"Fitting {MODELS.get_model_spec(key).label}",
                      on_done=self._fitted, on_cancelled=lambda: self.fit_button.setEnabled(True),
                      df=state.df, features=features, pre_dict=state.pre_spec.to_dict(),
                      model_key=key, model_params=params)

    def _fitted(self, result):
        self.fit_button.setEnabled(True)
        state = self.state
        state.split = result["split"]
        state.pipeline = result["pipeline"]
        state.prepared = result["prepared"]
        state.clusters = None if result["labels"] is None else np.asarray(result["labels"])
        state.model_key = result["model_key"]
        state.model_params = result["model_params"]
        state.train_spec = train_mod.TrainSpec(target=None, features=list(result["split"].feature_names),
                                               task="unsupervised")
        state.evaluation = {}
        state.importance = None
        state.cv = None
        state.log_line(f"fitted {MODELS.get_model_spec(result['model_key']).label} on "
                       f"{len(result['prepared']):,} rows → {result['prepared'].shape[1]} encoded columns")
        if result["silhouette"] is not None:
            state.log_line(f"silhouette score {result['silhouette']:.4f} (higher is better; "
                           "above ~0.5 means well-separated groups) — measured on "
                           f"{result.get('scored_on', 'prepared features')}")
        if result.get("embedding") is not None:
            state.log_line(f"latent code {np.asarray(result['embedding']).shape[1]}-dimensional "
                           "(a learned compression, not a linear one)")
        state.touch()
        self._show(result)

    def _show(self, result=None):
        state = self.state
        cards = []
        if result is not None:
            if result["silhouette"] is not None:
                cards.append(("silhouette", result["silhouette"], "higher is better"))
            if result["labels"] is not None:
                cards.append(("clusters found", len(set(np.asarray(result["labels"]).tolist())), ""))
            cards.append(("rows", len(result["prepared"]), "encoded columns "
                          f"{result['prepared'].shape[1]}"))
        self.cards.set_metrics(cards)
        if result is not None:
            self.table.set_frame(result["sizes"] if result["sizes"] is not None else result["explained"])
        try:
            figure = plots_mod.render("pca_scatter", state.plot_context(include_cv=False), {"max_points": 3000})
            self.canvas.show_figure(figure)
        except Exception as exc:  # noqa: BLE001 - the projection is a nicety
            state.log_line(f"could not draw the projection: {exc}")

    def refresh(self):
        token = self.data_token()
        if token != getattr(self, "_token", None):
            self._token = token
            columns = [str(c) for c in (self.state.df.columns if self.state.df is not None else [])]
            self.features_form.set_columns(columns)
        if self.model_combo.count() == 0:
            self.model_combo.blockSignals(True)
            for label, key, available in model_choices("unsupervised"):
                self.model_combo.addItem(label, key)
                if not available:
                    item = self.model_combo.model().item(self.model_combo.count() - 1)
                    if item is not None:
                        item.setEnabled(False)
            self.model_combo.blockSignals(False)
            self._model_changed()


# --------------------------------------------------------------------------
# 12. plots
# --------------------------------------------------------------------------

class PlotsPage(StepPage):
    title = "12 · Graphs"
    requires = "data"
    subtitle = ("Every graph the workbench can draw.  The ones that need a trained model are "
                "greyed out until you have one, and anything you draw can be queued for the report.")

    def build(self):
        self.family_combo = QComboBox()
        self.family_combo.addItems(plots_mod.PLOT_FAMILIES)
        self.family_combo.currentTextChanged.connect(self._fill_list)

        self.plot_list = QListWidget()
        self.plot_list.currentRowChanged.connect(self._plot_selected)

        self.param_form = FieldForm()
        self.draw_button = QPushButton("Draw")
        theme.tag(self.draw_button, "primary")
        self.draw_button.clicked.connect(self._draw)
        self.queue_button = QPushButton("Queue for the report")
        self.queue_button.clicked.connect(self._queue)
        self.queue_all_button = QPushButton("Queue every available plot")
        self.queue_all_button.clicked.connect(self._queue_all)
        self.save_button = QPushButton("Save as PNG…")
        self.save_button.clicked.connect(self._save_png)

        self.doc_label = QLabel()
        self.doc_label.setWordWrap(True)
        theme.tag(self.doc_label, "muted")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("family:"))
        left_layout.addWidget(self.family_combo)
        left_layout.addWidget(self.plot_list, 1)
        left_layout.addWidget(self.doc_label)
        left_layout.addWidget(self.param_form)
        row = QHBoxLayout()
        row.addWidget(self.draw_button)
        row.addWidget(self.queue_button)
        row.addStretch(1)
        left_layout.addLayout(row)

        self.canvas = PlotCanvas()

        self.queue_list = QListWidget()
        remove = QPushButton("Remove from the report")
        remove.clicked.connect(self._unqueue)
        clear = QPushButton("Clear the queue")
        clear.clicked.connect(self._clear_queue)
        queue_buttons = QHBoxLayout()
        queue_buttons.addWidget(self.queue_all_button)
        queue_buttons.addWidget(self.save_button)
        queue_buttons.addStretch(1)
        queue_buttons.addWidget(remove)
        queue_buttons.addWidget(clear)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, 1)
        right_layout.addLayout(queue_buttons)
        right_layout.addWidget(section_title("Figures queued for the report"))
        right_layout.addWidget(self.queue_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

        self._fill_list(self.family_combo.currentText())
        self._refresh_queue()

    # ------------------------------------------------------------------- list
    def _fill_list(self, family: str):
        self._available = {spec.key: spec for spec in plots_mod.available_plots(self.state.plot_context())}
        self._all = [spec for spec in plots_mod.PLOTS if spec.family == family]
        self.plot_list.blockSignals(True)
        self.plot_list.clear()
        for spec in self._all:
            available = spec.key in self._available
            item = QListWidgetItem(spec.label if available else f"{spec.label}  — unavailable")
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            if not available:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setToolTip(plots_mod.plot_requirement_hint(spec))
            self.plot_list.addItem(item)
        self.plot_list.blockSignals(False)
        for row in range(self.plot_list.count()):
            if self.plot_list.item(row).data(Qt.ItemDataRole.UserRole) in self._available:
                self.plot_list.setCurrentRow(row)
                break

    def _current_spec(self):
        item = self.plot_list.currentItem()
        if item is None:
            return None
        key = item.data(Qt.ItemDataRole.UserRole)
        return plots_mod.PLOT_INDEX.get(key)

    def _plot_selected(self, row: int):
        spec = self._current_spec()
        if spec is None:
            return
        self.param_form.set_fields(fields_from_plot(spec))
        self.param_form.set_columns(list(self.state.df.columns) if self.state.df is not None else [])
        try:
            defaults = plots_mod.default_options(spec, self.state.plot_context())
        except Exception:  # noqa: BLE001
            defaults = {}
        self.param_form.set_values(defaults)
        self.doc_label.setText(spec.doc or f"family: {spec.family}")
        self.draw_button.setEnabled(spec.key in self._available)

    # ------------------------------------------------------------------ draw
    def _draw(self):
        spec = self._current_spec()
        if spec is None:
            return
        opts = self.param_form.values()
        try:
            figure = plots_mod.render(spec.key, self.state.plot_context(), opts)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw that graph", str(exc))
            return
        self.canvas.show_figure(figure)
        detail = ", ".join(f"{k}={v}" for k, v in opts.items() if v not in (None, ""))
        self.state.log_line(f"drew {spec.label}" + (f" ({detail})" if detail else ""))

    def _save_png(self):
        if self.canvas.figure is None:
            self.warn("Nothing to save", "Draw a graph first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the figure", "figure.png",
                                              "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            self.canvas.figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save", f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"figure saved to {path}")

    # ----------------------------------------------------------------- queue
    def _queue(self):
        spec = self._current_spec()
        if spec is None:
            return
        view = {"kind": spec.key, "options": {k: v for k, v in self.param_form.values().items() if v not in (None, "")}}
        self.state.plot_views.append(view)
        self._refresh_queue()
        self.state.log_line(f"queued {spec.label} for the report")

    def _queue_all(self):
        available = list(self._available.values())
        for spec in available:
            try:
                options = plots_mod.default_options(spec, self.state.plot_context())
            except Exception:  # noqa: BLE001
                options = {}
            self.state.plot_views.append({"kind": spec.key,
                                          "options": {k: v for k, v in options.items() if v not in (None, "")}})
        self._refresh_queue()
        self.state.log_line(f"queued {len(available)} figures for the report")

    def _unqueue(self):
        row = self.queue_list.currentRow()
        if 0 <= row < len(self.state.plot_views):
            del self.state.plot_views[row]
            self._refresh_queue()

    def _clear_queue(self):
        self.state.plot_views = []
        self._refresh_queue()

    def _refresh_queue(self):
        self.queue_list.clear()
        for view in self.state.plot_views:
            spec = plots_mod.PLOT_INDEX.get(view.get("kind"))
            label = spec.label if spec else view.get("kind")
            detail = ", ".join(f"{k}={v}" for k, v in (view.get("options") or {}).items())
            self.queue_list.addItem(f"{label}" + (f"  ({detail})" if detail else ""))

    # --------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        state_token = (self.state.model_key, bool(self.state.has_model), self.state.clusters is not None)
        if (token, state_token) == getattr(self, "_token", None):
            return
        self._token = (token, state_token)
        self._fill_list(self.family_combo.currentText())


# --------------------------------------------------------------------------
# 19. one click
# --------------------------------------------------------------------------

class AutoPage(StepPage):
    title = "19 · Run everything"
    requires = "data"
    subtitle = ("One button: split, compare models, tune the winner, fit it, read the importances, "
                "draw the figures, and write the bundle, the report and the reproduction script.")

    FAST_MODELS = {
        "regression": ["linear", "ridge", "random_forest_reg", "hist_gb_reg", "knn_reg"],
        "binary": ["logistic", "random_forest_clf", "hist_gb_clf", "knn_clf", "gaussian_nb"],
        "multiclass": ["logistic", "random_forest_clf", "hist_gb_clf", "extra_trees_clf"],
    }

    def build(self):
        self.model_set = QComboBox()
        self.model_set.addItem("A fast, sensible set", "fast")
        self.model_set.addItem("Everything that fits this task", "all")
        self.model_set.addItem("Only the current model", "current")

        self.options_form = FieldForm([
            {"name": "tune", "label": "tune the winning model", "kind": "bool", "default": True,
             "choices": [], "optional": False, "help": ""},
            {"name": "write_bundle", "label": "save the model bundle", "kind": "bool", "default": True,
             "choices": [], "optional": False, "help": ""},
            {"name": "write_report", "label": "write the HTML report", "kind": "bool", "default": True,
             "choices": [], "optional": False, "help": ""},
            {"name": "write_script", "label": "export the reproduction script", "kind": "bool",
             "default": True, "choices": [], "optional": False, "help": ""},
        ])
        self.title_edit = QLineEdit("Machine learning report")
        self.out_dir = QLineEdit(os.path.join(self.state.project_dir, "workbench_output"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        theme.tag(self.target_label, "muted")

        self.run_button = QPushButton("Run everything")
        theme.tag(self.run_button, "primary")
        self.run_button.clicked.connect(self._run)
        open_report = QPushButton("Open the report")
        open_report.clicked.connect(self._open_report)
        open_folder = QPushButton("Open the output folder")
        open_folder.clicked.connect(self._open_folder)
        self.open_report_button = open_report
        self.open_folder_button = open_folder
        open_report.setEnabled(False)
        open_folder.setEnabled(False)

        self.cards = MetricCards()
        self.summary_view = KeyValueView()
        self.leaderboard = DataFrameView(max_rows=100)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        theme.tag(self.log_view, "mono")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(hint("With no data loaded, this loads the built-in example dataset "
                                   "first — so the button always produces something."))
        left_layout.addWidget(QLabel("models to try:"))
        left_layout.addWidget(self.model_set)
        left_layout.addWidget(self.options_form)
        left_layout.addWidget(QLabel("report title:"))
        left_layout.addWidget(self.title_edit)
        left_layout.addWidget(QLabel("output folder:"))
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(browse)
        left_layout.addLayout(out_row)
        left_layout.addWidget(self.target_label)
        left_layout.addWidget(self.run_button)
        buttons = QHBoxLayout()
        buttons.addWidget(open_report)
        buttons.addWidget(open_folder)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.log_view)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cards)
        right_layout.addWidget(section_title("Summary"))
        right_layout.addWidget(self.summary_view)
        right_layout.addWidget(section_title("Model comparison"))
        right_layout.addWidget(self.leaderboard, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

    # ------------------------------------------------------------------ setup
    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Choose an output folder", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    def _open_report(self):
        if self.state.report_path and os.path.exists(self.state.report_path):
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(self.state.report_path))

    def _open_folder(self):
        path = self.out_dir.text().strip()
        if path and os.path.isdir(path):
            import webbrowser

            webbrowser.open("file://" + os.path.abspath(path))

    def _prepare_spec(self):
        state = self.state
        spec = state.train_spec
        if spec.task == "unsupervised":
            return None
        if not spec.target or state.df is None or spec.target not in state.df.columns:
            target = state.suggested_target()
            if not target:
                return None
            spec.target = target
            spec.task = eda.detect_task(state.df[target])
            spec.primary_metric = M.primary_metric(spec.task)
        if not spec.features:
            spec.features = [c for c in state.df.columns if c != spec.target]
        if spec.task in ("binary", "multiclass"):
            counts = state.df[spec.target].value_counts()
            spec.stratify = bool(len(counts) and counts.min() >= 2)
        return spec

    def _keys(self, spec) -> list[str]:
        mode = self.model_set.currentData()
        available = [k for _, k, ok in model_choices(spec.task) if ok]
        if mode == "current" and self.state.model_key:
            return [self.state.model_key]
        if mode == "fast":
            preferred = self.FAST_MODELS.get(spec.task, [])
            keys = [k for k in preferred if k in available]
            return keys or available[:5]
        return available

    # -------------------------------------------------------------------- run
    def _run(self):
        state = self.state
        if state.df is None:
            from .data_steps import example_frame

            frame = example_frame()
            state.set_raw(frame, path="", fmt=None, table="example", options={})
            state.log_line("no data was loaded, so the built-in example dataset was used")
        spec = self._prepare_spec()
        if spec is None:
            self.warn("Cannot run", "This needs a supervised target column.  Load data and choose a "
                                    "target on the Model screen, or press this button with no data "
                                    "to use the built-in example.")
            return
        keys = self._keys(spec)
        if not keys:
            self.warn("No models", "No model is available for this task.")
            return
        state.train_spec = spec
        values = self.options_form.values()
        out_dir = self.out_dir.text().strip() or os.path.join(state.project_dir, "workbench_output")
        payload = {
            "df": state.df, "spec": spec.to_dict(), "pre": state.pre_spec.to_dict(),
            "model_keys": keys, "tune": bool(values.get("tune")), "out_dir": out_dir,
            "source_file": state.source_file, "source_format": state.source_format,
            "source_table": state.source_table, "cleaning_ops": state.clean_ops,
            "notes": getattr(state, "notes", ""), "title": self.title_edit.text().strip(),
            "write_bundle": bool(values.get("write_bundle")),
            "write_report": bool(values.get("write_report")),
            "write_script": bool(values.get("write_script")),
        }
        self.run_button.setEnabled(False)
        self.log_view.setPlainText(f"running with {len(keys)} candidate model(s): {', '.join(keys)}")
        state.log_line("one-click run started")
        self.run_task(auto_job, "Running everything", on_done=self._done,
                      on_cancelled=lambda: self.run_button.setEnabled(True), payload=payload)

    def _done(self, result):
        self.run_button.setEnabled(True)
        state = self.state
        state.train_spec = result["spec"]
        state.split = result["split"]
        state.pipeline = result["pipeline"]
        state.evaluation = result["evaluation"]
        state.cv = result["cv"]
        state.importance = result["importance"]
        state.prepared = result["prepared"]
        state.model_key = result["model_key"]
        state.model_params = result["model_params"]
        state.tuning = result["tuning"]
        state.bundle = result["bundle"]
        paths = result["paths"]
        state.bundle_path = paths.get("bundle", "")
        state.report_path = paths.get("report", "")
        state.script_path = paths.get("script", "")
        state.plot_views = []

        task = result["spec"].task
        primary = result["spec"].primary_metric
        test = (result["evaluation"].get("test") or {}).get(primary)
        cv_mean = ((result["cv"] or {}).get("primary") or {}).get("mean")
        self.cards.set_metrics([
            (M.label(task, primary), test, "held-out test set"),
            ("cross-validation", cv_mean, "mean over folds"),
            ("model", MODELS.get_model_spec(result["model_key"]).label,
             describe_result(result["spec"], result["split"])),
            ("figures", len(result["best_values"]) if result["best_values"] is not None else 0,
             "features swept"),
        ])
        pairs = [(f"{M.label(task, k)}", v) for k, v in (result["evaluation"].get("test") or {}).items()
                 if isinstance(v, (int, float))]
        pairs += [("parameters", ", ".join(f"{k}={v}" for k, v in result["model_params"].items()) or "defaults"),
                  ("rows", f"{result['split'].n_train:,} train / {result['split'].n_test:,} test")]
        for name, path in paths.items():
            pairs.append((name, path))
        self.summary_view.set_pairs(pairs)
        self.leaderboard.set_frame(result["leaderboard"])
        self.log_view.setPlainText("\n".join([f"chosen model: {result['model_key']}",
                                              *[f"note: {n}" for n in result["notes"]],
                                              *[f"wrote: {p}" for p in paths.values()]]))
        self.open_report_button.setEnabled(bool(state.report_path))
        self.open_folder_button.setEnabled(os.path.isdir(self.out_dir.text().strip()))
        for path in paths.values():
            state.log_line(f"one-click run wrote {path}")
        score_text = f"{test:.4f}" if isinstance(test, (int, float)) else "n/a"
        state.log_line(f"one-click run finished: {MODELS.get_model_spec(result['model_key']).label}, "
                       f"{M.label(task, primary)} = {score_text}")
        state.touch()

    # --------------------------------------------------------------- refresh
    def refresh(self):
        spec = self.state.train_spec
        target = spec.target
        task = spec.task
        if target:
            self.target_label.setText(f"target: {target} · task: {task} · "
                                      f"primary metric: {M.label(task, spec.primary_metric)}")
        else:
            self.target_label.setText("no target chosen yet — one will be detected automatically")
