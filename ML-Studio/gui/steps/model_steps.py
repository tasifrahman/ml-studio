"""The modelling screens: choose a model, tune it, read the results, save, predict.

Every long job in this file follows the same contract: a module-level function
taking ``(progress, cancel, ...)`` that returns a plain dict, run through
:class:`~gui.workers.TaskRunner`.  Keeping the jobs at module level (rather than
as methods) means they can be exercised without a window, which is how they were
tested.

The jobs are also imported by the one-click screen, so "run everything" uses
exactly the same code path as pressing the buttons by hand.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from core import eda, explain, loaders, metrics as M, models as MODELS, plots as plots_mod, preprocess
from core import train as train_mod
from core import tuning as tuning_mod

from .. import theme
from ..widgets import (
    DataFrameView, FieldForm, KeyValueView, MetricCards, StepPage,
    fields_from_model_params, hint, scrollable, section_title, separator,
)

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

TASK_LABELS = [
    ("regression", "Regression — predict a number"),
    ("binary", "Binary classification — two classes"),
    ("multiclass", "Multiclass classification — three or more classes"),
    ("unsupervised", "Unsupervised — clustering / projection"),
]


def metric_keys(task: str) -> list[str]:
    if task == "unsupervised":
        return []
    keys = train_mod.metrics_for(task, 0)
    primary = M.primary_metric(task)
    return ([primary] + [k for k in keys if k != primary]) if primary in keys else keys


def metric_choices(task: str) -> list[tuple[str, str]]:
    return [(f"{M.label(task, k)}  ({k})", k) for k in metric_keys(task)]


def model_choices(task: str) -> list[tuple[str, str, bool]]:
    """(label, key, available) for every model that serves this task."""
    out = []
    for spec in MODELS.models_for_task(task):
        available = MODELS.is_available(spec)
        label = spec.label
        if spec.optional and not available:
            label += f"   (needs {spec.optional})"
        out.append((label, spec.key, available))
    return out


def default_plot_requests(ctx: plots_mod.PlotContext, task: str, target: str | None,
                          limit: int = 12) -> list[tuple[str, dict]]:
    """A sensible figure set for a report: overview, then model diagnostics."""
    df = ctx.df
    numeric = plots_mod.numeric_columns(df) if df is not None else []
    requests: list[tuple[str, dict]] = [("target_dist", {"target": target})] if target else []
    requests.append(("missing_bar", {}))
    if len(numeric) >= 2:
        requests.append(("corr_heatmap", {"triangle": True}))
        requests.append(("scatter", {"x": numeric[0], "y": numeric[1] if numeric[1] != target else numeric[0]}))
    if numeric:
        requests.append(("hist", {"column": numeric[0]}))
        requests.append(("box", {"column": numeric[-1]}))
    if task in ("binary", "multiclass"):
        requests += [("confusion", {}), ("class_balance", {"target": target})]
        if ctx.evaluation and (ctx.evaluation or {}).get("test_proba") is not None:
            requests.append(("roc_curve", {}))
    elif task == "regression":
        requests += [("actual_vs_predicted", {}), ("residuals", {})]
    if ctx.importance is not None:
        requests.append(("feature_importance", {"top": 20}))
    if task != "unsupervised":
        requests.append(("learning_curve", {"cv": 5}))
        if numeric:
            requests.append(("partial_dependence", {"feature": numeric[0]}))
    # only neural networks have a loss history, so this is filtered out for everything else
    requests.append(("training_curve", {}))
    requests.append(("pca_scatter", {}))
    if ctx.clusters is not None:
        requests.append(("cluster_sizes", {}))
    # only keep the ones the context can actually satisfy
    available = {spec.key for spec in plots_mod.available_plots(ctx)}
    kept = [(kind, opts) for kind, opts in requests if kind in available]
    return kept[:limit]


def describe_result(spec: train_mod.TrainSpec, split: train_mod.Split) -> str:
    return (f"{len(split.feature_names)} features · {split.n_train:,} train rows · "
            f"{split.n_test:,} test rows · task {spec.task}")


# --------------------------------------------------------------------------
# background jobs
# --------------------------------------------------------------------------

def train_job(progress, cancel, df, spec_dict, pre_dict, model_key, model_params, do_cv: bool = True):
    spec = train_mod.TrainSpec.from_dict(spec_dict)
    pre_spec = preprocess.PreprocessSpec.from_dict(pre_dict)
    supervised = spec.task != "unsupervised"
    total = 4 + (1 if (do_cv and supervised) else 0)

    progress(0, total, "splitting the data")
    split = train_mod.make_split(df, spec)
    train_mod.check_cancel(cancel)

    label = MODELS.get_model_spec(model_key).label
    progress(1, total, f"fitting {label}")
    pipe, _ = train_mod.build_train_pipeline(pre_spec, model_key, model_params, spec.task)
    train_mod.fit_full(pipe, split, spec)
    train_mod.check_cancel(cancel)

    progress(2, total, "scoring the test set")
    evaluation = train_mod.evaluate_fitted(pipe, split, spec)

    cv = None
    index = 3
    if do_cv and supervised:
        progress(3, total, f"cross-validating ({spec.cv_folds} folds)")
        cv = train_mod.cross_validate_pipeline(pipe, split.X_train, split.y_train, spec,
                                               weights=split.w_train)
        index = 4
    train_mod.check_cancel(cancel)

    progress(index, total, "feature importances")
    importance = None
    if supervised:
        try:
            importance = explain.importance_frame(pipe, split.X_train, split.y_train, spec.task,
                                                  n_repeats=8)
        except Exception:  # noqa: BLE001 - importances are optional
            importance = None

    prepared = None
    try:
        prepared = pipe.named_steps["prepare"].transform(split.X_train)
    except Exception:  # noqa: BLE001
        pass

    return {"split": split, "pipeline": pipe, "evaluation": evaluation, "cv": cv,
            "importance": importance, "prepared": prepared, "model_key": model_key,
            "model_params": dict(model_params), "spec": spec, "dropped": list(MODELS.DROPPED_PARAMS)}


def compare_job(progress, cancel, df, spec_dict, pre_dict, model_keys):
    spec = train_mod.TrainSpec.from_dict(spec_dict)
    pre_spec = preprocess.PreprocessSpec.from_dict(pre_dict)
    split = train_mod.make_split(df, spec)
    frame = tuning_mod.comparison_table(model_keys, pre_spec, split.X_train, split.y_train, spec,
                                        weights=split.w_train, progress=progress, cancel=cancel)
    return {"frame": frame, "cv": frame.attrs.get("cv", {}), "split": split}


def tune_job(progress, cancel, df, spec_dict, pre_dict, model_key, grid, tune_spec_dict,
             adopt: bool = True):
    """Search the grid, then optionally make the winner the current model."""
    spec = train_mod.TrainSpec.from_dict(spec_dict)
    pre_spec = preprocess.PreprocessSpec.from_dict(pre_dict)
    tune_spec = tuning_mod.TuneSpec.from_dict(tune_spec_dict)
    tune_spec.param_grid = dict(grid or {})

    split = train_mod.make_split(df, spec)
    result = tuning_mod.tune(model_key, pre_spec, split.X_train, split.y_train, spec, tune_spec,
                             weights=split.w_train, progress=progress, cancel=cancel)
    result["tune_spec"] = tune_spec.to_dict()
    result["split"] = split
    result["model_key"] = model_key
    result["evaluation"] = None
    result["importance"] = None
    result["prepared"] = None

    best = result.get("best_estimator")
    if best is not None and adopt:
        train_mod.check_cancel(cancel)
        result["evaluation"] = train_mod.evaluate_fitted(best, split, spec)
        try:
            result["importance"] = explain.importance_frame(best, split.X_train, split.y_train,
                                                            spec.task, n_repeats=8)
            result["prepared"] = best.named_steps["prepare"].transform(split.X_train)
        except Exception:  # noqa: BLE001
            pass
    return result


def predict_job(progress, cancel, pipe, frame, task):
    progress(0, 1, "predicting")
    out = pd.DataFrame(frame).copy()
    predictions = np.asarray(pipe.predict(frame))
    out["prediction"] = predictions
    proba = None
    if task != "regression":
        try:
            proba = np.asarray(pipe.predict_proba(frame))
        except Exception:  # noqa: BLE001 - not every classifier offers probabilities
            proba = None
    if proba is not None and proba.ndim == 2:
        classes = getattr(pipe, "classes_", None)
        if classes is None:
            classes = getattr(getattr(pipe, "named_steps", {}).get("model", None), "classes_", None)
        if classes is None or len(classes) != proba.shape[1]:
            classes = list(range(proba.shape[1]))
        for i, cls in enumerate(classes):
            out[f"probability of {cls}"] = proba[:, i]
    progress(1, 1, "done")
    return out


def load_bundle_job(progress, cancel, path):
    progress(0, 1, f"reading {os.path.basename(path)}")
    bundle = train_mod.load_bundle(path)
    progress(1, 1, "done")
    return bundle


def report_job(progress, cancel, bundle, context, plot_requests, title, path, extra_sections=None,
               extra_figures=None):
    from core import report as report_mod

    html, warnings = report_mod.build_report(bundle, context, plot_requests,
                                             extra_sections=extra_sections, title=title,
                                             extra_figures=extra_figures, progress=progress)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return {"path": path, "warnings": list(warnings), "bytes": os.path.getsize(path)}


# --------------------------------------------------------------------------
# 5. model
# --------------------------------------------------------------------------

class ModelPage(StepPage):
    title = "5 · Model"
    requires = "data"
    subtitle = ("Choose what to predict, how to split, and which algorithm.  Everything that "
                "learns from the data — imputation, encoding, scaling — is fitted on the training "
                "split only and travels with the saved model.")

    def build(self):
        # ---- problem
        self.task_combo = QComboBox()
        for key, label in TASK_LABELS:
            self.task_combo.addItem(label, key)
        self.task_combo.currentIndexChanged.connect(self._task_changed)

        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.currentTextChanged.connect(self._target_changed)

        self.features_form = FieldForm([{
            "name": "features", "label": "feature columns", "kind": "columns", "default": [],
            "choices": [], "optional": True,
            "help": "blank means every column except the target",
        }])
        all_button = QPushButton("Use every column except the target")
        all_button.clicked.connect(self._all_features)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        theme.tag(self.warning_label, "warning")

        problem_box = QGroupBox("What are we predicting?")
        problem_layout = QVBoxLayout(problem_box)
        problem_layout.addWidget(self.task_combo)
        problem_layout.addWidget(QLabel("target column:"))
        problem_layout.addWidget(self.target_combo)
        problem_layout.addWidget(self.features_form)
        problem_layout.addWidget(all_button)
        problem_layout.addWidget(self.warning_label)

        # ---- split
        self.spec_form = FieldForm([
            {"name": "test_size", "label": "test share", "kind": "float", "default": 0.2,
             "choices": [], "optional": False, "help": "0.2 = hold back 20% of the rows"},
            {"name": "shuffle", "label": "shuffle before splitting", "kind": "bool", "default": True,
             "choices": [], "optional": False, "help": "turn off for time-ordered data"},
            {"name": "stratify", "label": "keep class balance in the split", "kind": "bool",
             "default": False, "choices": [], "optional": False, "help": "classification only"},
            {"name": "random_state", "label": "random seed", "kind": "int", "default": 42,
             "choices": [], "optional": False, "help": "fix it so the split is reproducible"},
            {"name": "cv_folds", "label": "cross-validation folds", "kind": "int", "default": 5,
             "choices": [], "optional": False, "help": ""},
            {"name": "primary_metric", "label": "primary metric", "kind": "choice", "default": "r2",
             "choices": [], "optional": False, "help": "the number used to rank candidates"},
            {"name": "scale_target", "label": "standardise the target", "kind": "bool", "default": False,
             "choices": [], "optional": False, "help": "regression only"},
            {"name": "drop_missing_target", "label": "drop rows with no target", "kind": "bool",
             "default": True, "choices": [], "optional": False, "help": ""},
            {"name": "sample_weight_column", "label": "row weights column", "kind": "column",
             "default": None, "choices": [], "optional": True, "help": ""},
        ])
        split_box = QGroupBox("Train / test split")
        split_layout = QVBoxLayout(split_box)
        split_layout.addWidget(self.spec_form)

        # ---- algorithm
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.model_notes = QLabel()
        self.model_notes.setWordWrap(True)
        theme.tag(self.model_notes, "muted")
        self.param_form = FieldForm()
        defaults = QPushButton("Reset to defaults")
        defaults.clicked.connect(self._reset_params)
        suggested = QPushButton("Show the suggested search grid")
        suggested.clicked.connect(self._show_suggested)
        param_buttons = QHBoxLayout()
        param_buttons.addWidget(defaults)
        param_buttons.addWidget(suggested)
        param_buttons.addStretch(1)

        model_box = QGroupBox("Algorithm")
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_notes)
        model_layout.addWidget(self.param_form)
        model_layout.addLayout(param_buttons)

        # ---- actions
        self.train_button = QPushButton("Train this model")
        theme.tag(self.train_button, "primary")
        self.train_button.clicked.connect(self._train)
        self.compare_button = QPushButton("Compare several models")
        self.compare_button.clicked.connect(self._compare)
        self.compare_form = FieldForm([{
            "name": "keys", "label": "models to compare", "kind": "columns", "default": [],
            "choices": [], "optional": True, "help": "leave blank for a sensible default set",
        }])
        actions_box = QGroupBox("Run")
        actions_layout = QVBoxLayout(actions_box)
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(separator())
        actions_layout.addWidget(self.compare_form)
        actions_layout.addWidget(self.compare_button)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(problem_box)
        left_layout.addWidget(split_box)
        left_layout.addWidget(model_box)
        left_layout.addWidget(actions_box)
        left_layout.addStretch(1)

        # ---- results
        self.cards = MetricCards()
        self.cv_view = KeyValueView()
        self.importance_table = DataFrameView(max_rows=200)
        self.leaderboard = DataFrameView(max_rows=200)
        self.result_notes = QLabel()
        self.result_notes.setWordWrap(True)
        theme.tag(self.result_notes, "muted")

        results = QTabWidget()
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.addWidget(self.cards)
        overview_layout.addWidget(self.result_notes)
        overview_layout.addWidget(section_title("Cross-validation"))
        overview_layout.addWidget(self.cv_view)
        overview_layout.addStretch(1)
        results.addTab(overview, "Test results")
        importance_tab = QWidget()
        importance_layout = QVBoxLayout(importance_tab)
        importance_layout.addWidget(self.importance_table, 1)
        results.addTab(importance_tab, "Feature importance")
        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        compare_layout.addWidget(self.leaderboard, 1)
        results.addTab(compare_tab, "Model comparison")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(results, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

        self._task_changed()
        self._model_changed()
        self.state.features_changed.connect(self._adopt_features)

    def _adopt_features(self, columns):
        self.features_form.set_values({"features": [str(c) for c in columns]})
        self.state.log_line(f"the feature list on this screen was updated to {len(columns)} column(s)")

    # ---------------------------------------------------------------- settings
    def _task_changed(self):
        task = self.task_combo.currentData()
        choices = metric_choices(task)
        current = self.spec_form.values().get("primary_metric")
        self.spec_form.update_field(
            "primary_metric", choices=[c[1] for c in choices],
            default=M.primary_metric(task) if task != "unsupervised" else None)
        keys = [c[1] for c in choices]
        self.spec_form.set_values({"primary_metric": current if current in keys
                                   else (keys[0] if keys else "r2")})
        if task != getattr(self, "_last_task", None):
            self._last_task = task
            # a class split wants balance; a regression split does not care
            self.spec_form.set_values({"stratify": task in ("binary", "multiclass")})

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for label, key, available in model_choices(task):
            self.model_combo.addItem(label, key)
            if not available:
                item = self.model_combo.model().item(self.model_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        if self.state.model_key:
            index = self.model_combo.findData(self.state.model_key)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._model_changed()

    def _target_changed(self):
        target = self.target_combo.currentText().strip()
        if not target or self.state.df is None or target not in self.state.df.columns:
            self.warning_label.setText("")
            return
        detected = eda.detect_task(self.state.df[target])
        flags = eda.suspicious_columns(self.state.df, target)
        self.warning_label.setText(("· " + "\n· ".join(flags)) if flags else "")
        if detected in ("binary", "multiclass", "regression"):
            index = self.task_combo.findData(detected)
            if index >= 0 and index != self.task_combo.currentIndex():
                self.task_combo.setCurrentIndex(index)
        self.features_form.set_values({"features": [c for c in self.state.df.columns if c != target]})

    def _all_features(self):
        if self.state.df is None:
            return
        target = self.target_combo.currentText().strip()
        self.features_form.set_values({"features": [c for c in self.state.df.columns if c != target]})

    def _model_changed(self):
        key = self.model_combo.currentData()
        if not key:
            return
        spec = MODELS.get_model_spec(key)
        self.param_form.set_fields(fields_from_model_params(spec.params))
        if self.state.model_key == key and self.state.model_params:
            self.param_form.set_values(self.state.model_params)
        self.model_notes.setText(spec.notes or "")

    def _reset_params(self):
        key = self.model_combo.currentData()
        if key:
            self.param_form.set_values(MODELS.default_params(key))

    def _show_suggested(self):
        key = self.model_combo.currentData()
        if not key:
            return
        grid = MODELS.suggested_grid(key)
        if not grid:
            QMessageBox.information(self, "No grid", "This model has no suggested search grid.")
            return
        QMessageBox.information(self, "Suggested grid for " + MODELS.get_model_spec(key).label,
                                tuning_mod.grid_to_text(grid))

    # ------------------------------------------------------------------ spec
    def _read_spec(self) -> train_mod.TrainSpec | None:
        values = self.spec_form.values()
        target = self.target_combo.currentText().strip() or None
        task = self.task_combo.currentData()
        if task != "unsupervised" and not target:
            self.warn("No target", "Choose the column you want to predict.")
            return None
        features = self.features_form.values().get("features") or []
        if task == "unsupervised" and not features:
            features = [c for c in (self.state.df.columns if self.state.df is not None else []) if c != target]
        return train_mod.TrainSpec(
            target=target, features=list(features) or None, task=task,
            test_size=float(values.get("test_size") or 0.2),
            shuffle=bool(values.get("shuffle")),
            stratify=bool(values.get("stratify")),
            random_state=int(values.get("random_state") or 42),
            cv_folds=int(values.get("cv_folds") or 5),
            primary_metric=values.get("primary_metric") or M.primary_metric(task),
            scale_target=bool(values.get("scale_target")),
            drop_missing_target=bool(values.get("drop_missing_target")),
            sample_weight_column=values.get("sample_weight_column"),
        )

    def current_model(self):
        return self.model_combo.currentData(), {
            k: v for k, v in self.param_form.values().items() if v is not None}

    # ----------------------------------------------------------------- train
    def _train(self):
        if not self.require_data():
            return
        spec = self._read_spec()
        if spec is None:
            return
        key, params = self.current_model()
        if not key:
            self.warn("No algorithm", "Pick an algorithm first.")
            return
        spec.features = spec.features or [c for c in self.state.df.columns if c != spec.target]
        self.state.train_spec = spec
        self.train_button.setEnabled(False)
        self.state.log_line(f"training {MODELS.get_model_spec(key).label} with "
                            f"{MODELS.describe_params(key, params)}")
        self.run_task(train_job, f"Training {MODELS.get_model_spec(key).label}",
                      on_done=self._trained,
                      on_cancelled=lambda: self.train_button.setEnabled(True),
                      df=self.state.df, spec_dict=spec.to_dict(),
                      pre_dict=self.state.pre_spec.to_dict(), model_key=key,
                      model_params=params, do_cv=True)

    def _trained(self, result):
        self.train_button.setEnabled(True)
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
        for message in result.get("dropped") or []:
            state.log_line(f"note: {message}")
        if result["evaluation"]:
            primary = state.train_spec.primary_metric
            score = (result["evaluation"].get("test") or {}).get(primary)
            state.log_line(f"trained {MODELS.get_model_spec(result['model_key']).label}: "
                           f"{M.label(state.train_spec.task, primary)} on the test set = {score:.4f}"
                           if isinstance(score, float) else "trained")
        state.touch()
        self._show_results()

    # --------------------------------------------------------------- compare
    def _compare(self):
        if not self.require_data():
            return
        spec = self._read_spec()
        if spec is None or spec.task == "unsupervised":
            if spec is not None:
                self.warn("Not available", "Model comparison needs a supervised task.")
            return
        spec.features = spec.features or [c for c in self.state.df.columns if c != spec.target]
        keys = self.compare_form.values().get("keys") or []
        if not keys:
            keys = [k for _, k, available in model_choices(spec.task) if available][:6]
        keys = [k for k in keys if k in MODELS.MODEL_INDEX]
        if not keys:
            self.warn("Nothing to compare", "Pick at least one model key.")
            return
        self.state.train_spec = spec
        self.compare_button.setEnabled(False)
        self.run_task(compare_job, "Comparing models", on_done=self._compared,
                      on_cancelled=lambda: self.compare_button.setEnabled(True),
                      df=self.state.df, spec_dict=spec.to_dict(),
                      pre_dict=self.state.pre_spec.to_dict(), model_keys=keys)

    def _compared(self, result):
        self.compare_button.setEnabled(True)
        frame = result["frame"]
        self.leaderboard.set_frame(frame, note="cross-validated on the training split")
        self.state.log_line(f"compared {len(frame)} models; best = "
                            f"{frame.iloc[0]['model'] if len(frame) else 'none'}")
        self.state.touch()

    # --------------------------------------------------------------- results
    def _show_results(self):
        state = self.state
        evaluation = state.evaluation or {}
        task = state.train_spec.task
        primary = state.train_spec.primary_metric
        test = evaluation.get("test") or {}
        train = evaluation.get("train") or {}

        cards = []
        if state.model_key:
            cards.append(("model", MODELS.get_model_spec(state.model_key).label,
                          describe_result(state.train_spec, state.split) if state.split else ""))
        if primary and primary in test:
            cards.append((M.label(task, primary), test.get(primary), "held-out test set"))
        if primary and primary in train:
            cards.append((f"train {M.label(task, primary)}", train.get(primary), "same rows it learned from"))
        if state.cv:
            cv_primary = (state.cv.get("primary") or {})
            cards.append(("cross-validation", cv_primary.get("mean"),
                          f"± {cv_primary.get('std'):.4f} over {cv_primary.get('n_folds')} folds"
                          if cv_primary.get("std") is not None else ""))
        gap = evaluation.get("overfit_gap")
        if gap is not None:
            cards.append(("train→test gap", gap,
                          "large positive = memorising" if abs(float(gap)) > 0.1 else "healthy"))
        self.cards.set_metrics(cards[:5])

        pairs = []
        for key, value in test.items():
            if isinstance(value, (int, float)):
                pairs.append((f"test · {M.label(task, key)}", value))
        for key, value in train.items():
            if isinstance(value, (int, float)):
                pairs.append((f"train · {M.label(task, key)}", value))
        self.cv_view.set_pairs(pairs)

        notes = []
        if state.split is not None:
            for problem in state.split.dropped.values():
                notes.append(str(problem))
        self.result_notes.setText(" · ".join(notes))
        if state.importance is not None:
            self.importance_table.set_frame(state.importance)

    # --------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        changed = token != getattr(self, "_token", None)
        self._token = token
        frame = self.state.df
        if frame is None:
            return
        if changed:
            columns = [str(c) for c in frame.columns]
            current = self.target_combo.currentText()
            self.target_combo.blockSignals(True)
            self.target_combo.clear()
            self.target_combo.addItems(columns)
            suggested = self.state.train_spec.target or self.state.suggested_target()
            if current in columns:
                self.target_combo.setCurrentText(current)
            elif suggested in columns:
                self.target_combo.setCurrentText(str(suggested))
            self.target_combo.blockSignals(False)
            self.features_form.set_columns(columns)
            self.spec_form.set_columns(columns)
            self._target_changed()
        # follow a model chosen elsewhere (tuning, one-click, or a reopened project)
        if self.state.model_key:
            index = self.model_combo.findData(self.state.model_key)
            if index >= 0 and index != self.model_combo.currentIndex():
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(index)
                self.model_combo.blockSignals(False)
                self._model_changed()
            if self.state.model_params:
                self.param_form.set_values(self.state.model_params)
        if self.state.evaluation is not None:
            self._show_results()


# --------------------------------------------------------------------------
# 6. tune
# --------------------------------------------------------------------------

class TunePage(StepPage):
    title = "6 · Tune"
    requires = "data"
    subtitle = ("Search a grid of hyperparameters with the same cross-validation the rest of the "
                "workbench uses.  The winner is refitted and becomes the current model.")

    def build(self):
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)

        self.grid_edit = QPlainTextEdit()
        self.grid_edit.setMinimumHeight(120)
        theme.tag(self.grid_edit, "mono")

        load_grid = QPushButton("Load the suggested grid")
        load_grid.clicked.connect(self._load_grid)
        check = QPushButton("Check the grid")
        check.clicked.connect(self._check_grid)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Exhaustive grid", "grid")
        self.method_combo.addItem("Random sample of the grid", "random")

        self.metric_combo = QComboBox()
        self.folds_spin = FieldForm([
            {"name": "n_iter", "label": "random samples (if sampling)", "kind": "int", "default": 30,
             "choices": [], "optional": False, "help": ""},
            {"name": "cv_folds", "label": "folds", "kind": "int", "default": 5, "choices": [],
             "optional": False, "help": ""},
            {"name": "max_candidates", "label": "candidate budget", "kind": "int", "default": 60,
             "choices": [], "optional": False, "help": "a forgotten 4-parameter grid can explode"},
            {"name": "random_state", "label": "random seed", "kind": "int", "default": 42,
             "choices": [], "optional": False, "help": ""},
        ])

        settings = QGroupBox("Search")
        settings_layout = QVBoxLayout(settings)
        settings_layout.addWidget(QLabel("model:"))
        settings_layout.addWidget(self.model_combo)
        settings_layout.addWidget(QLabel("search method:"))
        settings_layout.addWidget(self.method_combo)
        settings_layout.addWidget(QLabel("optimise this metric:"))
        settings_layout.addWidget(self.metric_combo)
        settings_layout.addWidget(self.folds_spin)
        settings_layout.addWidget(QLabel("grid — one parameter per line, values comma separated:"))
        settings_layout.addWidget(self.grid_edit)
        row = QHBoxLayout()
        row.addWidget(load_grid)
        row.addWidget(check)
        row.addStretch(1)
        settings_layout.addLayout(row)

        self.run_button = QPushButton("Run the search")
        theme.tag(self.run_button, "primary")
        self.run_button.clicked.connect(self._run)
        adopt = QPushButton("Use the best parameters on the Model screen")
        adopt.clicked.connect(self._adopt)
        actions = QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(adopt)
        actions.addStretch(1)

        self.cards = MetricCards()
        self.best_view = KeyValueView()
        self.results_table = DataFrameView(max_rows=300)
        self.failures = QPlainTextEdit()
        self.failures.setReadOnly(True)
        self.failures.setMaximumHeight(90)
        theme.tag(self.failures, "mono")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(settings)
        left_layout.addLayout(actions)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cards)
        right_layout.addWidget(section_title("Best candidate"))
        right_layout.addWidget(self.best_view)
        right_layout.addWidget(section_title("Every candidate"))
        right_layout.addWidget(self.results_table, 1)
        right_layout.addWidget(self.failures)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

    # ------------------------------------------------------------------ form
    def _task(self) -> str:
        return self.state.train_spec.task or "regression"

    def _model_changed(self):
        key = self.model_combo.currentData()
        if key:
            self._load_grid(quiet=True)

    def _load_grid(self, quiet: bool = False):
        key = self.model_combo.currentData()
        if not key:
            return
        grid = MODELS.suggested_grid(key) or MODELS.param_default_grid(key)
        if not grid:
            if not quiet:
                QMessageBox.information(self, "No grid", "This model has no suggested grid — write one.")
            return
        self.grid_edit.setPlainText(tuning_mod.grid_to_text(grid))

    def _check_grid(self):
        grid, problems = tuning_mod.parse_grid_text(self.grid_edit.toPlainText())
        if problems:
            QMessageBox.warning(self, "Grid problems", "\n".join(problems))
        else:
            total = int(np.prod([len(v) for v in grid.values()])) if grid else 1
            QMessageBox.information(self, "Grid looks fine",
                                    f"{len(grid)} parameter(s), {total:,} combination(s).")
        return grid

    def _run(self):
        if not self.require_data():
            return
        key = self.model_combo.currentData()
        if not key:
            return
        if self.state.df is None or not self.state.train_spec.target:
            self.warn("No target", "Choose a target on the Model screen first.")
            return
        grid, problems = tuning_mod.parse_grid_text(self.grid_edit.toPlainText())
        if problems:
            QMessageBox.warning(self, "Grid problems", "\n".join(problems))
        if not grid:
            if QMessageBox.question(self, "Empty grid",
                                    "No parameters are listed, so every candidate is identical. "
                                    "Run it anyway?") != QMessageBox.StandardButton.Yes:
                return
        values = self.folds_spin.values()
        tune_spec = {
            "method": self.method_combo.currentData(),
            "n_iter": int(values.get("n_iter") or 30),
            "cv_folds": int(values.get("cv_folds") or 5),
            "random_state": int(values.get("random_state") or 42),
            "metric": self.metric_combo.currentData() or M.primary_metric(self._task()),
            "max_candidates": int(values.get("max_candidates") or 60),
            "param_grid": grid,
        }
        spec = self.state.train_spec
        self.run_button.setEnabled(False)
        self.state.log_line(f"tuning {MODELS.get_model_spec(key).label} over "
                            f"{len(grid)} parameter(s), method {tune_spec['method']}")
        self.run_task(tune_job, f"Tuning {MODELS.get_model_spec(key).label}",
                      on_done=self._tuned, on_cancelled=lambda: self.run_button.setEnabled(True),
                      df=self.state.df, spec_dict=spec.to_dict(),
                      pre_dict=self.state.pre_spec.to_dict(), model_key=key,
                      grid=grid, tune_spec_dict=tune_spec, adopt=True)

    def _tuned(self, result):
        self.run_button.setEnabled(True)
        state = self.state
        state.tuning = result
        self.results_table.set_frame(result.get("results"))
        best = {f"param {k}": v for k, v in (result.get("best_params") or {}).items()}
        best["score"] = result.get("best_score")
        best["metric"] = result.get("metric_label")
        best["candidates scored"] = result.get("n_scored")
        best["candidates tried"] = result.get("n_candidates")
        best["seconds"] = round(result.get("elapsed") or 0, 2)
        if result.get("note"):
            best["note"] = result["note"]
        self.best_view.set_pairs(list(best.items()))
        self.cards.set_metrics([
            (result.get("metric_label") or "score", result.get("best_score"), "best candidate"),
            ("candidates", f"{result.get('n_scored')} / {result.get('n_candidates')}", "scored"),
            ("seconds", round(result.get("elapsed") or 0, 1), ""),
        ])
        self.failures.setPlainText("\n".join(result.get("failures") or []))

        if result.get("best_estimator") is not None and result.get("evaluation"):
            state.split = result.get("split")
            state.pipeline = result["best_estimator"]
            state.evaluation = result["evaluation"]
            state.cv = result.get("best_cv")
            state.importance = result.get("importance")
            state.prepared = result.get("prepared")
            state.model_key = result.get("model_key")
            state.model_params = dict(result.get("best_params") or {})
            score = result.get("best_score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            state.log_line(f"tuned model adopted: "
                           f"{MODELS.describe_params(state.model_key, state.model_params)} "
                           f"({result.get('metric_label')} = {score_text})")
        state.touch()

    def _adopt(self):
        result = self.state.tuning
        if not result or not result.get("best_params"):
            self.warn("Nothing to adopt", "Run a search first.")
            return
        self.state.model_key = result.get("model_key") or self.state.model_key
        self.state.model_params = dict(result["best_params"])
        self.state.note("best parameters copied to the Model screen")
        if result.get("best_estimator") is not None:
            self.state.pipeline = result["best_estimator"]
            self.state.evaluation = result.get("evaluation")
            self.state.cv = result.get("best_cv")
            self.state.split = result.get("split")
            self.state.touch()

    # --------------------------------------------------------------- refresh
    def refresh(self):
        token = (self.state.train_spec.task, self.state.model_key)
        if token == getattr(self, "_token", None):
            return
        self._token = token
        task = self._task()
        keys = [c[1] for c in metric_choices(task)]
        current = self.metric_combo.currentData()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for label, key in metric_choices(task):
            self.metric_combo.addItem(label, key)
        if current in keys:
            self.metric_combo.setCurrentIndex(keys.index(current))
        self.metric_combo.blockSignals(False)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for label, key, available in model_choices(task):
            self.model_combo.addItem(label, key)
            if not available:
                item = self.model_combo.model().item(self.model_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        if self.state.model_key:
            index = self.model_combo.findData(self.state.model_key)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._load_grid(quiet=True)


# --------------------------------------------------------------------------
# 7. evaluate
# --------------------------------------------------------------------------

class EvaluatePage(StepPage):
    title = "7 · Evaluate"
    requires = "model"
    subtitle = ("Every number the model produced: held-out metrics, training metrics, the "
                "overfitting gap, per-fold scores, confusion matrix or residual summary, and "
                "importances.")

    VIEWS = [
        ("Held-out test metrics", "test"),
        ("Training metrics", "train"),
        ("Cross-validation folds", "folds"),
        ("Confusion matrix", "confusion"),
        ("Per-class detail", "report"),
        ("Residual summary", "residual_summary"),
        ("Residual table", "residual_table"),
        ("Feature importances", "importance"),
        ("Coefficients", "coefficients"),
        ("Prediction table", "predictions"),
    ]

    def build(self):
        self.view_combo = QComboBox()
        for label, key in self.VIEWS:
            self.view_combo.addItem(label, key)
        show = QPushButton("Show")
        show.clicked.connect(self._show)
        refresh = QPushButton("Recompute from the saved model")
        refresh.clicked.connect(self._recompute)
        copy_button = QPushButton("Copy the table")
        copy_button.clicked.connect(lambda: self.table.copy_to_clipboard())

        controls = QHBoxLayout()
        controls.addWidget(self.view_combo)
        controls.addWidget(show)
        controls.addWidget(refresh)
        controls.addStretch(1)
        controls.addWidget(copy_button)

        self.cards = MetricCards()
        self.gap_label = QLabel()
        self.gap_label.setWordWrap(True)
        self.table = DataFrameView(max_rows=2000)

        self.body.addLayout(controls)
        self.body.addWidget(self.cards)
        self.body.addWidget(self.gap_label)
        self.body.addWidget(self.table, 1)

    # ------------------------------------------------------------------ views
    def _show(self, quiet: bool = False):
        if not self.state.has_model:
            if not quiet:
                self.require_model()
            return
        state = self.state
        key = self.view_combo.currentData()
        try:
            frame = self._frame_for(key)
        except Exception as exc:  # noqa: BLE001
            if quiet:
                return
            QMessageBox.critical(self, "Could not build that view", f"{type(exc).__name__}: {exc}")
            return
        if frame is None:
            if quiet:
                return
            QMessageBox.information(self, "Not applicable",
                                    "That view does not apply to this task or model.")
            return
        self.table.set_frame(frame)

    def _frame_for(self, key: str):
        state = self.state
        evaluation = state.evaluation or {}
        task = state.train_spec.task
        classification = task in ("binary", "multiclass")
        if key in ("confusion", "report") and not classification:
            return None
        if key in ("residual_summary", "residual_table") and task != "regression":
            return None
        if key == "test":
            return M.metric_frame(task, evaluation.get("test") or {})
        if key == "train":
            return M.metric_frame(task, evaluation.get("train") or {})
        if key == "folds":
            folds = (state.cv or {}).get("folds") or []
            if not folds:
                return pd.DataFrame({"note": ["no cross-validation was run"]})
            rows = []
            for i, fold in enumerate(folds, 1):
                row = {"fold": i}
                for metric_key, value in fold.items():
                    if isinstance(value, (int, float)):
                        row[M.label(task, metric_key)] = value
                rows.append(row)
            return pd.DataFrame(rows)
        if key == "confusion":
            if evaluation.get("test_truth") is None:
                return None
            return M.confusion(evaluation["test_truth"], evaluation["test_pred"],
                               (state.bundle or {}).get("labels") or state.split.labels)
        if key == "report":
            if evaluation.get("test_truth") is None:
                return None
            return M.classification_report(evaluation["test_truth"], evaluation["test_pred"])
        if key == "residual_summary":
            if evaluation.get("test_truth") is None:
                return None
            frame = M.regression_residuals(evaluation["test_truth"], evaluation["test_pred"])
            return frame.describe().T.reset_index(names="statistic")
        if key == "residual_table":
            if evaluation.get("test_truth") is None:
                return None
            return M.regression_residuals(evaluation["test_truth"], evaluation["test_pred"]).head(1000)
        if key == "importance":
            if state.importance is None:
                return None
            return state.importance
        if key == "coefficients":
            if state.pipeline is None:
                return None
            return explain.coefficients(state.pipeline)
        if key == "predictions":
            if evaluation.get("test_pred") is None:
                return None
            frame = pd.DataFrame({
                "actual": evaluation.get("test_truth"),
                "predicted": evaluation.get("test_pred"),
            })
            proba = evaluation.get("test_proba")
            if proba is not None and getattr(proba, "ndim", 1) == 2:
                for i in range(proba.shape[1]):
                    frame[f"probability {i}"] = proba[:, i]
            return frame
        return None

    def _recompute(self):
        """Re-score the stored pipeline — useful after reopening a project."""
        if not self.require_model():
            return
        state = self.state
        if state.split is None:
            try:
                state.make_split()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Could not rebuild the split", str(exc))
                return
        state.evaluation = train_mod.evaluate_fitted(state.pipeline, state.split, state.train_spec)
        if state.train_spec.task != "unsupervised":
            try:
                state.cv = train_mod.cross_validate_pipeline(
                    state.pipeline, state.split.X_train, state.split.y_train, state.train_spec,
                    weights=state.split.w_train)
            except Exception:  # noqa: BLE001
                pass
            try:
                state.importance = explain.importance_frame(state.pipeline, state.split.X_train,
                                                            state.split.y_train, state.train_spec.task)
            except Exception:  # noqa: BLE001
                pass
        state.note("metrics recomputed from the stored pipeline")
        self.refresh()

    # --------------------------------------------------------------- refresh
    def refresh(self):
        state = self.state
        evaluation = state.evaluation or {}
        task = state.train_spec.task
        primary = state.train_spec.primary_metric
        test = evaluation.get("test") or {}
        if not state.has_model:
            self.cards.set_metrics([("model", "not trained yet", "use the Model screen")])
            self.gap_label.setText("")
            return
        cards = [
            (M.label(task, primary), test.get(primary), "test set"),
            ("cross-validation", ((state.cv or {}).get("primary") or {}).get("mean"),
             "mean over folds"),
            ("features", len(state.split.feature_names) if state.split else None, "after preparation"),
            ("rows", f"{state.split.n_train:,} / {state.split.n_test:,}" if state.split else None,
             "train / test"),
        ]
        self.cards.set_metrics(cards)
        gap = evaluation.get("overfit_gap")
        if gap is None:
            self.gap_label.setText("")
        else:
            direction = "worse" if float(gap) > 0 else "better"
            verdict = ("that gap is large — the model is probably memorising the training rows"
                       if abs(float(gap)) > 0.1 else "that gap is small, which is what you want")
            self.gap_label.setText(f"The primary metric is {abs(float(gap)):.4f} {direction} on the test "
                                   f"set than on the training set — {verdict}.")
        self._show(quiet=True)


# --------------------------------------------------------------------------
# 8. save & export
# --------------------------------------------------------------------------

class SavePage(StepPage):
    title = "8 · Save & export"
    requires = "model"
    subtitle = ("The saved bundle holds the whole pipeline, so a prediction later goes through "
                "exactly the arithmetic that produced these numbers.")

    def build(self):
        # ---- bundle
        self.bundle_edit = QLineEdit()
        default_bundle = os.path.join(self.state.project_dir, "model.joblib")
        self.bundle_edit.setText(default_bundle)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(self.bundle_edit, "model.joblib",
                                                         "Model bundle (*.joblib)"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setMaximumHeight(70)
        self.notes_edit.setPlaceholderText("notes to store inside the bundle and the report…")
        self.save_button = QPushButton("Save the model")
        theme.tag(self.save_button, "primary")
        self.save_button.clicked.connect(self._save)
        load_button = QPushButton("Load a saved model…")
        load_button.clicked.connect(self._load)

        bundle_row = QHBoxLayout()
        bundle_row.addWidget(self.bundle_edit, 1)
        bundle_row.addWidget(browse)
        bundle_box = QGroupBox("Model bundle")
        bundle_layout = QVBoxLayout(bundle_box)
        bundle_layout.addLayout(bundle_row)
        bundle_layout.addWidget(self.notes_edit)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(load_button)
        buttons.addStretch(1)
        bundle_layout.addLayout(buttons)

        # ---- report
        self.report_edit = QLineEdit(os.path.join(self.state.project_dir, "report.html"))
        report_browse = QPushButton("Browse…")
        report_browse.clicked.connect(lambda: self._browse_into(self.report_edit, "report.html",
                                                                "HTML report (*.html)"))
        self.title_edit = QLineEdit("Machine learning report")
        self.figure_choice = QComboBox()
        self.figure_choice.addItem("Figures queued on the Plots screen", "queued")
        self.figure_choice.addItem("A sensible default set", "default")
        self.figure_choice.addItem("No figures (tables only)", "none")
        self.report_button = QPushButton("Write the HTML report")
        theme.tag(self.report_button, "primary")
        self.report_button.clicked.connect(self._report)
        self.excel_button = QPushButton("Export the tables to Excel…")
        self.excel_button.clicked.connect(self._excel)
        self.word_button = QPushButton("Export a Word document…")
        self.word_button.clicked.connect(self._word)
        report_row = QHBoxLayout()
        report_row.addWidget(self.report_edit, 1)
        report_row.addWidget(report_browse)
        office_row = QHBoxLayout()
        office_row.addWidget(self.excel_button)
        office_row.addWidget(self.word_button)
        office_row.addStretch(1)
        report_box = QGroupBox("Report")
        report_layout = QVBoxLayout(report_box)
        report_layout.addWidget(QLabel("title:"))
        report_layout.addWidget(self.title_edit)
        report_layout.addWidget(QLabel("figures:"))
        report_layout.addWidget(self.figure_choice)
        report_layout.addLayout(report_row)
        report_layout.addWidget(self.report_button)
        report_layout.addWidget(hint("Excel gets every table as its own sheet; Word gets the "
                                     "same story with the figures in place, for a client who "
                                     "does not want a browser."))
        report_layout.addLayout(office_row)

        # ---- script
        self.script_edit = QLineEdit(os.path.join(self.state.project_dir, "reproduce_model.py"))
        script_browse = QPushButton("Browse…")
        script_browse.clicked.connect(lambda: self._browse_into(self.script_edit, "reproduce_model.py",
                                                                "Python script (*.py)"))
        self.script_button = QPushButton("Export a reproduction script")
        self.script_button.clicked.connect(self._script)
        script_row = QHBoxLayout()
        script_row.addWidget(self.script_edit, 1)
        script_row.addWidget(script_browse)
        script_box = QGroupBox("Reproducible script")
        script_layout = QVBoxLayout(script_box)
        script_layout.addLayout(script_row)
        script_layout.addWidget(self.script_button)
        script_layout.addWidget(hint("The script imports this workbench's own core package, so its "
                                     "numbers match the report exactly."))

        # ---- project
        project_box = QGroupBox("Project file")
        project_layout = QVBoxLayout(project_box)
        save_project = QPushButton("Save a .mlproj project…")
        save_project.clicked.connect(lambda: self.window.save_project() if self.window else None)
        open_project = QPushButton("Open a .mlproj project…")
        open_project.clicked.connect(lambda: self.window.open_project() if self.window else None)
        project_buttons = QHBoxLayout()
        project_buttons.addWidget(save_project)
        project_buttons.addWidget(open_project)
        project_buttons.addStretch(1)
        project_layout.addLayout(project_buttons)
        project_layout.addWidget(hint("A project file stores the recipe — data source, cleaning steps, "
                                      "model and hyperparameters — not the data itself."))

        self.summary = DataFrameView(max_rows=60)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(bundle_box)
        left_layout.addWidget(report_box)
        left_layout.addWidget(script_box)
        left_layout.addWidget(project_box)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("What is inside the bundle"))
        right_layout.addWidget(self.summary, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.body.addWidget(splitter, 1)

    # ------------------------------------------------------------------ misc
    def _browse_into(self, edit: QLineEdit, default_name: str, filter_text: str):
        path, _ = QFileDialog.getSaveFileName(self, "Choose a file", edit.text() or default_name,
                                              filter_text)
        if path:
            edit.setText(path)

    def make_bundle(self) -> dict | None:
        state = self.state
        if not state.has_model or state.split is None:
            self.warn("Nothing to save", "Train a model first.")
            return None
        return train_mod.make_bundle(
            state.pipeline, state.train_spec, state.split, state.pre_spec,
            state.model_key, state.model_params, state.evaluation or {},
            cv=state.cv, tuning=state.tuning,
            source_file=state.source_file, cleaning_ops=state.clean_ops,
            notes=self.notes_edit.toPlainText().strip() or getattr(state, "notes", ""),
            feature_importances=self._importances(),
        )

    def _importances(self) -> dict:
        frame = self.state.importance
        if frame is None or not len(frame) or "feature" not in frame.columns:
            return {}
        column = "importance" if "importance" in frame.columns else frame.columns[1]
        return dict(zip(frame["feature"].astype(str), frame[column].astype(float)))

    # ------------------------------------------------------------------ save
    def _save(self):
        bundle = self.make_bundle()
        if bundle is None:
            return
        path = self.bundle_edit.text().strip()
        if not path:
            self.warn("No path", "Choose where to write the model.")
            return
        try:
            train_mod.save_bundle(bundle, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save", f"{type(exc).__name__}: {exc}")
            return
        self.state.bundle = bundle
        self.state.bundle_path = path
        size = os.path.getsize(path) / 1e6
        self.state.log_line(f"saved the model bundle to {path} ({size:.2f} MB)")
        self.summary.set_frame(train_mod.bundle_summary(bundle))
        QMessageBox.information(self, "Saved", f"Wrote\n{path}\n\n{size:.2f} MB")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a model bundle", "",
                                              "Model bundle (*.joblib);;All files (*)")
        if not path:
            return
        self.run_task(load_bundle_job, f"Loading {os.path.basename(path)}",
                      on_done=lambda bundle: self._loaded(bundle, path), path=path)

    def _loaded(self, bundle, path):
        state = self.state
        state.bundle = bundle
        state.bundle_path = path
        state.pipeline = bundle.get("pipeline")
        state.model_key = bundle.get("model_key")
        state.model_params = dict(bundle.get("model_params") or {})
        state.evaluation = {"test": bundle.get("metrics", {}).get("test") or {},
                            "train": bundle.get("metrics", {}).get("train") or {},
                            "overfit_gap": bundle.get("overfit_gap")}
        state.cv = bundle.get("cv")
        importances = bundle.get("feature_importances") or {}
        state.importance = (pd.DataFrame({"feature": list(importances),
                                          "importance": list(importances.values())})
                            if importances else None)
        state.log_line(f"loaded a saved model from {path} "
                       f"({bundle.get('model_label')}, task {bundle.get('task')})")
        for message in bundle.get("_warnings") or []:
            state.log_line(f"warning: {message}")
        self.summary.set_frame(train_mod.bundle_summary(bundle))
        self.bundle_edit.setText(path)
        state.touch()

    # ---------------------------------------------------------------- report
    def _report(self):
        if not self.require_model():
            return
        bundle = self.state.bundle or self.make_bundle()
        if bundle is None:
            return
        path = self.report_edit.text().strip()
        if not path:
            self.warn("No path", "Choose where to write the report.")
            return
        choice = self.figure_choice.currentData()
        if choice == "queued":
            requests = [(view.get("kind"), view.get("options") or {}) for view in self.state.plot_views]
            if not requests:
                requests = default_plot_requests(self.state.plot_context(), self.state.train_spec.task,
                                                 self.state.train_spec.target)
        elif choice == "default":
            requests = default_plot_requests(self.state.plot_context(), self.state.train_spec.task,
                                             self.state.train_spec.target)
        else:
            requests = []
        extra = self._extra_sections()
        self.report_button.setEnabled(False)
        self.run_task(report_job, "Writing the report", on_done=self._reported,
                      on_cancelled=lambda: self.report_button.setEnabled(True),
                      bundle=bundle, context=self.state.plot_context(), plot_requests=requests,
                      title=self.title_edit.text().strip() or "Machine learning report",
                      path=path, extra_sections=extra,
                      extra_figures=list(self.state.extra_figures))

    def _reported(self, result):
        self.report_button.setEnabled(True)
        self.state.report_path = result["path"]
        self.state.log_line(f"report written to {result['path']} "
                            f"({result['bytes'] / 1024:.0f} KB, {len(result['warnings'])} warning(s))")
        for message in result["warnings"]:
            self.state.log_line(f"warning: {message}")

    def _extra_sections(self) -> dict:
        sections = {}
        tuning = self.state.tuning
        if isinstance(tuning, dict) and tuning.get("results") is not None and len(tuning["results"]):
            sections["Hyperparameter search"] = tuning["results"]
        selection = self.state.selection
        if selection is not None and getattr(selection, "ranking", None) is not None \
                and len(selection.ranking):
            sections[f"Feature selection ({selection.method})"] = selection.ranking
        design = self.state.design
        if design is not None and getattr(design, "table", None) is not None and len(design.table):
            sections["Inverse design shortlist"] = design.table
        if self.state.clusters is not None:
            counts = pd.Series(self.state.clusters).value_counts().sort_index()
            sections["Cluster sizes"] = pd.DataFrame({"cluster": counts.index, "rows": counts.values})
        # Anything the engineering screens computed — a test, a fitted curve, an experiment,
        # a control chart — travels into the report through this one door, so no screen has
        # to know how the report is assembled.
        for heading, table in (self.state.extra_sections or {}).items():
            if isinstance(table, pd.DataFrame) and len(table):
                sections[heading] = table
        return sections

    # ----------------------------------------------------------------- office
    def _tables_for_office(self) -> dict:
        """Every table the project currently holds, keyed by sheet name."""
        state = self.state
        tables: dict = {"Summary": self.summary.current_frame()}
        evaluation = state.evaluation or {}
        for split in ("train", "test"):
            metrics = evaluation.get(split) or {}
            if metrics:
                tables[f"{split.title()} metrics"] = pd.DataFrame(
                    [{"metric": k, "value": v} for k, v in metrics.items()])
        if state.importance is not None and len(state.importance):
            tables["Feature importance"] = state.importance
        if state.cv:
            folds = state.cv.get("folds")
            if isinstance(folds, pd.DataFrame) and len(folds):
                tables["Cross-validation"] = folds
        tuning = state.tuning
        if isinstance(tuning, dict) and isinstance(tuning.get("results"), pd.DataFrame) \
                and len(tuning["results"]):
            tables["Hyperparameter search"] = tuning["results"]
        selection = state.selection
        if selection is not None and getattr(selection, "ranking", None) is not None \
                and len(selection.ranking):
            tables["Feature selection"] = selection.ranking
        design = state.design
        if design is not None and getattr(design, "table", None) is not None and len(design.table):
            tables["Inverse design"] = design.table
        if getattr(state, "curve_table", None) is not None and len(state.curve_table):
            tables["Curve fits"] = state.curve_table
        fit = getattr(state, "curve_fit", None)
        if fit is not None and getattr(fit, "converged", False):
            tables["Fitted coefficients"] = fit.parameters()
        doe = getattr(state, "doe", None)
        if doe is not None:
            tables["Experiment design"] = doe.worksheet()
        return tables

    def _office_path(self, default_name: str, caption: str, pattern: str) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self, caption, os.path.join(self.state.project_dir, default_name), pattern)
        return path

    def _excel(self):
        path = self._office_path("workbench_tables.xlsx", "Export the tables to Excel",
                                 "Excel workbook (*.xlsx)")
        if not path:
            return
        from core import export_office as EO

        tables = self._tables_for_office()
        for heading, table in (self.state.extra_sections or {}).items():
            if isinstance(table, pd.DataFrame) and len(table):
                tables[heading] = table
        notes = [self.notes_edit.toPlainText().strip(),
                 f"model: {self.state.model_key} ({self.state.model_params})",
                 f"target: {self.state.train_spec.target} · task: {self.state.train_spec.task}"]
        try:
            EO.write_excel(path, tables, notes=notes,
                           title=self.title_edit.text().strip() or "ML Workbench")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not write the workbook",
                                 f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"exported {len(tables)} table(s) to {path}")
        QMessageBox.information(self, "Exported",
                                f"Wrote {len(tables)} sheet(s) to\n{path}")

    def _word_blocks(self) -> list:
        """The document as (kind, …) tuples — the shape core.export_office.write_docx takes."""
        blocks: list = []
        subtitle = self._word_subtitle()
        if subtitle:
            blocks.append(("paragraph", subtitle))
        notes = self.notes_edit.toPlainText().strip()
        if notes:
            blocks.append(("paragraph", notes))
        summary = self.summary.current_frame()
        if isinstance(summary, pd.DataFrame) and len(summary):
            blocks.append(("heading", 2, "Model summary"))
            blocks.append(("table", summary.head(40)))
        for heading, table in (self.state.extra_sections or {}).items():
            if isinstance(table, pd.DataFrame) and len(table):
                blocks.append(("heading", 2, heading))
                blocks.append(("table", table.head(60)))
        for caption, png in (self.state.extra_figures or []):
            if not png:
                continue
            blocks.append(("image", png, caption))
        return blocks

    def _word(self):
        path = self._office_path("workbench_report.docx", "Export a Word document",
                                 "Word document (*.docx)")
        if not path:
            return
        from core import export_office as EO

        blocks = self._word_blocks()
        try:
            EO.write_docx(path, blocks, title=self.title_edit.text().strip() or "ML Workbench")
            report = EO.verify_docx(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not write the document",
                                 f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"exported a Word document to {path}")
        problems = report.get("problems") or []
        QMessageBox.information(
            self, "Exported",
            f"Wrote\n{path}\n\n{len(blocks)} block(s), "
            f"{len(self.state.extra_figures or [])} figure(s).\n"
            + ("Word-format check: ok" if not problems
               else "Word-format problems: " + "; ".join(problems[:3])))

    def _word_subtitle(self) -> str:
        state = self.state
        bits = [f"model: {state.model_key}" if state.model_key else "no model"]
        if state.train_spec.target:
            bits.append(f"target: {state.train_spec.target}")
        bits.append(f"task: {state.train_spec.task}")
        evaluation = state.evaluation or {}
        primary = state.train_spec.primary_metric
        score = (evaluation.get("test") or {}).get(primary)
        if isinstance(score, (int, float)):
            bits.append(f"{primary} = {score:.4f}")
        return " · ".join(str(b) for b in bits)

    # ---------------------------------------------------------------- script
    def _script(self):
        from core import export_script

        if not self.require_model():
            return
        path = self.script_edit.text().strip()
        if not path:
            self.warn("No path", "Choose where to write the script.")
            return
        try:
            export_script.write_script(
                path, model_key=self.state.model_key, model_params=self.state.model_params,
                train_spec=self.state.train_spec.to_dict(),
                preprocess_spec=self.state.pre_spec.to_dict(),
                cleaning_ops=self.state.clean_ops, source_file=self.state.source_file,
                source_format=self.state.source_format, source_table=self.state.source_table,
                notes=self.notes_edit.toPlainText().strip() or getattr(self.state, "notes", ""),
                workbench_root=self.state.project_dir,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not export", f"{type(exc).__name__}: {exc}")
            return
        self.state.script_path = path
        self.state.log_line(f"reproduction script written to {path}")
        QMessageBox.information(self, "Exported",
                                f"Wrote\n{path}\n\nRun it with the workbench interpreter:\n"
                                f"{self.state.project_dir}/.venv/bin/python {path}")

    # --------------------------------------------------------------- refresh
    def refresh(self):
        if self.state.bundle is not None:
            self.summary.set_frame(train_mod.bundle_summary(self.state.bundle))
        elif self.state.has_model:
            self.summary.set_frame(pd.DataFrame({
                "field": ["state"],
                "value": ["a model is trained but has not been saved yet"],
            }))


# --------------------------------------------------------------------------
# 9. predict
# --------------------------------------------------------------------------

class PredictPage(StepPage):
    title = "9 · Predict"
    requires = "model"
    subtitle = ("Feed new rows through the saved pipeline.  Missing values and unseen categories "
                "are handled the same way they were at training time.")

    def build(self):
        self.source_combo = QComboBox()
        self.source_combo.addItem("Type the values in by hand", "manual")
        self.source_combo.addItem("Load a file", "file")
        self.source_combo.addItem("Use the held-out test rows", "test")
        self.source_combo.currentIndexChanged.connect(self._source_changed)

        self.path_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.format_combo = QComboBox()
        for spec in loaders.available_formats():
            self.format_combo.addItem(spec.label, spec.key)
            if not spec.available:
                item = self.format_combo.model().item(self.format_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        self.load_button = QPushButton("Load these rows")
        self.load_button.clicked.connect(self._load_file)

        self.file_row = QWidget()
        file_row = QHBoxLayout(self.file_row)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse)
        file_row.addWidget(self.format_combo)
        file_row.addWidget(self.load_button)

        self.manual_form = FieldForm()
        self.add_row_button = QPushButton("Add this row")
        self.add_row_button.clicked.connect(self._add_manual_row)
        fill_test = QPushButton("Fill with a test row")
        fill_test.clicked.connect(self._fill_from_test)
        self.manual_buttons = QWidget()
        manual_buttons = QHBoxLayout(self.manual_buttons)
        manual_buttons.setContentsMargins(0, 0, 0, 0)
        manual_buttons.addWidget(self.add_row_button)
        manual_buttons.addWidget(fill_test)
        manual_buttons.addStretch(1)

        self.input_table = DataFrameView(max_rows=500)
        clear = QPushButton("Clear the input rows")
        clear.clicked.connect(self._clear_input)
        self.predict_button = QPushButton("Predict")
        theme.tag(self.predict_button, "primary")
        self.predict_button.clicked.connect(self._predict)
        self.export_button = QPushButton("Export the predictions…")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)

        actions = QHBoxLayout()
        actions.addWidget(self.predict_button)
        actions.addWidget(self.export_button)
        actions.addStretch(1)
        actions.addWidget(clear)

        source_box = QGroupBox("Where do the new rows come from?")
        source_layout = QVBoxLayout(source_box)
        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.file_row)
        source_layout.addWidget(self.manual_form)
        source_layout.addWidget(self.manual_buttons)

        self.pending = pd.DataFrame()
        self.result = None

        self.body.addWidget(source_box)
        self.body.addWidget(section_title("Rows waiting to be scored"))
        self.body.addWidget(self.input_table, 1)
        self.body.addLayout(actions)
        self.body.addWidget(section_title("Predictions"))
        self.output_table = DataFrameView(max_rows=500)
        self.body.addWidget(self.output_table, 1)

        self._source_changed()

    # ------------------------------------------------------------------ input
    def _source_changed(self):
        kind = self.source_combo.currentData()
        self.file_row.setVisible(kind == "file")
        self.manual_form.setVisible(kind == "manual")
        self.manual_buttons.setVisible(kind == "manual")
        if kind == "test":
            self._fill_from_test()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a data file", "", "All files (*)")
        if path:
            self.path_edit.setText(path)

    def _load_file(self):
        path = self.path_edit.text().strip()
        if not path:
            self.warn("No file", "Choose a file first.")
            return
        from core import loaders

        fmt = self.format_combo.currentData()
        try:
            tables = loaders.read_all(path, fmt=fmt)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read that file", f"{type(exc).__name__}: {exc}")
            return
        frame = next(iter(tables.values()))
        self.pending = frame
        self.input_table.set_frame(frame)
        self.state.log_line(f"loaded {len(frame):,} new rows from {os.path.basename(path)} for prediction")

    def _fill_from_test(self):
        split = self.state.split
        if split is None or not len(split.X_test):
            self.warn("No held-out rows", "Train a model first — its test split is the natural "
                                          "source of 'new' rows.")
            return
        self.pending = split.X_test.head(10).copy()
        self.input_table.set_frame(self.pending, note="held-out rows the model never saw")

    def _add_manual_row(self):
        values = self.manual_form.values()
        row = {k: v for k, v in values.items() if v not in (None, "")}
        if not row:
            self.warn("Empty row", "Fill in at least one field.")
            return
        self.pending = pd.concat([self.pending, pd.DataFrame([row])], ignore_index=True)
        self.input_table.set_frame(self.pending)

    def _clear_input(self):
        self.pending = pd.DataFrame()
        self.result = None
        self.input_table.set_frame(None)
        self.output_table.set_frame(None)
        self.export_button.setEnabled(False)

    # ---------------------------------------------------------------- predict
    def _predict(self):
        if not self.require_model():
            return
        frame = self.pending
        if frame is None or not len(frame):
            self.warn("Nothing to predict", "Add some rows first.")
            return
        self.predict_button.setEnabled(False)
        self.run_task(predict_job, "Predicting", on_done=self._predicted,
                      on_cancelled=lambda: self.predict_button.setEnabled(True),
                      pipe=self.state.pipeline, frame=frame, task=self.state.train_spec.task)

    def _predicted(self, result):
        self.predict_button.setEnabled(True)
        self.result = result
        self.output_table.set_frame(result)
        self.export_button.setEnabled(True)
        self.state.log_line(f"predicted {len(result):,} row(s)")

    def _export(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the predictions", "predictions.csv",
                                              "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            if path.lower().endswith(".xlsx"):
                self.result.to_excel(path, index=False)
            else:
                self.result.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not write", f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"predictions written to {path}")

    # --------------------------------------------------------------- refresh
    def refresh(self):
        token = (self.state.model_key, self.data_token())
        if token == getattr(self, "_token", None):
            return
        self._token = token
        split = self.state.split
        if split is None:
            return
        fields = []
        for column in split.feature_names:
            frame = self.state.df
            if frame is not None and column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column]):
                values = [str(v) for v in frame[column].dropna().unique()[:40]]
                fields.append({"name": column, "label": column, "kind": "choice", "default": None,
                               "choices": values, "optional": True, "help": ""})
            else:
                fields.append({"name": column, "label": column, "kind": "text", "default": None,
                               "choices": [], "optional": True, "help": ""})
        self.manual_form.set_fields(fields)
