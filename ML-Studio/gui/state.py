"""Shared application state.

One object holds everything the screens read and write: the loaded frame, the
cleaning steps, the split, the fitted pipeline, the metrics, and the log.  Steps
never talk to each other directly — they mutate this state and emit ``changed``,
so a screen can be added or reordered without touching the others.
"""
from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from core import clean as clean_mod
from core import eda, plots, preprocess, session as session_mod, train


class AppState(QObject):
    changed = pyqtSignal()
    logged = pyqtSignal(str)
    features_changed = pyqtSignal(list)
    preprocess_changed = pyqtSignal()

    def __init__(self, project_dir: str):
        super().__init__()
        self.project_dir = project_dir

        # source
        self.source_file: str = ""
        self.source_format: str | None = None
        self.source_table: str | None = None
        self.source_options: dict = {}
        self.raw_df: pd.DataFrame | None = None

        # cleaning
        self.clean_ops: list = []
        self.clean_log: list = []
        self.df: pd.DataFrame | None = None

        # modelling
        self.train_spec = train.TrainSpec()
        self.pre_spec = preprocess.PreprocessSpec()
        self.split: train.Split | None = None
        self.pipeline = None
        self.model_key: str | None = None
        self.model_params: dict = {}
        self.cv: dict | None = None
        self.evaluation: dict | None = None
        self.importance: pd.DataFrame | None = None
        self.prepared: pd.DataFrame | None = None
        self.bundle: dict | None = None
        self.tuning: dict | None = None
        self.selection = None
        self.design = None
        self.clusters: np.ndarray | None = None
        self.cluster_model_key: str | None = None

        # engineering analysis — statistics, curve fits, experiments, time series,
        # unit conversions and the reliability simulation.  These are *results*, not
        # recipe, so they are not cleared by reset_model(); they are cleared when the
        # data underneath them changes.
        self.stats_result: dict = {}
        self.curve_fit = None
        self.curve_table: pd.DataFrame | None = None
        self.doe = None
        self.doe_analysis: dict = {}
        self.timeseries: dict = {}
        self.monte_carlo = None
        self.unit_conversions: list = []

        # outputs
        self.bundle_path: str = ""
        self.report_path: str = ""
        self.script_path: str = ""
        self.plot_views: list = []      # [{"kind":..., "options":{...}}]
        self.project_path: str = ""
        self.log: list = []
        #: Figures and tables the engineering screens produced.  The report writer picks
        #: them up so a Monte-Carlo histogram or a control chart reaches the HTML report
        #: without the report needing to know which screen drew it.
        self.extra_figures: list = []   # [(caption, png bytes)]
        self.extra_sections: dict = {}  # {heading: DataFrame}

    # -------------------------------------------------------------- extras
    def remember_figure(self, caption: str, png: bytes):
        """Record a figure for the report, replacing any earlier one with the same caption."""
        self.extra_figures = [pair for pair in self.extra_figures if pair[0] != caption]
        self.extra_figures.append((caption, png))

    def remember_section(self, heading: str, table):
        self.extra_sections[heading] = table

    def forget_extras(self, prefix: str = ""):
        if not prefix:
            self.extra_figures = []
            self.extra_sections = {}
            return
        self.extra_figures = [p for p in self.extra_figures if not p[0].startswith(prefix)]
        self.extra_sections = {k: v for k, v in self.extra_sections.items()
                               if not k.startswith(prefix)}

    def clear_engineering(self):
        """Drop every engineering result, because the data they were computed from has moved."""
        self.stats_result = {}
        self.curve_fit = None
        self.curve_table = None
        self.doe = None
        self.doe_analysis = {}
        self.timeseries = {}
        self.monte_carlo = None
        self.forget_extras()

    # ------------------------------------------------------------------ log
    def log_line(self, message: str):
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        entry = f"[{stamp}] {message}"
        self.log.append(entry)
        self.logged.emit(entry)

    def note(self, message: str):
        """A log entry plus a state refresh — for things worth redrawing."""
        self.log_line(message)
        self.touch()

    def touch(self):
        self.changed.emit()

    # --------------------------------------------------- cross-screen updates
    def adopt_features(self, columns, message: str = ""):
        """Called when a search picks a feature subset for the Model screen."""
        self.train_spec.features = list(columns)
        self.features_changed.emit(list(columns))
        if message:
            self.log_line(message)
        self.touch()

    def adopt_preprocess(self, spec, message: str = ""):
        """Called when a recipe search picks preprocessing settings."""
        self.pre_spec = spec
        self.preprocess_changed.emit()
        self.reset_model()
        if message:
            self.log_line(message)
        self.touch()

    # --------------------------------------------------------------- access
    @property
    def has_data(self) -> bool:
        return self.df is not None and len(self.df) > 0

    @property
    def has_model(self) -> bool:
        return self.pipeline is not None and self.evaluation is not None

    def feature_columns(self) -> list:
        if not self.has_data:
            return []
        return [c for c in self.df.columns if c != self.train_spec.target]

    def numeric_columns(self) -> list:
        if not self.has_data:
            return []
        return eda.numeric_columns(self.df)

    def detect_task(self) -> str:
        if not self.has_data:
            return "regression"
        target = self.train_spec.target
        if not target or target not in self.df.columns:
            return "regression"
        return eda.detect_task(self.df[target])

    def suggested_target(self) -> str | None:
        return eda.suggested_target(self.df) if self.has_data else None

    # --------------------------------------------------------------- stages
    def set_raw(self, df: pd.DataFrame, path: str = "", fmt: str | None = None,
                table: str | None = None, options: dict | None = None):
        self.raw_df = df
        if path:
            self.source_file = path
        self.source_format = fmt or self.source_format
        self.source_table = table or self.source_table
        self.source_options = options or self.source_options
        self.df = df.copy()
        self.clean_log = []
        # A cleaning recipe names the columns of the file it was written for, so carrying it
        # over to a new file is never right: "Water / Cement" becomes a citation of a column
        # that no longer exists, and the failure surfaces much later as
        # "operation 1 failed: unknown name(s): Water".
        if self.clean_ops:
            dropped = len(self.clean_ops)
            self.clean_ops = []
            self.log_line(f"the previous cleaning recipe ({dropped} step(s)) belonged to the "
                          "file that was loaded before, so it has been cleared")
        self.reset_model()
        self.clear_engineering()
        if not self.train_spec.target or self.train_spec.target not in self.df.columns:
            self.train_spec.target = self.suggested_target()
        self.log_line(f"loaded {len(df):,} rows × {df.shape[1]} columns from {os.path.basename(path) or 'memory'}")
        self.touch()

    def apply_cleaning(self, ops: list | None = None):
        if self.raw_df is None:
            raise ValueError("load data first")
        self.clean_ops = list(self.clean_ops if ops is None else ops)
        frame, log = clean_mod.apply_ops(self.raw_df, self.clean_ops)
        self.df = frame
        self.clean_log = log
        self.reset_model()
        self.clear_engineering()
        self.log_line(f"cleaning applied: {len(log)} step(s) → {len(frame):,} rows × {frame.shape[1]} columns")
        self.touch()

    def reset_model(self):
        self.split = None
        self.pipeline = None
        self.cv = None
        self.evaluation = None
        self.importance = None
        self.prepared = None
        self.clusters = None
        self.tuning = None
        self.selection = None
        self.design = None
        self.bundle = None
        # A reliability simulation is a statement about a particular fitted model, so a new
        # fit invalidates it even though the data underneath has not moved.
        self.monte_carlo = None

    def make_split(self):
        if not self.has_data:
            raise ValueError("load data first")
        self.split = train.make_split(self.df, self.train_spec)
        if not self.train_spec.features:
            self.train_spec.features = list(self.split.feature_names)
        for problem in self.split.dropped.values():
            self.log_line(f"split note: {problem}")
        self.log_line(
            f"split ready: {self.split.n_train:,} train / {self.split.n_test:,} test, "
            f"{len(self.split.feature_names)} features, task = {self.train_spec.task}"
        )
        self.touch()
        return self.split

    def training_frame(self) -> pd.DataFrame:
        """The frame the split was taken from, target included."""
        if self.df is None:
            raise ValueError("load data first")
        features = train.resolve_features(self.df, self.train_spec)
        columns = features + ([self.train_spec.target] if self.train_spec.target else [])
        frame = self.df[columns].copy()
        if self.train_spec.drop_missing_target and self.train_spec.target in frame.columns:
            frame = frame[frame[self.train_spec.target].notna()]
        return frame

    # ---------------------------------------------------------------- plots
    def plot_context(self, include_cv: bool = True) -> plots.PlotContext:
        evaluation = dict(self.evaluation or {})
        if include_cv and self.cv:
            evaluation["cv"] = self.cv
        # Cluster labels only make sense next to the rows they were fitted on.  After
        # a supervised model is trained the split changes, so stale labels are dropped
        # rather than handed to a plot that would misalign them row for row.
        clusters = self.clusters
        if clusters is not None and self.split is not None and len(clusters) != len(self.split.X_train):
            clusters = None
        return plots.PlotContext(
            df=self.df,
            target=self.train_spec.target,
            task=self.train_spec.task,
            model=self.pipeline,
            X=self.split.X_train if self.split is not None else None,
            y=self.split.y_train if self.split is not None else None,
            evaluation=evaluation if evaluation else None,
            importance=self.importance,
            clusters=clusters,
            labels=self.split.labels if self.split is not None else [],
            preprocessor=self.pre_spec,
            model_key=self.model_key,
            prepared=self.prepared,
        )

    # -------------------------------------------------------------- session
    def to_session(self) -> session_mod.Session:
        return session_mod.Session(
            source_file=self.source_file,
            source_format=self.source_format,
            source_table=self.source_table,
            source_options=self.source_options,
            cleaning_ops=self.clean_ops,
            train_spec=self.train_spec.to_dict(),
            preprocess_spec=self.pre_spec.to_dict(),
            model_key=self.model_key,
            model_params=self.model_params,
            tune_spec=(self.tuning or {}).get("spec", {}) if isinstance(self.tuning, dict) else {},
            grid_text=getattr(self, "_grid_text", ""),
            select_spec=getattr(self, "_select_spec", {}),
            design_spec=getattr(self, "_design_spec", {}),
            plot_views=self.plot_views,
            bundle_path=self.bundle_path,
            report_path=self.report_path,
            notes=getattr(self, "notes", ""),
            log=self.log[-200:],
        )

    def apply_session(self, data: session_mod.Session, load_data: bool = True):
        """Restore a project.  Returns the list of things that could not be restored."""
        problems = []
        self.source_file = data.source_file
        self.source_format = data.source_format
        self.source_table = data.source_table
        self.source_options = data.source_options or {}
        self.clean_ops = data.cleaning_ops or []
        self.train_spec = train.TrainSpec.from_dict(data.train_spec)
        self.pre_spec = preprocess.PreprocessSpec.from_dict(data.preprocess_spec)
        self.model_key = data.model_key
        self.model_params = data.model_params or {}
        self._grid_text = data.grid_text
        self._select_spec = data.select_spec or {}
        self._design_spec = data.design_spec or {}
        self.plot_views = data.plot_views or []
        self.bundle_path = data.bundle_path or ""
        self.report_path = data.report_path or ""
        self.notes = data.notes or ""

        if load_data:
            if not self.source_file or not os.path.exists(self.source_file):
                problems.append(f"the data file is missing: {self.source_file or '(none recorded)'}")
            else:
                from core import loaders

                try:
                    tables = loaders.read_all(self.source_file, fmt=self.source_format, **self.source_options)
                    frame = tables.get(self.source_table) if self.source_table else next(iter(tables.values()))
                    self.raw_df = frame
                    self.df = frame.copy()
                    frame2, log = clean_mod.apply_ops(self.raw_df, self.clean_ops)
                    self.df = frame2
                    self.clean_log = log
                    self.log_line(f"reopened project: {len(frame2):,} rows × {frame2.shape[1]} columns")
                except Exception as exc:
                    problems.append(f"could not reload the data: {exc}")

        if self.bundle_path and os.path.exists(self.bundle_path):
            try:
                bundle = train.load_bundle(self.bundle_path)
                self.pipeline = bundle.get("pipeline")
                self.bundle = bundle
                self.evaluation = {"test": bundle.get("metrics", {}).get("test") or {},
                                   "train": bundle.get("metrics", {}).get("train") or {}}
                self.cv = bundle.get("cv")
                importances = bundle.get("feature_importances") or {}
                if importances:
                    self.importance = pd.DataFrame({"feature": list(importances),
                                                    "importance": list(importances.values())})
                self.log_line(f"restored saved model from {os.path.basename(self.bundle_path)}")
                for warning in bundle.get("_warnings") or []:
                    problems.append(warning)
            except Exception as exc:
                problems.append(f"could not restore the saved model: {exc}")
        elif data.bundle_path:
            problems.append(f"the saved model is missing: {data.bundle_path}")

        self.log_line(f"project settings restored ({session_mod.summarise(data)})")
        self.touch()
        return problems
