"""The data screens: load, explore, clean, and prepare features.

Loading is deliberately format-agnostic: every reader returns ``{table name:
DataFrame}``, so a multi-sheet workbook, a multi-table SQLite file and a single
CSV all arrive through the same picker.  The reader options shown for a format
come straight from ``core.loaders.FORMAT_OPTIONS``, so the form can never offer
a keyword the reader will silently ignore.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from core import clean as clean_mod
from core import eda, formula, loaders, preprocess

from .. import theme
from ..widgets import (
    DataFrameView, FieldForm, KeyValueView, PlotCanvas, StepPage, fields_from_op, hint,
    scrollable, section_title,
)


# --------------------------------------------------------------------------
# loader option form
# --------------------------------------------------------------------------

def _opt(name, label, kind="text", choices=None, default=None, help=""):
    return {"name": name, "label": label, "kind": kind, "choices": choices or [],
            "default": default, "optional": True, "help": help}


#: One line of context for each example that ships in ``examples/``, so the picker says
#: what a file is for rather than only what it is called.  A file nobody annotated still
#: appears — the list is built by reading the folder, not from this dictionary.
EXAMPLE_NOTES = {
    "customer_churn.csv": "the walk-through in this manual (4,000 customers)",
    "concrete_mix.csv": "1,200 trial mixes, cube strength and a C30/37 pass/fail target",
    "pile_settlement_monitoring.csv": "one pile, three years of weekly settlement readings",
}


def shipped_examples(folder: str):
    """Every data file in ``examples/``, as ``(label, path)`` pairs."""
    found = []
    if not os.path.isdir(folder):
        return found
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json")):
            continue
        path = os.path.join(folder, name)
        note = EXAMPLE_NOTES.get(name)
        found.append((f"{name} — {note}" if note else name, path))
    return found


_OPTION_FIELDS = {
    "sep": _opt("sep", "separator", help="leave blank to detect , ; tab or |"),
    "encoding": _opt("encoding", "encoding", "choice",
                     ["auto", "utf-8", "utf-8-sig", "latin-1", "cp1252"], "auto"),
    "header": _opt("header", "header row", help="row number holding the column names (blank = 0)"),
    "names": _opt("names", "column names", help="comma separated; use instead of a header row"),
    "index_col": _opt("index_col", "index column", help="column to use as the row index"),
    "skiprows": _opt("skiprows", "skip rows", help="number of leading rows to ignore"),
    "nrows": _opt("nrows", "limit rows", help="read only the first N rows"),
    "na_values": _opt("na_values", "extra missing markers", help="comma separated, e.g. 'NA, -, n/a'"),
    "decimal": _opt("decimal", "decimal mark", help="e.g. , for European files"),
    "quotechar": _opt("quotechar", "quote character"),
    "comment": _opt("comment", "comment character", help="everything after it is ignored"),
    "usecols": _opt("usecols", "read only these columns", help="comma separated"),
    "columns": _opt("columns", "read only these columns", help="comma separated"),
    "sheet_name": _opt("sheet_name", "sheet", help="sheet name or number; blank reads them all"),
    "orient": _opt("orient", "JSON layout", "choice",
                   [None, "records", "columns", "index", "split", "table", "values"]),
    "lines": _opt("lines", "one JSON object per line", "bool", default=False),
    "match": _opt("match", "match table containing", help="substring or regex"),
    "xpath": _opt("xpath", "xpath", help="path to the rows, e.g. ./record"),
    "query": _opt("query", "SQL query", help="a SELECT statement instead of a table name"),
    "table": _opt("table", "table name"),
    "key": _opt("key", "HDF5 key"),
}

_LIST_OPTIONS = {"names", "usecols", "columns"}


def loader_fields(fmt: str) -> list[dict]:
    names = loaders.FORMAT_OPTIONS.get(fmt, [])
    return [dict(_OPTION_FIELDS[n]) for n in names if n in _OPTION_FIELDS]


def read_options(values: dict) -> dict:
    """Turn the form's strings into the kwargs the readers expect."""
    out = {}
    for key, value in (values or {}).items():
        if value in (None, "", []):
            continue
        if key in _LIST_OPTIONS and isinstance(value, str):
            items = [v.strip() for v in value.split(",") if v.strip()]
            if items:
                out[key] = items
            continue
        out[key] = value
    return out


# --------------------------------------------------------------------------
# a dataset that always exists
# --------------------------------------------------------------------------

def example_frame(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """A synthetic concrete-mix dataset, so the app is usable with nothing on disk.

    It is built to be *modellable*: a mostly additive strength formula with a
    non-linear age term, one categorical column, and a sprinkling of missing
    values so the cleaning and imputation screens have real work to do.
    """
    rng = np.random.RandomState(seed)
    cement = rng.uniform(120, 540, n)
    water = rng.uniform(120, 230, n)
    fine = rng.uniform(590, 950, n)
    coarse = rng.uniform(800, 1150, n)
    superplasticiser = rng.uniform(0, 32, n)
    slag = rng.uniform(0, 360, n)
    ash = rng.uniform(0, 200, n)
    age = rng.choice([1, 3, 7, 14, 28, 56, 90, 180], n).astype(float)

    ratio = water / cement
    strength = (
        16.0
        + 0.062 * cement
        + 0.028 * slag
        + 0.021 * ash
        - 0.145 * water
        + 0.0009 * fine
        + 0.0012 * coarse
        + 0.28 * superplasticiser
        + 12.0 * np.log1p(age)
        - 24.0 * ratio
        + rng.normal(0, 3.2, n)
    )
    frame = pd.DataFrame({
        "Cement": cement,
        "Blast furnace slag": slag,
        "Fly ash": ash,
        "Water": water,
        "Superplasticiser": superplasticiser,
        "Fine aggregate": fine,
        "Coarse aggregate": coarse,
        "Age (days)": age,
        "Curing method": rng.choice(["air", "water", "steam"], n, p=[0.45, 0.40, 0.15]),
        "Compressive strength": np.clip(strength, 4.0, None),
    })
    frame.loc[rng.choice(n, 26, replace=False), "Water"] = np.nan
    frame.loc[rng.choice(n, 14, replace=False), "Fine aggregate"] = np.nan
    return frame.round(3)


# --------------------------------------------------------------------------
# background jobs
# --------------------------------------------------------------------------

def load_job(progress, cancel, path: str, fmt: str, options: dict):
    from core.train import check_cancel

    check_cancel(cancel)
    progress(0, 1, f"reading {os.path.basename(path) or fmt}")
    tables = loaders.read_all(path, fmt=fmt, **options)
    check_cancel(cancel)
    progress(1, 1, f"{len(tables)} table(s) read")
    return tables


# --------------------------------------------------------------------------
# 1. load
# --------------------------------------------------------------------------

class LoadPage(StepPage):
    title = "1 · Load data"
    subtitle = ("CSV, Excel, ODS, JSON, HTML, XML, Parquet, Feather, HDF5, SQLite, pickle or "
                "the clipboard — every sheet and table is offered separately.")

    def build(self):
        self._tables: dict = {}
        self._pending: dict = {}

        self.format_combo = QComboBox()
        for spec in loaders.available_formats():
            label = spec.label
            if not spec.available:
                label += f"   (needs {', '.join(spec.missing)})"
            self.format_combo.addItem(label, spec.key)
            if not spec.available:
                item = self.format_combo.model().item(self.format_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        self.format_combo.currentIndexChanged.connect(self._format_changed)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("path to a data file…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.load_button = QPushButton("Load")
        self.load_button.setDefault(True)
        self.load_button.clicked.connect(self._start_load)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(self.load_button)

        self.table_combo = QComboBox()
        self.table_combo.currentIndexChanged.connect(self._table_changed)
        self.table_row = QWidget()
        table_row = QHBoxLayout(self.table_row)
        table_row.setContentsMargins(0, 0, 0, 0)
        table_row.addWidget(QLabel("table:"))
        table_row.addWidget(self.table_combo, 1)
        self.table_row.setVisible(False)

        self.options_box = QGroupBox("reader options")
        options_layout = QVBoxLayout(self.options_box)
        self.options_form = FieldForm()
        options_layout.addWidget(self.options_form)
        options_layout.addWidget(hint("Blank fields are left to pandas.  Only the options that "
                                      "make sense for the chosen format are shown."))

        clipboard = QPushButton("Read the clipboard")
        clipboard.clicked.connect(lambda: self._start_clipboard())
        example = QPushButton("Load a built-in example dataset")
        example.clicked.connect(self._load_example)
        self.capability_label = QLabel()
        self.capability_label.setWordWrap(True)
        theme.tag(self.capability_label, "muted")

        # The example files that ship with the program, listed by reading the folder — so a
        # beginner can start from a real file without hunting for it in a file dialog, and a
        # newly added example appears here with no code change.
        self.example_combo = QComboBox()
        self.example_combo.setToolTip("the data files in the examples/ folder")
        # The labels describe the datasets ("concrete_mix.csv — 1,200 trial mixes, …"), so
        # one of them is 560 px of text.  A combo sizes itself to its longest item and that
        # became this screen's minimum width — which then became the *window's* minimum
        # width, and a window that cannot shrink to the work area cannot be maximized
        # properly.  Adjusting to a contents length instead lets the box shrink and elide
        # the label when the column is narrow; the whole label is still there in the drop
        # down and in the tooltip.
        self.example_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.example_combo.setMinimumContentsLength(20)
        for label, path in shipped_examples(os.path.join(self.state.project_dir, "examples")):
            self.example_combo.addItem(label, path)
        load_example = QPushButton("Load this example")
        load_example.clicked.connect(self._load_shipped)
        self.example_row = QWidget()
        example_row = QHBoxLayout(self.example_row)
        example_row.setContentsMargins(0, 0, 0, 0)
        example_row.addWidget(self.example_combo, 1)
        example_row.addWidget(load_example)
        self.example_row.setVisible(self.example_combo.count() > 0)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(path_row)
        left_layout.addWidget(self.table_row)
        left_layout.addWidget(self.options_box)
        buttons = QHBoxLayout()
        buttons.addWidget(clipboard)
        buttons.addWidget(example)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)
        left_layout.addWidget(self.example_row)
        left_layout.addWidget(self.capability_label)
        left_layout.addStretch(1)

        self.summary = KeyValueView()
        self.preview = DataFrameView(max_rows=2000)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("Loaded frame"))
        right_layout.addWidget(self.summary)
        right_layout.addWidget(self.preview, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(scrollable(right))
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

        self._format_changed()
        self._show_capabilities()

    # ------------------------------------------------------------------ ui
    def _show_capabilities(self):
        gaps = [s for s in loaders.available_formats() if not s.available]
        if gaps:
            text = "Unavailable here: " + "; ".join(f"{s.label} (install {', '.join(s.missing)})" for s in gaps)
        else:
            text = "Every reader in the catalogue is available."
        self.capability_label.setText(text)

    def _format_changed(self):
        fmt = self.format_combo.currentData()
        self.options_form.set_fields(loader_fields(fmt))
        self.table_row.setVisible(False)
        if fmt == "clipboard":
            self.path_edit.setPlaceholderText("(the clipboard is read directly — no file needed)")
        else:
            spec = loaders.FORMAT_INDEX.get(fmt)
            self.path_edit.setPlaceholderText(
                f"path to a {spec.label.lower()} file…" if spec else "path to a data file…")

    def _browse(self):
        fmt = self.format_combo.currentData()
        spec = loaders.FORMAT_INDEX.get(fmt)
        patterns = " ".join(f"*{e}" for e in (spec.extensions if spec else ())) or "*"
        path, _ = QFileDialog.getOpenFileName(self, "Choose a data file", "", f"Data files ({patterns});;All files (*)")
        if not path:
            return
        self.path_edit.setText(path)
        guessed = loaders.format_for_path(path)
        if guessed:
            index = self.format_combo.findData(guessed)
            if index >= 0 and index != self.format_combo.currentIndex():
                self.format_combo.setCurrentIndex(index)

    # --------------------------------------------------------------- loading
    def _start_load(self):
        path = self.path_edit.text().strip()
        fmt = self.format_combo.currentData()
        if not path:
            self.warn("No file", "Choose a file first, or use the built-in example dataset.")
            return
        if not os.path.exists(path):
            self.warn("File not found", f"There is nothing at\n{path}")
            return
        options = read_options(self.options_form.values())
        self._pending = {"path": path, "fmt": fmt, "options": options}
        self.load_button.setEnabled(False)
        self.run_task(load_job, f"Loading {os.path.basename(path)}", on_done=self._loaded,
                      on_cancelled=lambda: self.load_button.setEnabled(True),
                      path=path, fmt=fmt, options=options)

    def _start_clipboard(self):
        if not loaders.FORMAT_INDEX["clipboard"].available:
            self.warn("Clipboard unavailable",
                      "No clipboard tool was found.  Install wl-clipboard, xclip or xsel.")
            return
        options = read_options(self.options_form.values())
        self._pending = {"path": "clipboard", "fmt": "clipboard", "options": options}
        self.load_button.setEnabled(False)
        self.run_task(load_job, "Reading the clipboard", on_done=self._loaded,
                      on_cancelled=lambda: self.load_button.setEnabled(True),
                      path="clipboard", fmt="clipboard", options=options)

    def _load_shipped(self):
        """Load one of the files in ``examples/``, through the ordinary load path."""
        path = self.example_combo.currentData()
        if not path or not os.path.exists(path):
            self.warn("That example is missing",
                      f"There is nothing at\n{path}\n\nRe-copy the examples/ folder from "
                      "the project.")
            return
        self.path_edit.setText(path)
        guessed = loaders.format_for_path(path)
        if guessed:
            index = self.format_combo.findData(guessed)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
        self._start_load()

    def _load_example(self):
        frame = example_frame()
        self._tables = {"example": frame}
        self._pending = {"path": "", "fmt": None, "options": {}}
        self.table_row.setVisible(False)
        self._use_table("example")
        self.state.log_line("loaded the built-in example dataset (900 synthetic concrete mixes)")

    def _loaded(self, tables):
        self.load_button.setEnabled(True)
        self._tables = tables
        names = list(tables)
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems(names)
        self.table_combo.setCurrentIndex(0)
        self.table_combo.blockSignals(False)
        self.table_row.setVisible(len(names) > 1)
        if len(names) > 1:
            self.state.log_line(f"the file holds {len(names)} tables — pick one above")
        self._use_table(names[0])

    def _table_changed(self, index):
        if index < 0 or not self._tables:
            return
        self._use_table(self.table_combo.currentText())

    def _use_table(self, name: str):
        frame = self._tables.get(name)
        if frame is None:
            return
        self.state.set_raw(frame, path=self._pending.get("path", ""), fmt=self._pending.get("fmt"),
                           table=name, options=self._pending.get("options") or {})
        self.preview.set_frame(frame)
        self.summary.set_pairs(list(eda.frame_summary(frame).items()))
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        frame = self.state.raw_df
        if frame is not None:
            self.preview.set_frame(frame)


# --------------------------------------------------------------------------
# 2. explore
# --------------------------------------------------------------------------

class EdaPage(StepPage):
    title = "2 · Explore"
    requires = "data"
    subtitle = "Column profiles, missingness, distributions, correlations and leakage warnings."

    ANALYSES = [
        ("Profile every column", "profile"),
        ("Frame summary", "summary"),
        ("Missing values", "missing"),
        ("Describe numeric columns", "describe"),
        ("Categorical counts", "categorical"),
        ("Correlation matrix", "correlation"),
        ("Strongest correlations", "top_corr"),
        ("Class balance (target)", "balance"),
        ("Duplicated rows", "duplicates"),
        ("Leakage / sanity check", "leakage"),
    ]

    def build(self):
        self._analysis_combo = QComboBox()
        for label, key in self.ANALYSES:
            self._analysis_combo.addItem(label, key)
        run = QPushButton("Run")
        run.clicked.connect(self._run_analysis)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["pearson", "spearman", "kendall"])

        self.column_combo = QComboBox()
        self.column_combo.setEditable(True)
        self.column_combo.setMinimumWidth(150)

        plot_buttons = [
            ("Target distribution", "target_dist"),
            ("Missing map", "missing_bar"),
            ("Correlation heatmap", "corr_heatmap"),
            ("Histogram of column", "hist"),
            ("Box plot of column", "box"),
        ]
        self._plot_buttons = []
        for label, key in plot_buttons:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, k=key: self._run_plot(k))
            self._plot_buttons.append(button)

        controls = QHBoxLayout()
        controls.addWidget(self._analysis_combo)
        controls.addWidget(run)
        controls.addWidget(QLabel("correlation:"))
        controls.addWidget(self.method_combo)
        controls.addWidget(QLabel("column:"))
        controls.addWidget(self.column_combo)
        controls.addStretch(1)

        plots_row = QHBoxLayout()
        for button in self._plot_buttons:
            plots_row.addWidget(button)
        plots_row.addStretch(1)

        self.table = DataFrameView(max_rows=3000)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(96)
        theme.tag(self.notes, "mono")
        self.canvas = PlotCanvas()

        bottom = QSplitter(Qt.Orientation.Vertical)
        bottom.addWidget(self.table)
        bottom.addWidget(self.canvas)

        self.body.addLayout(controls)
        self.body.addLayout(plots_row)
        self.body.addWidget(self.notes)
        self.body.addWidget(bottom, 1)

    # ------------------------------------------------------------------ data
    def _data(self):
        if not self.require_data():
            return None
        return self.state.df

    def _say(self, text: str):
        self.notes.setPlainText(text)

    def _run_analysis(self):
        frame = self._data()
        if frame is None:
            return
        key = self._analysis_combo.currentData()
        target = self.state.train_spec.target
        try:
            if key == "profile":
                out = eda.profile(frame)
                self._say("type/dtype are detected, not taken from the file — 'numeric' means the "
                          "values parse as numbers even when the column arrived as text.")
            elif key == "summary":
                out = pd.DataFrame(list(eda.frame_summary(frame).items()), columns=["field", "value"])
                self._say("")
            elif key == "missing":
                out = eda.missing_summary(frame)
                out = out.sort_values("missing_%", ascending=False).reset_index(drop=True)
                self._say(f"{int(frame.isna().sum().sum()):,} missing cells in total.")
            elif key == "describe":
                out = eda.numeric_describe(frame).reset_index(names="column")
                self._say("1%, 25%, 50%, 75% and 99% percentiles, plus skew, kurtosis and zero counts.")
            elif key == "categorical":
                out = eda.categorical_describe(frame)
                self._say("the ten most common values per categorical column.")
            elif key == "correlation":
                out = eda.correlation(frame, method=self.method_combo.currentText()).reset_index(names="column")
                self._say(f"{self.method_combo.currentText()} correlation between numeric columns.")
            elif key == "top_corr":
                out = eda.top_correlations(eda.correlation(frame, method=self.method_combo.currentText()))
                self._say("strongest off-diagonal pairs — check these for redundancy before modelling.")
            elif key == "balance":
                if not target or target not in frame.columns:
                    self.warn("No target", "Choose a target column on the Model screen first.")
                    return
                out = eda.class_balance(frame[target])
                self._say(f"task detected as {eda.detect_task(frame[target])}.")
            elif key == "duplicates":
                mask = frame.duplicated()
                out = frame[mask].head(500)
                self._say(f"{int(mask.sum()):,} duplicated rows (ignoring the index). "
                          "Remove them on the Clean screen.")
            elif key == "leakage":
                flags = eda.suspicious_columns(frame, target)
                out = pd.DataFrame({"finding": flags}) if flags else pd.DataFrame(
                    {"finding": ["nothing suspicious found"]})
                self._say("constant columns, row-index columns, high-cardinality text, and features "
                          "almost perfectly correlated with the target.")
            else:
                return
        except Exception as exc:  # noqa: BLE001 - reported in the panel
            QMessageBox.critical(self, "Analysis failed", f"{type(exc).__name__}: {exc}")
            return
        self.table.set_frame(out if isinstance(out, pd.DataFrame) else pd.DataFrame(out))

    def _run_plot(self, kind: str):
        frame = self._data()
        if frame is None:
            return
        from core import plots as plots_mod

        opts = {}
        if kind in ("hist", "box"):
            column = self.column_combo.currentText().strip()
            numeric = plots_mod.numeric_columns(frame)
            if column not in frame.columns:
                if not numeric:
                    self.warn("No numeric column", "This dataset has no numeric column to plot.")
                    return
                column = numeric[0]
            opts["column"] = column
        if kind == "corr_heatmap":
            opts["method"] = self.method_combo.currentText()
        if kind == "target_dist":
            if not self.state.train_spec.target:
                self.warn("No target", "Choose a target column on the Model screen first.")
                return
            target = self.state.train_spec.target
            if target in frame.columns and not pd.api.types.is_numeric_dtype(frame[target]):
                kind = "class_balance"      # a class label has no distribution to plot
            opts["target"] = target
        try:
            figure = plots_mod.render(kind, self.state.plot_context(include_cv=False), opts)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Plot failed", str(exc))
            return
        self.canvas.show_figure(figure)

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None:
            return
        columns = [str(c) for c in frame.columns]
        current = self.column_combo.currentText()
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        self.column_combo.addItems(columns)
        if current in columns:
            self.column_combo.setCurrentText(current)
        self.column_combo.blockSignals(False)


# --------------------------------------------------------------------------
# 3. clean
# --------------------------------------------------------------------------

class CleanPage(StepPage):
    title = "3 · Clean"
    requires = "data"
    subtitle = ("Structural cleaning only — filters, dropped columns, dtypes, duplicates and "
                "recoding.  Anything that learns from data (imputation, encoding, scaling) lives "
                "on the Features screen so it is fitted on the training split alone.")

    def build(self):
        self.ops: list[dict] = []
        self._editing = -1

        self.op_combo = QComboBox()
        for spec in clean_mod.OP_SPECS:
            self.op_combo.addItem(spec.label, spec.op)
        self.op_combo.currentIndexChanged.connect(self._op_changed)

        self.form = FieldForm()
        self.op_doc = QLabel()
        self.op_doc.setWordWrap(True)
        theme.tag(self.op_doc, "muted")

        add = QPushButton("Add step")
        add.clicked.connect(self._add_step)
        self.replace_button = QPushButton("Replace selected")
        self.replace_button.clicked.connect(self._replace_step)
        self.replace_button.setEnabled(False)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(section_title("Build a step"))
        editor_layout.addWidget(self.op_combo)
        editor_layout.addWidget(self.form)
        editor_layout.addWidget(self.op_doc)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(self.replace_button)
        row.addStretch(1)
        editor_layout.addLayout(row)
        editor_layout.addStretch(1)

        self.op_list = QListWidget()
        self.op_list.currentRowChanged.connect(self._selected)
        up = QPushButton("Move up")
        up.clicked.connect(lambda: self._move(-1))
        down = QPushButton("Move down")
        down.clicked.connect(lambda: self._move(1))
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_step)
        apply_button = QPushButton("Apply to the data")
        apply_button.clicked.connect(self._apply)
        reset = QPushButton("Discard all steps")
        reset.clicked.connect(self._reset)
        for button in (apply_button, reset):
            theme.tag(button, "primary")

        list_panel = QWidget()
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(section_title("Queued steps"))
        list_layout.addWidget(self.op_list, 1)
        buttons = QGridLayout()
        buttons.addWidget(up, 0, 0)
        buttons.addWidget(down, 0, 1)
        buttons.addWidget(remove, 0, 2)
        buttons.addWidget(apply_button, 1, 0, 1, 2)
        buttons.addWidget(reset, 1, 2)
        list_layout.addLayout(buttons)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(editor)
        top.addWidget(list_panel)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        self.preview = DataFrameView(max_rows=2000)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        theme.tag(self.log_view, "mono")

        bottom = QSplitter(Qt.Orientation.Vertical)
        bottom.addWidget(self.preview)
        bottom.addWidget(self.log_view)

        self.body.addWidget(top)
        self.body.addWidget(bottom, 1)

        self._op_changed()

    # ------------------------------------------------------------------ build
    def _op_changed(self):
        spec = clean_mod.op_spec(self.op_combo.currentData())
        if spec is None:
            return
        self.form.set_fields(fields_from_op(spec))
        self.form.set_columns(self._columns())
        self.op_doc.setText(spec.doc)

    def _columns(self) -> list:
        frame = self.state.df
        return [str(c) for c in frame.columns] if frame is not None else []

    def _validate(self, op: dict) -> str | None:
        kind = op.get("op")
        if kind in ("drop_columns", "keep_columns", "sort") and not op.get("by") and not op.get("columns"):
            return "choose at least one column"
        if kind == "rename_column" and not (op.get("column") and op.get("new_name")):
            return "give both a column and its new name"
        if kind in ("cast_dtype", "replace_value", "map_values", "clip") and not op.get("column"):
            return "choose a column"
        if kind == "filter_rows":
            conditions = op.get("conditions") or []
            if not conditions:
                return "add at least one condition"
            for condition in conditions:
                test = condition.get("test")
                if test not in ("is_null", "not_null") and not condition.get("column"):
                    return "every condition needs a column"
        return None

    def _collect(self) -> dict | None:
        spec = clean_mod.op_spec(self.op_combo.currentData())
        if spec is None:
            return None
        values = self.form.values()
        if spec.op == "filter_rows":
            clause = values.pop("conditions", {}) or {}
            op = {"op": spec.op, **clause}
        else:
            op = {"op": spec.op, **values}
        problem = self._validate(op)
        if problem:
            self.warn("This step is incomplete", problem)
            return None
        return op

    def _add_step(self):
        op = self._collect()
        if op is None:
            return
        self.ops.append(op)
        self._refresh_list(len(self.ops) - 1)

    def _replace_step(self):
        if self._editing < 0:
            return
        op = self._collect()
        if op is None:
            return
        self.ops[self._editing] = op
        self._refresh_list(self._editing)

    def _remove_step(self):
        row = self.op_list.currentRow()
        if row < 0:
            return
        del self.ops[row]
        self._refresh_list(min(row, len(self.ops) - 1))

    def _move(self, delta: int):
        row = self.op_list.currentRow()
        target = row + delta
        if row < 0 or not (0 <= target < len(self.ops)):
            return
        self.ops[row], self.ops[target] = self.ops[target], self.ops[row]
        self._refresh_list(target)

    def _refresh_list(self, select: int = -1):
        self.op_list.blockSignals(True)
        self.op_list.clear()
        for i, op in enumerate(self.ops):
            self.op_list.addItem(f"{i + 1}. {clean_mod.describe_op(op)}")
        self.op_list.blockSignals(False)
        if 0 <= select < len(self.ops):
            self.op_list.setCurrentRow(select)

    def _selected(self, row: int):
        self._editing = row if 0 <= row < len(self.ops) else -1
        self.replace_button.setEnabled(self._editing >= 0)
        if self._editing < 0:
            return
        op = self.ops[self._editing]
        index = self.op_combo.findData(op.get("op"))
        if index >= 0 and index != self.op_combo.currentIndex():
            self.op_combo.blockSignals(True)
            self.op_combo.setCurrentIndex(index)
            self.op_combo.blockSignals(False)
            self._op_changed()
        values = dict(op)
        if op.get("op") == "filter_rows":
            values = {"conditions": {k: op.get(k) for k in ("conditions", "logic", "negate")}}
        self.form.set_values(values)

    def _apply(self):
        if not self.require_data():
            return
        try:
            self.state.apply_cleaning(self.ops)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cleaning failed", f"{type(exc).__name__}: {exc}")
            return
        self._show_result()

    def _reset(self):
        self.ops = []
        self._refresh_list()
        if self.state.raw_df is not None:
            self.state.apply_cleaning([])
            self._show_result()

    def _show_result(self):
        frame = self.state.df
        self.preview.set_frame(frame)
        lines = self.state.clean_log or ["no steps — the frame is the raw file"]
        if frame is not None:
            lines = list(lines) + ["", f"result: {len(frame):,} rows × {frame.shape[1]} columns"]
        self.log_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        columns = self._columns()
        self.form.set_columns(columns)
        self._show_result()


# --------------------------------------------------------------------------
# 4. features
# --------------------------------------------------------------------------

class FeaturesPage(StepPage):
    title = "4 · Prepare features"
    requires = "data"
    subtitle = ("How missing values, categories and scales are handled.  Everything here is "
                "fitted on the training split only and stored inside the saved model, so new data "
                "goes through exactly the same arithmetic.")

    def build(self):
        self.form = FieldForm(self._spec_fields())
        self.form.changed.connect(self._read_form)

        self.engineered = QTableWidget(0, 2)
        self.engineered.setHorizontalHeaderLabels(["new column name", "formula"])
        self.engineered.horizontalHeader().setStretchLastSection(True)
        add_row = QPushButton("Add formula")
        add_row.clicked.connect(lambda: self.engineered.insertRow(self.engineered.rowCount()))
        remove_row = QPushButton("Remove formula")
        remove_row.clicked.connect(self._remove_formula)
        check = QPushButton("Check formulas")
        check.clicked.connect(self._check_formulas)

        formula_box = QGroupBox("engineered features")
        formula_layout = QVBoxLayout(formula_box)
        formula_layout.addWidget(self.engineered)
        formula_buttons = QHBoxLayout()
        formula_buttons.addWidget(add_row)
        formula_buttons.addWidget(remove_row)
        formula_buttons.addWidget(check)
        formula_buttons.addStretch(1)
        formula_layout.addLayout(formula_buttons)
        formula_layout.addWidget(hint("Arithmetic over existing columns, e.g. Water / Cement. "
                                      "Available functions: log, log1p, exp, sqrt, abs, square, "
                                      "sign, sin, cos, tan, floor, ceil, round, clip, minimum, "
                                      "maximum, where."))

        apply_button = QPushButton("Apply to the pipeline")
        theme.tag(apply_button, "primary")
        apply_button.clicked.connect(self._apply)
        preview_button = QPushButton("Preview the transformed columns")
        preview_button.clicked.connect(self._preview)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)

        actions = QHBoxLayout()
        actions.addWidget(apply_button)
        actions.addWidget(preview_button)
        actions.addWidget(reset)
        actions.addStretch(1)

        self.recipe = QPlainTextEdit()
        self.recipe.setReadOnly(True)
        self.recipe.setMaximumHeight(110)
        theme.tag(self.recipe, "mono")

        self.preview_table = DataFrameView(max_rows=400)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(section_title("Recipe"))
        left_layout.addWidget(self.form)
        left_layout.addWidget(formula_box)
        left_layout.addLayout(actions)
        left_layout.addWidget(section_title("Current recipe"))
        left_layout.addWidget(self.recipe)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("Encoded features (preview)"))
        right_layout.addWidget(self.preview_table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.body.addWidget(splitter, 1)

        self.state.preprocess_changed.connect(self._reload_from_state)
        self._refresh_recipe()

    # ------------------------------------------------------------------ reload
    def _reload_from_state(self):
        """Adopt a recipe chosen elsewhere (the cleaning-recipe search, or a project file)."""
        self.form.set_values(self.state.pre_spec.to_dict())
        self.engineered.setRowCount(0)
        for item in self.state.pre_spec.engineer or []:
            row = self.engineered.rowCount()
            self.engineered.insertRow(row)
            self.engineered.setItem(row, 0, QTableWidgetItem(str(item.get("name", ""))))
            self.engineered.setItem(row, 1, QTableWidgetItem(str(item.get("formula", ""))))
        self._refresh_recipe()

    # ------------------------------------------------------------------ form
    def _spec_fields(self) -> list[dict]:
        def field(name, label, kind, default, choices=None, help=""):
            return {"name": name, "label": label, "kind": kind, "default": default,
                    "choices": choices or [], "optional": False, "help": help}

        return [
            field("numeric_impute", "missing numbers filled with", "choice", "median",
                  preprocess.NUMERIC_IMPUTE,
                  "knn and iterative are slower but use the other columns to guess a value"),
            field("numeric_constant", "…or this constant", "float", 0.0),
            field("categorical_impute", "missing categories filled with", "choice", "mode",
                  preprocess.CATEGORICAL_IMPUTE),
            field("categorical_constant", "…or this label", "text", "missing"),
            field("encoder", "categorical encoding", "choice", "onehot", preprocess.ENCODERS,
                  "onehot keeps names readable; target encoding must be fitted inside the pipeline "
                  "(it is) to avoid leaking the answer"),
            field("max_categories", "keep at most N categories", "int", 30,
                  help="the rest are pooled into __rare__"),
            field("target_encode_smoothing", "target-encoding smoothing", "float", 10.0),
            field("scale", "scaling", "choice", "standard", preprocess.SCALERS),
            field("outlier", "outlier treatment", "choice", "none", preprocess.OUTLIER_MODES,
                  help="fitted on the training split, then applied to new rows"),
            field("drop_constant", "drop constant columns", "bool", True),
            field("cast_numeric_like", "treat numeric-looking text as numbers", "bool", True),
            field("expand_datetime", "expand dates into year/month/day/…", "bool", False),
        ]

    def _read_form(self):
        values = self.form.values()
        self.state.pre_spec = preprocess.PreprocessSpec.from_dict({
            **self.state.pre_spec.to_dict(), **values,
            "outlier_columns": self.state.pre_spec.outlier_columns,
            "engineer": self._read_formulas(),
        })
        self._refresh_recipe()

    def _read_formulas(self) -> list[dict]:
        out = []
        for row in range(self.engineered.rowCount()):
            name_item = self.engineered.item(row, 0)
            formula_item = self.engineered.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            expr = formula_item.text().strip() if formula_item else ""
            if name and expr:
                out.append({"name": name, "formula": expr})
        return out

    def _remove_formula(self):
        row = self.engineered.currentRow()
        if row >= 0:
            self.engineered.removeRow(row)
            self._read_form()

    def _check_formulas(self):
        frame = self.state.df
        if frame is None:
            self.warn("No data", "Load a dataset first so the formulas can be checked.")
            return
        problems = []
        for item in self._read_formulas():
            problem = formula.check_expression(item["formula"], frame.columns)
            if problem:
                problems.append(f"{item['name']}: {problem}")
        if problems:
            QMessageBox.warning(self, "Formula problems", "\n".join(problems))
        else:
            QMessageBox.information(self, "Formulas look fine",
                                    "Every formula parses and only uses existing columns.")

    def _apply(self):
        self._read_form()
        self.state.reset_model()
        self.state.note("preprocessing recipe updated")
        for line in preprocess.describe_spec(self.state.pre_spec):
            self.state.log_line(f"  {line}")

    def _reset(self):
        self.state.pre_spec = preprocess.PreprocessSpec()
        self.engineered.setRowCount(0)
        self.form.set_values(self.state.pre_spec.to_dict())
        self._refresh_recipe()

    def _refresh_recipe(self):
        lines = preprocess.describe_spec(self.state.pre_spec)
        self.recipe.setPlainText("\n".join(lines))

    def _preview(self):
        if not self.require_data():
            return
        self._read_form()
        from core.preprocess import WorkbenchPreprocessor

        try:
            prep = WorkbenchPreprocessor(self.state.pre_spec, task=self.state.train_spec.task)
            prepared = prep.fit_transform(self.state.df, self.state.df.get(self.state.train_spec.target))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not transform", f"{type(exc).__name__}: {exc}")
            return
        self.preview_table.set_frame(prepared.head(400),
                                     note=f"{prepared.shape[1]} encoded columns (fitted on all rows, for preview only)")
        for problem in getattr(prep, "feature_problems_", []) or []:
            self.state.log_line(f"preprocess note: {problem}")

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        self.form.set_columns(self._columns())

    def _columns(self) -> list:
        frame = self.state.df
        return [str(c) for c in frame.columns] if frame is not None else []
