"""The engineering screens: tests, curve fitting, experiments, time series and units.

Five screens that between them remove almost every reason to open a Python prompt during a
civil-engineering study.  They are grouped into one file for the same reason the other
groups are: they share the table helpers, the plotting shims and the "what column?" combo
that each of them needs, and splitting them into five files would have produced five files
of imports.

The screens are deliberately *read-only* with respect to the model.  They analyse the
working frame and write their findings into :class:`~gui.state.AppState` for the report, but
none of them changes the data except the two that say so on the button
("Add … to the dataset", which goes through the cleaning pipeline so it can be undone and
is repeated by any project that is reopened).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import curvefit as CF
from core import doe as DOE
from core import eda
from core import formula as FORM
from core import plots_engineering as PEG
from core import stats as STATS
from core import timeseries as TS
from core import units as UNITS

from .. import theme
from ..widgets import (
    DataFrameView, FieldForm, KeyValueView, MetricCards, PlotCanvas, StepPage, hint,
    scrollable, section_title,
)


# --------------------------------------------------------------------------
# table helpers
# --------------------------------------------------------------------------

def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def fill_grid(table: QTableWidget, frame: pd.DataFrame):
    """Put a frame into an *editable* grid, keeping the header names."""
    table.setRowCount(0)
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for record in frame.itertuples(index=False):
        row = table.rowCount()
        table.insertRow(row)
        for i, value in enumerate(record):
            table.setItem(row, i, QTableWidgetItem(_text(value)))
    table.resizeColumnsToContents()


def read_grid(table: QTableWidget) -> pd.DataFrame:
    """Read an editable grid back into a frame, string-valued."""
    headers = []
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        headers.append(item.text() if item is not None else f"column {column + 1}")
    rows = []
    for row in range(table.rowCount()):
        values = []
        for column in range(table.columnCount()):
            item = table.item(row, column)
            values.append(item.text() if item is not None else "")
        rows.append(values)
    return pd.DataFrame(rows, columns=headers)


def numeric_column(combo: QComboBox) -> str:
    return str(combo.currentData() if combo.currentData() is not None else combo.currentText())


# --------------------------------------------------------------------------
# jobs that go on a worker thread
# --------------------------------------------------------------------------

def curve_fit_job(progress, cancel, x_values, y_values, keys):
    """Fit the chosen forms (or every form) and rank them."""
    from core.train import Cancelled

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    table, fits = CF.fit_all(x_values, y_values, keys=keys or None)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    return table, fits


def doe_analysis_job(progress, cancel, coded, response, order):
    from core.train import Cancelled

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    return DOE.effects_analysis(coded, response, order=int(order))


# --------------------------------------------------------------------------
# 13 · tests and statistics
# --------------------------------------------------------------------------

class StatisticsPage(StepPage):
    title = "13 · Tests & statistics"
    requires = "data"
    subtitle = ("Is that difference real, or is it scatter?  Compare two columns, describe one "
                "with a confidence interval, check whether a sample is normal, and work out how "
                "many readings you actually need.  The test is chosen for you, and the screen "
                "says why.")

    def build(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_compare(), "Compare groups or columns")
        self.tabs.addTab(self._build_describe(), "Describe one column")
        self.tabs.addTab(self._build_correlation(), "Correlations")
        self.tabs.addTab(self._build_sample_size(), "How many samples?")
        self.tabs.addTab(self._build_guide(), "Which test?")
        self.body.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------- compare
    def _build_compare(self):
        self.compare_a = QComboBox()
        self.compare_b = QComboBox()
        self.compare_paired = QComboBox()
        self.compare_paired.addItem("Independent samples (different rows)", False)
        self.compare_paired.addItem("Paired (each row is one pair)", True)

        run = QPushButton("Run the right test")
        theme.tag(run, "primary")
        run.clicked.connect(self._run_compare)

        self.compare_headline = QLabel()
        self.compare_headline.setWordWrap(True)
        theme.tag(self.compare_headline, "stat")
        self.compare_decision = QLabel()
        self.compare_decision.setWordWrap(True)
        theme.tag(self.compare_decision, "muted")
        self.compare_cards = MetricCards()
        self.compare_result = DataFrameView(max_rows=40)
        self.compare_groups = DataFrameView(max_rows=60)
        self.compare_posthoc = DataFrameView(max_rows=60)
        self.compare_canvas = PlotCanvas()
        self.compare_plot_button = QPushButton("Draw the groups")
        self.compare_plot_button.clicked.connect(self._draw_compare)

        form = QGroupBox("What are the two columns?")
        form_layout = QVBoxLayout(form)
        form_layout.addWidget(hint("Any combination works: two numbers (correlation), a number "
                                   "and a category (t-test / ANOVA), or two categories "
                                   "(chi-square).  The screen picks the test."))
        form_layout.addWidget(QLabel("column A:"))
        form_layout.addWidget(self.compare_a)
        form_layout.addWidget(QLabel("column B:"))
        form_layout.addWidget(self.compare_b)
        form_layout.addWidget(QLabel("the rows are:"))
        form_layout.addWidget(self.compare_paired)
        form_layout.addWidget(run)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(form)
        left_layout.addWidget(self.compare_headline)
        left_layout.addWidget(self.compare_decision)
        left_layout.addWidget(self.compare_cards)
        left_layout.addWidget(section_title("The test result"))
        left_layout.addWidget(self.compare_result)
        left_layout.addWidget(section_title("Group by group"))
        left_layout.addWidget(self.compare_groups)
        left_layout.addWidget(section_title("If there are more than two groups: post-hoc"))
        left_layout.addWidget(self.compare_posthoc)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("The picture"))
        right_layout.addWidget(self.compare_plot_button, 0, Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.compare_canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        return page

    def _run_compare(self):
        if not self.require_data():
            return
        frame = self.state.df
        a, b = numeric_column(self.compare_a), numeric_column(self.compare_b)
        if not a or not b:
            self.warn("Choose two columns", "Pick both a column A and a column B.")
            return
        if a == b:
            self.warn("Same column twice",
                      "Column A and column B are the same; choose two different columns.")
            return
        paired = bool(self.compare_paired.currentData())
        try:
            result = STATS.analyse_pair(frame, a, b, paired_rows=paired)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "The test could not be run", f"{type(exc).__name__}: {exc}")
            return

        self.state.stats_result = result
        self.compare_headline.setText(str(result.get("headline", "")))
        self.compare_decision.setText(str(result.get("decision", "")))
        self.compare_result.set_frame(result.get("result"))
        self.compare_groups.set_frame(result.get("groups"))
        self.compare_posthoc.set_frame(result.get("posthoc"))
        self._compare_frame = result.get("frame") if result.get("frame") is not None else \
            frame[[c for c in (a, b) if c in frame.columns]].copy()
        self._compare_pair = (a, b)
        self._fill_compare_cards(result)
        self.state.remember_section("Statistical test", result.get("result"))
        if result.get("groups") is not None:
            self.state.remember_section("Groups compared", result.get("groups"))
        self.state.log_line(f"statistics: {a} vs {b} → {result.get('headline', '')[:120]}")
        self._draw_compare()

    def _fill_compare_cards(self, result):
        """Four headline numbers, whatever the chosen test happened to produce."""
        items = []
        table = result.get("result")
        if isinstance(table, pd.DataFrame) and len(table):
            row = table.iloc[0]
            for key in row.index:
                if str(key).lower() in ("note",):
                    continue
                items.append((str(key), row[key], ""))
        if result.get("effect size"):
            items.append(("effect size", result["effect size"], "how big the difference is"))
        self.compare_cards.set_metrics(items[:5])

    def _draw_compare(self):
        frame = getattr(self, "_compare_frame", None)
        if frame is None or not len(frame):
            self.warn("Nothing to draw", "Run a comparison first.")
            return
        first, second = getattr(self, "_compare_pair", ("", ""))
        missing = [c for c in (first, second) if c not in frame.columns]
        if missing:
            self.warn("Nothing to draw", "Re-run the comparison first.")
            return

        def _numeric(column: str) -> bool:
            values = pd.to_numeric(frame[column], errors="coerce")
            return bool(values.notna().sum() >= max(3, 0.5 * len(frame)))

        try:
            if _numeric(first) and _numeric(second):
                data = frame[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
                fit = CF.fit(data[first].to_numpy(), data[second].to_numpy(), "linear")
                figure = PEG.curve_fit_plot(fit, xlabel=first, ylabel=second)
                self.state.remember_figure(f"Scatter and fit: {second} against {first}",
                                           _figure_bytes(figure))
            else:
                numeric = first if _numeric(first) else second
                group = second if numeric == first else first
                figure = PEG.group_comparison_plot(frame, group, numeric,
                                                   self.state.stats_result)
                self.state.remember_figure(f"Groups: {numeric} by {group}",
                                           _figure_bytes(figure))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", f"{type(exc).__name__}: {exc}")
            return
        self.compare_canvas.show_figure(figure)

    # ------------------------------------------------------------ describe
    def _build_describe(self):
        self.desc_column = QComboBox()
        run = QPushButton("Describe it")
        theme.tag(run, "primary")
        run.clicked.connect(self._run_describe)
        self.desc_table = DataFrameView(max_rows=20)
        self.desc_normality = DataFrameView(max_rows=10)
        self.desc_outliers = DataFrameView(max_rows=20)
        self.desc_cards = MetricCards()
        self.desc_note = QLabel()
        self.desc_note.setWordWrap(True)
        theme.tag(self.desc_note, "muted")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("The confidence interval is the honest headline number: it says how "
                              "precisely the mean is known, which a bare average never does."))
        layout.addWidget(QLabel("numeric column:"))
        layout.addWidget(self.desc_column)
        layout.addWidget(run, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.desc_cards)
        layout.addWidget(self.desc_note)
        layout.addWidget(section_title("Summary"))
        layout.addWidget(self.desc_table)
        layout.addWidget(section_title("Is it normally distributed?"))
        layout.addWidget(self.desc_normality)
        layout.addWidget(section_title("Outliers (Grubbs' test)"))
        layout.addWidget(self.desc_outliers)
        layout.addStretch(1)
        return page

    def _run_describe(self):
        if not self.require_data():
            return
        column = numeric_column(self.desc_column)
        frame = self.state.df
        if column not in frame.columns:
            self.warn("Choose a column", "Pick a numeric column to describe.")
            return
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 2:
            self.warn("Not enough values", f"“{column}” has fewer than two numbers.")
            return

        summary = STATS.describe(values)
        self.desc_table.set_frame(pd.DataFrame(
            [{"statistic": k, "value": v} for k, v in summary.items()]))
        normality = STATS.normality(values)
        self.desc_normality.set_frame(normality)
        self.desc_outliers.set_frame(STATS.grubbs(values))

        normal = STATS.is_normal(values)
        self.desc_cards.set_metrics([
            ("readings", f"{summary.get('n', len(values)):,}",
             f"{int(len(frame) - len(values)):,} blank in the dataset"),
            ("mean", summary.get("mean"), f"± {summary.get('se', float('nan')):,.4g} s.e."),
            ("95% confidence", f"{summary.get('ci95 low', float('nan')):,.4g} … "
                               f"{summary.get('ci95 high', float('nan')):,.4g}",
             "the mean is in this range"),
            ("normal at 5%?", "yes" if normal else "no",
             "decides which test is valid"),
        ])
        self.desc_note.setText(
            "A 95% confidence interval that is wide compared with the effect you are looking for "
            "is the honest answer to “do we have enough data?”.  "
            + ("The normality tests agree, so tests that assume a bell curve are usable."
               if normal else
               "At least one normality test rejects the bell curve, so prefer a rank-based test "
               "(Mann–Whitney, Kruskal–Wallis) or the median and interquartile range."))
        self.state.remember_section(f"Describe {column}", self.desc_table.current_frame())
        self.state.log_line(f"statistics: described {column} — "
                            f"{'normal' if normal else 'not normal'} at 5%")

    # --------------------------------------------------------- correlation
    def _build_correlation(self):
        self.corr_form = FieldForm([{
            "name": "columns", "label": "columns to correlate", "kind": "columns", "default": [],
            "choices": [], "optional": True,
            "help": "blank uses every numeric column",
        }])
        self.corr_method = QComboBox()
        self.corr_method.addItems(["pearson", "spearman", "kendall"])
        run = QPushButton("Correlate them")
        theme.tag(run, "primary")
        run.clicked.connect(self._run_correlation)
        draw = QPushButton("Draw the heat map")
        draw.clicked.connect(self._draw_correlation)
        self.corr_table = DataFrameView(max_rows=40)
        self.corr_pairs = DataFrameView(max_rows=60)
        self.corr_canvas = PlotCanvas()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(hint("Pearson measures a straight-line relationship; Spearman and "
                                   "Kendall measure any relationship that keeps going up or down, "
                                   "and are unaffected by outliers."))
        left_layout.addWidget(self.corr_form)
        left_layout.addWidget(QLabel("method:"))
        left_layout.addWidget(self.corr_method)
        buttons = QHBoxLayout()
        buttons.addWidget(run)
        buttons.addWidget(draw)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)
        left_layout.addWidget(section_title("Correlation matrix"))
        left_layout.addWidget(self.corr_table)
        left_layout.addWidget(section_title("Strongest pairs, with p-values"))
        left_layout.addWidget(self.corr_pairs)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.corr_canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        return page

    def _run_correlation(self):
        if not self.require_data():
            return
        frame = self.state.df
        columns = self.corr_form.values().get("columns") or []
        numeric = eda.numeric_columns(frame)
        columns = [c for c in (columns or numeric) if c in numeric]
        if len(columns) < 2:
            self.warn("Need at least two numbers",
                      "Choose two or more numeric columns to correlate.")
            return
        method = self.corr_method.currentText()
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").corr(method=method)
        self.corr_table.set_frame(matrix.round(4).reset_index(names="column"))
        self._corr_matrix = matrix
        self._corr_method = method

        rows = []
        for i, first in enumerate(columns):
            for second in columns[i + 1:]:
                try:
                    info = STATS.correlation_p(
                        pd.to_numeric(frame[first], errors="coerce"),
                        pd.to_numeric(frame[second], errors="coerce"), method)
                except Exception:  # noqa: BLE001
                    continue
                # A ranked list of correlations is useless without saying which two columns
                # produced each number, and correlation_p() only ever sees two anonymous
                # series, so the names are added here.
                info = {"first column": first, "second column": second, **info}
                rows.append(info)
        pairs = pd.DataFrame(rows)
        if len(pairs):
            ordered = ["first column", "second column"] + \
                [c for c in pairs.columns if c not in ("first column", "second column")]
            pairs = pairs[ordered].reindex(
                pairs["r"].abs().sort_values(ascending=False).index)
        self.corr_pairs.set_frame(pairs)
        self.state.remember_section(f"{method.title()} correlations", pairs)
        self.state.log_line(f"statistics: correlated {len(columns)} columns ({method})")
        self._draw_correlation()

    def _draw_correlation(self):
        matrix = getattr(self, "_corr_matrix", None)
        if matrix is None:
            self.warn("Nothing to draw", "Run the correlation first.")
            return
        try:
            figure = PEG.correlation_plot(matrix, getattr(self, "_corr_method", "pearson"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.state.remember_figure("Correlation heat map", _figure_bytes(figure))
        self.corr_canvas.show_figure(figure)

    # --------------------------------------------------------- sample size
    def _build_sample_size(self):
        self.sample_form = FieldForm([
            {"name": "question", "label": "what am I estimating?", "kind": "choice",
             "default": "compare two means",
             "choices": ["compare two means", "compare two proportions", "power I already have"],
             "optional": False,
             "help": "“power I already have” answers “with this many readings, what could I "
                     "have detected?”"},
            {"name": "effect", "label": "difference worth detecting (σ or %)", "kind": "float",
             "default": 0.5, "choices": [], "optional": False,
             "help": "for means this is in standard deviations; 0.2 small, 0.5 medium, 0.8 large"},
            {"name": "p2", "label": "second proportion (proportions only)", "kind": "float",
             "default": 0.6, "choices": [], "optional": True, "help": ""},
            {"name": "n_per_group", "label": "readings per group I already have", "kind": "int",
             "default": 20, "choices": [], "optional": True, "help": ""},
            {"name": "power", "label": "chance of detecting it (power)", "kind": "float",
             "default": 0.8, "choices": [], "optional": False, "help": "0.8 is the usual floor"},
            {"name": "alpha", "label": "significance level", "kind": "float", "default": 0.05,
             "choices": [], "optional": False, "help": ""},
        ])
        run = QPushButton("Work it out")
        theme.tag(run, "primary")
        run.clicked.connect(self._run_sample_size)
        self.sample_result = DataFrameView(max_rows=10)
        self.sample_note = QLabel()
        self.sample_note.setWordWrap(True)
        theme.tag(self.sample_note, "muted")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("Run this *before* the testing programme, not after.  It is the "
                              "difference between “the difference was not significant” and "
                              "“there was never enough data to see it”."))
        layout.addWidget(self.sample_form)
        layout.addWidget(run, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.sample_result)
        layout.addWidget(self.sample_note)
        layout.addStretch(1)
        return page

    def _run_sample_size(self):
        values = self.sample_form.values()
        question = values.get("question")
        effect = float(values.get("effect") or 0.0)
        power = float(values.get("power") or 0.8)
        alpha = float(values.get("alpha") or 0.05)
        try:
            if question == "compare two means":
                info = STATS.sample_size_mean(effect, power=power, alpha=alpha)
                table = pd.DataFrame([{"quantity": k, "value": v} for k, v in info.items()])
                note = (f"To stand a {power:.0%} chance of detecting a difference of "
                        f"{effect:g} standard deviations, you need about "
                        f"{info.get('n per group', '?')} readings in each group "
                        f"({2 * (info.get('n per group') or 0)} in total).  Halving the "
                        "difference you care about roughly quadruples the readings.")
            elif question == "compare two proportions":
                second = float(values.get("p2") or 0.6)
                info = STATS.sample_size_proportion(max(effect, 1e-6), second,
                                                    power=power, alpha=alpha)
                table = pd.DataFrame([{"quantity": k, "value": v} for k, v in info.items()])
                note = (f"For proportions of {effect:.0%} against {second:.0%}, about "
                        f"{info.get('n per group', '?')} readings in each group are needed.")
            else:
                n_per_group = int(values.get("n_per_group") or 20)
                info = STATS.power_for_mean(effect, n_per_group, alpha=alpha)
                table = pd.DataFrame([{"quantity": k, "value": v} for k, v in info.items()])
                note = (f"With {n_per_group} readings per group you could detect a difference of "
                        f"about {info.get('detectable difference', float('nan')):,.3g} standard "
                        f"deviations with {power:.0%} power.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not work it out", f"{type(exc).__name__}: {exc}")
            return
        self.sample_result.set_frame(table)
        self.sample_note.setText(note)
        self.state.remember_section("Sample size calculation", table)
        self.state.log_line(f"statistics: sample size for “{question}” worked out")

    # --------------------------------------------------------------- guide
    def _build_guide(self):
        table = DataFrameView(max_rows=40)
        table.set_frame(STATS.test_guide())
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(hint("Read down the “when to use it” column.  The workbench picks the "
                              "test itself on the first tab, but it is worth knowing why."))
        layout.addWidget(table, 1)
        return page

    def _tab_changed(self, _index: int):
        self.refresh()

    # ------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None or not len(frame):
            return
        columns = [str(c) for c in frame.columns]
        numeric = [str(c) for c in eda.numeric_columns(frame)]
        for combo in (self.compare_a, self.compare_b):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        if self.compare_b.count() > 1 and self.compare_b.currentIndex() == 0:
            self.compare_b.setCurrentIndex(1)
        current = self.desc_column.currentText()
        self.desc_column.clear()
        self.desc_column.addItems(numeric)
        if current in numeric:
            self.desc_column.setCurrentText(current)
        self.corr_form.set_columns(columns)
        if not (self.corr_form.values().get("columns") or []):
            self.corr_form.set_values({"columns": numeric[:8]})


# --------------------------------------------------------------------------
# 14 · curve fitting
# --------------------------------------------------------------------------

class CurveFitPage(StepPage):
    title = "14 · Curve fitting"
    requires = "data"
    subtitle = ("Fit the standard engineering equations to a scatter — strength against age, "
                "settlement against load, flow against head — and get the coefficients, their "
                "confidence intervals, the R² and a ranking that penalises extra parameters.")

    def build(self):
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.form_combo = QComboBox()
        self.form_combo.addItem("Fit every form and rank them", "")
        for key, form in CF.FORMS.items():
            self.form_combo.addItem(form.label, key)
        self.solve_edit = QLineEdit()
        self.solve_edit.setPlaceholderText("e.g. 30")
        solve = QPushButton("Solve")
        solve.clicked.connect(self._solve)
        self.interp_form = FieldForm([
            {"name": "points", "label": "x values to interpolate", "kind": "text",
             "default": "", "choices": [], "optional": True,
             "help": "comma-separated, e.g. 3, 7, 14, 28"},
            {"name": "method", "label": "how", "kind": "choice",
             "default": "pchip (no overshoot)",
             "choices": ["linear", "nearest", "pchip (no overshoot)", "cubic"],
             "optional": False,
             "help": "pchip is the safe default — a cubic spline can invent values the "
                     "material cannot reach"},
        ])
        interp = QPushButton("Interpolate")
        interp.clicked.connect(self._interpolate)

        fit_button = QPushButton("Fit")
        theme.tag(fit_button, "primary")
        fit_button.clicked.connect(self._fit)
        residual = QPushButton("Plot the residuals")
        residual.clicked.connect(self._residuals)
        ranking = QPushButton("Plot the ranking")
        ranking.clicked.connect(self._ranking_plot)

        controls = QGroupBox("What to fit")
        control_layout = QVBoxLayout(controls)
        control_layout.addWidget(hint("Both columns must be numeric.  A date column is turned "
                                      "into days automatically, so “strength against age” works."))
        row = QHBoxLayout()
        row.addWidget(QLabel("x (input):"))
        row.addWidget(self.x_combo, 1)
        row.addWidget(QLabel("y (measured):"))
        row.addWidget(self.y_combo, 1)
        control_layout.addLayout(row)
        control_layout.addWidget(QLabel("equation:"))
        control_layout.addWidget(self.form_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(fit_button)
        buttons.addWidget(residual)
        buttons.addWidget(ranking)
        buttons.addStretch(1)
        control_layout.addLayout(buttons)

        self.cards = MetricCards()
        self.equation_label = QLabel()
        self.equation_label.setWordWrap(True)
        theme.tag(self.equation_label, "mono")
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        theme.tag(self.verdict, "muted")
        self.params_table = DataFrameView(max_rows=20)
        self.ranking_table = DataFrameView(max_rows=20)
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        theme.tag(self.warnings_label, "muted")
        self.canvas = PlotCanvas()

        solve_row = QHBoxLayout()
        solve_row.addWidget(QLabel("what x gives y ="))
        solve_row.addWidget(self.solve_edit)
        solve_row.addWidget(solve)
        solve_row.addStretch(1)
        self.solve_result = QLabel()
        self.solve_result.setWordWrap(True)
        self.interp_note = QLabel()
        self.interp_note.setWordWrap(True)
        theme.tag(self.interp_note, "muted")
        self.interp_result = DataFrameView(max_rows=40)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls)
        left_layout.addWidget(self.cards)
        left_layout.addWidget(self.equation_label)
        left_layout.addWidget(self.verdict)
        left_layout.addWidget(self.warnings_label)
        left_layout.addWidget(section_title("Coefficients"))
        left_layout.addWidget(self.params_table)
        left_layout.addWidget(section_title("Every form, ranked by AICc"))
        left_layout.addWidget(self.ranking_table)
        left_layout.addWidget(section_title("What x gives a given y?"))
        left_layout.addLayout(solve_row)
        left_layout.addWidget(self.solve_result)
        left_layout.addWidget(section_title("Interpolate a table"))
        left_layout.addWidget(self.interp_form)
        left_layout.addWidget(interp, 0, Qt.AlignmentFlag.AlignLeft)
        left_layout.addWidget(self.interp_note)
        left_layout.addWidget(self.interp_result)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)

    # ------------------------------------------------------------------ fit
    def _columns(self):
        frame = self.state.df
        x = numeric_column(self.x_combo)
        y = numeric_column(self.y_combo)
        if not x or not y:
            raise ValueError("choose both an x column and a y column")
        if x == y:
            raise ValueError("x and y are the same column")
        data = frame[[x, y]].copy()
        for column in (x, y):
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = (data[column] - data[column].min()).dt.total_seconds() / 86400.0
            else:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna()
        if len(data) < 3:
            raise ValueError("fewer than three complete rows to fit")
        return x, y, data

    def _fit(self):
        if not self.require_data():
            return
        try:
            x, y, data = self._columns()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot fit", str(exc))
            return
        key = self.form_combo.currentData()
        if key:
            fit = CF.fit(data[x].to_numpy(), data[y].to_numpy(), key)
            table = pd.DataFrame([fit.summary()])
            fits = {key: fit}
        else:
            table, fits = CF.fit_all(data[x].to_numpy(), data[y].to_numpy())
            fit = fits.get(table.iloc[0]["key"]) if len(table) else None
        if fit is None or not len(table):
            self.warn("Nothing fitted", "No equation could be fitted to these two columns.")
            return

        self.state.curve_fit = fit
        self.state.curve_table = table
        self._fits = fits
        self._pair = (x, y)

        self.ranking_table.set_frame(table)
        self.params_table.set_frame(fit.parameters())
        self.equation_label.setText(f"{fit.label}:    {fit.equation}")
        usable = bool(fit.converged and np.all(np.isfinite(fit.params)))
        self.cards.set_metrics([
            ("equation", fit.label, fit.equation),
            ("R²", f"{fit.r2:,.4f}" if np.isfinite(fit.r2) else "—",
             "1.0 is a perfect fit"),
            ("AICc", f"{fit.aicc:,.1f}" if np.isfinite(fit.aicc) else "—",
             "lower is better, extra parameters cost"),
            ("residual scatter", f"{fit.residual_sd:,.4g}"
             if np.isfinite(fit.residual_sd) else "—",
             f"on {fit.n} points, {fit.dof} degrees of freedom"),
        ])
        self.verdict.setText(self._verdict(table, fit))
        self.warnings_label.setText("  ·  ".join(fit.warnings))
        self._draw_fit(fit)
        self.state.remember_section("Curve fit", table)
        self.state.remember_section(f"Coefficients — {fit.label}", fit.parameters())
        self.state.log_line(f"curve fit: {x} → {y}, best {fit.label} R² = {fit.r2:,.4f}")
        if not usable:
            QMessageBox.information(self, "The fit did not converge",
                                    f"{fit.label} could not be fitted to this data.\n\n{fit.note}")

    @staticmethod
    def _verdict(table: pd.DataFrame, fit) -> str:
        if len(table) < 2:
            return ""
        second = table.iloc[1]
        gap = second.get("AICc", float("nan")) - fit.aicc
        if not np.isfinite(gap):
            return ""
        if gap < 2:
            return (f"“{second['form']}” is within {gap:,.1f} AICc of the winner, so the two are "
                    "statistically indistinguishable — choose the one that makes physical sense.")
        return (f"The next best equation (“{second['form']}”) is {gap:,.0f} AICc behind, so the "
                "choice is clear.")

    def _draw_fit(self, fit=None):
        fit = fit or self.state.curve_fit
        if fit is None:
            return
        x, y = getattr(self, "_pair", ("x", "y"))
        try:
            targets = []
            text = self.solve_edit.text().strip()
            if text:
                targets = [float(part) for part in text.replace(";", ",").split(",")
                           if part.strip()]
            figure = PEG.curve_fit_plot(fit, xlabel=x, ylabel=y, targets=targets)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.canvas.show_figure(figure)
        self.state.remember_figure(f"Fitted curve: {y} against {x} ({fit.label})",
                                   _figure_bytes(figure))

    def _residuals(self):
        fit = self.state.curve_fit
        if fit is None:
            self.warn("Nothing fitted", "Fit a curve first.")
            return
        try:
            figure = PEG.residual_plot(fit)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.canvas.show_figure(figure)
        self.state.remember_figure("Residuals of the fitted curve", _figure_bytes(figure))

    def _ranking_plot(self):
        table = self.state.curve_table
        if table is None or not len(table):
            self.warn("Nothing fitted", "Fit a curve first.")
            return
        try:
            figure = PEG.fit_ranking_plot(table)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.canvas.show_figure(figure)
        self.state.remember_figure("Equations ranked by AICc", _figure_bytes(figure))

    def _solve(self):
        fit = self.state.curve_fit
        if fit is None:
            self.warn("Nothing fitted", "Fit a curve first.")
            return
        text = self.solve_edit.text().strip()
        try:
            target = float(text)
        except ValueError:
            self.warn("Enter a number", f"“{text}” is not a number to solve for.")
            return
        info = fit.solve_for_x(target)
        if "x" not in info:
            self.solve_result.setText(f"No answer: {info.get('note', 'the curve never reaches it')}")
            return
        x, y = getattr(self, "_pair", ("x", "y"))
        self.solve_result.setText(
            f"{x} = <b>{info['x']:,.4g}</b> gives {y} = {target:,.6g}.  "
            f"(The band at that point spans {info['prediction low']:,.4g} to "
            f"{info['prediction high']:,.4g}, so treat it as a range, not a point.)")
        self.state.remember_section(f"Solved for {x}",
                                    pd.DataFrame([info]))
        self._draw_fit(fit)

    def _interpolate(self):
        if not self.require_data():
            return
        try:
            x, y, data = self._columns()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot interpolate", str(exc))
            return
        values = self.interp_form.values()
        text = str(values.get("points") or "").strip()
        if not text:
            self.warn("No points", "Type one or more x values to interpolate, separated by commas.")
            return
        try:
            targets = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
        except ValueError:
            self.warn("Not numbers", "The interpolation points must be numbers separated by commas.")
            return
        method = str(values.get("method", "linear")).split(" ")[0]
        # A real report repeats its x values (every batch is cured for 1, 3, 7, 14 and 28
        # days), and an interpolator needs one y per x — so say so rather than let SciPy
        # fail with a message about "strictly increasing".
        repeats = CF.repeated_x(data[x].to_numpy())
        try:
            table = CF.interpolate_frame(data[x].to_numpy(), data[y].to_numpy(), targets, method)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not interpolate", f"{type(exc).__name__}: {exc}")
            return
        self.interp_result.set_frame(table)
        extrapolated = int((table["inside the measured range?"] != "yes").sum()) \
            if "inside the measured range?" in table.columns else 0
        note = ""
        if repeats:
            note = (f"  {repeats:,} repeated x values were averaged onto one point each — "
                    "without that, a spline through a trial report will not run.")
            self.state.log_line(f"curve fit: {repeats:,} repeated x value(s) averaged")
        self.interp_note.setText(
            f"{len(targets)} point(s) interpolated"
            + (f", {extrapolated} outside the measured range" if extrapolated else
               ", all inside the measured range")
            + note)
        if extrapolated:
            self.interp_note.setText(self.interp_note.text()
                                     + "  An extrapolated value is not a measurement.")
        self.state.remember_section(f"Interpolated {y}", table)
        self.state.log_line(f"curve fit: interpolated {len(targets)} point(s) "
                            f"({extrapolated} outside the measured range)")

    # -------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None or not len(frame):
            return
        columns = [str(c) for c in frame.columns]
        numeric = [str(c) for c in eda.numeric_columns(frame)]
        for combo in (self.x_combo, self.y_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        if self.y_combo.count() > 1 and self.y_combo.currentIndex() == 0:
            self.y_combo.setCurrentIndex(min(1, self.y_combo.count() - 1))


# --------------------------------------------------------------------------
# 15 · design of experiments
# --------------------------------------------------------------------------

class DoePage(StepPage):
    title = "15 · Design of experiments"
    requires = "data"
    subtitle = ("Lay out a test programme so that every effect can be separated, randomise the "
                "run order, and then read the results: which factors mattered, by how much, and "
                "which of it was noise.")

    KINDS = [
        ("Full factorial (every combination)", "full"),
        ("Fractional factorial (screen many factors)", "fractional"),
        ("Central composite (curved surface)", "ccd"),
        ("Box–Behnken (curved surface, no corners)", "box_behnken"),
        ("Latin hypercube (space filling, for simulations)", "lhs"),
        ("Sobol sequence (space filling)", "sobol"),
        ("Halton sequence (space filling)", "halton"),
    ]

    def build(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plan(), "1 · Plan the experiment")
        self.tabs.addTab(self._build_analyse(), "2 · Read the results")
        self.body.addWidget(self.tabs, 1)

    # ---------------------------------------------------------------- plan
    def _build_plan(self):
        self.kind_combo = QComboBox()
        for label, key in self.KINDS:
            self.kind_combo.addItem(label, key)
        self.kind_combo.currentIndexChanged.connect(self._kind_changed)

        self.factors_table = QTableWidget(0, 3)
        self.factors_table.setHorizontalHeaderLabels(["factor", "lowest / 1st level", "highest / 2nd level"])
        self.factors_table.horizontalHeader().setStretchLastSection(True)
        add = QPushButton("Add a factor")
        add.clicked.connect(lambda: self.factors_table.insertRow(self.factors_table.rowCount()))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_factor)
        from_data = QPushButton("Fill from numeric columns")
        from_data.clicked.connect(self._factors_from_data)

        self.centre_spin = QSpinBox()
        self.centre_spin.setRange(1, 20)
        self.centre_spin.setValue(5)
        self.alpha_combo = QComboBox()
        self.alpha_combo.addItem("Rotatable (α = (2ᵏ)^¼)", "rotatable")
        self.alpha_combo.addItem("Face centred (α = 1, stays inside the range)", "face")
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(4, 2000)
        self.runs_spin.setValue(20)
        self.generators_edit = QLineEdit()
        self.generators_edit.setPlaceholderText("one per line or comma-separated: D=ABC")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)

        self.build_button = QPushButton("Build the design")
        theme.tag(self.build_button, "primary")
        self.build_button.clicked.connect(self._build)
        self.save_button = QPushButton("Save the run sheet…")
        self.save_button.clicked.connect(self._save_run_sheet)

        options = QGroupBox("Design")
        options_layout = QVBoxLayout(options)
        options_layout.addWidget(hint("Give each factor a name and its two levels.  Anything "
                                      "between them is interpolated, so the middle setting is "
                                      "covered too."))
        options_layout.addWidget(self.kind_combo)
        options_layout.addWidget(self.factors_table)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addWidget(from_data)
        row.addStretch(1)
        options_layout.addLayout(row)
        detail = QHBoxLayout()
        detail.addWidget(QLabel("centre points:"))
        detail.addWidget(self.centre_spin)
        detail.addWidget(QLabel("α:"))
        detail.addWidget(self.alpha_combo)
        detail.addWidget(QLabel("runs:"))
        detail.addWidget(self.runs_spin)
        options_layout.addLayout(detail)
        options_layout.addWidget(QLabel("fraction generators (fractional designs):"))
        options_layout.addWidget(self.generators_edit)
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("random seed (run order):"))
        seed_row.addWidget(self.seed_spin)
        seed_row.addStretch(1)
        options_layout.addLayout(seed_row)
        buttons = QHBoxLayout()
        buttons.addWidget(self.build_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        options_layout.addLayout(buttons)

        self.plan_cards = MetricCards()
        self.plan_note = QLabel()
        self.plan_note.setWordWrap(True)
        theme.tag(self.plan_note, "muted")
        self.quality_view = DataFrameView(max_rows=20)
        self.run_sheet = DataFrameView(max_rows=400)
        self.run_sheet.setToolTip("Edit the response column here after the tests, then switch to tab 2")
        self.canvas = PlotCanvas()
        draw = QPushButton("Show the design")
        draw.clicked.connect(self._draw_design)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(options)
        left_layout.addWidget(self.plan_cards)
        left_layout.addWidget(self.plan_note)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("What the design can and cannot separate"))
        right_layout.addWidget(self.quality_view)
        right_layout.addWidget(section_title("Run sheet — the order to run them in"))
        right_layout.addWidget(self.run_sheet, 1)
        right_layout.addWidget(draw, 0, Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        return page

    def _kind_changed(self, _index: int = 0):
        key = self.kind_combo.currentData()
        self.centre_spin.setVisible(key in ("ccd", "box_behnken"))
        self.alpha_combo.setVisible(key == "ccd")
        self.runs_spin.setVisible(key in ("lhs", "sobol", "halton"))
        self.generators_edit.setVisible(key == "fractional")

    def _remove_factor(self):
        row = self.factors_table.currentRow()
        if row >= 0:
            self.factors_table.removeRow(row)

    def _factors_from_data(self):
        frame = self.state.df
        if frame is None or not len(frame):
            self.warn("No data", "Load a dataset first.")
            return
        numeric = eda.numeric_columns(frame)
        if not numeric:
            self.warn("No numeric columns", "This dataset has no numeric column to use as a factor.")
            return
        self.factors_table.setRowCount(0)
        for column in numeric[:6]:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not len(values):
                continue
            row = self.factors_table.rowCount()
            self.factors_table.insertRow(row)
            self.factors_table.setItem(row, 0, QTableWidgetItem(str(column)))
            self.factors_table.setItem(row, 1, QTableWidgetItem(f"{values.quantile(0.1):g}"))
            self.factors_table.setItem(row, 2, QTableWidgetItem(f"{values.quantile(0.9):g}"))
        self.factors_table.resizeColumnsToContents()
        self.state.log_line(f"experiment: {self.factors_table.rowCount()} factors taken from the data")

    def _factor_spec(self) -> dict:
        spec = {}
        for row in range(self.factors_table.rowCount()):
            name_item = self.factors_table.item(row, 0)
            name = name_item.text().strip() if name_item is not None else ""
            if not name:
                continue
            low_item = self.factors_table.item(row, 1)
            high_item = self.factors_table.item(row, 2)
            low = low_item.text().strip() if low_item is not None else ""
            high = high_item.text().strip() if high_item is not None else ""
            try:
                spec[name] = (float(low), float(high))
            except ValueError:
                spec[name] = [low, high]
        return spec

    def _build(self):
        if not self.require_data():
            return
        spec = self._factor_spec()
        if len(spec) < 2:
            self.warn("Not enough factors",
                      "Give at least two factors a name and two levels each.")
            return
        key = self.kind_combo.currentData()
        seed = int(self.seed_spin.value())
        # Every design except a full factorial is built in coded units and decoded back onto
        # the real levels afterwards, so all of them need numeric low/high.
        numeric = {name: (float(level[0]), float(level[1]))
                   for name, level in spec.items()
                   if isinstance(level, (list, tuple)) and len(level) == 2
                   and _is_number(level[0]) and _is_number(level[1])}
        if key != "full" and len(numeric) < len(spec):
            bad = [name for name in spec if name not in numeric]
            self.warn("Levels must be numbers",
                      "A " + self.kind_combo.currentText().split(" (")[0].lower()
                      + " design needs a numeric lowest and highest level for every factor.\n\n"
                      + "Not numeric: " + ", ".join(bad)
                      + "\n\nUse “Full factorial” if you want text levels.")
            return
        try:
            if key == "full":
                design = DOE.full_factorial(spec, seed)
            elif key == "fractional":
                names = list(numeric)
                text = self.generators_edit.text().strip()
                generators = [part.strip() for part in text.replace("\n", ",").split(",")
                              if part.strip()]
                if not generators:
                    suggestion = DOE.recommended_fraction(len(names))
                    if suggestion is None:
                        raise ValueError("type the generators, e.g. D=ABC")
                    generators = suggestion[1]
                    self.generators_edit.setText(", ".join(generators))
                design = DOE.fractional_factorial(names, generators, seed)
            elif key == "ccd":
                design = DOE.central_composite(list(numeric), self.alpha_combo.currentData(),
                                               int(self.centre_spin.value()), seed)
            elif key == "box_behnken":
                design = DOE.box_behnken(list(numeric), int(self.centre_spin.value()), seed)
            else:
                design = DOE.space_filling(numeric, int(self.runs_spin.value()), key, seed)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not build that design",
                                 f"{type(exc).__name__}: {exc}")
            return

        # A design built in coded units is shown decoded onto the levels the technician dials in.
        if key in ("ccd", "box_behnken", "fractional"):
            design.frame = _decode_axis(design.frame, numeric)
        design.factors = spec
        sheet = design.worksheet()
        self.state.doe = design
        self.run_sheet.set_frame(sheet)
        quality = DOE.design_quality(design)
        self.quality_view.set_frame(pd.DataFrame(
            [{"property": k, "value": v} for k, v in quality.items()]))
        self.plan_cards.set_metrics([
            ("design", design.kind, ""),
            ("runs", len(design), "in randomised order"),
            ("factors", len(design.coded.columns), ""),
            ("blinding", "response column empty",
             "fill it in after the tests"),
        ])
        notes = [design.note] + list(design.warnings)
        self.plan_note.setText("  ·  ".join(n for n in notes if n))
        self._draw_design()
        self.state.remember_section("Experiment design", sheet)
        self.state.remember_section("Design quality", quality if isinstance(quality, pd.DataFrame)
                                    else pd.DataFrame([quality]))
        self.state.log_line(f"experiment: {design.kind}, {len(design)} runs, "
                            f"{len(design.coded.columns)} factors")
        self.tabs.setCurrentIndex(1)

    def _draw_design(self):
        design = self.state.doe
        if design is None:
            self.warn("No design yet", "Build a design first.")
            return
        try:
            figure = PEG.design_plot(design)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw the design", str(exc))
            return
        self.canvas.show_figure(figure)
        self.state.remember_figure(f"Experiment design ({design.kind})", _figure_bytes(figure))

    def _save_run_sheet(self):
        design = self.state.doe
        if design is None:
            self.warn("No design yet", "Build a design first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the run sheet",
                                              os.path.join(self.state.project_dir,
                                                           "run_sheet.csv"),
                                              "CSV (*.csv);;Excel workbook (*.xlsx)")
        if not path:
            return
        sheet = design.worksheet()
        try:
            if path.lower().endswith(".xlsx"):
                from core import export_office as EO

                EO.write_excel(path, {"Run sheet": sheet},
                               notes=[design.kind, design.note,
                                      "Fill in the response column, then load this file back "
                                      "into the workbench."],
                               title="Experiment run sheet")
            else:
                sheet.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save", f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"experiment: run sheet saved to {path}")
        QMessageBox.information(self, "Saved", f"Wrote\n{path}\n\nRun these in the order listed.")

    # ------------------------------------------------------------- analyse
    def _build_analyse(self):
        self.response_combo = QComboBox()
        self.response_combo.addItem("The response column (paste it into the table)", "__table__")
        self.order_combo = QComboBox()
        self.order_combo.addItem("Main effects only", 1)
        self.order_combo.addItem("Main effects + two-way interactions", 2)
        analyse = QPushButton("Analyse the experiment")
        theme.tag(analyse, "primary")
        analyse.clicked.connect(self._analyse)
        pareto_button = QPushButton("Plot the Pareto")
        pareto_button.clicked.connect(self._draw_pareto)
        main_button = QPushButton("Plot the main effects")
        main_button.clicked.connect(self._draw_main_effects)

        self.response_grid = QTableWidget(0, 2)
        self.response_grid.setHorizontalHeaderLabels(["run", "response"])
        paste = QPushButton("Paste from the clipboard")
        paste.clicked.connect(self._paste_responses)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.response_grid.setRowCount(0))

        entry = QGroupBox("The measured results")
        entry_layout = QVBoxLayout(entry)
        entry_layout.addWidget(hint("One response per run, in the same order as the run sheet.  "
                                    "Paste a column straight from Excel."))

        analyse_layout = QVBoxLayout()
        analyse_layout.addWidget(hint("Read the Pareto first: it says which factor mattered.  "
                                      "The p-values say whether the eye is being fooled."))
        analyse_layout.addWidget(QLabel("response comes from:"))
        analyse_layout.addWidget(self.response_combo)
        analyse_layout.addWidget(QLabel("model:"))
        analyse_layout.addWidget(self.order_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(analyse)
        buttons.addWidget(pareto_button)
        buttons.addWidget(main_button)
        buttons.addStretch(1)
        analyse_layout.addLayout(buttons)

        self.analysis_cards = MetricCards()
        self.analysis_headline = QLabel()
        self.analysis_headline.setWordWrap(True)
        theme.tag(self.analysis_headline, "stat")
        self.analysis_note = QLabel()
        self.analysis_note.setWordWrap(True)
        theme.tag(self.analysis_note, "muted")
        self.effects_table = DataFrameView(max_rows=40)
        self.pareto_table = DataFrameView(max_rows=40)
        self.main_table = DataFrameView(max_rows=20)
        self.canvas_analyse = PlotCanvas()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(entry)
        left_layout.addWidget(self.response_grid)
        row = QHBoxLayout()
        row.addWidget(paste)
        row.addWidget(clear)
        row.addStretch(1)
        left_layout.addLayout(row)
        left_layout.addLayout(analyse_layout)
        left_layout.addWidget(self.analysis_cards)
        left_layout.addWidget(self.analysis_headline)
        left_layout.addWidget(self.analysis_note)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("Terms, strongest first"))
        right_layout.addWidget(self.pareto_table)
        right_layout.addWidget(section_title("Full coefficient table"))
        right_layout.addWidget(self.effects_table)
        right_layout.addWidget(section_title("Average response at each level"))
        right_layout.addWidget(self.main_table)

        # Three tables and a chart do not fit in one column at a sensible window height, and
        # the chart is the thing the button was pressed for.  So they share the height and
        # the tables scroll, rather than the chart being squeezed to a sliver.
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(scrollable(right))
        right_split.addWidget(self.canvas_analyse)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)
        right_split.setSizes([340, 300])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        return page

    def _paste_responses(self):
        from PyQt6.QtWidgets import QApplication

        text = QApplication.clipboard().text().strip()
        if not text:
            self.warn("Clipboard is empty", "Copy the response column from Excel first.")
            return
        values = [part for part in text.replace("\r", "\n").split("\n") if part.strip()]
        self.response_grid.setRowCount(0)
        for i, value in enumerate(values, 1):
            row = self.response_grid.rowCount()
            self.response_grid.insertRow(row)
            self.response_grid.setItem(row, 0, QTableWidgetItem(str(i)))
            self.response_grid.setItem(row, 1, QTableWidgetItem(value.strip().split()[-1]))
        self.state.log_line(f"experiment: {len(values)} response value(s) pasted")

    def _responses(self):
        """The measured response, either pasted in or taken from a column of the dataset."""
        design = self.state.doe
        source = self.response_combo.currentData()
        if source and source != "__table__":
            series = pd.to_numeric(self.state.df[source], errors="coerce")
            if design is not None and len(series) == len(design):
                return series.to_numpy(dtype=float)
            return series.dropna().to_numpy(dtype=float)
        grid = read_grid(self.response_grid)
        if not len(grid):
            return np.array([])
        column = grid.columns[-1]
        return pd.to_numeric(grid[column], errors="coerce").to_numpy(dtype=float)

    def _analyse(self):
        design = self.state.doe
        if design is None:
            self.warn("No design yet", "Build a design on the first tab.")
            return
        response = self._responses()
        if len(response) != len(design):
            self.warn("The counts do not match",
                      f"The design has {len(design)} runs but there are {len(response)} "
                      "responses.  Paste one value per run.")
            return
        order = int(self.order_combo.currentData())
        result = DOE.effects_analysis(design.coded, response, order=order)
        if "table" not in result:
            self.warn("Could not analyse it", result.get("note", "the model could not be fitted"))
            return
        self.state.doe_analysis = result
        self.effects_table.set_frame(result["table"])
        self.pareto_table.set_frame(pareto := result["pareto"])
        self.main_table.set_frame(DOE.main_effects(design.coded, response))
        self.analysis_headline.setText(str(result.get("headline", "")))
        r2 = result.get("R²")
        self.analysis_cards.set_metrics([
            ("runs", result.get("n"), ""),
            ("R²", f"{r2:,.4f}" if r2 is not None and np.isfinite(r2) else "—",
             "how much of the variation is explained"),
            ("adjusted R²", f"{result.get('adjusted R²'):,.4f}"
             if result.get("adjusted R²") is not None and np.isfinite(result.get("adjusted R²"))
             else "—", "penalised for the number of terms"),
            ("strongest term", str(pareto.iloc[0]["term"]) if len(pareto) else "—",
             f"{pareto.iloc[0]['% contribution']:.0f}% of the explained variation"
             if len(pareto) else ""),
        ])
        self.analysis_note.setText(str(result.get("note", "")))
        self.state.remember_section("Experiment analysis", result["table"])
        self.state.remember_section("Term contributions", pareto)
        self._draw_pareto()
        self.state.log_line(f"experiment: analysed {result.get('n')} runs — "
                            f"{result.get('headline', '')[:110]}")

    def _draw_pareto(self):
        result = self.state.doe_analysis
        if not result or "pareto" not in result:
            self.warn("Nothing analysed yet", "Analyse the experiment first.")
            return
        try:
            figure = PEG.doe_pareto_plot(result["pareto"])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.canvas_analyse.show_figure(figure)
        self.state.remember_figure("Experiment Pareto — what mattered",
                                   _figure_bytes(figure))

    def _draw_main_effects(self):
        design = self.state.doe
        if design is None:
            self.warn("No design yet", "Build a design first.")
            return
        response = self._responses()
        if len(response) != len(design):
            self.warn("The counts do not match", "Paste one response per run first.")
            return
        try:
            figure = PEG.doe_main_effects_plot(DOE.main_effects(design.coded, response))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", str(exc))
            return
        self.canvas_analyse.show_figure(figure)
        self.state.remember_figure("Experiment main effects", _figure_bytes(figure))

    # ------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None or not len(frame):
            return
        current = self.response_combo.currentData()
        self.response_combo.blockSignals(True)
        self.response_combo.clear()
        self.response_combo.addItem("The response column (paste it into the table)", "__table__")
        for column in frame.columns:
            self.response_combo.addItem(f"Column in the dataset: {column}", str(column))
        index = self.response_combo.findData(current)
        self.response_combo.setCurrentIndex(index if index >= 0 else 0)
        self.response_combo.blockSignals(False)
        if not self.factors_table.rowCount():
            self._kind_changed()
            self._factors_from_data()


def _decode_axis(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Turn a coded response-surface design into the real levels a technician dials in."""
    out = frame.copy()
    for name, levels in spec.items():
        if name not in out.columns:
            continue
        try:
            low, high = float(levels[0]), float(levels[1])
        except (TypeError, ValueError, IndexError):
            continue
        centre, half = (low + high) / 2, (high - low) / 2 or 1.0
        out[name] = centre + out[name].astype(float) * half
    return out


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# 16 · time series and control charts
# --------------------------------------------------------------------------

class TimeSeriesPage(StepPage):
    title = "16 · Time series & control"
    requires = "data"
    subtitle = ("Monitoring records: is it going up, is it seasonal, is the process still in "
                "control, and is it capable of holding the specification?  Every analysis is "
                "reported as a sentence as well as a table.")

    ANALYSES = [
        ("Trend and direction of travel", "trend"),
        ("Seasonality — average by month", "profile"),
        ("Resample onto a regular calendar", "resample"),
        ("Moving average and rolling spread", "rolling"),
        ("Autocorrelation (ACF and PACF)", "acf"),
        ("Decompose into trend, season and remainder", "decompose"),
        ("Control chart", "control"),
        ("Process capability (Cp / Cpk)", "capability"),
        ("Gaps, cadence and duplicates", "gaps"),
    ]

    def build(self):
        self.time_combo = QComboBox()
        self.value_combo = QComboBox()
        self.analysis_combo = QComboBox()
        for label, key in self.ANALYSES:
            self.analysis_combo.addItem(label, key)
        self.analysis_combo.setCurrentIndex(6)
        self.analysis_combo.currentIndexChanged.connect(self._analysis_changed)

        self.rule_combo = QComboBox()
        for key, (_offset, label) in TS.RESAMPLE_RULES.items():
            self.rule_combo.addItem(label, key)
        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems(["mean", "median", "min", "max", "sum", "std", "count"])
        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 60)
        self.window_spin.setValue(3)
        self.period_spin = QSpinBox()
        self.period_spin.setRange(2, 366)
        self.period_spin.setValue(12)
        self.period_combo = QComboBox()
        for key in ("month", "weekday", "quarter", "year", "hour"):
            self.period_combo.addItem(key, key)
        self.chart_combo = QComboBox()
        for key, label in TS.CHART_KINDS.items():
            self.chart_combo.addItem(label, key)
        self.subgroup_spin = QSpinBox()
        self.subgroup_spin.setRange(2, 10)
        self.subgroup_spin.setValue(5)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 4.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(3.0)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.05, 1.0)
        self.lambda_spin.setSingleStep(0.05)
        self.lambda_spin.setValue(0.2)
        self.lsl_edit = QLineEdit()
        self.lsl_edit.setPlaceholderText("lower spec, e.g. 25")
        self.usl_edit = QLineEdit()
        self.usl_edit.setPlaceholderText("upper spec, e.g. 45")

        run = QPushButton("Run it")
        theme.tag(run, "primary")
        run.clicked.connect(self._run)
        self.add_button = QPushButton("Add these lag and rolling columns to the dataset")
        self.add_button.clicked.connect(self._add_features)

        self.cards = MetricCards()
        self.sentence = QLabel()
        self.sentence.setWordWrap(True)
        theme.tag(self.sentence, "stat")
        self.table = DataFrameView(max_rows=300)
        self.violations = DataFrameView(max_rows=200)
        self.canvas = PlotCanvas()

        controls = QGroupBox("The series")
        control_layout = QVBoxLayout(controls)
        control_layout.addWidget(hint("Pick the column holding the date and the column holding "
                                      "the measurement.  The date column can be text — it is "
                                      "parsed, and anything unreadable is reported, not ignored."))
        row = QHBoxLayout()
        row.addWidget(QLabel("date / time:"))
        row.addWidget(self.time_combo, 1)
        row.addWidget(QLabel("measurement:"))
        row.addWidget(self.value_combo, 1)
        control_layout.addLayout(row)
        control_layout.addWidget(self.analysis_combo)
        detail = QHBoxLayout()
        detail.addWidget(QLabel("resample:"))
        detail.addWidget(self.rule_combo)
        detail.addWidget(self.aggregation_combo)
        detail.addWidget(QLabel("window:"))
        detail.addWidget(self.window_spin)
        control_layout.addLayout(detail)
        detail2 = QHBoxLayout()
        detail2.addWidget(QLabel("season length:"))
        detail2.addWidget(self.period_spin)
        detail2.addWidget(self.period_combo)
        detail2.addWidget(QLabel("chart:"))
        detail2.addWidget(self.chart_combo)
        control_layout.addLayout(detail2)
        detail3 = QHBoxLayout()
        detail3.addWidget(QLabel("subgroup:"))
        detail3.addWidget(self.subgroup_spin)
        detail3.addWidget(QLabel("limit (σ):"))
        detail3.addWidget(self.sigma_spin)
        detail3.addWidget(QLabel("EWMA λ:"))
        detail3.addWidget(self.lambda_spin)
        control_layout.addLayout(detail3)
        specs = QHBoxLayout()
        specs.addWidget(QLabel("specification limits:"))
        specs.addWidget(self.lsl_edit)
        specs.addWidget(self.usl_edit)
        control_layout.addLayout(specs)
        buttons = QHBoxLayout()
        buttons.addWidget(run)
        buttons.addWidget(self.add_button)
        buttons.addStretch(1)
        control_layout.addLayout(buttons)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls)
        left_layout.addWidget(self.cards)
        left_layout.addWidget(self.sentence)
        left_layout.addWidget(section_title("Signals — what a rule fired on"))
        left_layout.addWidget(self.violations)
        left_layout.addWidget(section_title("Table"))
        left_layout.addWidget(self.table, 1)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)
        self._analysis_changed()

    def _analysis_changed(self, _index: int = 0):
        key = self.analysis_combo.currentData()
        for widget in (self.rule_combo, self.aggregation_combo):
            widget.setVisible(key in ("resample", "rolling"))
        self.window_spin.setVisible(key in ("rolling", "decompose"))
        self.period_spin.setVisible(key in ("profile", "decompose"))
        self.period_combo.setVisible(key == "profile")
        for widget in (self.chart_combo, self.subgroup_spin, self.sigma_spin, self.lambda_spin):
            widget.setVisible(key == "control")
        self.lsl_edit.setVisible(key == "capability")
        self.usl_edit.setVisible(key == "capability")
        self.add_button.setVisible(key in ("rolling", "acf", "trend"))

    def _series(self):
        frame = self.state.df
        time_column = numeric_column(self.time_combo)
        value_column = numeric_column(self.value_combo)
        if not time_column or not value_column:
            raise ValueError("choose both a date column and a measurement column")
        if time_column == value_column:
            raise ValueError("the date column and the measurement are the same column")
        return frame, time_column, value_column

    def _run(self):
        if not self.require_data():
            return
        try:
            frame, time_column, value_column = self._series()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot run it", str(exc))
            return
        key = self.analysis_combo.currentData()
        self.violations.set_frame(None)
        self.sentence.setText("")
        try:
            figure, table, sentence, cards, violations = self._compute(
                key, frame, time_column, value_column)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "That analysis failed",
                                 f"{type(exc).__name__}: {exc}")
            return
        self.table.set_frame(table)
        if violations is not None:
            self.violations.set_frame(violations)
        self.sentence.setText(sentence)
        self.cards.set_metrics(cards)
        if figure is not None:
            self.canvas.show_figure(figure)
            caption = self.analysis_combo.currentText()
            self.state.remember_figure(f"{caption}: {value_column}", _figure_bytes(figure))
        else:
            self.canvas.clear()
        if table is not None and len(table):
            self.state.remember_section(
                f"{self.analysis_combo.currentText()} — {value_column}", table)
        self.state.timeseries = {
            "analysis": key,
            "analysis label": self.analysis_combo.currentText(),
            "time": time_column,
            "value": value_column,
            "sentence": sentence,
            # The headline cards as a plain dictionary, so a later screen (or the report)
            # can quote "the slope is +3.2 mm/year" without re-running the analysis.
            "summary": {str(label): value for label, value, _caption in cards},
        }
        self.state.log_line(f"time series: {self.analysis_combo.currentText()} on {value_column}")

    def _compute(self, key, frame, time_column, value_column):
        """Return (figure, table, sentence, cards, violations) for one analysis."""
        if key == "trend":
            info = TS.trend(frame, time_column, value_column)
            if "note" in info and "slope per year" not in info:
                return None, pd.DataFrame([info]), str(info.get("note", "")), \
                    [("readings", info.get("n", 0), "")], None
            table = pd.DataFrame([{"quantity": k, "value": v} for k, v in info.items()
                                  if k not in ("column", "sentence")])
            figure = PEG.trend_plot(
                pd.to_datetime(frame[time_column], errors="coerce"),
                pd.to_numeric(frame[value_column], errors="coerce"), info, value_column)
            cards = [
                ("slope", f"{info['slope per year']:+,.4g}",
                 f"per year, 95% CI {info['slope 95% low']:+,.3g} to {info['slope 95% high']:+,.3g}"),
                ("Mann–Kendall", f"p = {info['Mann–Kendall p']:,.4g}", info.get("verdict", "")),
                ("Theil–Sen slope", f"{info['Sen slope per year']:+,.4g}", "per year, robust"),
                ("total change", f"{info['total change']:+,.4g}",
                 f"over {info['span (years)']:,.2f} years"),
            ]
            return figure, table, info.get("sentence", ""), cards, None

        if key == "profile":
            table = TS.profile_by_period(frame, time_column, value_column,
                                         self.period_combo.currentData())
            if not len(table):
                raise ValueError("there are no complete rows to average")
            figure = PEG.profile_plot(table)
            stats = ["mean", "median", "std", "min", "max", "readings"]
            period = table.columns[0]
            available = [c for c in stats if c in table.columns]
            table = table[[period] + available]
            busiest = table.loc[table["mean"].idxmax()] if "mean" in table.columns else None
            quietest = table.loc[table["mean"].idxmin()] if "mean" in table.columns else None
            sentence = ""
            if busiest is not None:
                sentence = (f"{period} averages from {quietest['mean']:,.4g} "
                            f"({quietest[period]}) to {busiest['mean']:,.4g} "
                            f"({busiest[period]}).")
            cards = [("periods", len(table), f"by {period}"),
                     ("highest", f"{busiest['mean']:,.4g}" if busiest is not None else "—",
                      str(busiest[period]) if busiest is not None else ""),
                     ("lowest", f"{quietest['mean']:,.4g}" if quietest is not None else "—",
                      str(quietest[period]) if quietest is not None else "")]
            return figure, table, sentence, cards, None

        if key == "resample":
            table = TS.resample(frame, time_column, self.rule_combo.currentData(),
                                self.aggregation_combo.currentText())
            if not len(table):
                raise ValueError("nothing left after resampling")
            figure = None
            sentence = (f"Resampled to {self.rule_combo.currentText()}: {len(table)} intervals, "
                        f"{int(table['readings'].sum())} readings in total.")
            cards = [("intervals", len(table), ""),
                     ("readings", int(table["readings"].sum()), ""),
                     ("busiest interval", int(table["readings"].max()), "readings"),
                     ("gaps", int((table["readings"] == 0).sum()), "intervals with no reading")]
            return figure, table, sentence, cards, None

        if key == "rolling":
            table = TS.rolling(frame, time_column, int(self.window_spin.value()), value_column)
            if not len(table):
                raise ValueError("nothing to smooth")
            figure = None
            sentence = (f"Added a {int(self.window_spin.value())}-reading moving average and its "
                        "spread.  The last row is the current state of the process.")
            cards = [("window", int(self.window_spin.value()), "readings"),
                     ("readings", len(table), "")]
            return figure, table, sentence, cards, None

        if key == "acf":
            values = pd.to_numeric(frame[value_column], errors="coerce").dropna().to_numpy()
            # Enough lags to see a seasonal cycle.  A weekly monitoring record has a
            # three-year shape, and a fixed 24 lags would stop at half a year and hide the
            # annual hump — which is the one thing the chart is usually opened to find.
            lags = min(len(values) - 2, max(24, min(104, int(round(len(values) / 3)))))
            table = TS.autocorrelation(values, nlags=max(5, lags))
            if not len(table):
                raise ValueError("need at least five readings for an autocorrelation")
            figure = PEG.acf_plot(table)
            outside = int((table["outside the 95% band?"] == "yes").sum()) - 1
            sentence = (f"{max(0, outside)} lag(s) outside the 95% band"
                        + (" — the readings are not independent, so a plain regression "
                           "understates the uncertainty." if outside > 0 else
                           " — the readings behave independently."))
            cards = [("readings", len(values), ""),
                     ("lags shown", len(table) - 1, ""),
                     ("significant lags", max(0, outside), "outside the 95% band")]
            return figure, table, sentence, cards, None

        if key == "decompose":
            result = TS.seasonal_decompose(frame, time_column, value_column,
                                           int(self.period_spin.value()))
            if "note" in result and "observed" not in result:
                raise ValueError(result["note"])
            figure = PEG.decomposition_plot(result)
            table = pd.DataFrame({
                "position": np.arange(1, len(result["observed"]) + 1),
                "observed": result["observed"], "trend": result["trend"],
                "seasonal": result["seasonal"], "remainder": result["residual"],
            })
            sentence = (f"Seasonal strength {result['seasonal strength']:.2f}, trend strength "
                        f"{result['trend strength']:.2f} (above 0.6 is a strong, visible pattern). "
                        f"Seasonal amplitude {result['seasonal amplitude']:,.4g}.")
            cards = [("period", result["period"], "readings"),
                     ("seasonal strength", f"{result['seasonal strength']:.3f}", ""),
                     ("trend strength", f"{result['trend strength']:.3f}", ""),
                     ("seasonal amplitude", f"{result['seasonal amplitude']:,.4g}", "")]
            return figure, table, sentence, cards, None

        if key == "control":
            result = TS.control_chart(
                frame, time_column, value_column, self.chart_combo.currentData(),
                subgroup=int(self.subgroup_spin.value()), sigma_limits=float(self.sigma_spin.value()),
                ewma_lambda=float(self.lambda_spin.value()))
            if "points" not in result:
                raise ValueError(result.get("note", "the chart could not be built"))
            figure = PEG.control_chart_plot(
                {**result, "title": f"{value_column} — {self.chart_combo.currentText()}"},
                value_column=value_column)
            violations = pd.DataFrame(result["violations"])
            if len(violations) and "point" in violations.columns:
                # "point 112" is unusable on its own — the reader has to count rows to find
                # out when it happened.  The chart's own cleaned frame knows the date.
                times = list(result["raw"][time_column])

                def when(point):
                    try:
                        position = int(point) - 1
                    except (TypeError, ValueError):
                        return ""
                    return times[position] if 0 <= position < len(times) else ""

                violations.insert(1, "when", [when(p) for p in violations["point"]])
            table = result["points"]
            count = len(violations)
            rate = count / max(1, len(table)) * 100
            sentence = (
                f"{count} signal(s) in {len(table)} points ({rate:.1f}%).  "
                + ("Nothing to investigate — the process looks stable."
                   if count == 0 else
                   "Each signal is listed below with the rule it broke; the first one is where "
                   "to start looking."))
            cards = [("chart", self.chart_combo.currentText().split(" (")[0], ""),
                     ("centre", f"{result['centre']:,.4g}", ""),
                     ("σ", f"{result.get('sigma', float('nan')):,.4g}", result.get("basis", "")[:40]),
                     ("signals", count, f"{rate:.1f}% of points")]
            return figure, table, sentence, cards, violations

        if key == "capability":
            lower = self._optional_float(self.lsl_edit)
            upper = self._optional_float(self.usl_edit)
            result = TS.process_capability(frame, time_column, value_column, lower, upper)
            table = pd.DataFrame([{"quantity": k, "value": v} for k, v in result.items()])
            sentence = result.get("note", "")
            if "Cpk" in result:
                sentence = (f"Cpk = {result['Cpk']:,.3f} — {result.get('verdict', '')}. "
                            f"About {result.get('expected outside spec %', 0):,.3g}% of readings "
                            "are expected to fall outside the specification.")
            cards = [("Cp", f"{result.get('Cp', float('nan')):,.3f}" if "Cp" in result else "—",
                      "spread against the specification"),
                     ("Cpk", f"{result.get('Cpk', float('nan')):,.3f}" if "Cpk" in result else "—",
                      "spread and centring together"),
                     ("mean", f"{result.get('mean', float('nan')):,.4g}", ""),
                     ("outside spec", f"{result.get('expected outside spec %', float('nan')):,.3g}%",
                      "from the normal fit")]
            return None, table, sentence, cards, None

        if key == "gaps":
            table = TS.gaps(frame, time_column)
            if not len(table):
                raise ValueError("no gaps to report")
            sentence = (f"{len(table)} gap(s) in the record.  A gap is where the cadence changes, "
                        "which usually means a missed reading rather than a real change.")
            longest = float(table["days"].max()) if "days" in table.columns else float("nan")
            cards = [("gaps", len(table), ""), ("longest gap", f"{longest:,.4g}", "days")]
            return None, table, sentence, cards, None

        raise ValueError(f"unknown analysis {key!r}")

    @staticmethod
    def _optional_float(edit: QLineEdit) -> float | None:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _add_features(self):
        if not self.require_data():
            return
        try:
            frame, time_column, value_column = self._series()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot add them", str(exc))
            return
        window = int(self.window_spin.value())
        # A quoted name is how the arithmetic engine refers to a column whose heading has a
        # space or a bracket in it, which is the normal case in a real test report.
        reference = FORM.quote(value_column)
        additions = [
            (f"{value_column} (lag {1})", f"lag({reference}, 1)"),
            (f"{value_column} (mean of last {window})",
             f"rolling_mean({reference}, {window})"),
            (f"{value_column} (change per step)", f"diff({reference})"),
        ]
        existing = {str(name) for name in frame.columns}
        ops = []
        if time_column in frame.columns:
            ops.append({"op": "sort", "by": [time_column], "ascending": True})
        for name, expression in additions:
            if name in existing:
                continue
            ops.append({"op": "derive_column", "name": name, "formula": expression})
        if not any(op["op"] == "derive_column" for op in ops):
            self.warn("Already there",
                      "Those lag and rolling columns already exist in the dataset.")
            return
        added = sum(1 for op in ops if op["op"] == "derive_column")
        self.state.clean_ops = list(self.state.clean_ops) + ops
        try:
            self.state.apply_cleaning()
        except Exception as exc:  # noqa: BLE001
            self.state.clean_ops = self.state.clean_ops[:len(self.state.clean_ops) - len(ops)]
            QMessageBox.critical(self, "Could not add them", f"{type(exc).__name__}: {exc}")
            return
        self._token = None
        # Nothing here is destructive and nothing is hidden: the new columns are recorded as
        # cleaning steps, exactly like a formula added on the Features screen, so they come
        # back by themselves when the project is reopened and the Clean screen can remove
        # them.  A confirmation box would only be a second thing to click — but silently
        # retraining nothing would be worse, so say what happened.
        self.state.log_line(f"time series: added {added} lag/rolling column(s) to the dataset")
        QMessageBox.information(
            self, "Added the columns",
            f"Added {added} new column(s) to the dataset:\n\n"
            + "\n".join(f"  ·  {name}" for name, _ in additions if name not in existing)
            + "\n\nThey are recorded as cleaning steps, so they reappear automatically when "
              "this project is reopened, and you can remove them on “3 · Clean”.\n\n"
              "The dataset has changed, so a model trained before now needs retraining.")

    # ------------------------------------------------------------- refresh
    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None or not len(frame):
            return
        columns = [str(c) for c in frame.columns]
        numeric = [str(c) for c in eda.numeric_columns(frame)]
        current = self.time_combo.currentText()
        self.time_combo.clear()
        self.time_combo.addItems(columns)
        if current in columns:
            self.time_combo.setCurrentText(current)
        current = self.value_combo.currentText()
        self.value_combo.clear()
        self.value_combo.addItems(columns)
        if current in columns:
            self.value_combo.setCurrentText(current)
        if current not in columns:
            for guess in numeric:
                if guess != self.time_combo.currentText():
                    self.value_combo.setCurrentText(guess)
                    break
        # A date-looking column is the best guess for the time axis.
        for column in columns:
            lowered = column.lower()
            if any(word in lowered for word in ("date", "time", "day", "month", "year")):
                self.time_combo.setCurrentText(column)
                break


# --------------------------------------------------------------------------
# 17 · units and constants
# --------------------------------------------------------------------------

class UnitsPage(StepPage):
    title = "17 · Units & constants"
    subtitle = ("Convert between the units the client uses and the units the calculation needs, "
                "convert a whole column in one go, and look up the physical constants and "
                "material properties that are usually retyped from a reference book.")

    def build(self):
        self.quantity_combo = QComboBox()
        for name in UNITS.quantities():
            self.quantity_combo.addItem(UNITS.label(name), name)
        self.quantity_combo.currentIndexChanged.connect(self._quantity_changed)

        self.from_combo = QComboBox()
        self.to_combo = QComboBox()
        self.value_edit = QLineEdit("1")
        swap = QPushButton("⇄ swap")
        swap.clicked.connect(self._swap)
        go = QPushButton("Convert")
        theme.tag(go, "primary")
        go.clicked.connect(self._convert)
        self.result_label = QLabel("—")
        theme.tag(self.result_label, "stat")
        self.result_label.setWordWrap(True)

        converter = QGroupBox("One value")
        converter_layout = QVBoxLayout(converter)
        converter_layout.addWidget(QLabel("what is it?"))
        converter_layout.addWidget(self.quantity_combo)
        row = QHBoxLayout()
        row.addWidget(self.value_edit, 1)
        row.addWidget(self.from_combo, 2)
        row.addWidget(swap)
        row.addWidget(self.to_combo, 2)
        row.addWidget(go)
        converter_layout.addLayout(row)
        converter_layout.addWidget(self.result_label)

        self.column_combo = QComboBox()
        self.column_from = QComboBox()
        self.column_to = QComboBox()
        column_go = QPushButton("Convert the column and add it to the dataset")
        column_go.clicked.connect(self._convert_column)
        column_box = QGroupBox("A whole column")
        column_layout = QVBoxLayout(column_box)
        column_layout.addWidget(hint("Adds a new column rather than overwriting the old one, so "
                                     "both the original and the converted version stay in the "
                                     "project and the report."))
        column_layout.addWidget(self.column_combo)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("currently in:"))
        row2.addWidget(self.column_from, 1)
        row2.addWidget(QLabel("convert to:"))
        row2.addWidget(self.column_to, 1)
        column_layout.addLayout(row2)
        column_layout.addWidget(column_go, 0, Qt.AlignmentFlag.AlignLeft)

        self.factors_view = DataFrameView(max_rows=60)
        self.history_view = DataFrameView(max_rows=60)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(converter)
        left_layout.addWidget(column_box)
        left_layout.addWidget(section_title("Conversions done in this session"))
        left_layout.addWidget(self.history_view)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(section_title("Every unit in this quantity, and its factor"))
        right_layout.addWidget(self.factors_view, 1)
        right_layout.addWidget(section_title("Constants and material properties"))
        constants = DataFrameView(max_rows=60)
        constants.set_frame(pd.DataFrame(UNITS.CONSTANTS,
                                         columns=["name", "value", "unit"]))
        right_layout.addWidget(constants, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.body.addWidget(splitter, 1)
        self._quantity_changed()

    def _quantity_changed(self, _index: int = 0):
        quantity = self.quantity_combo.currentData()
        if not quantity:
            return
        units = UNITS.units_for(quantity)
        for combo in (self.from_combo, self.to_combo, self.column_from, self.column_to):
            combo.clear()
            combo.addItems(units)
        self.to_combo.setCurrentIndex(min(1, len(units) - 1))
        self.column_to.setCurrentIndex(min(1, len(units) - 1))
        self.factors_view.set_frame(UNITS.factors_frame(quantity))
        self.result_label.setText("—")

    def _swap(self):
        first = self.from_combo.currentText()
        self.from_combo.setCurrentText(self.to_combo.currentText())
        self.to_combo.setCurrentText(first)

    def _convert(self):
        quantity = self.quantity_combo.currentData()
        text = self.value_edit.text().strip().replace(",", "")
        try:
            value = float(text)
        except ValueError:
            self.warn("Not a number", f"“{text}” is not a number to convert.")
            return
        try:
            result = UNITS.convert(value, self.from_combo.currentText(),
                                   self.to_combo.currentText(), quantity)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not convert", f"{type(exc).__name__}: {exc}")
            return
        self.result_label.setText(
            f"{UNITS.format_value(value)} {self.from_combo.currentText()} = "
            f"<b>{UNITS.format_value(result)} {self.to_combo.currentText()}</b>")
        row = {"quantity": UNITS.label(quantity), "value": value,
               "from": self.from_combo.currentText(), "result": result,
               "to": self.to_combo.currentText()}
        self.state.unit_conversions = (self.state.unit_conversions + [row])[-40:]
        self.history_view.set_frame(pd.DataFrame(self.state.unit_conversions))
        self.state.log_line(f"units: {UNITS.format_value(value)} "
                            f"{self.from_combo.currentText()} = "
                            f"{UNITS.format_value(result)} {self.to_combo.currentText()}")

    def _convert_column(self):
        if not self.require_data():
            return
        quantity = self.quantity_combo.currentData()
        column = numeric_column(self.column_combo)
        if not column:
            self.warn("Choose a column", "Pick the column to convert.")
            return
        frame = self.state.df
        if column not in frame.columns:
            self.warn("No such column", f"“{column}” is not in the dataset.")
            return
        if UNITS.is_affine(quantity):
            self.warn(
                "Not that quantity",
                f"“{UNITS.label(quantity)}” is an offset scale, not a plain multiplication — "
                "0 °C is not 0 °F — so converting a whole column is not offered.\n\n"
                "Convert the readings one at a time above, or convert them before loading "
                "the file.")
            return
        from_unit, to_unit = self.column_from.currentText(), self.column_to.currentText()
        if from_unit == to_unit:
            self.warn("Nothing to do", "The column is already in that unit.")
            return
        # A unit with a superscript or a slash would make a poor column heading, so the new
        # column is named after the source and the plain-ASCII form of the target unit —
        # "Cube strength (MPa) in US_ton_per_ft3" rather than "… in US ton/ft³".
        name = f"{column} ({to_unit})"
        factor = UNITS.convert(0.0, from_unit, to_unit, quantity)
        scale = UNITS.convert(1.0, from_unit, to_unit, quantity) - factor
        formula = (f"({FORM.quote(column)}) * {scale!r}" if abs(scale - 1.0) > 1e-12
                   else f"({FORM.quote(column)})")
        ops = list(self.state.clean_ops) + [
            {"op": "derive_column", "name": name, "formula": formula}]
        self.state.clean_ops = ops
        try:
            self.state.apply_cleaning()
        except Exception as exc:  # noqa: BLE001
            self.state.clean_ops = ops[:-1]
            QMessageBox.critical(self, "Could not convert the column",
                                 f"{type(exc).__name__}: {exc}")
            return
        self._token = None
        row = {"quantity": UNITS.label(quantity), "value": f"{column}: every row",
               "from": from_unit, "result": f"new column “{name}”", "to": to_unit}
        self.state.unit_conversions = (self.state.unit_conversions + [row])[-40:]
        self.history_view.set_frame(pd.DataFrame(self.state.unit_conversions))
        self.state.log_line(f"units: added “{name}” converting {column} from {from_unit} "
                            f"to {to_unit}")
        QMessageBox.information(
            self, "Added the column",
            f"Added “{name}” to the dataset.\n\nIt was recorded as a cleaning step, so it "
            "reappears automatically when the project is reopened.")

    def refresh(self):
        token = self.data_token()
        if token == getattr(self, "_token", None):
            return
        self._token = token
        frame = self.state.df
        if frame is None or not len(frame):
            return
        numeric = [str(c) for c in eda.numeric_columns(frame)]
        current = self.column_combo.currentText()
        self.column_combo.clear()
        self.column_combo.addItems(numeric)
        if current in numeric:
            self.column_combo.setCurrentText(current)
        if self.state.unit_conversions:
            self.history_view.set_frame(pd.DataFrame(self.state.unit_conversions))


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

def _figure_bytes(figure, dpi: int = 130) -> bytes:
    """PNG bytes for the report, then let the figure go."""
    from core import plots as plots_mod

    try:
        return plots_mod.figure_png_bytes(figure, dpi=dpi)
    except Exception:  # noqa: BLE001 - a figure that cannot be encoded must not break the screen
        return b""
