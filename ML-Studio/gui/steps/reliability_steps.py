"""The uncertainty and reliability screen.

One screen, because one idea: *how sure are we of this answer?*  A model that predicts a
28-day strength of 34 MPa is nearly useless on its own; what the designer needs is the
distribution of the strength given everything that is uncertain, the probability it falls
below the 30 MPa the specification demands, and which of the inputs is responsible for
most of that risk.  That is what this screen produces, by pushing thousands of random
draws of the inputs through the very same fitted pipeline that produced the prediction.

The simulation is a statement about a *particular* model, so a new fit empties it (see
:meth:`~gui.state.AppState.reset_model`).  Everything it produces is also handed to
:class:`~gui.state.AppState` for the report, so the histogram and the reliability index
reach the HTML report without the report knowing this screen exists.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core import eda
from core import montecarlo as MC
from core import plots_engineering as PEG

from .. import theme
from ..widgets import (
    DataFrameView, MetricCards, PlotCanvas, StepPage, hint, scrollable, section_title,
)

#: The editable inputs grid.  A combo box lives in the distribution column, because
#: choosing a distribution has to recompute that row's parameters.
COLUMN_VARY, COLUMN_NAME, COLUMN_DIST, COLUMN_PARAMS = 0, 1, 2, 3

CHART_KINDS = [
    ("Outcome distribution", "histogram"),
    ("What the answer is sensitive to", "tornado"),
    ("Influence of each input", "influence"),
    ("Probability of exceeding a limit", "exceedance"),
    ("The inputs as they were drawn", "inputs"),
]


def monte_carlo_job(progress, cancel, pipe, frame, spec, features, task):
    """The worker-thread body: sample, predict, and summarise."""
    from core.train import check_cancel

    result = MC.run(pipe, frame, spec, features, task=task,
                    progress=progress, cancel=cancel)
    check_cancel(cancel)
    if progress is not None:
        progress(len(spec.inputs) + 1, len(spec.inputs) + 2, "sweeping one input at a time")
    tornado = MC.tornado(pipe, frame, spec, features, task=task)
    check_cancel(cancel)
    return {"result": result, "tornado": tornado}


class ReliabilityPage(StepPage):
    title = "18 · Uncertainty & reliability"
    requires = "model"
    requires_note = ("Reliability works by pushing thousands of random draws of the inputs "
                     "through a fitted model, so it needs a model first.  Train one on "
                     "“5 · Model”, or press Ctrl+R to run the whole workflow.")
    subtitle = ("An answer is only as useful as its margin.  Vary every uncertain input at "
                "once, run the model thousands of times, and read off the distribution, the "
                "probability of missing a limit, the reliability index β, and which input is "
                "driving the risk.")

    def build(self):
        self._tornado = None
        self._last_spec = None

        # ------------------------------------------------------------- inputs
        self.grid = QTableWidget(0, 4)
        self.grid.setHorizontalHeaderLabels(["vary?", "input column", "how it varies",
                                             "its parameters"])
        self.grid.horizontalHeader().setStretchLastSection(True)
        self.grid.setColumnWidth(COLUMN_VARY, 55)
        self.grid.setColumnWidth(COLUMN_NAME, 180)
        self.grid.setColumnWidth(COLUMN_DIST, 210)
        self.grid.verticalHeader().setDefaultSectionSize(26)

        reset = QPushButton("Re-read the spread from the data")
        reset.clicked.connect(self._reload_grid)
        all_button = QPushButton("Vary everything")
        all_button.clicked.connect(lambda: self._set_all(True))
        none_button = QPushButton("Vary nothing")
        none_button.clicked.connect(lambda: self._set_all(False))

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(200, 200_000)
        self.samples_spin.setSingleStep(500)
        self.samples_spin.setValue(5000)
        self.samples_spin.setToolTip("More runs give a smoother distribution; 5,000 is "
                                     "plenty for a 1% probability, 50,000 for 0.1%")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("The same seed gives the same numbers, so a result can be "
                                  "reproduced and quoted")
        self.corr_check = QCheckBox("keep the correlations between the inputs")
        self.corr_check.setToolTip("A strong mix and a high cement content move together. "
                                   "Switching this on preserves that link instead of drawing "
                                   "each input independently, which is the honest option.")
        self.fill_combo = QComboBox()
        for key, label in (("median", "the typical value (median)"),
                           ("mode", "the most common value (mode)"),
                           ("mean", "the average (mean)"),
                           ("zero", "zero")):
            self.fill_combo.addItem(label, key)
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("e.g. 30")
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("the outcome must stay ABOVE it (a strength, a capacity)",
                                     "below")
        self.direction_combo.addItem("the outcome must stay BELOW it (a settlement, a "
                                     "deflection)", "above")
        self.demand_edit = QLineEdit()
        self.demand_edit.setPlaceholderText("optional — a demand for a safety factor")

        self.run_button = QPushButton("Run the simulation")
        theme.tag(self.run_button, "primary")
        self.run_button.clicked.connect(self._run)
        self.export_button = QPushButton("Export every table…")
        self.export_button.clicked.connect(self._export)
        self.setup_button = QPushButton("Save this setup…")
        self.setup_button.clicked.connect(self._save_setup)
        load_setup = QPushButton("Load a setup…")
        load_setup.clicked.connect(self._load_setup)

        inputs_box = QGroupBox("What is uncertain")
        inputs_layout = QVBoxLayout(inputs_box)
        inputs_layout.addWidget(hint(
            "Tick the inputs that are genuinely uncertain, and say how each one varies.  "
            "Unticked inputs are held at their typical value for the whole run, so the "
            "result answers “given that everything else is normal, what does this one thing "
            "do?”."))
        inputs_layout.addWidget(self.grid, 1)
        row = QHBoxLayout()
        row.addWidget(reset)
        row.addWidget(all_button)
        row.addWidget(none_button)
        row.addStretch(1)
        inputs_layout.addLayout(row)

        settings_box = QGroupBox("How to run it")
        settings_layout = QVBoxLayout(settings_box)
        first = QHBoxLayout()
        first.addWidget(QLabel("simulations:"))
        first.addWidget(self.samples_spin)
        first.addWidget(QLabel("random seed:"))
        first.addWidget(self.seed_spin)
        first.addStretch(1)
        settings_layout.addLayout(first)
        settings_layout.addWidget(self.corr_check)
        second = QHBoxLayout()
        second.addWidget(QLabel("hold the other inputs at:"))
        second.addWidget(self.fill_combo, 1)
        settings_layout.addLayout(second)
        third = QHBoxLayout()
        third.addWidget(QLabel("limit value:"))
        third.addWidget(self.threshold_edit)
        third.addWidget(QLabel("demand (for a safety factor):"))
        third.addWidget(self.demand_edit)
        settings_layout.addLayout(third)
        settings_layout.addWidget(self.direction_combo)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.setup_button)
        buttons.addWidget(load_setup)
        buttons.addStretch(1)
        settings_layout.addLayout(buttons)

        # ------------------------------------------------------------ results
        self.cards = MetricCards()
        self.sentence = QLabel()
        self.sentence.setWordWrap(True)
        theme.tag(self.sentence, "stat")
        self.story = QLabel()
        self.story.setWordWrap(True)
        theme.tag(self.story, "muted")
        self.summary_view = DataFrameView(max_rows=6)
        self.percentile_view = DataFrameView(max_rows=20)
        self.influence_view = DataFrameView(max_rows=40)
        self.tornado_view = DataFrameView(max_rows=40)
        self.safety_view = DataFrameView(max_rows=20)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        theme.tag(self.notes, "warning")

        self.chart_combo = QComboBox()
        for label, key in CHART_KINDS:
            self.chart_combo.addItem(label, key)
        self.chart_combo.currentIndexChanged.connect(self._draw)
        self.canvas = PlotCanvas()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(inputs_box)
        left_layout.addWidget(settings_box)
        left_layout.addStretch(1)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addWidget(self.cards)
        middle_layout.addWidget(self.sentence)
        middle_layout.addWidget(self.story)
        middle_layout.addWidget(self.notes)
        middle_layout.addWidget(section_title("The outcome distribution"))
        middle_layout.addWidget(self.summary_view)
        middle_layout.addWidget(section_title("Percentiles"))
        middle_layout.addWidget(self.percentile_view)
        middle_layout.addWidget(section_title("Which input moves the answer (one at a time)"))
        middle_layout.addWidget(self.tornado_view)
        middle_layout.addWidget(section_title("Which input the answer moves with (all at once)"))
        middle_layout.addWidget(self.influence_view)
        middle_layout.addWidget(section_title("Safety factor against the demand"))
        middle_layout.addWidget(self.safety_view)
        middle_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.chart_combo)
        right_layout.addWidget(self.canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scrollable(left))
        splitter.addWidget(scrollable(middle))
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 3)
        self.body.addWidget(splitter, 1)

    # ------------------------------------------------------------- the grid
    def _reload_grid(self):
        """One row per feature the model uses, with the spread read off the data."""
        frame = self.state.df
        features = self._features()
        if frame is None or not features:
            return
        self.grid.setRowCount(0)
        for column in features:
            row = self.grid.rowCount()
            self.grid.insertRow(row)

            vary = QTableWidgetItem()
            vary.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            vary.setCheckState(Qt.CheckState.Checked)
            vary.setToolTip("untick to hold this input at its typical value")
            self.grid.setItem(row, COLUMN_VARY, vary)

            name = QTableWidgetItem(str(column))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.grid.setItem(row, COLUMN_NAME, name)

            combo = QComboBox()
            for key in MC.DISTRIBUTIONS:
                combo.addItem(MC.distribution_label(key), key)
            chosen = "normal"
            series = frame[column] if column in frame.columns else pd.Series(dtype=float)
            try:
                chosen = MC.choose_distribution(series)
            except Exception:  # noqa: BLE001 - a wrong guess is not worth an error
                chosen = "normal"
            index = combo.findData(chosen)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.currentIndexChanged.connect(
                lambda _i, r=row: self._distribution_changed(r))
            self.grid.setCellWidget(row, COLUMN_DIST, combo)

            self._write_params(row)
            if MC.is_categorical(series):
                # A column of names has no curve to fit, so it is held at one value rather
                # than sampled.  Saying so here is better than a row that silently does
                # nothing, and the box is still editable to pick a different category.
                vary.setCheckState(Qt.CheckState.Unchecked)
                vary.setToolTip("a category cannot be sampled from a curve — it is held at "
                                "the value in the box, which you can change")

    def _write_params(self, row: int):
        """Recompute a row's parameter text for its current distribution."""
        combo = self.grid.cellWidget(row, COLUMN_DIST)
        name_item = self.grid.item(row, COLUMN_NAME)
        if combo is None or name_item is None:
            return
        distribution = combo.currentData()
        column = name_item.text()
        frame = self.state.df
        series = pd.Series(dtype=float)
        if frame is not None and column in frame.columns:
            series = frame[column]
        try:
            params = MC.default_params(series, distribution)
        except Exception:  # noqa: BLE001
            params = {}
        text = ", ".join(f"{k} = {_short(v)}" for k, v in params.items())
        item = QTableWidgetItem(text)
        item.setToolTip(MC.distribution_help(distribution))
        if MC.is_categorical(series):
            item.setToolTip(item.toolTip()
                            + "  ·  this column holds names, so it is held at the value "
                              "shown rather than sampled")
        self.grid.setItem(row, COLUMN_PARAMS, item)

    def _distribution_changed(self, row: int):
        self._write_params(row)

    def _set_all(self, state: bool):
        for row in range(self.grid.rowCount()):
            item = self.grid.item(row, COLUMN_VARY)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked if state
                                   else Qt.CheckState.Unchecked)

    def _features(self) -> list:
        features = list(self.state.train_spec.features or [])
        if features:
            return features
        frame = self.state.df
        if frame is None:
            return []
        target = self.state.train_spec.target
        return [str(c) for c in eda.numeric_columns(frame) if c != target]

    def _spec(self):
        """Read the grid and the settings into a :class:`MonteCarloSpec`."""
        specs = []
        for row in range(self.grid.rowCount()):
            vary = self.grid.item(row, COLUMN_VARY)
            if vary is None or vary.checkState() != Qt.CheckState.Checked:
                continue
            name_item = self.grid.item(row, COLUMN_NAME)
            combo = self.grid.cellWidget(row, COLUMN_DIST)
            if name_item is None or combo is None:
                continue
            column = name_item.text()
            distribution = combo.currentData()
            params = {}
            params_item = self.grid.item(row, COLUMN_PARAMS)
            if params_item is not None and params_item.text().strip():
                for part in params_item.text().split(","):
                    if "=" not in part:
                        continue
                    key, _, value = part.partition("=")
                    params[key.strip()] = _param_value(value.strip())
            specs.append(MC.InputSpec(column=column, distribution=distribution, params=params))
        if not specs:
            raise ValueError("tick at least one input to vary")
        # A row whose parameters do not describe a distribution must be caught here, with a
        # sentence, rather than deep inside numpy as a KeyError on "mean".
        for item in specs:
            try:
                item.draw(3, np.random.default_rng(0))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"“{item.column}” is set to {MC.distribution_label(item.distribution)}, "
                    f"but its parameters are not complete ({exc}).  Fill in: "
                    + ", ".join(MC.distribution_params(item.distribution))
                    + "  — or change the distribution.") from exc
        if self.corr_check.isChecked() and len(specs) < 2:
            pass    # one input has nothing to correlate with; harmless
        threshold = _number(self.threshold_edit.text().strip())
        spec = MC.MonteCarloSpec(
            inputs=specs,
            n_samples=int(self.samples_spin.value()),
            seed=int(self.seed_spin.value()),
            keep_correlations=bool(self.corr_check.isChecked()),
            fill=self.fill_combo.currentData(),
            threshold=threshold,
            direction=self.direction_combo.currentData(),
        )
        return spec

    # ------------------------------------------------------------------ run
    def _run(self):
        if not self.require_model():
            return
        if self.state.pipeline is None:
            self.warn("No model", "Train a model first — this screen drives one.")
            return
        if not self.grid.rowCount():
            self._reload_grid()
        try:
            spec = self._spec()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot run it", str(exc))
            return
        if spec.threshold is None:
            self.warn(
                "Give a limit value",
                "The probability of missing a limit is the point of this screen, so type the "
                "limit the design has to meet (a required strength, a permitted settlement) "
                "in the “limit value” box.\n\nWithout one, you still get the distribution and "
                "the sensitivity, but no reliability.")
        task = "classification" if self.state.train_spec.task != "regression" else "regression"
        self._last_spec = spec
        self.run_button.setEnabled(False)
        self.state.log_line(f"reliability: running {spec.n_samples:,} simulations")

        def done(payload):
            self.run_button.setEnabled(True)
            self.state.monte_carlo = payload["result"]
            self._tornado = payload["tornado"]
            self._show(payload["result"], payload["tornado"])

        def failed(*_args):
            self.run_button.setEnabled(True)

        self.run_task(monte_carlo_job, "The simulation", on_done=done,
                      on_cancelled=failed,
                      pipe=self.state.pipeline, frame=self.state.df, spec=spec,
                      features=self._features(), task=task)

    def _show(self, result, tornado):
        summary = result.summary
        self.summary_view.set_frame(pd.DataFrame(
            [{"quantity": k, "value": v} for k, v in summary.items()]))
        self.percentile_view.set_frame(result.percentiles)
        self.influence_view.set_frame(result.influence)
        self.tornado_view.set_frame(tornado)

        cards = [("simulations", f"{summary.get('n', 0):,}", ""),
                 ("typical outcome", f"{summary.get('mean', float('nan')):,.4g}",
                  f"sd {summary.get('sd', float('nan')):,.4g}")]
        if result.probability is not None:
            cards.append(("probability beyond the limit",
                          f"{result.probability * 100:,.2f}%",
                          f"the limit is {result.threshold:,.6g}"))
        if result.reliability_index is not None and np.isfinite(result.reliability_index):
            cards.append(("reliability index β",
                          f"{result.reliability_index:,.2f}",
                          "bigger is safer; below 3 is a warning"))
        self.cards.set_metrics(cards[:5])

        self.sentence.setText(result.sentence())
        self.story.setText(self._story(result, tornado))
        notes = list(result.notes)
        # Which inputs were *not* sampled — an unticked row, or a column of names that has no
        # curve to fit.  A tally of the varied inputs alone would let a held one look like an
        # input that was studied and found not to matter.
        varied = set(map(str, result.influence["input"])) if len(result.influence) else set()
        held = [c for c in self._features() if str(c) not in varied]
        if held:
            notes.append("Held at one value, so not sampled: " + ", ".join(held)
                         + ".  The result is conditional on those values — untick or change "
                           "one and run it again to see how much they matter.")
        skipped = list(getattr(tornado, "attrs", {}).get("skipped") or [])
        if skipped:
            notes.append("Varied, but not on the one-at-a-time chart because it has no "
                         "quantiles to move between: " + ", ".join(skipped))
        self.notes.setText("  ·  ".join(notes))

        # ---- safety factors against a chosen demand
        demand = _number(self.demand_edit.text().strip())
        if demand:
            info = MC.safety_factors(result.values, demand)
            self.safety_view.set_frame(pd.DataFrame(
                [{"quantity": k, "value": v} for k, v in info.items() if k != "factors"]))
            self.state.remember_section("Safety factors", self.safety_view.current_frame())
        else:
            self.safety_view.set_frame(None)

        self.state.remember_section("Monte-Carlo outcome distribution",
                                    self.summary_view.current_frame())
        self.state.remember_section("Monte-Carlo percentiles", result.percentiles)
        self.state.remember_section("Sensitivity, one input at a time", tornado)
        self.state.remember_section("Influence of each input", result.influence)
        self.state.log_line(f"reliability: {result.sentence()[:150]}")
        self._draw()

    def _story(self, result, tornado) -> str:
        parts = []
        if result.influence is not None and len(result.influence) and \
                "rank correlation" in result.influence.columns:
            first = result.influence.iloc[0]
            parts.append(f"“{first['input']}” matters most "
                         f"(rank correlation {first['rank correlation']:+.2f}).")
        if tornado is not None and len(tornado) and "swing" in tornado.columns:
            top = tornado.iloc[0]
            share = top.get("% of total swing", float("nan"))
            parts.append(f"Moving “{top['input']}” alone across its likely range shifts the "
                         f"answer by {top['swing']:,.4g}"
                         + (f", {share:,.0f}% of the total movement." if np.isfinite(share) else "."))
        if result.probability is not None and result.reliability_index is not None:
            if result.reliability_index < 0:
                parts.append("The mean is already on the wrong side of the limit, so this "
                             "design does not work at these settings.")
            elif result.reliability_index < 1.5:
                parts.append(f"β = {result.reliability_index:.2f} is a real risk — a change of "
                             "something small would tip it over.")
            elif result.reliability_index < 3.0:
                parts.append(f"β = {result.reliability_index:.2f} is moderate; typical "
                             "structural code targets are 3.0 to 4.0.")
            else:
                parts.append(f"β = {result.reliability_index:.2f} is comfortable by structural "
                             "code standards.")
        if result.summary.get("coefficient of variation %") is not None:
            cv = result.summary["coefficient of variation %"]
            if np.isfinite(cv):
                parts.append(f"The answer varies by {cv:,.1f}% of its own mean.")
        return "  ".join(parts)

    # ---------------------------------------------------------------- charts
    def _draw(self):
        result = getattr(self.state, "monte_carlo", None)
        if result is None:
            return
        kind = self.chart_combo.currentData()
        task = self.state.train_spec.task
        label = "probability of the positive class" if task != "regression" else "predicted outcome"
        try:
            figure = self._figure_for(kind, result, label)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not draw it", f"{type(exc).__name__}: {exc}")
            return
        if figure is None:
            self.canvas.clear()
            return
        self.canvas.show_figure(figure)
        self.state.remember_figure(f"Uncertainty — {self.chart_combo.currentText()}",
                                   _figure_bytes(figure))

    def _figure_for(self, kind, result, label):
        if kind == "histogram":
            return PEG.simulation_histogram(result.values, result.threshold,
                                            result.direction, label)
        if kind == "tornado":
            if self._tornado is None or not len(self._tornado):
                raise PEG.PlotError("run the simulation first")
            baseline = float(self._tornado["baseline"].iloc[0]) \
                if "baseline" in self._tornado.columns else None
            return PEG.tornado_plot(self._tornado, baseline)
        if kind == "influence":
            return PEG.influence_plot(result.influence)
        if kind == "exceedance":
            return PEG.exceedance_plot(MC.exceedance_curve(result.values), label)
        if kind == "inputs":
            spec = getattr(self, "_last_spec", None)
            return PEG.distribution_review_plot(result.frame,
                                                spec.inputs if spec is not None else None)
        return None

    # --------------------------------------------------------------- exports
    def _export(self):
        result = self.state.monte_carlo
        if result is None:
            self.warn("Nothing to export", "Run the simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the simulation", os.path.join(self.state.project_dir,
                                                        "uncertainty.xlsx"),
            "Excel workbook (*.xlsx)")
        if not path:
            return
        from core import export_office as EO

        sheets = {
            "Outcome summary": self.summary_view.current_frame(),
            "Percentiles": result.percentiles,
            "Sensitivity one at a time": self._tornado,
            "Influence": result.influence,
            "Exceedance curve": MC.exceedance_curve(result.values),
            "Inputs as drawn": result.frame,
        }
        sheets = {name: table for name, table in sheets.items()
                  if isinstance(table, pd.DataFrame) and len(table)}
        try:
            EO.write_excel(path, sheets,
                           notes=[result.sentence()] + list(result.notes),
                           title="Uncertainty and reliability")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not write the workbook",
                                 f"{type(exc).__name__}: {exc}")
            return
        self.state.log_line(f"reliability: exported {len(sheets)} table(s) to {path}")
        QMessageBox.information(self, "Exported",
                               f"Wrote {len(sheets)} sheet(s) to\n{path}")

    def _save_setup(self):
        try:
            spec = self._spec()
        except Exception as exc:  # noqa: BLE001
            self.warn("Cannot save the setup", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the simulation setup",
            os.path.join(self.state.project_dir, "uncertainty.json"), "JSON (*.json)")
        if not path:
            return
        payload = spec.to_dict()
        payload["task"] = self.state.train_spec.task
        payload["model"] = self.state.model_key
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save it", str(exc))
            return
        self.state.log_line(f"reliability: setup saved to {path}")

    def _load_setup(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a simulation setup", self.state.project_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not read it", str(exc))
            return
        self.samples_spin.setValue(int(payload.get("n_samples", 5000)))
        self.seed_spin.setValue(int(payload.get("seed", 42)))
        self.corr_check.setChecked(bool(payload.get("keep_correlations", False)))
        index = self.fill_combo.findData(payload.get("fill", "median"))
        self.fill_combo.setCurrentIndex(index if index >= 0 else 0)
        index = self.direction_combo.findData(payload.get("direction", "below"))
        self.direction_combo.setCurrentIndex(index if index >= 0 else 0)
        threshold = payload.get("threshold")
        self.threshold_edit.setText("" if threshold is None else str(threshold))
        wanted = {item["column"]: item for item in payload.get("inputs", [])}
        for row in range(self.grid.rowCount()):
            name_item = self.grid.item(row, COLUMN_NAME)
            vary = self.grid.item(row, COLUMN_VARY)
            combo = self.grid.cellWidget(row, COLUMN_DIST)
            if name_item is None or combo is None:
                continue
            item = wanted.get(name_item.text())
            if vary is not None:
                vary.setCheckState(Qt.CheckState.Checked if item else Qt.CheckState.Unchecked)
            if not item:
                continue
            index = combo.findData(item.get("distribution", "normal"))
            if index >= 0:
                combo.setCurrentIndex(index)
            params = item.get("params") or {}
            self.grid.setItem(row, COLUMN_PARAMS, QTableWidgetItem(
                ", ".join(f"{k} = {_short(v)}" for k, v in params.items())))
        self.state.log_line(f"reliability: setup loaded from {path}")

    # -------------------------------------------------------------- refresh
    def refresh(self):
        token = (self.data_token(), self.state.model_key,
                 None if self.state.pipeline is None else id(self.state.pipeline))
        if token == getattr(self, "_token", None):
            return
        self._token = token
        if self.state.df is None or not len(self.state.df):
            return
        # A new fit invalidates the old simulation, so start the grid again from the data.
        if self.state.monte_carlo is None:
            self._reload_grid()
            self._default_threshold()
        else:
            self._show(self.state.monte_carlo, self._tornado)

    def _default_threshold(self):
        target = self.state.train_spec.target
        frame = self.state.df
        if frame is None or not target or target not in frame.columns:
            return
        if self.state.train_spec.task == "regression":
            values = pd.to_numeric(frame[target], errors="coerce").dropna()
            if len(values):
                self.threshold_edit.setText(f"{values.quantile(0.25):.4g}")
                index = self.direction_combo.findData("below")
                self.direction_combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            self.threshold_edit.setText("0.5")
            index = self.direction_combo.findData("above")
            self.direction_combo.setCurrentIndex(index if index >= 0 else 0)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _number(text):
    if text is None or text == "":
        return None
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def _param_value(text):
    """A parameter typed in the grid: a number if it is one, otherwise the words.

    ``constant`` has to accept a *name* as well as a number — "hold the curing method at
    steam" is a real question — and turning that into ``None`` would quietly simulate the
    wrong thing instead of the thing that was asked for.
    """
    number = _number(text)
    return number if number is not None else str(text).strip()


def _short(value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{value:,.4g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        return f"{len(value)} value(s)"
    return str(value)


def _figure_bytes(figure, dpi: int = 130) -> bytes:
    from core import plots as plots_mod

    try:
        return plots_mod.figure_png_bytes(figure, dpi=dpi)
    except Exception:  # noqa: BLE001
        return b""
