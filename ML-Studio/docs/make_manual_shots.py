"""Regenerate the screenshots used by USER_MANUAL.md.

The manual is rendered inside the application, so its images have to live in the tree
next to it.  Rather than hand-cropping them, this script drives the real window through
the manual's own example project and photographs the result — so the pictures cannot
drift away from what the program actually does.

Run it from the project root with the project's own interpreter::

    ./.venv/bin/python docs/make_manual_shots.py

It needs no display; Qt renders offscreen.  Re-run it after any change to a screen, and
commit the PNGs it writes to ``docs/manual/``.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "manual")
sys.path.insert(0, PROJECT)

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTableWidgetItem  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.steps import (  # noqa: E402
    AutoPage, CleanPage, EdaPage, EvaluatePage, FeaturesPage, LoadPage, ModelPage,
    PlotsPage, PredictPage, SavePage, StartPage, TunePage,
)
from gui.steps.data_steps import load_job  # noqa: E402
from gui.steps.model_steps import train_job  # noqa: E402

WINDOW = (1380, 900)
#: The nine mix variables the concrete example is really about.  The identifer and the cast
#: date are not mix design, so the example model does not use them.
MIX_FEATURES = ["Cement", "Blast furnace slag", "Fly ash", "Water", "Superplasticiser",
                "Fine aggregate", "Coarse aggregate", "Age (days)", "Curing method"]
#: The manual is read in a dialog about 1080 px wide, and Qt lays images out at their
#: natural size — a full-size grab would force a horizontal scrollbar on every page.  So
#: the window is rendered at a comfortable size and the picture is then scaled to fit.
MAX_IMAGE_WIDTH = 1000


def wait_idle(app, window, timeout: float = 600.0):
    """Spin the event loop until the shared worker slot is free again."""
    import time

    deadline = time.time() + timeout
    while window.runner.busy and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    app = QApplication(sys.argv[:1])
    window = MainWindow(PROJECT)
    window.resize(*WINDOW)
    window.show()
    app.processEvents()
    state = window.state

    # Some buttons finish by telling you what they did in a message box.  That is right for
    # a person and fatal for a script — a modal dialog with no user blocks the event loop
    # forever — so the timer closes any that appear.  The action has already happened.
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox

    def _dismiss():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.isVisible():
                print(f"    (closed a message box: {widget.text()[:60]!r})")
                widget.close()

    dismisser = QTimer()
    dismisser.timeout.connect(_dismiss)
    dismisser.start(60)

    def shoot(page_class, name: str):
        page = window._go_to(page_class)
        for hook in ("on_enter", "refresh", "update_requirements"):
            if hasattr(page, hook):
                getattr(page, hook)()
        app.processEvents()

        # A screen with more on it than fits scrolls — that is deliberate, because a window
        # that insists on being tall enough for its longest screen cannot be maximized on a
        # small display.  A *picture* of a scrollbar and half a form is not much use in a
        # manual though, so if the screen does not fit, the window is made taller for the
        # photograph and put back afterwards.  The picture is scaled to MAX_IMAGE_WIDTH
        # either way, so a taller grab is still comfortable to read in the dialog.
        grow_to = WINDOW[1]
        area = getattr(page, "_scroll", None)
        if area is not None and area.verticalScrollBar().maximum() > 0:
            chrome = WINDOW[1] - area.viewport().height()
            grow_to = min(area.widget().sizeHint().height() + chrome + 6, 1400)
        if grow_to != window.height():
            window.resize(WINDOW[0], grow_to)
            for _ in range(3):
                app.processEvents()

        pixmap = window.grab()
        if pixmap.width() > MAX_IMAGE_WIDTH:
            pixmap = pixmap.scaledToWidth(MAX_IMAGE_WIDTH,
                                           Qt.TransformationMode.SmoothTransformation)
        path = os.path.join(OUT, f"{name}.png")
        pixmap.save(path)
        print(f"wrote {path}  ({pixmap.width()}x{pixmap.height()})")
        if window.height() != WINDOW[1]:
            window.resize(*WINDOW)
            app.processEvents()

    # 1 — the opening screen, with nothing loaded yet
    shoot(StartPage, "start")

    # 2 — load the example project
    load = window._go_to(LoadPage)
    tables = load_job(lambda *a: None, None,
                      os.path.join(PROJECT, "examples", "customer_churn.csv"), "csv", {})
    load._loaded(tables)
    app.processEvents()
    shoot(LoadPage, "load")

    # 3 — a real column profile, so the picture is not an empty table
    eda = window._go_to(EdaPage)
    eda._run_analysis()
    wait_idle(app, window)
    shoot(EdaPage, "explore")

    shoot(CleanPage, "clean")
    shoot(FeaturesPage, "features")

    # 4 — choose the target and train
    model = window._go_to(ModelPage)
    model.target_combo.setCurrentText("churned")
    model._all_features()
    state.train_spec = model._read_spec()
    key, params = model.current_model()
    result = train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                       state.pre_spec.to_dict(), key, params, True)
    model._trained(result)
    app.processEvents()
    shoot(ModelPage, "model")

    # 5 — the results
    shoot(EvaluatePage, "evaluate")
    shoot(TunePage, "tune")
    shoot(SavePage, "save")
    shoot(PredictPage, "predict")

    # 6 — one graph actually drawn
    plots_page = window._go_to(PlotsPage)
    plots_page.refresh()
    plots_page._draw()
    wait_idle(app, window)
    shoot(PlotsPage, "graphs")

    shoot(AutoPage, "auto")

    # 7 — the neural network, so the deep-learning section shows its real settings
    model = window._go_to(ModelPage)
    index = model.model_combo.findData("torch_mlp_clf")
    if index >= 0:
        model.model_combo.setCurrentIndex(index)
        app.processEvents()
        state.train_spec = model._read_spec()
        key, params = model.current_model()
        result = train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                           state.pre_spec.to_dict(), key, params, True)
        model._trained(result)
        app.processEvents()
        shoot(ModelPage, "network")
    else:
        print("!! torch is not available — skipping the network picture")

    # ---------------------------------------------------------------- engineering
    # Screens 13 to 18 answer the questions that arrive as a number: how likely is a batch
    # to miss the specification, is the pile still moving, which factors matter.  They read
    # best on engineering data, so the rest of the pictures load the trial report and the
    # monitoring record.  Everything above needs the churn classifier, so it stays first.
    from gui.steps import (  # noqa: E402
        CurveFitPage, DoePage, ReliabilityPage, StatisticsPage, TimeSeriesPage, UnitsPage,
    )
    from gui.steps.engineering_steps import fill_grid  # noqa: E402

    def load_file(name: str):
        tables = load_job(lambda *a: None, None,
                          os.path.join(PROJECT, "examples", name), "csv", {})
        load._loaded(tables)
        app.processEvents()

    def train(target: str, features=None, key: str = "random_forest_reg"):
        page = window._go_to(ModelPage)
        index = page.model_combo.findData(key)
        if index >= 0:
            page.model_combo.setCurrentIndex(index)
        page.target_combo.setCurrentText(target)
        if features:
            # The reliability screen's picture only makes sense on a model built from the
            # mix variables — an id column and the cast date are not mix design, and feeding
            # them in moves the strength distribution for the wrong reason.
            page.features_form.set_values({"features": list(features)})
        else:
            page._all_features()
        state.train_spec = page._read_spec()
        chosen, params = page.current_model()
        result = train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                           state.pre_spec.to_dict(), chosen, params, True)
        page._trained(result)
        app.processEvents()
        return result

    load_file("concrete_mix.csv")

    # 8 — statistics on the trial report: a real test, so the panel is not empty
    stats = window._go_to(StatisticsPage)
    stats.compare_a.setCurrentText("Curing method")
    stats.compare_b.setCurrentText("Compressive strength")
    stats._run_compare()
    app.processEvents()
    shoot(StatisticsPage, "stats")

    # 9 — every curve form ranked on the one relationship every engineer knows
    fit = window._go_to(CurveFitPage)
    fit.x_combo.setCurrentText("Age (days)")
    fit.y_combo.setCurrentText("Compressive strength")
    fit.form_combo.setCurrentIndex(fit.form_combo.findData(""))     # rank them all
    fit.solve_edit.setText("50")
    fit._fit()
    app.processEvents()
    shoot(CurveFitPage, "curvefit")

    # 10 — an experiment planned and then analysed against a response that has a real effect
    doe = window._go_to(DoePage)
    doe.kind_combo.setCurrentIndex(doe.kind_combo.findData("full"))
    table = doe.factors_table
    table.setRowCount(3)
    for row, (name, low, high) in enumerate(
            [("Cement", 300, 450), ("Water", 150, 190), ("Age (days)", 7, 56)]):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(str(low)))
        table.setItem(row, 2, QTableWidgetItem(str(high)))
    doe._build()
    design = state.doe
    coded = design.coded
    names = list(coded.columns)
    rng = np.random.default_rng(11)
    response = (44.0 + 5.0 * coded[names[0]] - 2.5 * coded[names[1]]
                + 2.0 * coded[names[0]] * coded[names[1]]
                + 1.0 * coded[names[2]] + rng.normal(0, 0.25, len(design)))
    fill_grid(doe.response_grid, pd.DataFrame({"measured response": np.round(response, 3)}))
    doe.response_combo.setCurrentIndex(doe.response_combo.findData("__table__"))
    doe.order_combo.setCurrentIndex(1)
    doe._analyse()
    app.processEvents()
    shoot(DoePage, "doe")

    # 11 — the model the reliability screen drives
    train("Compressive strength", MIX_FEATURES)
    reliability = window._go_to(ReliabilityPage)
    reliability.samples_spin.setValue(5000)
    reliability.threshold_edit.setText("37")
    reliability.direction_combo.setCurrentIndex(
        reliability.direction_combo.findData("below"))
    reliability._run()
    wait_idle(app, window)
    shoot(ReliabilityPage, "reliability")

    # 12 — units, with one conversion of a number and one of a whole column
    units = window._go_to(UnitsPage)
    units.quantity_combo.setCurrentIndex(units.quantity_combo.findData("pressure"))
    units.value_edit.setText("30")
    units.from_combo.setCurrentText("MPa")
    units.to_combo.setCurrentText("psi")
    units._convert()
    if "Compressive strength" in [units.column_combo.itemText(i)
                                  for i in range(units.column_combo.count())]:
        units.column_combo.setCurrentText("Compressive strength")
        units.column_from.setCurrentText("MPa")
        units.column_to.setCurrentText("psi")
        units._convert_column()
    app.processEvents()
    shoot(UnitsPage, "units")

    # 13 — the monitoring record, charted the way it has to be charted
    load_file("pile_settlement_monitoring.csv")
    series = window._go_to(TimeSeriesPage)
    series.time_combo.setCurrentText("date")
    series.value_combo.setCurrentText("settlement_mm")
    series.analysis_combo.setCurrentIndex(series.analysis_combo.findData("trend"))
    series._run()
    series.window_spin.setValue(4)
    series._add_features()
    app.processEvents()
    series.analysis_combo.setCurrentIndex(series.analysis_combo.findData("control"))
    series.chart_combo.setCurrentIndex(series.chart_combo.findData("individuals"))
    series.value_combo.setCurrentText("settlement_mm (change per step)")
    series._run()
    app.processEvents()
    shoot(TimeSeriesPage, "timeseries")

    print(f"\n{len(os.listdir(OUT))} images in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
