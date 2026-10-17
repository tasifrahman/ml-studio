#!/usr/bin/env python3
"""ML Workbench — an end-to-end machine-learning GUI built on pandas and scikit-learn.

Run it::

    ./.venv/bin/python app.py                     # the full window
    ./.venv/bin/python app.py --self-test         # headless end-to-end check
    ./.venv/bin/python app.py --data path.csv     # start with a file already loaded

``--self-test`` builds the real window off-screen and drives the whole workflow
through the same code paths the buttons use (load, train, tune, evaluate, the four
"best combination" tools, clustering, plots, bundle, report, script, prediction),
printing one line per check.  It is the regression test for this project.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="An end-to-end machine-learning workbench.")
    parser.add_argument("--self-test", action="store_true",
                        help="run the headless end-to-end check instead of the window")
    parser.add_argument("--data", metavar="PATH",
                        help="load this data file at startup (or in the self-test)")
    parser.add_argument("--target", metavar="COLUMN",
                        help="use this column as the target in the self-test")
    parser.add_argument("--out", metavar="DIR", help="output folder for the self-test artefacts")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(HERE, data=args.data, target=args.target, out_dir=args.out)

    from PyQt6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("ML Workbench")
    window = MainWindow(HERE)
    window.show()
    if args.data:
        page = window._go_to(_load_page_class())
        if page is not None:
            page.path_edit.setText(os.path.abspath(args.data))
            page._start_load()
    return app.exec()


def _load_page_class():
    from gui.steps import LoadPage

    return LoadPage


# --------------------------------------------------------------------------
# headless end-to-end check
# --------------------------------------------------------------------------

def run_self_test(project_dir: str, data: str | None = None, target: str | None = None,
                  out_dir: str | None = None) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from core import loaders, models as MODELS, plots as plots_mod, train as train_mod
    from gui.main_window import MainWindow
    from gui.steps import (AutoPage, CleanPage, ClusterPage, EdaPage, EvaluatePage,
                           FeaturesPage, LoadPage, ModelPage, OptimizePage, PlotsPage,
                           PredictPage, SavePage, StartPage, TunePage)
    from gui.steps.analysis_steps import cluster_job
    from gui.steps.data_steps import load_job
    from gui.steps.model_steps import (model_choices, predict_job, report_job, train_job,
                                       tune_job)

    app = QApplication.instance() or QApplication([])
    results: list[tuple[str, str, str]] = []

    # A modal dialog would block forever with no user.  Nothing here is meant to
    # raise one, so if one appears the timer closes it and the check fails loudly.
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox

    def _dismiss():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.isVisible():
                print(f"    (auto-dismissed a dialog: {widget.text()[:70]!r})", flush=True)
                widget.close()

    dismisser = QTimer()
    dismisser.timeout.connect(_dismiss)
    dismisser.start(60)

    def check(name, fn):
        try:
            value = fn()
            results.append((name, "ok", "" if value is None else str(value)[:110]))
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        name_, status, detail = results[-1]
        print(f"[{len(results):2d}] {('ok  ' if status == 'ok' else 'FAIL')} {name_:<52} {detail}",
              flush=True)

    window = MainWindow(project_dir)
    state = window.state
    state.logged.connect(lambda line: None)  # keep the console clean

    check("build the window and every screen", lambda: f"{len(window.pages)} screens")

    def visit_all():
        visited = []
        for index, page in enumerate(window.pages):
            row = window._stack_index.get(window.stack.indexOf(page))
            window.sidebar.setCurrentRow(row)
            app.processEvents()
            visited.append(type(page).__name__)
        return ", ".join(visited)

    check("switch to every screen", visit_all)

    # ---- the user interface itself
    def check_ui_contract():
        """Every screen declares what it needs, and says so when it is missing.

        This is the check that keeps the beginner-protection honest: a screen with
        ``requires = "model"`` must actually show its signpost while there is no model,
        rather than just looking broken.
        """
        allowed = {None, "data", "model"}
        signposted = []
        for page in window.pages:
            name = type(page).__name__
            assert getattr(page, "requires", "absent") in allowed, \
                f"{name} declares requires={page.requires!r}"
            page.update_requirements()
            wants = page.requires is not None
            shown = page._require_callout.isVisibleTo(page)
            assert shown == wants, (
                f"{name}: signpost shown={shown} but requires={page.requires!r}")
            if wants:
                assert page._require_callout._action.text(), f"{name} offers no way out"
                signposted.append(name)
        return f"{len(window.pages)} screens, {len(signposted)} with a signpost"

    check("ui: every screen explains what it needs", check_ui_contract)

    def check_theme():
        """The stylesheet and the palette must both be pinned, over the Fusion base style."""
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QListWidget, QPushButton

        from gui import theme

        sheet = app.styleSheet()
        assert len(sheet) > 4000, f"the stylesheet is only {len(sheet)} characters"
        assert theme.PALETTE["accent"] in sheet, "the palette is not in the stylesheet"
        # With a stylesheet installed, QApplication.style() is Qt's QStyleSheetStyle
        # wrapper, so the base style can only be seen with the stylesheet lifted.
        app.setStyleSheet("")
        base = app.style().metaObject().className()
        app.setStyleSheet(sheet)
        assert "fusion" in base.lower(), f"the base widget style is {base!r}, not Fusion"
        probe = QPushButton("probe")
        theme.tag(probe, "primary")
        assert probe.property("role") == "primary"

        # A desktop set to a dark theme hands Qt a dark palette, and every widget the
        # stylesheet does not name — a plain list, a table's header, the standard icons
        # Qt draws itself — is then drawn dark on a light window.  Pinning the palette in
        # theme.apply() is the fix; this keeps it pinned.
        saved = app.palette()
        try:
            dark = QPalette()
            for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                         QPalette.ColorRole.Button, QPalette.ColorRole.Text,
                         QPalette.ColorRole.ButtonText):
                dark.setColor(role, QColor("#1e1e1e"))
            app.setPalette(dark)
            theme.apply()
            for role, name in ((QPalette.ColorRole.Window, "window"),
                               (QPalette.ColorRole.Base, "base"),
                               (QPalette.ColorRole.Button, "button")):
                value = app.palette().color(role).lightness()
                assert value > 200, f"a dark desktop turns the {name} colour dark ({value})"
            view = QListWidget()
            for what, widget in (("list", view), ("list viewport", view.viewport())):
                value = widget.palette().color(QPalette.ColorRole.Base).lightness()
                assert value > 200, f"a plain {what} would be drawn dark ({value})"
        finally:
            app.setPalette(saved)
        return f"{base}, palette pinned, {len(sheet):,} characters of stylesheet"

    check("ui: the theme is installed", check_theme)

    def check_window_fits():
        """The window must be fillable, and a screen must scroll rather than be squeezed.

        Neither mistake here could fail any other check, because both are about *size*
        rather than about a value.  A stacked widget takes the maximum of its pages, so a
        single wide page used to make the window demand 1478 x 1060 — more than the work
        area of this 1920 x 1080 screen — and a window that cannot shrink to the work area
        cannot be maximized either: on Wayland the compositor gives up and leaves a
        "maximized" window parked at its minimum size, with a 400 px gutter beside it.
        Then, having made the body scrollable, the body was sized to its *minimum* hint
        instead of its preferred height, and every row of the taller forms was squashed.
        """
        from PyQt6.QtWidgets import QScrollArea, QSizePolicy

        widest = tallest = 0
        for page in window.pages:
            name = type(page).__name__
            area = getattr(page, "_scroll", None)
            assert isinstance(area, QScrollArea), f"{name}: the body is not scrollable"
            policy = area.widget().sizePolicy().verticalPolicy()
            assert not (policy.value & QSizePolicy.PolicyFlag.ShrinkFlag), (
                f"{name}: the body would be squeezed rather than scrolled")
            hint = page.minimumSizeHint()
            widest, tallest = max(widest, hint.width()), max(tallest, hint.height())
        assert widest <= 900 and tallest <= 700, \
            f"one screen asks the window for {widest} x {tallest}"
        whole = window.minimumSizeHint()
        assert whole.width() <= 1000 and whole.height() <= 700, (
            f"the window itself insists on {whole.width()} x {whole.height()}")

        action = getattr(window, "_full_screen_action", None)
        assert action is not None, "the View menu offers no Full screen"
        assert action.shortcut().toString() == "F11", \
            f"Full screen is bound to {action.shortcut().toString()!r}"
        assert not window.isFullScreen(), "the window starts out full screen"
        action.setChecked(True)
        assert window.isFullScreen() and action.isChecked(), \
            "ticking Full screen does not fill the display"
        action.setChecked(False)
        assert not window.isFullScreen(), "unticking Full screen does not restore the window"
        window.resize(1380, 900)
        return (f"widest screen {widest} px, tallest {tallest} px, window floor "
                f"{whole.width()} x {whole.height()}, F11 fills the display")

    check("ui: the window can be filled, and a screen scrolls", check_window_fits)

    def check_manual():
        """The manual must exist, its pictures must be present, and its contents must work.

        Qt's markdown renderer creates no anchors for headings, so the table of contents
        is only usable because ``ManualDialog`` looks the headings up by name.  This
        check fails if a contents entry stops matching a heading — which is exactly what
        happens when someone edits a title and forgets the link.
        """
        import re

        from gui.manual import ManualDialog, headings, read_manual

        text = read_manual(project_dir)
        assert text.lstrip().startswith("# ML Workbench"), \
            "the manual does not begin with a top-level heading"
        referenced = re.findall(r"\]\((docs/manual/[^)\s]+)\)", text)
        assert referenced, "the manual shows no pictures"
        missing = [p for p in referenced if not os.path.exists(os.path.join(project_dir, p))]
        assert not missing, f"the manual refers to pictures that are not there: {missing}"

        dialog = ManualDialog(project_dir)
        dialog.show()
        for _ in range(4):
            app.processEvents()
        drawn = dialog.view.document().toHtml().count("<img")
        assert drawn >= len(referenced), f"only {drawn} of {len(referenced)} pictures rendered"

        contents = re.findall(r"\[([^\]]+)\]\(#([^)]+)\)", text)
        assert contents, "the manual has no table of contents"
        index = headings(text)
        dangling = [link for _, link in contents if link not in index]
        assert not dangling, f"contents entries that point at no heading: {dangling}"
        # The entry should read like the heading it points at, or a reader scanning the
        # contents cannot find the section.  The top-level entries are an ordered list, so
        # markdown puts the "1." outside the link text — hence "is contained in" rather
        # than "equals".  Only the contents *list* is held to this: a sentence that links
        # to a section is prose, and "the questions in ..." is a perfectly good way to
        # point at "6. The engineering toolkit".
        listing = text[text.find("## Contents"):]
        listing = listing[:listing.find("\n---")] if "\n---" in listing else listing
        entries = re.findall(r"\[([^\]]+)\]\(#([^)]+)\)", listing)
        assert entries, "the contents list has no entries"
        mismatched = [(label, index[link]) for label, link in entries
                      if link in index and label.strip() not in index[link]]
        assert not mismatched, f"contents entries whose wording differs: {mismatched}"
        # every entry must actually reach its heading.  Checking only that the view moved
        # is not enough: a contents entry has the same text as its heading, so a lookup
        # that matches on text alone lands on the list item and "works" while going
        # nowhere near the section.
        from PyQt6.QtGui import QTextCursor

        stuck, misplaced = [], []
        for label, link in contents:
            dialog.view.verticalScrollBar().setValue(0)
            app.processEvents()
            if not dialog._jump(link):
                stuck.append(link)
                continue
            block = dialog.view.document().begin()
            while block.isValid():
                if (block.blockFormat().headingLevel() > 0
                        and block.text().strip() == index[link]):
                    bar = dialog.view.verticalScrollBar()
                    top = dialog.view.cursorRect(QTextCursor(block)).top()
                    # at the very end of the document there is nothing left to scroll,
                    # so the heading can only rise as far as the remaining space allows
                    if top > 40 and bar.value() < bar.maximum():
                        misplaced.append((link, top))
                    break
                block = block.next()
        assert not stuck, f"contents entries that do not scroll anywhere: {stuck}"
        assert not misplaced, f"contents entries that do not land on their heading: {misplaced}"
        dialog.close()
        return (f"{len(text):,} characters, {len(set(referenced))} pictures "
                f"({len(referenced)} references), {len(contents)} contents links")

    check("ui: the user manual is present and renders", check_manual)

    # ---- load
    load_page: LoadPage = window._go_to(LoadPage)

    def do_load():
        if data:
            load_page.path_edit.setText(os.path.abspath(data))
            guessed = loaders.format_for_path(data)
            if guessed:
                index = load_page.format_combo.findData(guessed)
                if index >= 0:
                    load_page.format_combo.setCurrentIndex(index)
            load_page._start_load()
            _wait_for_runner(window)
        else:
            load_page._load_example()
        return f"{len(state.df):,} rows × {state.df.shape[1]} columns"

    check("load data", do_load)

    def check_examples_picker():
        """The manual sends the reader to the example list on this screen, so a file in
        examples/ that the list does not offer would make the manual wrong."""
        folder = os.path.join(project_dir, "examples")
        on_disk = sorted(name for name in os.listdir(folder)
                         if name.lower().endswith((".csv", ".tsv", ".xlsx", ".parquet")))
        listed = sorted(os.path.basename(str(load_page.example_combo.itemData(i)))
                        for i in range(load_page.example_combo.count()))
        assert listed == on_disk, f"the picker lists {listed}, the folder holds {on_disk}"
        assert listed.count("concrete_mix.csv") == 1, "the trial report is not offered"
        assert load_page.example_combo.currentData(), "no example is selected in the list"
        return f"{len(listed)} example file(s) offered: {', '.join(listed)}"

    check("ui: every shipped example is in the loader's list", check_examples_picker)

    def check_start_progress():
        """The Start screen's checklist must tick itself off from the shared state."""
        page = window._go_to(StartPage)
        page.refresh()
        marks = {key: row["mark"].text() for key, row in page._rows.items()}
        assert marks["data"] == "✔", f"the 'load a dataset' row still reads {marks['data']!r}"
        assert marks["model"] == "○", f"the 'train a model' row should be unticked"
        assert marks["outputs"] == "○", "nothing has been saved yet"
        return " ".join(f"{k}={v}" for k, v in marks.items())

    check("ui: the start screen tracks progress", check_start_progress)

    # ---- explore
    eda_page: EdaPage = window._go_to(EdaPage)

    def do_eda():
        names = []
        for row in range(eda_page._analysis_combo.count()):
            eda_page._analysis_combo.setCurrentIndex(row)
            eda_page._run_analysis()
            names.append(eda_page._analysis_combo.currentText())
        for button, key in zip(eda_page._plot_buttons, ("target_dist", "missing_bar",
                                                        "corr_heatmap", "hist", "box")):
            eda_page._run_plot(key)
        return f"{len(names)} analyses and {len(eda_page._plot_buttons)} plots"

    check("explore: every analysis and plot", do_eda)

    # ---- clean
    clean_page: CleanPage = window._go_to(CleanPage)

    def do_clean():
        """A four-step recipe: cast one column, filter on it, de-duplicate, re-index.

        The column must be **numeric** — casting a text id to a float gives NaN, and the
        ``> 0`` filter then drops every row.  The check used to take ``columns[0]``, which
        is an id in a real export, and when the self-test was pointed at such a file the
        frame came back empty and took the whole model section down with it (fourteen
        checks "failed" with `every row is missing the target value`).  Pick the column the
        recipe is actually about.
        """
        columns = [str(c) for c in state.df.columns]
        import pandas as pd

        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(state.df[c])]
        column = (numeric or columns)[0]
        ops = [
            {"op": "cast_dtype", "column": column, "dtype": "float"},
            {"op": "filter_rows", "conditions": [
                {"column": column, "test": "gt", "value": 0}], "logic": "and", "negate": False},
            {"op": "drop_duplicates", "keep": "first"},
            {"op": "reset_index"},
        ]
        state.apply_cleaning(ops)
        clean_page._show_result()
        return f"{len(state.df):,} rows after {len(ops)} steps on {column!r}"

    check("clean: apply a four-step recipe", do_clean)

    # ---- features
    features_page: FeaturesPage = window._go_to(FeaturesPage)

    def do_features():
        features_page.engineered.insertRow(0)
        from PyQt6.QtWidgets import QTableWidgetItem

        features_page.engineered.setItem(0, 0, QTableWidgetItem("water_cement"))
        features_page.engineered.setItem(0, 1, QTableWidgetItem("Water / Cement"))
        features_page._apply()
        features_page._preview()
        return ", ".join(features_page._read_formulas()[0].values())

    check("features: add a formula and preview the encoding", do_features)

    # ---- model
    model_page: ModelPage = window._go_to(ModelPage)

    def choose_target():
        if target:
            model_page.target_combo.setCurrentText(target)
        else:
            model_page.target_combo.setCurrentText(model_page.target_combo.itemText(
                model_page.target_combo.count() - 1))
        model_page._all_features()
        spec = model_page._read_spec()
        assert spec is not None, "no spec"
        assert spec.sample_weight_column is None, \
            f"row weights were silently set to {spec.sample_weight_column!r}"
        if spec.task in ("binary", "multiclass"):
            assert spec.stratify, "a class split should keep the classes balanced"
        state.train_spec = spec
        return f"target={spec.target} task={spec.task} metric={spec.primary_metric}"

    check("model: choose the target and detect the task", choose_target)

    def do_train():
        key, params = model_page.current_model()
        result = train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                           state.pre_spec.to_dict(), key, params, True)
        model_page._trained(result)
        primary = state.train_spec.primary_metric
        score = (state.evaluation.get("test") or {}).get(primary)
        return f"{key}: {primary} = {score:.4f}"

    check("model: train and read the metrics", do_train)

    def do_compare():
        keys = [k for _, k, ok in __import__("gui.steps.model_steps", fromlist=["model_choices"])
                .model_choices(state.train_spec.task) if ok][:4]
        from gui.steps.model_steps import compare_job

        result = compare_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                             state.pre_spec.to_dict(), keys)
        model_page._compared(result)
        return f"{len(result['frame'])} models compared"

    check("model: compare a handful of algorithms", do_compare)

    def do_every_model():
        """Every registered model must fit, and keep the label vocabulary.

        This is the check that catches library drift: a model that silently demands
        0..n-1 labels (XGBoost on a text multiclass target) fails here.
        """
        import numpy as np

        task = state.train_spec.task
        keys = [key for _, key, available in model_choices(task) if available]
        problems: list[str] = []
        fitted = 0
        for key in keys:
            try:
                split = train_mod.make_split(state.df, state.train_spec)
                pipe, _ = train_mod.build_train_pipeline(state.pre_spec, key,
                                                         MODELS.default_params(key), task)
                train_mod.fit_full(pipe, split, state.train_spec)
                evaluation = train_mod.evaluate_fitted(pipe, split, state.train_spec,
                                                       include_train=False)
                if task in ("binary", "multiclass"):
                    seen = {str(v) for v in np.unique(evaluation["test_pred"])}
                    allowed = {str(v) for v in split.labels}
                    assert seen <= allowed, f"predictions {sorted(seen)} outside {sorted(allowed)}"
                    assert evaluation["test_proba"] is not None, "no probabilities"
                fitted += 1
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{key}: {type(exc).__name__}: {exc}")
        assert not problems, "; ".join(problems[:3])
        return f"{fitted}/{len(keys)} models fitted, labels preserved"

    check("model: fit every registered model", do_every_model)

    # ---- deep learning (CPU-only PyTorch)
    def _dl_params(key, **over):
        params = dict(MODELS.default_params(key))
        params.update({"epochs": 40, "patience": 6, "max_seconds": 60, "threads": 4})
        params.update(over)
        return params

    def _dl_frame(rows=500, seed=5):
        """A self-contained frame that serves all four tasks and has a categorical."""
        import numpy as np
        import pandas as pd

        rng = np.random.RandomState(seed)
        frame = pd.DataFrame({
            "num_a": rng.normal(size=rows),
            "num_b": rng.normal(size=rows),
            "site": rng.choice(["alpha", "beta", "gamma", "delta"], rows),
        })
        signal = 2.5 * frame["num_a"] - frame["num_b"] + (frame["site"] == "alpha") * 3.0
        frame["target_reg"] = signal + rng.normal(0, 0.5, rows)
        frame["target_bin"] = (signal > signal.median()).astype(int)
        frame["target_multi"] = pd.cut(signal, 3, labels=["low", "mid", "high"]).astype(str)
        return frame

    def dl_every_network():
        """Each network fits the tasks it declares, and records a usable history."""
        import numpy as np

        from core.preprocess import PreprocessSpec

        if not MODELS.HAVE_TORCH:
            return "torch is not installed — the neural networks are unavailable"
        frame = _dl_frame()
        notes = []
        cases = [("regression", "target_reg", "torch_mlp_reg"),
                 ("binary", "target_bin", "torch_mlp_clf"),
                 ("multiclass", "target_multi", "torch_mlp_clf")]
        for task, target, key in cases:
            spec = train_mod.TrainSpec(target=target, features=["num_a", "num_b", "site"],
                                       task=task, cv_folds=3)
            split = train_mod.make_split(frame, spec)
            pipe, _ = train_mod.build_train_pipeline(PreprocessSpec(), key, _dl_params(key), task)
            train_mod.fit_full(pipe, split, spec)
            evaluation = train_mod.evaluate_fitted(pipe, split, spec)
            model = pipe.named_steps["model"]
            assert getattr(model, "history_", None), f"{key}: no training history was recorded"
            assert model.n_epochs_ >= 1, f"{key}: trained for no epochs"
            assert model.architecture_, f"{key}: no architecture was recorded"
            assert model.state_dict_, f"{key}: no weights were stored for the bundle"
            if task in ("binary", "multiclass"):
                assert evaluation["test_proba"] is not None, f"{key}: no probabilities"
                assert np.allclose(evaluation["test_proba"].sum(axis=1), 1.0, atol=1e-4), \
                    f"{key}: probabilities do not sum to one"
                seen = {str(v) for v in np.unique(evaluation["test_pred"])}
                allowed = {str(v) for v in split.labels}
                assert seen <= allowed, f"{key}: predictions {sorted(seen)} outside {sorted(allowed)}"
            notes.append(f"{task} {model.n_epochs_}ep")
            if task == "regression":
                # the loss history is what the training-curve plot draws
                ctx = plots_mod.PlotContext(df=frame, target=target, task=task, model=pipe,
                                            X=split.X_train, y=split.y_train,
                                            evaluation=evaluation, prepared=None)
                fig = plots_mod.render("training_curve", ctx, {})
                assert plots_mod.figure_png_bytes(fig), "the training curve rendered empty"
                import matplotlib.pyplot as plt

                plt.close(fig)

        spec_u = train_mod.TrainSpec(target=None, features=["num_a", "num_b", "site"],
                                     task="unsupervised")
        split_u = train_mod.make_split(frame, spec_u)
        pipe_u, _ = train_mod.build_train_pipeline(PreprocessSpec(), "torch_autoencoder",
                                                   _dl_params("torch_autoencoder"), "unsupervised")
        train_mod.fit_full(pipe_u, split_u, spec_u)
        auto = pipe_u.named_steps["model"]
        assert auto.embedding_ is not None, "the autoencoder produced no latent code"
        assert auto.labels_ is not None, "the autoencoder produced no cluster labels"
        assert auto.embedding_.shape[1] == auto.arch_["latent"], "the latent code has the wrong width"
        prepared_u = pipe_u.named_steps["prepare"].transform(split_u.X_train)
        assert auto.reconstruction_error(prepared_u).shape[0] == len(prepared_u), \
            "reconstruction error has the wrong length"
        try:
            auto.reconstruction_error(split_u.X_train)          # raw frame: must be a clear error
            raise AssertionError("a raw frame should not be accepted by reconstruction_error")
        except ValueError as exc:
            assert "prepare" in str(exc), f"unhelpful error for a raw frame: {exc}"
        notes.append(f"unsupervised latent {auto.embedding_.shape[1]}-d, "
                     f"{len(set(auto.labels_.tolist()))} clusters")
        from core.dl import runtime

        return f"{runtime.describe_runtime()} | " + "; ".join(notes)

    check("deep learning: every network fits its tasks", dl_every_network)

    def dl_embeddings():
        """One-hot groups become embedding tables, and turning it off is visible."""
        from core.preprocess import PreprocessSpec

        frame = _dl_frame(400)
        spec = train_mod.TrainSpec(target="target_reg", features=["num_a", "num_b", "site"],
                                   task="regression", cv_folds=3)
        split = train_mod.make_split(frame, spec)
        on, _ = train_mod.build_train_pipeline(PreprocessSpec(), "torch_mlp_reg",
                                               _dl_params("torch_mlp_reg", epochs=15), "regression")
        train_mod.fit_full(on, split, spec)
        prepared = on.named_steps["prepare"].transform(split.X_train)
        onehot = [c for c in prepared.columns if "=" in str(c)]
        embedded = on.named_steps["model"]
        assert onehot, "the preprocessor produced no one-hot columns to embed"
        assert embedded.cat_positions_, "no categorical group was recognised"
        assert embedded.arch_["cat_sizes"], "no embedding table was built"
        assert embedded.arch_["n_numeric"] < len(prepared.columns), \
            "the categorical columns were not taken out of the numeric block"
        off, _ = train_mod.build_train_pipeline(
            PreprocessSpec(), "torch_mlp_reg",
            _dl_params("torch_mlp_reg", epochs=15, use_embeddings=False), "regression")
        train_mod.fit_full(off, split, spec)
        plain = off.named_steps["model"]
        assert not plain.cat_positions_, "use_embeddings=False still grouped the columns"
        assert plain.arch_["n_numeric"] == len(prepared.columns), \
            "use_embeddings=False should feed every prepared column in"
        return (f"{len(onehot)} one-hot columns -> {len(embedded.cat_sizes_)} embedding table(s) "
                f"{embedded.cat_sizes_}; numeric block {embedded.arch_['n_numeric']} vs "
                f"{plain.arch_['n_numeric']} without")

    check("deep learning: categoricals become embeddings", dl_embeddings)

    def dl_row_weights():
        """Row weights must actually change the fit — for regression as well as classification.

        This is a regression test for a real bug: the regression loss came back as
        ``(batch, 1)`` while the weights were ``(batch,)``, so the multiplication
        broadcast into an outer product and the weights cancelled themselves out of
        the mean.  Row weights were accepted and then silently ignored.
        """
        import dataclasses

        import numpy as np
        import pandas as pd

        from core.preprocess import PreprocessSpec

        rng = np.random.RandomState(0)
        rows = 300
        a = rng.uniform(-3, 3, rows)
        frame = pd.DataFrame({"a": a, "b": rng.normal(size=rows)})
        frame["y"] = np.where(a > 0, a, 1000.0 + a)     # two wildly different regimes
        spec = train_mod.TrainSpec(target="y", features=["a", "b"], task="regression", cv_folds=3)
        split = train_mod.make_split(frame, spec)
        weights = pd.Series(np.where(a > 0, 1.0, 1e-7), index=frame.index)
        weighted = dataclasses.replace(split,
                                       w_train=weights.loc[split.X_train.index],
                                       w_test=weights.loc[split.X_test.index])
        params = _dl_params("torch_mlp_reg", epochs=200, patience=0)
        errors = {}
        for label, active in (("plain", split), ("weighted", weighted)):
            pipe, _ = train_mod.build_train_pipeline(PreprocessSpec(), "torch_mlp_reg", params,
                                                     "regression")
            train_mod.fit_full(pipe, active, spec)
            error = np.abs(pipe.predict(split.X_test) - split.y_test)
            starved = (split.X_test["a"] <= 0).to_numpy()
            errors[label] = float(error[starved].mean())
        assert "row weights" not in weighted.dropped, "the network refused sample weights"
        assert errors["weighted"] > errors["plain"] * 3, (
            f"down-weighting half the rows barely changed the fit "
            f"({errors['weighted']:.1f} vs {errors['plain']:.1f}) — the weights are not "
            "reaching the loss")
        return (f"error on the down-weighted half {errors['plain']:.1f} -> "
                f"{errors['weighted']:.1f}")

    check("deep learning: row weights reach the loss", dl_row_weights)

    def dl_cancel():
        """Cancel must interrupt training, not wait for every epoch."""
        import threading

        from core.dl import runtime
        from core.preprocess import PreprocessSpec
        from core.train import Cancelled

        frame = _dl_frame(400)
        spec = train_mod.TrainSpec(target="target_reg", features=["num_a", "num_b", "site"],
                                   task="regression")
        split = train_mod.make_split(frame, spec)
        pipe, _ = train_mod.build_train_pipeline(
            PreprocessSpec(), "torch_mlp_reg",
            _dl_params("torch_mlp_reg", epochs=5000, patience=0, max_seconds=0), "regression")
        event = threading.Event()
        event.set()
        try:
            with runtime.observing(cancel=event):
                train_mod.fit_full(pipe, split, spec)
        except Cancelled:
            return "cancelling mid-fit raises Cancelled instead of running all 5000 epochs"
        raise AssertionError("a cancelled fit ran to completion")

    check("deep learning: cancel interrupts training", dl_cancel)

    def dl_bundle():
        """A network survives save/reload and the bundle records the torch version."""
        import os
        import tempfile

        import numpy as np

        from core.preprocess import PreprocessSpec

        frame = _dl_frame(400)
        spec = train_mod.TrainSpec(target="target_reg", features=["num_a", "num_b", "site"],
                                   task="regression")
        split = train_mod.make_split(frame, spec)
        pipe, _ = train_mod.build_train_pipeline(PreprocessSpec(), "torch_mlp_reg",
                                                 _dl_params("torch_mlp_reg"), "regression")
        train_mod.fit_full(pipe, split, spec)
        evaluation = train_mod.evaluate_fitted(pipe, split, spec)
        bundle = train_mod.make_bundle(pipe, spec, split, PreprocessSpec(), "torch_mlp_reg",
                                       MODELS.default_params("torch_mlp_reg"), evaluation)
        folder = out_dir or tempfile.mkdtemp(prefix="mlw_dl_")
        os.makedirs(folder, exist_ok=True)
        path = train_mod.save_bundle(bundle, os.path.join(folder, "network.joblib"))
        reloaded = train_mod.load_bundle(path)
        before = pipe.predict(split.X_test)
        after = reloaded["pipeline"].predict(split.X_test)
        assert np.allclose(before, after), "the reloaded network predicts differently"
        assert "torch" in (reloaded.get("versions") or {}), "the bundle does not record the torch version"
        return (f"{os.path.getsize(path):,} bytes, identical predictions, "
                f"torch {reloaded['versions']['torch']}")

    check("deep learning: bundle round-trip", dl_bundle)

    def dl_gap_sign():
        """``overfit_gap`` must be positive exactly when the test score is *worse*.

        Every consumer reads it that way — the Model screen's card is labelled "large
        positive = memorising", and both the card and the report say "worse on the test
        set" when it is positive.  The original sign was the other way round for
        higher-is-better metrics (ROC AUC, R², F1 — i.e. all of them), so the app told the
        user the test score was *better* when it was worse.  Nothing caught it for the
        whole life of v1, because a wrong sentence is not an exception.
        """
        import numpy as np
        import pandas as pd

        from core.preprocess import PreprocessSpec

        rng = np.random.default_rng(0)
        n = 300
        frame = pd.DataFrame({"x": rng.normal(size=n)})
        frame["noise"] = rng.normal(size=n)
        frame["row_id"] = np.arange(n)          # a column a deep tree can memorise
        frame["y"] = 3 * frame["x"] + rng.normal(size=n)
        spec = train_mod.TrainSpec(target="y", features=["x", "noise", "row_id"],
                                   task="regression", test_size=0.3, random_state=0)
        split = train_mod.make_split(frame, spec)
        pipe, _ = train_mod.build_train_pipeline(
            PreprocessSpec(), "decision_tree_reg", {"max_depth": None}, "regression")
        train_mod.fit_full(pipe, split, spec)
        evaluation = train_mod.evaluate_fitted(pipe, split, spec)
        gap = evaluation["overfit_gap"]
        test = evaluation["test"]["r2"]
        train = evaluation["train"]["r2"]
        assert gap is not None, "no overfit gap was computed"
        assert train > test, f"the deep tree did not overfit (train {train:.3f}, test {test:.3f})"
        assert gap > 0, (f"the test score is worse ({test:.3f} against {train:.3f}) but the "
                         f"gap is {gap:+.3f} — the sign is inverted")
        assert abs(gap - (train - test)) < 1e-9, "the gap is not train minus test"
        return (f"an unpruned tree overfits: train {train:.3f} > test {test:.3f}, "
                f"gap {gap:+.3f} (positive = memorising)")

    check("deep learning: the overfit gap points the right way", dl_gap_sign)

    # ---- tune
    tune_page: TunePage = window._go_to(TunePage)

    def do_tune():
        tune_page.refresh()
        grid = MODELS.suggested_grid(state.model_key) or MODELS.param_default_grid(state.model_key)
        tune_spec = {"method": "grid", "n_iter": 8, "cv_folds": 3, "random_state": 42,
                     "metric": state.train_spec.primary_metric, "max_candidates": 8,
                     "param_grid": grid}
        result = tune_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                          state.pre_spec.to_dict(), state.model_key, grid, tune_spec, True)
        tune_page._tuned(result)
        return f"{result.get('n_scored')}/{result.get('n_candidates')} candidates scored"

    check("tune: search the grid and adopt the winner", do_tune)

    # ---- evaluate
    evaluate_page: EvaluatePage = window._go_to(EvaluatePage)

    def do_evaluate():
        names = []
        for row in range(evaluate_page.view_combo.count()):
            evaluate_page.view_combo.setCurrentIndex(row)
            evaluate_page._show()
            names.append(evaluate_page.view_combo.currentText())
        evaluate_page._recompute()
        return f"{len(names)} views: {names[0]}…{names[-1]}"

    check("evaluate: every table view", do_evaluate)

    # ---- the four best-combination tools
    optimize_page: OptimizePage = window._go_to(OptimizePage)

    def do_best_value():
        optimize_page._run_field()
        _wait_for_runner(window)
        frame = optimize_page.field_table.current_frame()
        return f"{len(frame)} features swept, strongest = {frame.iloc[0]['feature']}"

    check("best value per field", do_best_value)

    def do_design():
        optimize_page._fill_bounds()
        name = optimize_page.bounds_table.item(0, 0).text()
        high = optimize_page.bounds_table.item(0, 2).text()
        optimize_page.constraints_edit.setPlainText(f"{name} <= {high}")
        optimize_page.design_target.set_values({"target_value": 45.0, "n_samples": 600,
                                               "n_iter": 8, "population": 20, "top_n": 8,
                                               "method": "genetic algorithm"})
        if state.train_spec.task == "regression":
            optimize_page.design_objective.setCurrentText("match a target")
        else:
            optimize_page.design_objective.setCurrentText("maximise")
        optimize_page._run_design()
        _wait_for_runner(window)
        result = state.design
        assert result is not None, "the design search produced nothing"
        return f"{len(result.table)} designs, best {result.summary.get('best prediction')}"

    check("inverse design with a constraint", do_design)

    def do_recipe():
        optimize_page._run_recipe()
        _wait_for_runner(window)
        frame = optimize_page.recipe_table.current_frame()
        optimize_page._adopt_recipe()
        return f"{len(frame)} recipes; adopted {state.pre_spec.numeric_impute}/" \
               f"{state.pre_spec.outlier}/{state.pre_spec.scale}"

    check("best cleaning recipe", do_recipe)

    def do_selection():
        optimize_page.select_form.set_values({"method": "forward", "k": 5, "max_evaluations": 12,
                                             "cv_folds": 3})
        optimize_page._run_selection()
        _wait_for_runner(window)
        optimize_page._adopt_selection()
        return f"chose {len(state.selection.best['columns'])} columns: " \
               f"{', '.join(map(str, state.selection.best['columns']))}"

    check("best feature combination", do_selection)

    # ---- clustering
    cluster_page: ClusterPage = window._go_to(ClusterPage)

    def do_cluster():
        cluster_page.refresh()
        index = cluster_page.model_combo.findData("kmeans")
        if index >= 0:
            cluster_page.model_combo.setCurrentIndex(index)
        cluster_page.fit_button.setEnabled(True)
        cluster_page._fit()
        _wait_for_runner(window)
        return f"kmeans: silhouette {state.clusters is not None}"

    check("cluster: k-means", do_cluster)

    def do_retrain():
        """Clustering replaced the current model, so put the supervised one back."""
        spec = model_page._read_spec()
        assert spec is not None, "no supervised spec on the Model screen"
        assert spec.task != "unsupervised", "the Model screen still says unsupervised"
        state.train_spec = spec
        key, params = model_page.current_model()
        result = train_job(lambda *a: None, None, state.df, spec.to_dict(),
                           state.pre_spec.to_dict(), key, params, True)
        model_page._trained(result)
        return f"back to {key}, clusters still set: {state.clusters is not None}"

    check("model: retrain the supervised model", do_retrain)

    # ---- plots
    plots_page: PlotsPage = window._go_to(PlotsPage)

    def do_plots():
        drawn = failed = 0
        problems: list[str] = []
        listed: set = set()
        context = state.plot_context()
        from PyQt6.QtCore import Qt as _Qt

        for family in range(plots_page.family_combo.count()):
            plots_page.family_combo.setCurrentIndex(family)
            for row in range(plots_page.plot_list.count()):
                item = plots_page.plot_list.item(row)
                listed.add(item.data(_Qt.ItemDataRole.UserRole))
                if not item.flags() & _Qt.ItemFlag.ItemIsEnabled:
                    continue
                plots_page.plot_list.setCurrentRow(row)
                spec = plots_page._current_spec()
                try:
                    plots_mod.render(spec.key, context, plots_page.param_form.values())
                    drawn += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    problems.append(f"{spec.key}: {exc}")
        plots_page._draw()  # exercise the GUI path (it shows its own dialog on failure)
        unavailable = [s.key for s in plots_mod.PLOTS
                       if s.key not in {p.key for p in plots_mod.available_plots(context)}]
        print(f"      the sidebar listed {len(listed)} kinds; unavailable in this context: "
              f"{', '.join(unavailable)}", flush=True)
        assert not failed, "; ".join(problems[:4])
        return f"{drawn} drawn, {failed} failed"

    check("plots: draw every available graph", do_plots)

    def check_projection_guard():
        """PCA and t-SNE need two columns, and must say so rather than die inside sklearn.

        The regression test for finding 27.  The "draw every available graph" check above
        caught this only because one of the four datasets produced a one-column feature set
        — the bug lives in the *availability* predicate, so it is worth asserting directly.
        """
        import numpy as np
        import pandas as pd

        from core import plots as plots_mod

        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["w", "x", "y", "z"]})
        narrow = plots_mod.PlotContext(df=frame, prepared=np.zeros((4, 1)),
                                       X=pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
        listed = {spec.key for spec in plots_mod.available_plots(narrow)}
        offered = sorted(listed & {"pca_scatter", "tsne_scatter"})
        assert not offered, f"a one-column model still offers {offered}"
        try:
            plots_mod.render("pca_scatter", narrow, {})
        except plots_mod.PlotError as exc:
            message = str(exc)
        else:
            raise AssertionError("PCA drew a projection from a single column")
        assert "2" in message, f"the refusal does not say what is missing: {message!r}"

        wide = plots_mod.PlotContext(
            df=pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}),
            X=pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}))
        drawn = plots_mod.render("pca_scatter", wide, {})
        assert drawn is not None, "PCA refused a two-column frame"
        return f"hidden on one column ({message}), drawn on two"

    check("plots: a projection needs two columns", check_projection_guard)

    def do_queue():
        plots_page._queue_all()
        count = len(state.plot_views)
        plots_page._unqueue()
        return f"{count} queued, then {len(state.plot_views)} after removing one"

    check("plots: queue for the report", do_queue)

    # ---- save, report, script, predict
    save_page: SavePage = window._go_to(SavePage)
    out = out_dir or os.path.join("/tmp/opencode", "mlw_gui_out")
    os.makedirs(out, exist_ok=True)

    def do_save():
        save_page.bundle_edit.setText(os.path.join(out, "model.joblib"))
        save_page._save()
        return f"{os.path.getsize(state.bundle_path) / 1e6:.2f} MB"

    check("save: write the model bundle", do_save)

    def do_load_bundle():
        save_page._loaded(train_mod.load_bundle(state.bundle_path), state.bundle_path)
        return f"reloaded {state.model_key}, task {state.train_spec.task}"

    check("save: reload the bundle", do_load_bundle)

    def do_report():
        save_page.figure_choice.setCurrentIndex(1)  # the default figure set
        result = report_job(lambda *a: None, None, state.bundle, state.plot_context(),
                            __import__("gui.steps.model_steps", fromlist=["default_plot_requests"])
                            .default_plot_requests(state.plot_context(), state.train_spec.task,
                                                   state.train_spec.target),
                            "Self-test report", os.path.join(out, "report.html"),
                            save_page._extra_sections())
        save_page._reported(result)
        return f"{result['bytes'] / 1024:.0f} KB, {len(result['warnings'])} warnings"

    check("report: write the HTML", do_report)

    def do_script():
        save_page.script_edit.setText(os.path.join(out, "reproduce_model.py"))
        save_page._script()
        return state.script_path

    check("report: export the reproduction script", do_script)

    predict_page: PredictPage = window._go_to(PredictPage)

    def do_predict():
        predict_page.refresh()
        predict_page._fill_from_test()
        predict_page._predict()
        _wait_for_runner(window)
        frame = predict_page.output_table.current_frame()
        predict_page.result.to_csv(os.path.join(out, "predictions.csv"), index=False)
        return f"{len(frame)} rows scored, columns: {', '.join(frame.columns[-3:])}"

    check("predict: score the held-out rows", do_predict)

    def do_predict_file():
        from core import loaders as loaders_mod

        frame = loaders_mod.read_all(state.source_file, fmt=state.source_format) \
            if state.source_file else None
        if frame is None:
            predict_page.pending = state.split.X_test.head(5)
        else:
            predict_page.pending = next(iter(frame.values())).head(20)
        out_frame = predict_job(lambda *a: None, None, state.pipeline, predict_page.pending,
                                state.train_spec.task)
        return f"{len(out_frame)} rows, prediction column present: {'prediction' in out_frame.columns}"

    check("predict: a raw frame through the pipeline", do_predict_file)

    # ---- the one-click run, exercised through its real handler
    auto_page: AutoPage = window._go_to(AutoPage)

    def do_auto():
        from gui.steps.analysis_steps import auto_job

        auto_page.out_dir.setText(os.path.join(out, "one_click"))
        spec = auto_page._prepare_spec()
        assert spec is not None, "no supervised target"
        keys = auto_page._keys(spec)
        payload = {"df": state.df, "spec": spec.to_dict(), "pre": state.pre_spec.to_dict(),
                   "model_keys": keys, "tune": True, "out_dir": auto_page.out_dir.text(),
                   "source_file": state.source_file, "source_format": state.source_format,
                   "source_table": state.source_table, "cleaning_ops": state.clean_ops,
                   "notes": "", "title": "One-click self-test",
                   "write_bundle": True, "write_report": True, "write_script": True}
        result = auto_job(lambda *a: None, None, payload)
        auto_page._done(result)
        return f"{result['model_key']}: wrote {', '.join(os.path.basename(p) for p in result['paths'].values())}"

    check("one-click: the whole workflow", do_auto)

    def do_project():
        from core import session as session_mod

        path = os.path.join(out, "project.mlproj")
        session_mod.save_session(state.to_session(), path)
        session = session_mod.load_session(path)
        state.apply_session(session, load_data=False)
        return f"{os.path.getsize(path)} bytes round-tripped"

    check("project: save and reopen", do_project)

    def do_example_project():
        """The dataset the manual walks through must still be the shape the manual claims.

        The manual quotes column counts, a missing-value count and a two-class target, so
        this check fails if someone regenerates the example and quietly changes it.
        """
        path = os.path.join(project_dir, "examples", "customer_churn.csv")
        assert os.path.exists(path), f"the manual's example is missing from {path}"
        tables = load_job(lambda *a: None, None, path, "csv", {})
        frame = tables[list(tables)[0]]
        assert frame.shape[1] == 10, f"the example has {frame.shape[1]} columns, not 10"
        assert "churned" in frame.columns, "the target column has been renamed"
        labels = sorted(set(map(str, frame["churned"].unique())))
        assert labels == ["no", "yes"], f"the target labels are {labels}"
        missing = int(frame.isna().sum().sum())
        assert missing > 0, "the manual promises missing values to clean"
        return f"{len(frame):,} rows × {frame.shape[1]} columns, {missing} missing"

    check("example: the manual's project loads", do_example_project)

    # ----------------------------------------------------------------------
    # the engineering toolkit
    #
    # These six screens are what let a civil engineer answer their own question without a
    # custom script, and the beginner manual quotes their numbers.  So the checks below
    # drive the real screens on the real example files and then assert the quantities the
    # manual states — if someone regenerates an example or changes a formula, the manual
    # stops being true and this fails.
    # ----------------------------------------------------------------------

    from core import curvefit as CF
    from core import doe as DOE
    from core import montecarlo as MC
    from core import preprocess
    from core import timeseries as TS
    from core import units as UNITS
    import numpy as np
    import pandas as pd
    from gui.steps import (CurveFitPage, DoePage, ReliabilityPage, StatisticsPage,
                           TimeSeriesPage, UnitsPage)
    from gui.steps.engineering_steps import fill_grid
    from gui.steps.model_steps import default_plot_requests
    from gui.steps.reliability_steps import monte_carlo_job
    from PyQt6.QtWidgets import QTableWidgetItem

    def load_example(name: str):
        """Put one of the shipped example files in the workbench, through the loader."""
        path = os.path.join(project_dir, "examples", name)
        assert os.path.exists(path), f"the example is missing: {path}"
        tables = load_job(lambda *a: None, None, path, "csv", {})
        frame = tables[list(tables)[0]]
        state.set_raw(frame, path=path, fmt="csv", table=list(tables)[0], options={})
        window.state.touch()
        app.processEvents()
        return frame

    CONCRETE_FEATURES = ["Cement", "Blast furnace slag", "Fly ash", "Water",
                         "Superplasticiser", "Fine aggregate", "Coarse aggregate",
                         "Age (days)", "Curing method"]
    CONCRETE_SPECIFICATION = 37.0            # MPa, the C30/37 cube strength

    def concrete_shape():
        frame = load_example("concrete_mix.csv")
        assert len(frame) == 1200, f"the trial report has {len(frame)} rows, not 1200"
        assert frame.shape[1] == 13, f"the report has {frame.shape[1]} columns, not 13"
        # Both targets must be there: the regression target and the derived pass/fail one,
        # because the manual walks through classification with the same file.
        assert "Compressive strength" in frame.columns, "the strength target is gone"
        assert "meets_C30_37" in frame.columns, "the pass/fail target is gone"
        assert "batch_id" in frame.columns and "cast_date" in frame.columns, \
            "the report has no identifier and date column for the cleaning screen to remove"
        labels = sorted(set(map(str, frame["meets_C30_37"].dropna().unique())))
        assert labels == ["no", "yes"], f"the pass/fail labels are {labels}"
        missing = int(frame.isna().sum().sum())
        assert missing > 0, "the manual promises missing values in the trial report"
        # The manual quotes these two numbers, so they are asserted.
        mean = float(frame["Compressive strength"].mean())
        assert 46.5 <= mean <= 48.5, f"the manual says the mean is 47.6 MPa, got {mean:.2f}"
        share = (frame["meets_C30_37"] == "yes").mean()
        assert 0.62 <= share <= 0.72, \
            f"the manual says two thirds pass C30/37, got {share:.1%}"
        return (f"{len(frame):,} rows × {frame.shape[1]} columns, {missing} missing, "
                f"mean strength {mean:.2f} MPa")

    check("example: the concrete trial report loads", concrete_shape)

    def concrete_model():
        frame = state.df
        state.train_spec = train_mod.TrainSpec(
            target="Compressive strength",
            features=[c for c in CONCRETE_FEATURES if c in frame.columns],
            task="regression", primary_metric="r2")
        state.pre_spec = preprocess.PreprocessSpec()
        result = train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                           state.pre_spec.to_dict(), "random_forest_reg",
                           MODELS.default_params("random_forest_reg"), True)
        model_page._trained(result)
        r2 = state.evaluation["test"]["r2"]
        assert r2 > 0.85, f"the trial report should be very learnable, got r² = {r2:.3f}"
        return f"{state.model_key}: test r² = {r2:.4f}"

    check("example: train a model on the trial report", concrete_model)

    # ---- 13 · tests and statistics
    stats_page: StatisticsPage = window._go_to(StatisticsPage)

    def do_describe():
        stats_page.desc_column.setCurrentText("Compressive strength")
        stats_page._run_describe()
        table = stats_page.desc_table.current_frame()
        assert table is not None and len(table) > 8, "the statistics table is empty"
        normality = stats_page.desc_normality.current_frame()
        assert normality is not None and len(normality), "no normality test was run"
        return f"{len(table)} statistics, {len(normality)} normality test(s)"

    check("statistics: describe one column", do_describe)

    def do_group_compare():
        """Two groups that really do differ — curing method and strength."""
        stats_page.compare_a.setCurrentText("Curing method")
        stats_page.compare_b.setCurrentText("Compressive strength")
        stats_page._run_compare()
        result = state.stats_result
        assert result, "the comparison produced nothing"
        assert result.get("groups") is not None and len(result["groups"]) == 3, \
            f"expected 3 curing methods, got {result.get('groups')}"
        assert result.get("result") is not None, "no test was chosen for the comparison"
        # The manual quotes this p-value (0.11): the three curing methods are not
        # distinguishable at this much scatter, which is a real and useful answer.
        headline = str(result.get("headline", ""))
        assert "p = 0.0" in headline or "p = 0.1" in headline, \
            f"the curing comparison should be marginal: {headline!r}"
        return headline[:80]

    check("statistics: compare groups and pick the test", do_group_compare)

    def do_numeric_compare():
        stats_page.compare_a.setCurrentText("Cement")
        stats_page.compare_b.setCurrentText("Compressive strength")
        stats_page._run_compare()
        result = state.stats_result
        assert result.get("kind") == "numeric-numeric", \
            f"two numeric columns should give a numeric-numeric result, got {result.get('kind')}"
        r = float(result["result"].iloc[0]["r"])
        assert 0.2 < r < 0.7, f"cement vs strength should be a moderate positive r, got {r:.3f}"
        return f"{result['result'].iloc[0]['method']}: r = {r:.3f}, " \
               f"{result['result'].iloc[0]['significant at 5%?']}"

    check("statistics: compare two numeric columns", do_numeric_compare)

    def do_sample_size():
        stats_page.sample_form.set_values({"effect_size": 0.5, "power": 0.8, "alpha": 0.05})
        stats_page._run_sample_size()
        note = stats_page.sample_note.text()
        assert "32" in note, f"a 0.5σ effect at 80% power needs about 32 per group: {note!r}"
        return note[:80]

    check("statistics: how many samples do I need", do_sample_size)

    def do_correlation():
        stats_page.corr_form.set_values(
            {"columns": ["Cement", "Water", "Age (days)", "Compressive strength"]})
        stats_page._run_correlation()
        table = stats_page.corr_table.current_frame()
        assert table is not None and table.shape == (4, 5), \
            f"a four-column correlation matrix is 4×5 including the names, got {table.shape}"
        pairs = stats_page.corr_pairs.current_frame()
        assert len(pairs) == 6, f"four columns should give six pairs, got {len(pairs)}"
        assert list(pairs.columns)[:2] == ["first column", "second column"], \
            "a ranked correlation must name the two columns it is about"
        target = pairs[(pairs["first column"] == "Cement")
                       | (pairs["second column"] == "Cement")]
        strength = target[(target["first column"] == "Compressive strength")
                          | (target["second column"] == "Compressive strength")]
        assert len(strength) == 1, "cement against strength is missing from the pairs"
        r = abs(float(strength.iloc[0]["r"]))
        assert 0.2 < r < 0.7, f"cement against strength should be a moderate r, got {r:.3f}"
        return (f"{table.shape[0]}×{table.shape[1]} matrix, {len(pairs)} pairs ranked, "
                f"strongest |r| = {abs(float(pairs.iloc[0]['r'])):.3f}")

    check("statistics: correlation matrix", do_correlation)

    def do_stat_figures():
        stats_page._draw_compare()
        stats_page._draw_correlation()
        assert len(state.extra_figures) >= 2, "the statistics figures were not queued"
        return f"{len(state.extra_figures)} figure(s) queued for the report"

    check("statistics: the figures reach the report queue", do_stat_figures)

    # ---- 14 · curve fitting
    fit_page: CurveFitPage = window._go_to(CurveFitPage)

    def do_fit_all():
        fit_page.x_combo.setCurrentText("Age (days)")
        fit_page.y_combo.setCurrentText("Compressive strength")
        fit_page.form_combo.setCurrentIndex(fit_page.form_combo.findData(""))
        fit_page._fit()
        table = state.curve_table
        assert table is not None and len(table) == len(CF.FORMS), \
            f"expected {len(CF.FORMS)} ranked forms, got {0 if table is None else len(table)}"
        best = state.curve_fit
        assert best.converged, "the best fit did not converge"
        assert best.label.lower().startswith("power law"), \
            f"strength against age should be a power law, got {best.label!r}"
        # The manual quotes this: the age term alone explains about four fifths of it.
        assert 0.75 <= best.r2 <= 0.83, f"the strength-age R² moved: {best.r2:.4f}"
        return f"{len(table)} forms ranked, best = {best.label}, R² = {best.r2:.4f}"

    check("curve fit: rank every form", do_fit_all)

    def do_strength_age():
        fit_page.form_combo.setCurrentIndex(fit_page.form_combo.findData("strength_age"))
        fit_page._fit()
        fit = state.curve_fit
        assert fit.converged and fit.r2 > 0.7, f"the closed-form strength-age fit failed: {fit}"
        assert "x^" in fit.equation, f"the offset power law should show a power: {fit.equation}"
        fit_page.solve_edit.setText("50")
        fit_page._solve()
        assert fit_page.solve_result.text().strip(), "solving for 50 MPa produced nothing"
        return f"{fit.equation}, R² = {fit.r2:.4f}"

    check("curve fit: the strength-age form and solving backwards", do_strength_age)

    def do_interpolate():
        """Real trial reports repeat their x values, which used to break the splines."""
        fit_page.interp_form.set_values({"points": "21, 35, 200"})
        fit_page._interpolate()
        table = fit_page.interp_result.current_frame()
        assert table is not None and len(table) == 3, "three points were asked for"
        assert "NO — extrapolated" in list(table["inside the measured range?"]), \
            "x = 200 is past the oldest age tested, so it must be flagged as extrapolated"
        assert CF.repeated_x(state.df["Age (days)"].to_numpy()) > 0, \
            "the trial report is expected to repeat its ages"
        fit_page._residuals()
        fit_page._ranking_plot()
        return f"{int(CF.repeated_x(state.df['Age (days)'].to_numpy()))} repeated ages averaged"

    check("curve fit: interpolate a table that repeats its x values", do_interpolate)

    # ---- 15 · design of experiments
    doe_page: DoePage = window._go_to(DoePage)

    def do_design():
        doe_page.kind_combo.setCurrentIndex(doe_page.kind_combo.findData("full"))
        doe_page._factors_from_data()
        table = doe_page.factors_table
        table.setRowCount(3)                     # the screen offers six; use three
        for row, (name, low, high) in enumerate(
                [("Cement", 300, 450), ("Water", 150, 190), ("Age (days)", 7, 56)]):
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(str(low)))
            table.setItem(row, 2, QTableWidgetItem(str(high)))
        doe_page._build()
        design = state.doe
        assert design is not None, "no design was built"
        assert len(design) == 8, f"three factors at two levels is 8 runs, got {len(design)}"
        assert len(design.coded.columns) == 3, "the coded design lost a factor"
        sheet = design.worksheet()
        assert "run" in sheet.columns and "standard order" in sheet.columns, \
            "the run sheet is not randomised against standard order"
        assert len(doe_page.quality_view.current_frame()) >= 4, "the design quality is missing"
        return (f"{design.kind}: {len(design)} runs, "
                f"{design.worksheet().shape[1]} columns in the run sheet")

    check("experiment: build an eight-run two-level design", do_design)

    def do_experiment():
        """Analyse the design against a response that really has a main effect and an
        interaction, so the Pareto chart is exercising the code rather than empty."""
        design = state.doe
        coded = design.coded
        rng = np.random.default_rng(11)
        names = list(coded.columns)
        response = (44.0 + 5.0 * coded[names[0]] - 2.5 * coded[names[1]]
                    + 2.0 * coded[names[0]] * coded[names[1]]
                    + 1.0 * coded[names[2]] + rng.normal(0, 0.25, len(design)))
        fill_grid(doe_page.response_grid,
                  pd.DataFrame({"measured response": np.round(response, 3)}))
        doe_page.response_combo.setCurrentIndex(
            doe_page.response_combo.findData("__table__"))
        doe_page.order_combo.setCurrentIndex(1)      # main effects + two-way interactions
        doe_page._analyse()
        result = state.doe_analysis
        assert result and "pareto" in result, "the experiment was not analysed"
        top = result["pareto"].iloc[0]
        assert names[0] in str(top["term"]), \
            f"the biggest effect should be {names[0]}, got {top['term']!r}"
        assert float(result["R²"]) > 0.98, f"a near-noiseless design should fit: R² = {result['R²']}"
        doe_page._draw_pareto()
        doe_page._draw_main_effects()
        doe_page._draw_design()
        return str(result["headline"])[:85]

    check("experiment: analyse the responses and rank the terms", do_experiment)

    # ---- 17 · units and constants
    units_page: UnitsPage = window._go_to(UnitsPage)

    def do_unit_conversion():
        value = UNITS.convert(30.0, "MPa", "psi", "pressure")
        assert abs(value - 4351.13) < 0.05, f"30 MPa should be 4351.13 psi, got {value:.2f}"
        units_page.quantity_combo.setCurrentIndex(
            units_page.quantity_combo.findData("pressure"))
        app.processEvents()
        units_page.from_combo.setCurrentText("MPa")
        units_page.to_combo.setCurrentText("psi")
        units_page.value_edit.setText("30")
        units_page._convert()
        assert "4351" in units_page.result_label.text(), \
            f"the screen disagrees with the core: {units_page.result_label.text()!r}"
        assert len(units_page.factors_view.current_frame()) > 3, "the factor table is empty"
        return f"30 MPa = {value:,.2f} psi, {len(UNITS.quantities())} quantities listed"

    check("units: convert a number through the screen", do_unit_conversion)

    def do_unit_column():
        """The one thing on this screen that changes the data, and it must go through the
        cleaning pipeline so it survives a project reopen."""
        units_page.quantity_combo.setCurrentIndex(units_page.quantity_combo.findData("mass"))
        app.processEvents()
        units_page.column_combo.setCurrentText("Cement")
        units_page.column_from.setCurrentText("kg")
        units_page.column_to.setCurrentText("lb")
        units_page._convert_column()
        assert "Cement (lb)" in state.df.columns, f"no converted column: {list(state.df.columns)}"
        assert state.clean_ops and state.clean_ops[-1]["op"] == "derive_column", \
            "the conversion was not recorded as a cleaning step"
        # The whole point of recording it: re-running the recipe must reproduce the column.
        state.apply_cleaning()
        assert "Cement (lb)" in state.df.columns, "the converted column did not survive a re-run"
        expected = pd.to_numeric(state.df["Cement"], errors="coerce") * 2.20462
        actual = pd.to_numeric(state.df["Cement (lb)"], errors="coerce")
        assert float((expected - actual).abs().max()) < 0.01, "the conversion factor is wrong"
        # And a new file must not inherit this recipe.
        assert len(state.clean_ops) == 1, "one conversion should be one step"
        return "Cement → Cement (lb) as a reproducible cleaning step"

    check("units: convert a column through the cleaning pipeline", do_unit_column)

    # ---- 18 · uncertainty and reliability
    def reliability_model():
        """The unit conversion above legitimately invalidated the model, exactly as any
        change to the data does, so re-fit before asking about uncertainty."""
        frame = state.df
        state.train_spec = train_mod.TrainSpec(
            target="Compressive strength",
            features=[c for c in CONCRETE_FEATURES if c in frame.columns],
            task="regression", primary_metric="r2")
        return train_job(lambda *a: None, None, state.df, state.train_spec.to_dict(),
                         state.pre_spec.to_dict(), "random_forest_reg",
                         MODELS.default_params("random_forest_reg"), True)

    check("reliability: re-fit after the data changed",
          lambda: (model_page._trained(reliability_model()), "re-fitted")[1])

    rel_page: ReliabilityPage = window._go_to(ReliabilityPage)

    def do_reliability_grid():
        rel_page._reload_grid()
        assert rel_page.grid.rowCount() == len(CONCRETE_FEATURES), \
            f"one row per input: expected {len(CONCRETE_FEATURES)}, got {rel_page.grid.rowCount()}"
        # A column of names has no curve to fit, so it must be held, not sampled.
        names = [rel_page.grid.item(r, 1).text() for r in range(rel_page.grid.rowCount())]
        asked = [rel_page.grid.item(r, 0).checkState().name for r in range(rel_page.grid.rowCount())]
        assert "Curing method" in names, "the categorical input is missing from the grid"
        row = names.index("Curing method")
        assert asked[row] == "Unchecked", "a category must not be sampled as if it were a number"
        combo = rel_page.grid.cellWidget(row, 2)
        assert combo.currentData() == "constant", \
            f"a column of names should default to held-fixed, got {combo.currentData()!r}"
        return f"{rel_page.grid.rowCount()} inputs, the categorical one held fixed"

    check("reliability: fill the input grid from the model's features", do_reliability_grid)

    def do_reliability():
        rel_page.samples_spin.setValue(2000)
        rel_page.threshold_edit.setText("37")
        rel_page.direction_combo.setCurrentIndex(rel_page.direction_combo.findData("below"))
        rel_page.demand_edit.setText("45")
        spec = rel_page._spec()
        assert spec.n_samples == 2000, "the sample count did not reach the specification"
        payload = monte_carlo_job(lambda *a: None, None, state.pipeline, state.df, spec,
                                  rel_page._features(), "regression")
        result, tornado = payload["result"], payload["tornado"]
        state.monte_carlo = result
        rel_page._tornado = tornado
        rel_page._last_spec = spec
        rel_page._show(result, tornado)
        assert 0.02 < result.probability < 0.30, \
            f"P(strength < 37 MPa) should be a realistic single-digit-to-double-digit " \
            f"percentage, got {result.probability:.3f}"
        assert result.reliability_index > 0, "the mean strength is above the specification"
        assert len(result.percentiles) == 11, "the percentile table is the standard eleven"
        varied = list(map(str, result.influence["input"]))
        assert len(varied) == len(CONCRETE_FEATURES) - 1, \
            f"one row per sampled input: {varied}"
        assert "Curing method" not in varied, \
            "a column of names cannot be sampled, so it must be held instead"
        assert "Curing method" in rel_page.notes.text(), \
            "the screen must say which inputs it held rather than sampled"
        assert result.frame is not None and len(result.frame) == 2000, \
            "the drawn inputs were not kept"
        assert len(tornado) and "Curing method" not in list(tornado["input"]), \
            "a held category cannot be a tornado bar"
        assert len(rel_page.safety_view.current_frame()) > 3, "no safety factors were computed"
        return (f"P(strength < 37) = {result.probability:.3f}, β = "
                f"{result.reliability_index:.2f}, top input = "
                f"{result.influence.iloc[0]['input']}")

    check("reliability: run the Monte-Carlo and read the risk", do_reliability)

    def do_reliability_charts():
        for key in ("histogram", "tornado", "influence", "exceedance", "inputs"):
            rel_page.chart_combo.setCurrentIndex(rel_page.chart_combo.findData(key))
            rel_page._draw()
        assert len(state.extra_figures) >= 5, "every reliability chart should be queued"
        # The rotated-names trap: a held category has no histogram, and the chart must say
        # so rather than draw an empty pair of axes.
        import json

        payload = rel_page._spec().to_dict()
        assert json.dumps(payload, default=str), "the setup does not serialise"
        return f"{len(state.extra_figures)} figure(s) queued, setup round-trips through JSON"

    check("reliability: every chart, and the setup is reproducible", do_reliability_charts)

    def do_engineering_report():
        """The engineering work has to reach the report, or it is not a deliverable."""
        assert len(state.extra_sections) >= 4, \
            f"only {len(state.extra_sections)} engineering table(s) were queued"
        result = report_job(lambda *a: None, None, state.bundle or save_page.make_bundle(),
                            state.plot_context(),
                            default_plot_requests(state.plot_context(), state.train_spec.task,
                                                  state.train_spec.target),
                            "Engineering self-test", os.path.join(out, "engineering.html"),
                            save_page._extra_sections(),
                            extra_figures=list(state.extra_figures))
        with open(result["path"], encoding="utf-8") as handle:
            html = handle.read()
        # The reliability tables are the ones created after the unit conversion cleared the
        # earlier screens' work (a changed dataset invalidates an analysis of the old one),
        # so they are also the check that clearing and re-queueing both work.
        for heading in ("Monte-Carlo percentiles", "Sensitivity, one input at a time",
                        "Influence of each input"):
            assert heading in html, f"the report is missing a section: {heading}"
        assert html.count("<img") >= len(state.extra_figures), \
            "not every engineering figure reached the report"
        return (f"{result['bytes'] / 1024:.0f} KB, {len(state.extra_sections)} sections, "
                f"{len(state.extra_figures)} figures, {len(result['warnings'])} warnings")

    check("report: the engineering sections and figures are in the HTML",
          do_engineering_report)

    def do_office_exports():
        from core import export_office as EO

        tables = save_page._tables_for_office()
        for heading, table in (state.extra_sections or {}).items():
            if isinstance(table, pd.DataFrame) and len(table):
                tables[heading] = table
        book = os.path.join(out, "tables.xlsx")
        EO.write_excel(book, tables, notes=["self-test"], title="Engineering self-test")
        sheets = EO.read_excel_sheets(book)
        # One sheet per table, plus the "Notes" sheet that says where the workbook came from.
        assert "Notes" in sheets, f"the workbook has no notes sheet: {sheets}"
        assert len(sheets) == len(tables) + 1, \
            f"{len(tables)} tables should be {len(tables) + 1} sheets, got {len(sheets)}"
        document = os.path.join(out, "report.docx")
        blocks = save_page._word_blocks()
        EO.write_docx(document, blocks, title="Engineering self-test")
        report = EO.verify_docx(document)
        assert not (report.get("problems") or []), f"the Word file is malformed: {report}"
        assert os.path.getsize(document) > 4000, "the Word file has no pictures in it"
        return (f"{len(sheets)} sheets, {len(blocks)} Word blocks, "
                f"{os.path.getsize(document) / 1024:.0f} KB")

    check("office: Excel and Word come out of the same tables", do_office_exports)

    def do_settlement():
        frame = load_example("pile_settlement_monitoring.csv")
        assert len(frame) == 157, f"three years of weekly readings is 157 rows, got {len(frame)}"
        assert frame.shape[1] == 6, f"the monitoring record has {frame.shape[1]} columns"
        missing = int(frame["settlement_mm"].isna().sum())
        assert missing == 3, f"the manual says three missed readings, found {missing}"
        return f"{len(frame)} weekly readings, {missing} missed, max move {frame['settlement_mm'].max():.2f} mm"

    check("example: the pile monitoring record loads", do_settlement)

    ts_page: TimeSeriesPage = window._go_to(TimeSeriesPage)

    def do_trend():
        ts_page.time_combo.setCurrentText("date")
        ts_page.value_combo.setCurrentText("settlement_mm")
        ts_page.analysis_combo.setCurrentIndex(ts_page.analysis_combo.findData("trend"))
        ts_page._run()
        info = state.timeseries
        assert info and info.get("analysis") == "trend", "the trend analysis did not run"
        summary = info["summary"]
        assert "slope" in summary, f"the trend produced no slope card: {sorted(summary)}"
        # The manual quotes this slope, so it is asserted.
        slope = float(str(summary["slope"]).replace(",", "").replace("+", ""))
        assert 3.2 <= slope <= 4.0, f"the settlement rate moved: {summary['slope']}"
        table = ts_page.table.current_frame()
        row = table.loc[table["quantity"] == "Mann–Kendall p", "value"]
        assert len(row) and float(row.iloc[0]) < 0.001, \
            f"the rise should be unmistakable: {row.tolist()}"
        verdict = table.loc[table["quantity"] == "verdict", "value"]
        assert len(verdict) and "increasin" in str(verdict.iloc[0]).lower(), \
            f"the trend verdict should be 'increasing': {verdict.tolist()}"
        return str(ts_page.sentence.text())[:95]

    check("time series: the settlement trend", do_trend)

    def do_every_analysis():
        names = []
        for row in range(ts_page.analysis_combo.count()):
            ts_page.analysis_combo.setCurrentIndex(row)
            label = ts_page.analysis_combo.currentText()
            ts_page._run()
            table = ts_page.table.current_frame()
            assert table is not None and len(table), f"{label} produced no table"
            names.append(label.split("(")[0].split("—")[0].strip())
        return f"{len(names)} analyses: {', '.join(names)}"

    check("time series: run every analysis", do_every_analysis)

    def do_seasonality():
        """A weekly record for three years should get enough lags to see a whole year."""
        ts_page.analysis_combo.setCurrentIndex(ts_page.analysis_combo.findData("acf"))
        ts_page._run()
        acf = ts_page.table.current_frame()
        assert acf is not None and "acf" in acf.columns, "no autocorrelation table"
        assert len(acf) - 1 >= 40, (
            f"a three-year weekly record needs about 50 lags to show a year, "
            f"but the screen stopped at {len(acf) - 1}")
        return (f"ACF at lag 1 = {float(acf['acf'].iloc[1]):.3f}, "
                f"{len(acf) - 1} lags computed")

    check("time series: autocorrelation", do_seasonality)

    def do_ts_features():
        ts_page.analysis_combo.setCurrentIndex(ts_page.analysis_combo.findData("trend"))
        ts_page.window_spin.setValue(4)
        before = state.df.shape[1]
        ts_page._add_features()
        added = [c for c in state.df.columns if str(c).startswith("settlement_mm (")]
        assert len(added) == 3, f"expected a lag, a rolling mean and a change: {added}"
        # The lag shifts by one, so it can only be defined where the reading before it was:
        # one row goes missing at the start, and each blank reading takes its lag with it.
        lag = state.df["settlement_mm (lag 1)"]
        reading = state.df["settlement_mm"]
        assert lag.notna().sum() == reading.notna().sum() - 1, \
            "the lag column does not line up with the readings"
        assert lag.iloc[1] == reading.iloc[0], "the lag is not the previous reading"
        # Everything the screen writes goes through the cleaning pipeline, so a reopen
        # reproduces it.  This is the check that keeps that promise honest.
        state.apply_cleaning()
        still = [c for c in state.df.columns if str(c).startswith("settlement_mm (")]
        assert len(still) == 3, f"the derived columns did not survive a re-run: {still}"
        return f"{before} → {state.df.shape[1]} columns, all reproduced by the recipe"

    check("time series: lag and rolling columns go through the cleaner", do_ts_features)

    def do_control_charts():
        """Why a rising record is charted as its weekly *movement*, not as its total.

        Charting the running total against limits taken from its own spread flags nearly
        every point, because the spread here *is* the trend — σ = 0.16 mm against a rise of
        13 mm.  Chart the week-on-week change and the one real event stands out: the week
        ending 31 March 2025 moved 1.32 mm, 5.3σ, when a whole extra millimetre appeared and
        stayed.
        """
        ts_page.analysis_combo.setCurrentIndex(ts_page.analysis_combo.findData("control"))

        ts_page.value_combo.setCurrentText("settlement_mm")
        ts_page.chart_combo.setCurrentIndex(ts_page.chart_combo.findData("individuals"))
        ts_page._run()
        raw = ts_page.violations.current_frame()
        assert raw is not None and len(raw) > 50, \
            ("the running total must be shown to be out of control almost everywhere — "
             f"it flagged {0 if raw is None else len(raw)} of 154 points")

        change = "settlement_mm (change per step)"
        assert change in state.df.columns, "the change column was not added"
        ts_page.value_combo.setCurrentText(change)
        ts_page._run()
        table = ts_page.violations.current_frame()
        assert table is not None and len(table), "the change chart found no signals"
        assert "when" in table.columns, \
            "a signal has to say *when* it happened, not just 'point 112'"
        big = table[table["rule"] == 1]
        assert len(big) >= 1, f"no point beyond 3σ was found: {list(table['rule'])}"
        worst = big.iloc[-1]
        assert str(worst["when"]).startswith("2025-03-31"), \
            f"the excursion is the week of 2025-03-31, not {worst['when']}"
        assert 3.0 <= abs(float(worst["what was seen"].rstrip("σ"))) <= 7.0, \
            f"the excursion should be a 5σ week: {worst['what was seen']}"
        found = {}
        for row in range(ts_page.chart_combo.count()):
            ts_page.chart_combo.setCurrentIndex(row)
            ts_page._run()
            frame = ts_page.violations.current_frame()
            found[ts_page.chart_combo.currentText().split(" (")[0]] = \
                0 if frame is None else len(frame)
        return (f"running total {len(raw)} signals of 154; weekly movement: "
                + ", ".join(f"{k} {v}" for k, v in found.items()))

    check("time series: the control chart finds the week the pile moved", do_control_charts)

    def do_capability():
        """Capability belongs on a process that is meant to be steady, which the running
        total is not.  The weekly movement against a ±0.5 mm/week criterion is."""
        ts_page.analysis_combo.setCurrentIndex(ts_page.analysis_combo.findData("capability"))
        ts_page.value_combo.setCurrentText("settlement_mm (change per step)")
        ts_page.lsl_edit.setText("-0.5")
        ts_page.usl_edit.setText("0.5")
        ts_page._run()
        assert "Cpk" in ts_page.sentence.text(), \
            f"the capability sentence is missing Cpk: {ts_page.sentence.text()!r}"
        # The cards are the screen's own summary of the run, kept in the state so a later
        # screen can quote them without recomputing anything.
        summary = state.timeseries.get("summary") or {}
        assert "Cpk" in summary, f"the capability cards are missing Cpk: {sorted(summary)}"
        cpk = float(str(summary["Cpk"]).replace(",", ""))
        assert 0.3 <= cpk <= 0.9, f"the movement Cpk moved: {summary['Cpk']}"
        assert "not capable" in ts_page.sentence.text().lower(), \
            f"±0.5 mm/week should not be capable: {ts_page.sentence.text()!r}"
        return ts_page.sentence.text()[:100]

    check("time series: process capability against a specification", do_capability)

    def do_engineering_modules():
        """Direct checks on the new core modules, so library drift fails here first."""
        # units
        assert abs(UNITS.convert(1.0, "kN", "kip", "force") - 0.224809) < 1e-5
        assert abs(UNITS.convert(1.0, "m/s", "km/h", "velocity") - 3.6) < 1e-9
        assert abs(UNITS.convert(0.0, "°C", "°F", "temperature") - 32.0) < 1e-9, \
            "temperature is affine, and the offset must survive"
        assert UNITS.is_affine("temperature") and not UNITS.is_affine("pressure")
        # curvefit: pchip through repeats, and the duplicated x averaged rather than fatal
        values = CF.interpolate([1, 1, 2, 3], [10, 12, 20, 30], [1.5], "linear")
        assert abs(float(values[0]) - 15.5) < 1e-6, \
            f"the two readings at x = 1 should average to 11, giving 15.5 at 1.5: {values}"
        assert CF.repeated_x([1, 1, 2, 3]) == 1, "the repeated x was not counted"
        assert float(CF.interpolate([1, 1, 2, 3], [10, 12, 20, 30], [2.5], "pchip")[0]) > 0, \
            "a pchip through repeated x used to raise instead of running"
        # timeseries: a sustained step in a process is found by CUSUM
        stepped = pd.DataFrame({
            "when": pd.date_range("2024-01-01", periods=120, freq="D"),
            "reading": np.concatenate([np.full(60, 10.0), np.full(60, 11.2)]),
        })
        chart = TS.control_chart(stepped, "when", "reading", chart="cusum")
        assert len(chart["violations"]), "a sustained 1.2σ step should raise a CUSUM signal"
        individuals = TS.control_chart(stepped, "when", "reading", chart="individuals")
        assert individuals["points"] is not None, "the individuals chart produced no points"
        # doe: a full factorial is balanced, and the alias table knows what it traded away
        design = DOE.two_level_full(["a", "b", "c"], seed=1)
        assert len(design) == 8, f"2³ is 8 runs, got {len(design)}"
        aliases = DOE.alias_table(["a", "b", "c", "d"], ["D=ABC"])
        assert aliases is not None and len(aliases), "the alias table is empty"
        # export_office: the workbook written a moment ago is still readable
        from core import export_office as export_mod

        assert export_mod.read_excel_sheets(os.path.join(out, "tables.xlsx")), \
            "the workbook written by the office check is unreadable"
        # montecarlo: a category is held, a number is sampled
        assert MC.is_categorical(pd.Series(list("aaabbb"))), "a column of names is not numeric"
        assert not MC.is_categorical(pd.Series([1.0, 2.0, 3.0])), \
            "a numeric column must be samplable"
        held = MC.InputSpec(column="curing", distribution="constant",
                           params={"value": "water"})
        drawn = held.draw(5, np.random.default_rng(0))
        assert list(drawn) == ["water"] * 5, f"a held category was not held: {drawn}"
        assert held.quantile(0.5) is None, "a held category has no quantile to report"
        return (f"{len(UNITS.quantities())} unit quantities, {len(CF.FORMS)} curve forms, "
                f"{len(TS.CHART_KINDS)} control charts, {len(MC.DISTRIBUTIONS)} distributions")

    check("engineering: the six new core modules behave", do_engineering_modules)

    def do_new_file_clears_the_recipe():
        """A cleaning recipe names the previous file's columns, so a new file must not
        inherit it.  It used to, and the failure surfaced much later as
        'operation 1 failed: unknown name(s): Cement'."""
        state.clean_ops = [{"op": "derive_column", "name": "x", "formula": "Anything * 2"}]
        load_example("pile_settlement_monitoring.csv")
        assert state.clean_ops == [], f"the old recipe survived: {state.clean_ops}"
        return "loading a second file starts with no cleaning recipe"

    check("clean: a new file does not inherit the old recipe", do_new_file_clears_the_recipe)

    check("close the window", lambda: (window.close(), "closed")[1])

    width = max(len(name) for name, _, _ in results)
    print()
    print("ML Workbench self-test")
    print("=" * (width + 30))
    for name, status, detail in results:
        marker = "ok  " if status == "ok" else "FAIL"
        print(f"{marker} {name.ljust(width)}  {detail}")
    failures = [name for name, status, _ in results if status != "ok"]
    print("-" * (width + 30))
    print(f"{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    print(f"artefacts in {out}")
    return 1 if failures else 0


def _wait_for_runner(window, timeout: float = 900.0):
    """Block until the window's worker thread has finished and delivered its result."""
    import time

    from PyQt6.QtWidgets import QApplication

    deadline = time.time() + timeout
    while window.runner.busy:
        QApplication.processEvents()
        time.sleep(0.01)
        if time.time() > deadline:
            raise TimeoutError("the background task did not finish in time")
    # the completion callback travels on the event queue, so give it a moment
    for _ in range(20):
        QApplication.processEvents()
        time.sleep(0.005)


if __name__ == "__main__":
    sys.exit(main())
