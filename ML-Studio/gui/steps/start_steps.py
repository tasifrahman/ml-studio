"""The screen the application opens on: what this is, and what to do first.

A workbench with fourteen screens is intimidating the first time, so the window now
opens here instead of on the file dialog.  The page does three things: it says what the
program is for, it offers a one-click way to get data in front of the user, and it
keeps a live checklist of the four things that actually have to happen.  The checklist
reads the shared state, so it ticks itself off as the user works — no screen has to
cooperate with it.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .. import theme
from ..manual import ManualDialog
from ..widgets import Callout, StepPage, section_title

#: The beginner's path: (key, what to do, which screen does it, where to go).
CHECKLIST = [
    ("data", "Load a dataset — any CSV will do",
     "1 · Load data", "load"),
    ("target", "Say which column you want to predict",
     "5 · Model", "model"),
    ("model", "Train a model and read its score",
     "5 · Model", "model"),
    ("outputs", "Save the model, the report or the script",
     "8 · Save & export", "save"),
]

#: The four stages of the workflow, as the sidebar groups them.
STAGES = [
    ("Data", "Load a file, look at it, fix what is broken, choose the feature columns."),
    ("Model", "Pick the target and the algorithm, train, tune, evaluate, save."),
    ("Analyse", "Search for the best combination, cluster the rows, draw graphs."),
    ("Finish", "Run the whole recipe end-to-end and produce the report."),
]


class StartPage(StepPage):
    title = "Start here"
    subtitle = ("Everything happens in the sidebar on the left.  If you have never trained "
                "a model before, read the manual once and follow the example — it takes "
                "about ten minutes.")

    def build(self):
        self.welcome = Callout(
            "Load a dataset, say which column you want to predict, train a model, then "
            "read the report.  Nothing here is destructive and nothing is saved until you "
            "ask for it — you can always go back and change your mind.",
            "info", title="What this program does")
        self.welcome.set_action("Open the user manual — a full walkthrough with an example",
                                self._open_manual)
        self.body.addWidget(self.welcome)

        # ------------------------------------------------------------ quick start
        self.body.addWidget(section_title("Try it right now"))
        quick = QHBoxLayout()
        quick.setSpacing(8)
        self.example_button = QPushButton("Load the built-in example dataset")
        theme.tag(self.example_button, "primary")
        self.example_button.setToolTip("900 synthetic concrete mixes — no file needed")
        self.example_button.clicked.connect(self._load_builtin)
        self.churn_button = QPushButton("Load the example project (customer churn)")
        self.churn_button.setToolTip(
            "examples/customer_churn.csv — the dataset the user manual walks through")
        self.churn_button.clicked.connect(self._load_churn)
        quick.addWidget(self.example_button)
        quick.addWidget(self.churn_button)
        quick.addStretch(1)
        self.body.addLayout(quick)

        # ------------------------------------------------------------- checklist
        self.body.addWidget(section_title("Your progress"))
        card = QFrame()
        theme.tag(card, "card")
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)

        self._rows: dict[str, dict] = {}
        for row, (key, what, where, target) in enumerate(CHECKLIST):
            mark = theme.tag(QLabel(), "muted")
            mark.setFixedWidth(18)
            label = theme.tag(QLabel(what), "callout-body")
            place = theme.tag(QLabel(where), "muted")
            jump = QPushButton("Go")
            theme.tag(jump, "ghost")
            jump.clicked.connect(lambda _=False, t=target: self._open(t))
            grid.addWidget(mark, row, 0)
            grid.addWidget(label, row, 1)
            grid.addWidget(place, row, 2)
            grid.addWidget(jump, row, 3)
            self._rows[key] = {"mark": mark, "label": label, "place": place}

        self.body.addWidget(card)

        # ----------------------------------------------------------- the big map
        self.body.addWidget(section_title("How the screens fit together"))
        stages = QWidget()
        stage_layout = QVBoxLayout(stages)
        stage_layout.setContentsMargins(2, 0, 2, 0)
        stage_layout.setSpacing(4)
        for name, blurb in STAGES:
            row = QHBoxLayout()
            row.setSpacing(8)
            name_label = theme.tag(QLabel(name), "callout-title")
            name_label.setFixedWidth(70)
            row.addWidget(name_label)
            row.addWidget(theme.tag(QLabel(blurb), "muted"), 1)
            stage_layout.addLayout(row)
        self.body.addWidget(stages)

        self.tip = Callout(
            "Ctrl+R runs the whole recipe — load, clean, train, evaluate, report — and "
            "stops before saving anything.  Press Esc to cancel it.",
            "tip", title="The fast path")
        self.body.addWidget(self.tip)
        self.body.addStretch(1)

    # --------------------------------------------------------------- actions
    def _open(self, target: str):
        """Switch to one of the screens the checklist points at."""
        if self.window is None:
            return None
        from . import LoadPage, ModelPage, SavePage

        page_class = {"load": LoadPage, "model": ModelPage, "save": SavePage}[target]
        return self.window._go_to(page_class)

    def _open_manual(self):
        ManualDialog(self.state.project_dir, self.window or self).exec()

    def _load_builtin(self):
        page = self._open("load")
        if page is not None and hasattr(page, "_load_example"):
            page._load_example()

    def _load_churn(self):
        path = os.path.join(self.state.project_dir, "examples", "customer_churn.csv")
        if not os.path.exists(path):
            self.warn("The example is missing",
                      f"There is no example dataset at\n{path}\n\n"
                      "Re-copy the examples/ folder from the project.")
            return
        page = self._open("load")
        if page is None:
            return
        index = page.format_combo.findData("csv")
        if index >= 0:
            page.format_combo.setCurrentIndex(index)
        page.path_edit.setText(path)
        page._start_load()

    # --------------------------------------------------------------- refresh
    def _done(self, key: str) -> bool:
        state = self.state
        if key == "data":
            return bool(state.has_data)
        if key == "target":
            return bool(state.train_spec.target)
        if key == "model":
            return bool(state.has_model)
        if key == "outputs":
            return bool(state.bundle_path or state.report_path or state.script_path)
        return False

    def refresh(self):
        for key, row in self._rows.items():
            done = self._done(key)
            row["mark"].setText("✔" if done else "○")
            theme.tag(row["mark"], "callout-title" if done else "muted")
            row["label"].setEnabled(not done)
            row["place"].setEnabled(not done)
        self.churn_button.setEnabled(not self._done("data"))
