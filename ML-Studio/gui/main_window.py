"""The application window: a grouped sidebar, a stack of screens, and one status bar.

The window owns three long-lived objects and nothing else: :class:`AppState` (the
shared data), :class:`TaskRunner` (the single worker slot) and the list of screens.
Every screen is constructed with those two objects, so no screen needs to know
about any other — the only cross-screen traffic goes through the state's signals.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QStackedWidget, QStyle, QVBoxLayout,
    QWidget,
)

from core import session as session_mod

from . import theme
from .manual import ManualDialog
from .state import AppState
from .steps import (
    AutoPage, LoadPage, ModelPage, SavePage, StartPage, TunePage, build_steps,
)
from .widgets import LogPane
from .workers import TaskRunner

#: A standard Qt icon per screen.  These come from the widget style, so there are no
#: image files to ship and nothing to go missing.  A page with no entry simply gets no
#: icon, which is why the lookup is by name and never assumes a match.
SIDEBAR_ICONS = {
    "StartPage": "SP_DirHomeIcon",
    "LoadPage": "SP_DialogOpenButton",
    "EdaPage": "SP_FileDialogInfoView",
    "CleanPage": "SP_DialogResetButton",
    "FeaturesPage": "SP_FileDialogDetailedView",
    "ModelPage": "SP_MediaPlay",
    "TunePage": "SP_BrowserReload",
    "EvaluatePage": "SP_FileDialogContentsView",
    "SavePage": "SP_DialogSaveButton",
    "PredictPage": "SP_ArrowForward",
    "OptimizePage": "SP_MediaSeekForward",
    "ClusterPage": "SP_FileDialogListView",
    "PlotsPage": "SP_FileDialogDetailedView",
    "StatisticsPage": "SP_FileDialogInfoView",
    "CurveFitPage": "SP_FileDialogContentsView",
    "DoePage": "SP_FileDialogDetailedView",
    "TimeSeriesPage": "SP_BrowserReload",
    "UnitsPage": "SP_FileDialogListView",
    "ReliabilityPage": "SP_MessageBoxWarning",
    "AutoPage": "SP_DialogApplyButton",
}


class MainWindow(QMainWindow):
    def __init__(self, project_dir: str, parent=None):
        super().__init__(parent)
        theme.apply()          # before any widget exists, so Fusion draws them all
        self.project_dir = project_dir
        self.state = AppState(project_dir)
        self.runner = TaskRunner(self)
        self._stack_index: dict[int, int] = {}

        self.pages, groups = build_steps(self.state, self.runner, self)
        self._build_ui(groups)
        self._build_menu()
        self._connect()

        self.setWindowTitle(f"ML Workbench — {project_dir}")
        # Opened at a comfortable size but never larger than the display, and — just as
        # important — never *demanding* more than a laptop can show.  The old minimum was
        # the tallest page's minimum plus the chrome (1478 x 1060 here), which is bigger
        # than the work area of a 1080p screen: the window manager then cannot maximize it
        # properly.  Pages scroll now, so a small floor is honest.
        self.setMinimumSize(880, 560)
        screen = QApplication.primaryScreen()
        width, height = 1380, 900
        if screen is not None:
            area = screen.availableGeometry()
            width = max(self.minimumWidth(), min(width, area.width() - 60))
            height = max(self.minimumHeight(), min(height, area.height() - 60))
        self.resize(width, height)
        self._go_to(StartPage if not self.state.has_data else LoadPage)
        self._refresh_pages()
        self._status_changed()

    # ------------------------------------------------------------------- ui
    def _build_ui(self, groups):
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(246)
        self.sidebar.setIconSize(QSize(16, 16))
        self.stack = QStackedWidget()

        for group, pages in groups:
            header = QListWidgetItem(group.upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setBold(True)
            font.setPointSize(max(7, font.pointSize() - 1))
            header.setFont(font)
            header.setForeground(Qt.GlobalColor.gray)
            self.sidebar.addItem(header)
            for page in pages:
                index = self.stack.addWidget(page)
                item = QListWidgetItem(f"  {page.title}")
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(page.subtitle)
                icon = self._page_icon(page)
                if icon is not None:
                    item.setIcon(icon)
                self.sidebar.addItem(item)
                self._stack_index[index] = self.sidebar.count() - 1

        self.sidebar.currentItemChanged.connect(self._sidebar_changed)

        # the rail: a small brand block above the list, so the window has a header
        rail = QWidget()
        rail.setObjectName("rail")
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(0)
        rail_layout.addWidget(self._brand_block())
        rail_layout.addWidget(self.sidebar, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(rail)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        self.setCentralWidget(splitter)

        # status bar: what is loaded, what is running, and a way to stop it
        self.status_label = QLabel("no data loaded")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.runner.cancel)
        bar = self.statusBar()
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.cancel_button)

        self.log_pane = LogPane()
        self.log_dock = QDockWidget("Activity", self)
        self.log_dock.setWidget(self.log_pane)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea |
                                     Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setVisible(False)
        self.resizeDocks([self.log_dock], [190], Qt.Orientation.Vertical)

    def _brand_block(self) -> QWidget:
        box = QWidget()
        box.setObjectName("railBrand")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 15, 16, 12)
        layout.setSpacing(2)
        layout.addWidget(theme.tag(QLabel("ML Workbench"), "nav-title"))
        layout.addWidget(theme.tag(QLabel("deep-learning edition"), "nav-sub"))
        return box

    def _page_icon(self, page):
        """The standard icon for a screen, or None when it has none."""
        name = SIDEBAR_ICONS.get(type(page).__name__)
        pixmap = getattr(QStyle.StandardPixmap, name, None) if name else None
        if pixmap is None:
            return None
        return self.style().standardIcon(pixmap)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "Load data…", self._load_data, QKeySequence.StandardKey.Open)
        file_menu.addSeparator()
        self._action(file_menu, "Save project…", self.save_project, QKeySequence.StandardKey.Save)
        self._action(file_menu, "Open project…", self.open_project)
        file_menu.addSeparator()
        self._action(file_menu, "Save the model…", lambda: self._go_to(SavePage))
        self._action(file_menu, "Export the report…", lambda: self._go_to(SavePage))
        self._action(file_menu, "Export a reproduction script…", lambda: self._go_to(SavePage))
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        run_menu = self.menuBar().addMenu("&Run")
        self._action(run_menu, "Train the current model", self._train_now, "Ctrl+T")
        self._action(run_menu, "Tune the current model", lambda: self._go_to(TunePage), "Ctrl+U")
        self._action(run_menu, "Run the whole workflow", self._run_everything, "Ctrl+R")
        run_menu.addSeparator()
        self._action(run_menu, "Cancel the running task", self.runner.cancel, "Esc")

        view_menu = self.menuBar().addMenu("&View")
        toggle = QAction("Show the activity log", self)
        toggle.setCheckable(True)
        toggle.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(toggle.setChecked)
        view_menu.addAction(toggle)
        view_menu.addSeparator()
        self._full_screen_action = QAction("Full screen", self)
        self._full_screen_action.setCheckable(True)
        self._full_screen_action.setShortcut("F11")
        self._full_screen_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self._full_screen_action.toggled.connect(self._toggle_full_screen)
        view_menu.addAction(self._full_screen_action)

        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "Start here (beginner's guide)", self._show_start,
                     QKeySequence.StandardKey.HelpContents)
        self._action(help_menu, "Open the user manual", self._show_manual)
        self._action(help_menu, "Keyboard shortcuts", self._show_shortcuts)
        help_menu.addSeparator()
        self._action(help_menu, "About", self._about)

    # ------------------------------------------------------------------- help
    def _toggle_full_screen(self, on: bool):
        """Fill the display (F11), remembering where the window was.

        ``showFullScreen`` rather than ``showMaximized``: maximizing is the compositor's
        business and on Wayland it silently does nothing when the window's minimum size
        does not fit the work area, which is what made this look broken.  Full screen is
        a request the client can always satisfy.
        """
        if on:
            self._was_maximized = self.isMaximized()
            self._normal_geometry = self.geometry()
            self.showFullScreen()
        else:
            self.showNormal()
            if getattr(self, "_was_maximized", False):
                self.showMaximized()
            elif getattr(self, "_normal_geometry", None) is not None:
                self.setGeometry(self._normal_geometry)
        self.state.log_line("full screen on" if on else "full screen off")

    def _sync_full_screen_action(self):
        """Keep the View menu honest when the state changed without going through it."""
        action = getattr(self, "_full_screen_action", None)
        if action is not None and action.isChecked() != bool(self.windowState()
                                                             & Qt.WindowState.WindowFullScreen):
            action.blockSignals(True)
            action.setChecked(bool(self.windowState() & Qt.WindowState.WindowFullScreen))
            action.blockSignals(False)

    def changeEvent(self, event):  # noqa: N802 - Qt API
        if event.type() == event.Type.WindowStateChange:
            self._sync_full_screen_action()
        super().changeEvent(event)

    def _show_start(self):
        self._go_to(StartPage)

    def _show_manual(self):
        dialog = ManualDialog(self.project_dir, self)
        dialog.exec()

    def _show_shortcuts(self):
        rows = [
            ("Ctrl+R", "Run the whole workflow on the Auto screen"),
            ("Ctrl+T", "Train the model that is selected right now"),
            ("Ctrl+U", "Go to the tuning screen"),
            ("Ctrl+O", "Load a data file"),
            ("Ctrl+S", "Save the project"),
            ("F11", "Full screen — fill the display with the program"),
            ("F1", "This screen — the beginner's guide"),
            ("Esc", "Cancel whatever is running"),
        ]
        text = "\n".join(f"{key:<10}{what}" for key, what in rows)
        QMessageBox.information(self, "Keyboard shortcuts", text)

    def _action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def _connect(self):
        self.state.logged.connect(self.log_pane.append)
        self.state.changed.connect(self._refresh_pages)
        self.state.changed.connect(self._status_changed)
        self.runner.busy_changed.connect(self._busy_changed)
        self.runner.progressed.connect(self._progressed)

    # ------------------------------------------------------------- navigation
    def _sidebar_changed(self, current, _previous):
        if current is None:
            return
        index = current.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        self.stack.setCurrentIndex(int(index))
        page = self.stack.currentWidget()
        if page is not None and hasattr(page, "on_enter"):
            page.on_enter()
        if page is not None and hasattr(page, "update_requirements"):
            # refresh the signpost on entry, so a screen opened before any data was
            # loaded still explains itself rather than looking broken
            page.update_requirements()

    def _go_to(self, page_class):
        for index, widget in enumerate(self.pages):
            if isinstance(widget, page_class):
                row = self._stack_index.get(self.stack.indexOf(widget))
                if row is not None:
                    self.sidebar.setCurrentRow(row)
                return widget
        return None

    def _refresh_pages(self):
        for page in self.pages:
            try:
                page.refresh()
            except Exception as exc:  # noqa: BLE001 - one broken screen must not stop the rest
                self.state.log_line(f"{type(page).__name__}.refresh failed: "
                                    f"{type(exc).__name__}: {exc}")
            try:
                page.update_requirements()
            except Exception as exc:  # noqa: BLE001 - the signpost is a nicety, not a feature
                self.state.log_line(f"{type(page).__name__}.update_requirements failed: "
                                    f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ state
    def _status_changed(self):
        state = self.state
        if not state.has_data:
            self.status_label.setText("no data loaded")
            return
        bits = [f"{len(state.df):,} rows × {state.df.shape[1]} columns"]
        if state.source_file:
            bits.append(os.path.basename(state.source_file))
        elif state.source_table == "example":
            bits.append("built-in example dataset")
        if state.train_spec.target:
            bits.append(f"target {state.train_spec.target}")
        if state.has_model:
            from core import models as MODELS

            bits.append(f"{MODELS.get_model_spec(state.model_key).label} trained"
                        if state.model_key else "model trained")
        self.status_label.setText("  ·  ".join(bits))

    def _busy_changed(self, busy: bool, label: str):
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
            self.statusBar().showMessage("", 4000)
        elif label:
            self.statusBar().showMessage(f"{label}…")

    def _progressed(self, current: int, total: int, message: str):
        if current < 0:
            # message-only update: keep the bar where the job put it (used for the
            # per-epoch lines a neural network emits while it trains)
            if message:
                self.statusBar().showMessage(message)
            return
        if total > 0:
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(current))
        else:
            self.progress.setRange(0, 0)
        if message:
            self.statusBar().showMessage(message)

    # ------------------------------------------------------------------ menus
    def _load_data(self):
        self._go_to(LoadPage)

    def _train_now(self):
        page = self._go_to(ModelPage)
        if page is not None:
            page._train()

    def _run_everything(self):
        page = self._go_to(AutoPage)
        if page is not None:
            page._run()

    def _about(self):
        from core import train as train_mod
        from core.dl import runtime

        versions = train_mod.library_versions()
        text = "\n".join(f"{k} {v}" for k, v in versions.items())
        QMessageBox.information(
            self, "About ML Workbench",
            "An end-to-end machine-learning workbench: load, explore, clean, prepare, "
            "train, tune, evaluate, optimise, plot, report.\n\n"
            f"Deep learning:\n{runtime.describe_runtime()}\n\n"
            f"Project folder:\n{self.project_dir}\n\nLibrary versions:\n{text}")

    # --------------------------------------------------------------- projects
    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save the project", self.state.project_path
                                              or os.path.join(self.project_dir, "project.mlproj"),
                                              "ML Workbench project (*.mlproj)")
        if not path:
            return
        try:
            session_mod.save_session(self.state.to_session(), path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save the project", f"{type(exc).__name__}: {exc}")
            return
        self.state.project_path = path
        self.state.log_line(f"project saved to {path}")
        self.setWindowTitle(f"ML Workbench — {os.path.basename(path)}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a project", self.project_dir,
                                              "ML Workbench project (*.mlproj);;All files (*)")
        if not path:
            return
        try:
            session = session_mod.load_session(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open the project", f"{type(exc).__name__}: {exc}")
            return
        try:
            self.state.apply_session(session)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "The project could not be restored",
                                 f"{type(exc).__name__}: {exc}\n\nThe recipe was read, but the "
                                 "data could not be reloaded.  Load the data file by hand.")
        self.state.preprocess_changed.emit()
        self.state.features_changed.emit(list(self.state.train_spec.features or []))
        self.state.project_path = path
        self.state.log_line(f"project opened from {path}")
        self.setWindowTitle(f"ML Workbench — {os.path.basename(path)}")
        self.state.touch()

    # ------------------------------------------------------------------ close
    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self.runner.busy:
            answer = QMessageBox.question(self, "A task is still running",
                                          "Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.cancel()
        event.accept()
