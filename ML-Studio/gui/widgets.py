"""Reusable widgets: data tables, a generic parameter form, and a plot canvas.

The parameter form is the important one.  Every screen in this application is
driven by the declarative specs in :mod:`core` — ``models.Param``,
``plots.PlotSpec.params``, ``clean.OpSpec.params`` — so one builder turns any of
them into real widgets.  Adding a hyperparameter to the registry makes it appear
in the tuning and modelling screens with no GUI code at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QTableView, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget,
)

from core import clean as clean_mod

from . import theme


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def hint(text: str, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    return theme.tag(label, "muted")


def section_title(text: str) -> QLabel:
    return theme.tag(QLabel(text), "section")


def separator() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    return theme.tag(line, "rule")


class Callout(QFrame):
    """A bordered note used for guidance, empty states and warnings.

    Three tones, chosen by meaning rather than colour: ``info`` for context, ``tip``
    for the thing a beginner would otherwise miss, ``warn`` for a real problem.  An
    optional action button turns it into a signpost — "this screen needs a model →
    take me there".
    """

    TONES = {"info": "Note", "tip": "Tip", "warn": "Heads up"}

    def __init__(self, text: str = "", tone: str = "info", title: str | None = None,
                 parent=None):
        super().__init__(parent)
        theme.tag(self, "callout", tone=tone)
        self._callback = None

        self._title = theme.tag(QLabel(), "callout-title")
        self._body = theme.tag(QLabel(), "callout-body")
        self._body.setWordWrap(True)
        self._action = QPushButton()
        theme.tag(self._action, "ghost")
        self._action.setVisible(False)
        self._action.clicked.connect(self._fire)

        inner = QVBoxLayout(self)
        inner.setContentsMargins(12, 10, 12, 10)
        inner.setSpacing(4)
        inner.addWidget(self._title)
        inner.addWidget(self._body)
        inner.addWidget(self._action, 0, Qt.AlignmentFlag.AlignLeft)

        self.set_text(text, tone, title)

    def _fire(self):
        if self._callback is not None:
            self._callback()

    def set_text(self, text: str, tone: str | None = None, title: str | None = None):
        if tone is not None:
            theme.retag(self, tone=tone)
        heading = self.TONES.get(tone or "info", "") if title is None else title
        self._title.setText(heading)
        self._title.setVisible(bool(heading))
        self._body.setText(text)
        self.setVisible(bool(text))
        return self

    def set_action(self, text: str, callback=None):
        self._callback = callback
        self._action.setText(text or "")
        self._action.setVisible(bool(text))
        return self


def scrollable(inner: QWidget) -> QScrollArea:
    """Put a panel in a scroll area, with a floor under its width.

    The floor is what keeps a form readable.  A form's input boxes are only as wide as
    the column they sit in, so a splitter that squeezes the panel below the form's
    natural width clips the inputs and adds a horizontal scrollbar, which is what
    happened before this existed.  The cap keeps that floor from making the window
    impossible to fit: at 560 the floors of the panels on screens 1 and 18 and the chart
    toolbar added up past the width of a 1400-px window, so those screens scrolled
    sideways for no good reason.  A form's fields are line edits — 440 px is comfortably
    wider than the label plus a spin box, and the panel's own content is what is used
    when it is wider than the floor.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setFrameShape(QFrame.Shape.NoFrame)
    needed = inner.minimumSizeHint().width()
    if needed > 0:
        area.setMinimumWidth(min(needed + 22, 440))
    return area


def fmt_value(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:,.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------
# data table
# --------------------------------------------------------------------------

class PandasModel(QAbstractTableModel):
    """A QAbstractTableModel over a DataFrame — handles 100k rows without blinking."""

    def __init__(self, frame: pd.DataFrame | None = None, max_rows: int = 20000, parent=None):
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._max_rows = max_rows
        self._display = self._frame.head(max_rows)

    def set_frame(self, frame: pd.DataFrame | None):
        self.beginResetModel()
        self._frame = frame if frame is not None else pd.DataFrame()
        self._display = self._frame.head(self._max_rows)
        self.endResetModel()

    @property
    def truncated(self) -> bool:
        return len(self._frame) > len(self._display)

    def rowCount(self, parent=QModelIndex()):  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._display)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._display.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            try:
                value = self._display.iat[index.row(), index.column()]
            except (IndexError, KeyError):
                return None
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "—"
            if isinstance(value, (float, np.floating)):
                text = f"{value:,.4f}" if abs(value) >= 1e-4 else f"{value:.4g}"
            elif isinstance(value, (int, np.integer)):
                text = f"{value:,}"
            else:
                text = str(value)
            return text
        if role == Qt.ItemDataRole.TextAlignmentRole and self._display.dtypes.iloc[index.column()].kind in "ifu":
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole and index.row() % 2 == 1:
            return QColor("#fafcfd")
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._display.columns[section])
        return str(section + 1)


class DataFrameView(QWidget):
    """Table + row filter + CSV export, sized for a workbench panel."""

    def __init__(self, max_rows: int = 20000, parent=None):
        super().__init__(parent)
        self.model = PandasModel(max_rows=max_rows)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)

        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("filter rows…")
        self.filter_box.textChanged.connect(self.proxy.setFilterFixedString)
        self.filter_box.setMaximumWidth(240)

        self.status = QLabel("no data")
        theme.tag(self.status, "muted")

        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy the visible rows to the clipboard as tab-separated text")
        copy_button.clicked.connect(self.copy_to_clipboard)
        export_button = QPushButton("Export CSV…")
        export_button.clicked.connect(self.export_csv)
        autosize = QPushButton("Fit columns")
        autosize.clicked.connect(self.autosize_columns)

        bar = QHBoxLayout()
        bar.addWidget(self.filter_box)
        bar.addWidget(self.status)
        bar.addStretch(1)
        bar.addWidget(autosize)
        bar.addWidget(copy_button)
        bar.addWidget(export_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.table, 1)

    def set_frame(self, frame: pd.DataFrame | None, note: str = ""):
        self.model.set_frame(frame)
        if frame is None or not len(frame):
            self.status.setText("no data")
            return
        text = f"{len(frame):,} rows × {frame.shape[1]} columns"
        if self.model.truncated:
            text += f" · showing first {len(self.model._display):,}"
        if note:
            text += f" · {note}"
        self.status.setText(text)
        self.autosize_columns()

    def current_frame(self) -> pd.DataFrame:
        return self.model._display

    def autosize_columns(self):
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        for column in range(self.model.columnCount()):
            if header.sectionSize(column) > 260:
                header.resizeSection(column, 260)

    def copy_to_clipboard(self):
        frame = self.current_frame()
        if frame.empty:
            return
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText(frame.to_csv(sep="\t", index=False))

    def export_csv(self):
        frame = self.current_frame()
        if frame.empty:
            QMessageBox.information(self, "Nothing to export", "There are no rows to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export table", "table.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Exported", f"Written to\n{path}")


class KeyValueView(QWidget):
    """A two-column field/value table, for summaries and bundle details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["field", "value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def set_pairs(self, pairs):
        if isinstance(pairs, pd.DataFrame):
            pairs = list(pairs.itertuples(index=False, name=None))
        self.table.setRowCount(0)
        for row in pairs or []:
            index = self.table.rowCount()
            self.table.insertRow(index)
            for column, value in enumerate(row[:2]):
                item = QTableWidgetItem(fmt_value(value))
                self.table.setItem(index, column, item)


# --------------------------------------------------------------------------
# log pane
# --------------------------------------------------------------------------

class LogPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        theme.tag(self.view, "mono")
        clear = QPushButton("Clear")
        clear.clicked.connect(self.view.clear)
        bar = QHBoxLayout()
        bar.addWidget(section_title("Activity log"))
        bar.addStretch(1)
        bar.addWidget(clear)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

    def append(self, text: str):
        self.view.appendPlainText(text)
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())


# --------------------------------------------------------------------------
# the generic parameter form
# --------------------------------------------------------------------------

class ColumnPicker(QDialog):
    def __init__(self, columns, preselected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose columns")
        self.setMinimumSize(360, 460)
        self.list = QListWidget()
        preselected = set(preselected or [])
        for column in columns:
            item = QListWidgetItem(str(column))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if column in preselected else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        all_button = QPushButton("Select all")
        all_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        none_button = QPushButton("Clear")
        none_button.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        bar = QHBoxLayout()
        bar.addWidget(all_button)
        bar.addWidget(none_button)
        bar.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.list, 1)
        layout.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def selected(self) -> list:
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]


class ConditionEditor(QWidget):
    """Rows of (column, test, value, value2) with an AND/OR switch."""

    def __init__(self, columns=None, parent=None):
        super().__init__(parent)
        self.columns = list(columns or [])
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows: list[tuple] = []

        self.logic = QComboBox()
        self.logic.addItems(["all conditions (AND)", "any condition (OR)"])
        self.negate = QCheckBox("invert (keep rows that do NOT match)")
        switches = QHBoxLayout()
        switches.addWidget(QLabel("combine:"))
        switches.addWidget(self.logic)
        switches.addWidget(self.negate)
        switches.addStretch(1)

        add = QPushButton("Add condition")
        add.clicked.connect(lambda: self.add_row())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(switches)
        layout.addLayout(self.rows_layout)
        layout.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        if not self.columns:
            self.add_row()

    def set_columns(self, columns):
        self.columns = list(columns or [])
        for row in self.rows:
            combo, _, _, _ = row
            current = combo.currentText()
            combo.clear()
            combo.addItems(self.columns)
            if current in self.columns:
                combo.setCurrentText(current)

    def add_row(self, column=None, test="eq", value="", value2=""):
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        column_combo = QComboBox()
        column_combo.setEditable(True)
        column_combo.addItems(self.columns)
        if column:
            column_combo.setCurrentText(str(column))
        column_combo.setMinimumWidth(130)
        test_combo = QComboBox()
        for key, label in clean_mod.FILTER_TESTS.items():
            test_combo.addItem(label, key)
        index = test_combo.findData(test)
        if index >= 0:
            test_combo.setCurrentIndex(index)
        value_edit = QLineEdit(str(value))
        value_edit.setPlaceholderText("value")
        value2_edit = QLineEdit(str(value2))
        value2_edit.setPlaceholderText("and (for 'between')")
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda: self.remove_row(holder))
        for widget in (column_combo, test_combo, value_edit, value2_edit):
            row.addWidget(widget)
        row.addWidget(remove)
        self.rows_layout.addWidget(holder)
        self.rows.append((column_combo, test_combo, value_edit, value2_edit))
        holder.setProperty("row_widget", True)
        self._holder_map = getattr(self, "_holder_map", {})
        self._holder_map[holder] = (column_combo, test_combo, value_edit, value2_edit)

    def remove_row(self, holder):
        mapping = getattr(self, "_holder_map", {})
        entry = mapping.pop(holder, None)
        if entry and entry in self.rows:
            self.rows.remove(entry)
        holder.setParent(None)
        holder.deleteLater()

    def value(self) -> dict:
        """The whole clause: conditions plus how they combine."""
        conditions = []
        for column_combo, test_combo, value_edit, value2_edit in self.rows:
            column = column_combo.currentText().strip()
            test = test_combo.currentData()
            if not column and test not in ("is_null", "not_null"):
                continue
            condition = {"column": column, "test": test, "value": value_edit.text()}
            if test == "between":
                condition["value2"] = value2_edit.text()
            conditions.append(condition)
        return {
            "conditions": conditions,
            "logic": "or" if self.logic.currentIndex() == 1 else "and",
            "negate": bool(self.negate.isChecked()),
        }

    def set_value(self, value):
        """Accept either a bare list of conditions or a full clause dict."""
        if isinstance(value, dict):
            conditions = value.get("conditions") or []
            self.logic.blockSignals(True)
            self.logic.setCurrentIndex(1 if value.get("logic") == "or" else 0)
            self.logic.blockSignals(False)
            self.negate.setChecked(bool(value.get("negate", False)))
        else:
            conditions = value or []
        for holder in list(getattr(self, "_holder_map", {}).keys()):
            self.remove_row(holder)
        for condition in conditions:
            self.add_row(condition.get("column"), condition.get("test", "eq"),
                         condition.get("value", ""), condition.get("value2", ""))


class FieldForm(QWidget):
    """Builds widgets from a list of normalised field descriptions.

    A field dict is ``{name, label, kind, default, choices, optional, help}`` where
    ``kind`` is one of text, textarea, int, float, bool, choice, column, columns,
    mapping, conditions.
    """

    changed = pyqtSignal()

    def __init__(self, fields=None, columns=None, parent=None):
        super().__init__(parent)
        self.fields: list[dict] = []
        self.widgets: dict[str, QWidget] = {}
        self.columns = list(columns or [])
        self.form = QFormLayout(self)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        if fields:
            self.set_fields(fields)

    # ---------------------------------------------------------------- build
    def set_fields(self, fields: list[dict]):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields = list(fields or [])
        self.widgets = {}
        for field in self.fields:
            widget = self._build_widget(field)
            self.widgets[field["name"]] = widget
            label = field.get("label") or field["name"]
            if field.get("optional"):
                label += " (optional)"
            if field.get("help"):
                widget.setToolTip(field["help"])
            self.form.addRow(label + ":", widget)

    def set_columns(self, columns):
        self.columns = list(columns or [])
        for field in self.fields:
            widget = self.widgets.get(field["name"])
            if field["kind"] == "column" and isinstance(widget, QComboBox):
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems(self.columns)
                if current in self.columns or current == "":
                    widget.setCurrentText(current)   # keep an optional column unset
                widget.blockSignals(False)
            elif field["kind"] == "conditions" and isinstance(widget, ConditionEditor):
                widget.set_columns(self.columns)

    def _build_widget(self, field: dict) -> QWidget:
        kind = field["kind"]
        default = field.get("default")
        choices = field.get("choices") or []

        if kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(default))
            box.stateChanged.connect(lambda *_: self.changed.emit())
            return box

        if kind == "choice":
            combo = QComboBox()
            combo.setEditable(field.get("editable", False))
            for choice in choices:
                combo.addItem("(none)" if choice is None else str(choice), choice)
            if default is not None:
                index = combo.findData(default)
                if index < 0:
                    index = combo.findText(str(default))
                if index >= 0:
                    combo.setCurrentIndex(index)
            elif choices:
                combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
            return combo

        if kind == "column":
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(self.columns)
            if default is not None:
                combo.setCurrentText(str(default))
            elif field.get("optional"):
                # an *optional* column starts unset — silently picking the first
                # column would, for instance, weight every row by it during training
                combo.setCurrentText("")
            combo.currentTextChanged.connect(lambda *_: self.changed.emit())
            return combo

        if kind == "columns":
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            values = default
            if isinstance(values, (list, tuple)):
                edit.setText(", ".join(str(v) for v in values))
            elif values:
                edit.setText(str(values))
            edit.setPlaceholderText("comma separated")
            edit.setCursorPosition(0)
            button = QPushButton("…")
            button.setFixedWidth(30)

            def pick():
                preselected = [v.strip() for v in edit.text().split(",") if v.strip()]
                dialog = ColumnPicker(self.columns, preselected, self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    edit.setText(", ".join(dialog.selected()))
                    edit.setCursorPosition(0)

            button.clicked.connect(pick)
            layout.addWidget(edit, 1)
            layout.addWidget(button)
            holder.setProperty("line_edit", edit)
            edit.textChanged.connect(lambda *_: self.changed.emit())
            holder._edit = edit  # type: ignore[attr-defined]
            return holder

        if kind == "int":
            spin = QSpinBox()
            spin.setRange(-10_000_000, 10_000_000)
            if default is not None:
                spin.setValue(int(float(default)))
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            return spin

        if kind == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            spin.setSingleStep(0.1)
            if default is not None:
                spin.setValue(float(default))
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            return spin

        if kind == "textarea":
            edit = QTextEdit()
            edit.setAcceptRichText(False)
            if default:
                edit.setPlainText(str(default))
            edit.setMinimumHeight(120)
            edit.textChanged.connect(lambda *_: self.changed.emit())
            return edit

        if kind == "mapping":
            edit = QLineEdit()
            if isinstance(default, dict):
                edit.setText("; ".join(f"{k}:{v}" for k, v in default.items()))
            elif default:
                edit.setText(str(default))
            edit.setPlaceholderText("old:new; another:replacement")
            edit.textChanged.connect(lambda *_: self.changed.emit())
            return edit

        if kind == "conditions":
            editor = ConditionEditor(self.columns)
            if default:
                editor.set_value(default)
            editor.setMinimumHeight(90)
            return editor

        # plain text
        edit = QLineEdit()
        if default is not None:
            edit.setText(str(default))
        edit.textChanged.connect(lambda *_: self.changed.emit())
        return edit

    # ---------------------------------------------------------------- read
    def values(self) -> dict:
        out = {}
        for field in self.fields:
            out[field["name"]] = self._read(field)
        return out

    def _read(self, field: dict):
        widget = self.widgets.get(field["name"])
        kind = field["kind"]
        if widget is None:
            return None
        if kind == "bool":
            return widget.isChecked()
        if kind == "choice":
            data = widget.currentData()
            text = widget.currentText()
            if data is None and text in ("", "(none)"):
                return None
            return data if data is not None else text
        if kind == "column":
            text = widget.currentText().strip()
            return text or None
        if kind == "columns":
            edit = getattr(widget, "_edit", None)
            text = edit.text() if edit is not None else ""
            return [v.strip() for v in text.split(",") if v.strip()]
        if kind in ("int", "float"):
            return widget.value()
        if kind == "textarea":
            return widget.toPlainText()
        if kind == "mapping":
            return widget.text()
        if kind == "conditions":
            return widget.value()
        text = widget.text()
        if text == "" and field.get("optional"):
            return None
        return text

    def set_values(self, values: dict):
        values = values or {}
        for field in self.fields:
            name = field["name"]
            if name not in values:
                continue
            self._write(field, values[name])

    def update_field(self, name: str, **changes):
        """Rebuild one row (its choices, say) while keeping every other value.

        ``set_fields`` recreates every widget, which throws away what the user has
        typed; this is for the common case of one field's choice list depending on
        another field — the metric list depending on the task, say.
        """
        index = next((i for i, field in enumerate(self.fields) if field["name"] == name), None)
        if index is None:
            return
        values = self.values()
        self.fields[index] = {**self.fields[index], **changes}
        self.widgets.pop(name, None)
        self.form.removeRow(index)
        field = self.fields[index]
        widget = self._build_widget(field)
        self.widgets[name] = widget
        label = field.get("label") or name
        if field.get("optional"):
            label += " (optional)"
        if field.get("help"):
            widget.setToolTip(field["help"])
        self.form.insertRow(index, label + ":", widget)
        self.set_values({name: values.get(name)})

    def _write(self, field: dict, value):
        widget = self.widgets.get(field["name"])
        if widget is None:
            return
        kind = field["kind"]
        widget.blockSignals(True)
        try:
            if kind == "bool":
                widget.setChecked(bool(value))
            elif kind == "choice":
                index = widget.findData(value)
                if index < 0:
                    index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif kind == "column":
                widget.setCurrentText("" if value is None else str(value))
            elif kind == "columns":
                edit = getattr(widget, "_edit", None)
                if edit is not None:
                    if isinstance(value, (list, tuple)):
                        edit.setText(", ".join(str(v) for v in value))
                    else:
                        edit.setText("" if value is None else str(value))
                    edit.setCursorPosition(0)
            elif kind in ("int", "float"):
                if value is not None:
                    widget.setValue(float(value) if kind == "float" else int(float(value)))
            elif kind == "textarea":
                widget.setPlainText("" if value is None else str(value))
            elif kind == "mapping":
                if isinstance(value, dict):
                    widget.setText("; ".join(f"{k}:{v}" for k, v in value.items()))
                else:
                    widget.setText("" if value is None else str(value))
            elif kind == "conditions":
                widget.set_value(value)
            else:
                widget.setText("" if value is None else str(value))
        finally:
            widget.blockSignals(False)


# --------------------------------------------------------------------------
# field adapters
# --------------------------------------------------------------------------

def fields_from_model_params(params) -> list[dict]:
    """Turn ``models.Param`` objects into FieldForm descriptions."""
    out = []
    for param in params:
        kind = {"int": "int", "float": "float", "bool": "bool", "choice": "choice",
                "str": "text", "none": "text"}.get(param.kind, "text")
        out.append({
            "name": param.name,
            "label": param.label,
            "kind": kind,
            "default": param.default,
            "choices": list(param.choices),
            "optional": False,
            "help": param.help or (f"range {param.low} … {param.high}" if param.low is not None else ""),
        })
    return out


def fields_from_plot(spec) -> list[dict]:
    out = []
    for param in spec.params:
        out.append({
            "name": param["name"],
            "label": param.get("label") or param["name"],
            "kind": param.get("kind", "text"),
            "default": param.get("default"),
            "choices": list(param.get("choices") or []),
            "optional": param.get("optional", True),
            "help": param.get("help", ""),
        })
    return out


def fields_from_op(op_spec) -> list[dict]:
    out = []
    for param in op_spec.params:
        kind = {"text": "text", "number": "text", "bool": "bool", "choice": "choice",
                "columns": "columns", "column": "column", "mapping": "mapping",
                "conditions": "conditions"}.get(param["kind"], "text")
        out.append({
            "name": param["name"],
            "label": param["label"],
            "kind": kind,
            "default": param["default"],
            "choices": list(param["choices"]),
            "optional": param["optional"],
            "help": param["help"],
        })
    return out


# --------------------------------------------------------------------------
# plot canvas
# --------------------------------------------------------------------------

class PlotCanvas(QWidget):
    """Shows one matplotlib figure at a time, with the usual navigation toolbar.

    Each new figure gets its own ``FigureCanvasQTAgg`` widget rather than having
    the old canvas re-pointed at a different figure.  Re-pointing works, but it
    leaves the figure manager and the toolbar disagreeing about which figure they
    belong to; rebuilding the pair is a few more lines and never misbehaves.  The
    figure being replaced is closed so pyplot does not accumulate figures for the
    life of the window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = self._make_placeholder()
        self._layout.addWidget(self._placeholder, 1)

    @staticmethod
    def _make_placeholder() -> QLabel:
        label = theme.tag(QLabel("no figure drawn yet"), "muted")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    # ------------------------------------------------------------------ api
    def show_figure(self, figure):
        """Display a figure built anywhere (including on a worker thread)."""
        self._release()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def clear(self):
        self._release()
        self._placeholder = self._make_placeholder()
        self._layout.addWidget(self._placeholder, 1)

    def save_figure(self, path: str, dpi: int = 200) -> str:
        if self.figure is None:
            raise ValueError("there is no figure to save")
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return path

    def png_bytes(self, dpi: int = 140) -> bytes:
        from core import plots as plots_mod

        if self.figure is None:
            raise ValueError("there is no figure to save")
        return plots_mod.figure_png_bytes(self.figure, dpi=dpi)

    # -------------------------------------------------------------- private
    def _release(self):
        for widget in (self.toolbar, self.canvas, self._placeholder):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self.toolbar = self.canvas = self._placeholder = None
        if self.figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.figure)
            except Exception:
                pass
            self.figure = None


# --------------------------------------------------------------------------
# headline metrics
# --------------------------------------------------------------------------

class MetricCards(QWidget):
    """A row of big numbers with a small caption under each."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._cards: list[QWidget] = []

    def set_metrics(self, items):
        for card in self._cards:
            self._layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        for item in items or []:
            label, value, caption = (list(item) + ["", "", ""])[:3]
            card = QFrame()
            theme.tag(card, "card")
            inner = QVBoxLayout(card)
            inner.setContentsMargins(12, 10, 12, 10)
            inner.setSpacing(2)
            top = theme.tag(QLabel(str(label).upper()), "stat-label")
            middle = theme.tag(QLabel(fmt_value(value)), "stat")
            bottom = theme.tag(QLabel(str(caption or "")), "muted")
            bottom.setWordWrap(True)
            inner.addWidget(top)
            inner.addWidget(middle)
            inner.addWidget(bottom)
            self._layout.addWidget(card, 1)
            self._cards.append(card)


# --------------------------------------------------------------------------
# the base class every screen inherits
# --------------------------------------------------------------------------

class StepPage(QWidget):
    """Common scaffolding for one screen of the workbench.

    A page declares a title and a subtitle, builds its widgets in :meth:`build`,
    and is refreshed by the window whenever :class:`AppState` changes.  Long work
    is handed to the shared :class:`~gui.workers.TaskRunner` through
    :meth:`run_task`, which also reports failures in one consistent way.

    A page may also declare ``requires`` — ``"data"`` or ``"model"`` — and the window
    then shows a signpost at the top of the screen whenever that prerequisite is
    missing, with a button that jumps to the screen which supplies it.  That is the
    whole beginner-protection mechanism: no screen is ever silently useless.
    """

    title = "Step"
    subtitle = ""
    #: ``None``, ``"data"`` or ``"model"`` — what must exist for this screen to work.
    requires: str | None = None
    #: Optional replacement sentence for the signpost.
    requires_note = ""

    def __init__(self, state, runner, window=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.runner = runner
        self.window = window

        heading = theme.tag(QLabel(self.title), "page-title")
        blurb = theme.tag(QLabel(self.subtitle), "page-subtitle") if self.subtitle else None
        if blurb is not None:
            blurb.setWordWrap(True)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(heading)
        if blurb is not None:
            header.addWidget(blurb)

        self._require_callout = Callout("", "warn")

        # The body goes inside a scroll area, and this is load-bearing rather than a
        # nicety.  A page builds a form, a table or a splitter with a chart, and the
        # *worst* page decides the whole window's minimum size — a stacked widget takes
        # the maximum of its pages.  That made the window demand 1478 x 1060 on a
        # 1920 x 1080 screen, and a window that cannot shrink to the work area cannot be
        # maximized either: Mutter gives up and leaves a "maximized" window parked at its
        # minimum size in the middle of the display, with a 436 px gutter beside it.  With
        # the body scrollable, the window's own minimum stays small and it always fills the
        # screen.
        #
        # The vertical size policy is the part that is easy to get wrong.  A resizable
        # scroll area sizes its widget to ``max(viewport, qSmartMinSize(widget))``, and
        # qSmartMinSize returns the *minimum* size hint when the policy carries the shrink
        # flag — which every default widget policy does.  A page's minimum size hint is
        # smaller than the height its form actually needs (a box layout will overrule an
        # individual minimum when it is over-constrained), so a plain QWidget body was
        # sized to that minimum and every row of the form was squeezed and clipped.
        # MinimumExpanding has *no* shrink flag, so the floor becomes the size hint — the
        # height the page really wants — and the body scrolls instead of compressing.
        # Horizontally the default is kept, so a table or a canvas may still be narrowed
        # rather than forcing a sideways scrollbar.
        self.body = QVBoxLayout()
        self.body.setSpacing(11)
        content = QWidget()
        content.setLayout(self.body)
        content.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.MinimumExpanding)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(content)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(11)
        outer.addLayout(header)
        outer.addWidget(self._require_callout)
        outer.addWidget(separator())
        outer.addWidget(self._scroll, 1)
        self.build()

    # ------------------------------------------------------------ prerequisites
    def _requirement(self):
        """Return ``(kind, sentence)`` when this screen is missing something, else None."""
        if self.requires == "data" and not self.state.has_data:
            return ("data", self.requires_note or (
                "This screen works on a loaded dataset, and none is loaded yet. "
                "Open one on “1 · Load data” — the built-in example dataset needs no file."))
        if self.requires == "model" and not self.state.has_model:
            return ("model", self.requires_note or (
                "This screen works on a trained model, and none is trained yet. "
                "Train one on “5 · Model”, or press Ctrl+R to run the whole workflow "
                "automatically and stop here."))
        return None

    def update_requirements(self):
        """Show or hide the prerequisite signpost.  The window calls this on every refresh."""
        missing = self._requirement()
        if missing is None:
            self._require_callout.setVisible(False)
            return
        kind, sentence = missing
        target = "1 · Load data" if kind == "data" else "5 · Model"
        self._require_callout.set_text(sentence, "warn")
        self._require_callout.set_action(f"Take me to {target}", lambda k=kind: self._goto(k))
        self._require_callout.setVisible(True)

    def _goto(self, kind: str):
        if self.window is None:
            return
        from .steps import LoadPage, ModelPage  # late import: steps import this module
        self.window._go_to(LoadPage if kind == "data" else ModelPage)

    # ----------------------------------------------------------------- hooks
    def build(self):
        """Create the widgets; called once from the constructor."""

    def refresh(self):
        """Re-read the shared state; called whenever it changes."""

    def on_enter(self):
        """Called when the page becomes visible."""

    # --------------------------------------------------------------- helpers
    def data_token(self):
        """A cheap fingerprint of the working frame, for 'did the data change?' checks."""
        frame = self.state.df
        if frame is None:
            return None
        return (id(frame), frame.shape, tuple(map(str, frame.columns)))

    def log(self, message: str):
        self.state.log_line(message)

    def note(self, message: str):
        self.state.note(message)

    def warn(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def require_data(self) -> bool:
        if self.state.has_data:
            return True
        self.warn("No data yet", "Load a dataset on the Data screen first.")
        return False

    def require_model(self) -> bool:
        if self.state.has_model:
            return True
        self.warn("No trained model yet",
                  "Train a model on the Model screen first — this screen needs one.")
        return False

    def run_task(self, fn, label: str, on_done=None, on_progress=None, on_cancelled=None,
                 *args, **kwargs):
        """Start ``fn(progress, cancel, *args)`` on a worker thread."""

        def on_error(message, detail):
            self.state.log_line(f"{label} failed: {message}")
            for line in (detail or "").strip().splitlines()[-6:]:
                self.state.log_line(f"    {line}")
            QMessageBox.critical(self, f"{label} failed", message)

        return self.runner.run(fn, *args, label=label, on_done=on_done,
                               on_error=on_error, on_progress=on_progress,
                               on_cancelled=on_cancelled, **kwargs)

