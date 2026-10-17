"""The visual language of the workbench: one palette, one stylesheet, applied once.

Everything cosmetic lives here so the screens never hard-code a colour.  Two rules
make the rest of the GUI code simple:

* **Roles, not colours.**  A widget says what it *is* — ``theme.tag(button,
  "primary")``, ``theme.tag(label, "muted")`` — and the stylesheet decides how that
  looks.  Re-theming is then a change to this file and nothing else.
* **The Fusion style.**  Qt's native style on Linux is drawn by the desktop theme and
  ignores large parts of a stylesheet, so the app is pinned to Fusion, which honours
  it.  That is what makes the look identical on every machine.

Two Qt details are worth knowing before editing the stylesheet, because both produce
confusing symptoms:

* ``tag`` repolishes the widget.  Qt caches a widget's resolved style, so a property
  selector only takes effect after ``unpolish``/``polish`` — without that, a role set
  after the widget is shown appears to do nothing.
* A ``QComboBox``, ``QSpinBox`` or ``QCheckBox`` loses its indicator the moment the
  stylesheet gives it a custom border: Qt clips the sub-control down to a sliver and
  you get an empty bracket where the arrow was.  The only reliable fix is to supply
  the indicator as an image, so the few needed are drawn from the palette at startup
  (see :func:`_assets`) rather than checked in as files.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QWidget

# --------------------------------------------------------------------------
# colours
# --------------------------------------------------------------------------

PALETTE = {
    "ink": "#16202b",          # headings and primary text
    "body": "#33414f",         # ordinary text
    "muted": "#5f7183",        # secondary text, hints, captions
    "faint": "#8b9aa8",        # disabled and decorative text
    "line": "#d9e1e9",         # hairline borders
    "line_strong": "#c3cfdb",  # input borders
    "surface": "#ffffff",      # panels and cards
    "canvas": "#f6f8fb",       # window background
    "sidebar": "#eef3f8",      # navigation rail
    "sidebar_hover": "#e3ecf6",
    "accent": "#2563a8",       # the one interactive colour
    "accent_dark": "#1b4c84",
    "accent_soft": "#e8f0fa",
    "ok": "#1f7a52",
    "ok_soft": "#eaf5ef",
    "warn": "#a4560f",
    "warn_soft": "#fdf4e9",
    "danger": "#b3261e",
    "danger_soft": "#fdecea",
}

#: Preferred UI families, best first.  Qt falls back automatically, and DejaVu Sans is
#: always present on this machine, so text is never missing.
FONT_FAMILIES = ("Inter", "Segoe UI", "Noto Sans", "Ubuntu", "Cantarell", "DejaVu Sans")

MONO_FAMILIES = ("JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Menlo", "Consolas",
                 "monospace")


# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------

def palette() -> QPalette:
    """The application palette, built from :data:`PALETTE`.

    Pinning the *style* is not enough.  Fusion takes its colours from the application
    palette, and Qt seeds that from the desktop's colour scheme — so on a desktop set to
    a dark theme every widget the stylesheet does not name comes out dark on a light
    window: a plain ``QListWidget`` viewport, a table's header, a toolbar, and the
    standard icons Qt draws itself (which is why the sidebar's icons went pale).  Setting
    the palette explicitly is what makes the look the same on every machine, and it is
    the only fix that covers widgets this file never mentions.
    """
    ink = QColor(PALETTE["ink"])
    body = QColor(PALETTE["body"])
    muted = QColor(PALETTE["muted"])
    faint = QColor(PALETTE["faint"])
    line = QColor(PALETTE["line"])
    line_strong = QColor(PALETTE["line_strong"])
    surface = QColor(PALETTE["surface"])
    canvas = QColor(PALETTE["canvas"])
    accent = QColor(PALETTE["accent"])
    white = QColor("#ffffff")

    out = QPalette()
    roles = {
        QPalette.ColorRole.Window: canvas,
        QPalette.ColorRole.WindowText: body,
        QPalette.ColorRole.Base: surface,
        QPalette.ColorRole.AlternateBase: QColor("#fbfcfe"),
        QPalette.ColorRole.Text: body,
        QPalette.ColorRole.PlaceholderText: faint,
        QPalette.ColorRole.Button: surface,
        QPalette.ColorRole.ButtonText: body,
        QPalette.ColorRole.BrightText: white,
        QPalette.ColorRole.Light: white,
        QPalette.ColorRole.Midlight: canvas,
        QPalette.ColorRole.Mid: line_strong,
        QPalette.ColorRole.Dark: muted,
        QPalette.ColorRole.Shadow: QColor("#a9b8c7"),
        QPalette.ColorRole.Highlight: accent,
        QPalette.ColorRole.HighlightedText: white,
        QPalette.ColorRole.Link: accent,
        QPalette.ColorRole.LinkVisited: QColor(PALETTE["accent_dark"]),
        QPalette.ColorRole.ToolTipBase: surface,
        QPalette.ColorRole.ToolTipText: ink,
    }
    for role, color in roles.items():
        out.setColor(role, color)

    # Qt draws greyed-out text and icons from the disabled group, so it has to be set too
    # or disabled rows come out black-on-grey.
    for role, color in ((QPalette.ColorRole.WindowText, faint),
                        (QPalette.ColorRole.Text, faint),
                        (QPalette.ColorRole.ButtonText, faint),
                        (QPalette.ColorRole.Base, canvas),
                        (QPalette.ColorRole.Button, canvas),
                        (QPalette.ColorRole.Highlight, line),
                        (QPalette.ColorRole.HighlightedText, white)):
        out.setColor(QPalette.ColorGroup.Disabled, role, color)
    return out

# --------------------------------------------------------------------------
# generated indicator images
# --------------------------------------------------------------------------

_ASSET_CACHE: dict = {}


def _draw_arrow(color: str, direction: str, width: int = 18, height: int = 12) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    cx, cy = width / 2.0, height / 2.0
    if direction == "down":
        points = [(cx - 4, cy - 2), (cx + 4, cy - 2), (cx, cy + 3)]
    else:
        points = [(cx - 4, cy + 2), (cx + 4, cy + 2), (cx, cy - 3)]
    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    painter.end()
    return image


def _draw_check(color: str, size: int = 16) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(QPolygonF([QPointF(4.0, 8.4), QPointF(6.8, 11.2), QPointF(12.0, 4.8)]))
    painter.end()
    return image


def _assets() -> dict | None:
    """Write the indicator images and return their paths, or ``None`` if impossible.

    They are drawn from the palette each time the stylesheet is built and named after a
    hash of the colours, so a palette change can never leave a stale arrow behind.  A
    read-only temp directory is the one thing that can break this, which is why the
    caller has a fallback.
    """
    key = hashlib.sha1(
        f"{PALETTE['muted']}{PALETTE['accent']}{PALETTE['surface']}{PALETTE['line_strong']}"
        .encode("utf-8")).hexdigest()[:8]
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]

    directory = os.path.join(tempfile.gettempdir(), f"ml-workbench-theme-{key}")
    try:
        os.makedirs(directory, exist_ok=True)
        wanted = {
            "arrow_down": _draw_arrow(PALETTE["muted"], "down"),
            "arrow_up": _draw_arrow(PALETTE["muted"], "up"),
            "check": _draw_check("#ffffff"),
            "box": None,
        }
        for name, image in wanted.items():
            if image is None:
                continue
            if not image.save(os.path.join(directory, f"{name}.png")):
                return None
        paths = {name: _url(os.path.join(directory, f"{name}.png"))
                 for name in wanted if wanted[name] is not None}
    except OSError:
        return None
    _ASSET_CACHE[key] = paths
    return paths


def _url(path: str) -> str:
    return "url(" + path.replace("\\", "/") + ")"


# --------------------------------------------------------------------------
# the stylesheet
# --------------------------------------------------------------------------

#: Rounded inputs, spin boxes and combo boxes with indicators drawn from images.
_INPUT_RULES = """
QLineEdit, QPlainTextEdit, QTextEdit {
    background: %(surface)s;
    border: 1px solid %(line_strong)s;
    border-radius: 6px;
    padding: 4px 7px;
    selection-background-color: %(accent)s;
    selection-color: #ffffff;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus { border: 1px solid %(accent)s; }
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    background: %(canvas)s; color: %(faint)s;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background: %(surface)s;
    border: 1px solid %(line_strong)s;
    border-radius: 6px;
    padding: 4px 7px;
    selection-background-color: %(accent)s;
    selection-color: #ffffff;
}
QSpinBox, QDoubleSpinBox { padding-right: 23px; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid %(accent)s; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background: %(canvas)s; color: %(faint)s;
}
QComboBox::drop-down {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; border: none; background: transparent;
}
QComboBox::down-arrow { image: %(arrow_down)s; width: 11px; height: 7px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 20px;
    border: none; border-left: 1px solid %(line)s;
    border-top-right-radius: 6px; background: transparent;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
    border: none; border-left: 1px solid %(line)s;
    border-bottom-right-radius: 6px; background: transparent;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: %(sidebar_hover)s;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: %(arrow_up)s; width: 11px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: %(arrow_down)s; width: 11px; height: 7px; }
QComboBox QAbstractItemView {
    background: %(surface)s; border: 1px solid %(line_strong)s;
    selection-background-color: %(accent_soft)s; selection-color: %(ink)s;
    outline: none; padding: 3px;
}
QComboBox QLineEdit { border: none; background: transparent; padding: 0; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid %(line_strong)s; border-radius: 4px;
    background: %(surface)s;
}
QCheckBox::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked {
    background: %(accent)s; border-color: %(accent)s; image: %(check)s;
}
QCheckBox:disabled { color: %(faint)s; }
"""

#: Used only when the indicator images could not be written.  A square native frame
#: looks plain but works, which beats a rounded box with an empty bracket in it.
_INPUT_FALLBACK = """
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: %(surface)s;
    border: 1px solid %(line_strong)s;
    padding: 4px 7px;
    selection-background-color: %(accent)s;
    selection-color: #ffffff;
}
QComboBox QAbstractItemView {
    background: %(surface)s; border: 1px solid %(line_strong)s;
    selection-background-color: %(accent_soft)s; selection-color: %(ink)s;
    outline: none; padding: 3px;
}
"""

STYLESHEET = """
QWidget {
    color: %(body)s;
    font-size: 13px;
}
QMainWindow, QDialog { background: %(canvas)s; }
QStackedWidget { background: %(canvas)s; }

/* ---------------------------------------------------------------- text roles */
QLabel[role="page-title"]    { color: %(ink)s; font-size: 21px; font-weight: 700; }
QLabel[role="page-subtitle"] { color: %(muted)s; font-size: 12.5px; }
QLabel[role="section"]       { color: %(ink)s; font-size: 14px; font-weight: 700; }
QLabel[role="muted"]         { color: %(muted)s; font-size: 12px; }
QLabel[role="warning"]       { color: %(warn)s; font-size: 12px; }
QLabel[role="stat"]          { color: %(ink)s; font-size: 19px; font-weight: 700; }
QLabel[role="stat-label"]    { color: %(muted)s; font-size: 10.5px; font-weight: 600; }
QLabel[role="badge"]         { color: %(accent)s; background: %(accent_soft)s;
                               border-radius: 9px; padding: 2px 9px;
                               font-size: 11px; font-weight: 700; }

/* -------------------------------------------------------------------- buttons */
QPushButton {
    background: %(surface)s;
    border: 1px solid %(line_strong)s;
    border-radius: 6px;
    padding: 6px 13px;
    color: %(body)s;
    min-height: 15px;
}
QPushButton:hover   { background: %(sidebar_hover)s; border-color: %(accent)s; }
QPushButton:pressed { background: %(accent_soft)s; }
QPushButton:disabled { color: %(faint)s; background: %(canvas)s; border-color: %(line)s; }

QPushButton[role="primary"] {
    background: %(accent)s; border: 1px solid %(accent_dark)s;
    color: #ffffff; font-weight: 700; padding: 7px 16px;
}
QPushButton[role="primary"]:hover   { background: %(accent_dark)s; }
QPushButton[role="primary"]:pressed { background: #163f6e; }
QPushButton[role="primary"]:disabled { background: #a9bdd4; border-color: #a9bdd4; color: #eef3f8; }

QPushButton[role="ghost"] {
    background: transparent; border: 1px solid transparent;
    color: %(accent)s; font-weight: 600; padding: 4px 8px;
}
QPushButton[role="ghost"]:hover { background: %(accent_soft)s; }

/* --------------------------------------------------------------------- inputs */
%(input_rules)s

/* ------------------------------------------------------------------- grouping */
QGroupBox {
    background: %(surface)s;
    border: 1px solid %(line)s;
    border-radius: 8px;
    margin-top: 15px;
    padding: 12px 12px 10px 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 11px; padding: 0 6px; color: %(ink)s; background: %(canvas)s;
}

QFrame[role="card"] {
    background: %(surface)s; border: 1px solid %(line)s; border-radius: 8px;
}
QFrame[role="rule"] { background: %(line)s; border: none; }
QFrame[role="callout"] { border-radius: 8px; border: 1px solid %(line)s; }
QFrame[role="callout"][tone="info"] { background: %(accent_soft)s; border-color: #c2d8ef; }
QFrame[role="callout"][tone="tip"]  { background: %(ok_soft)s;     border-color: #c3e2d2; }
QFrame[role="callout"][tone="warn"] { background: %(warn_soft)s;   border-color: #eed6b4; }
QLabel[role="callout-title"] { font-weight: 700; color: %(ink)s; font-size: 12.5px; }
QLabel[role="callout-body"]  { color: %(body)s; font-size: 12.5px; }

/* ---------------------------------------------------------------------- tabs */
QTabWidget::pane {
    border: 1px solid %(line)s; border-radius: 8px; background: %(surface)s; top: -1px;
}
QTabBar::tab {
    background: transparent; color: %(muted)s;
    padding: 7px 15px; margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
}
QTabBar::tab:hover:!selected { background: %(sidebar_hover)s; color: %(body)s; }
QTabBar::tab:selected {
    background: %(surface)s; border-color: %(line)s; border-bottom-color: %(surface)s;
    color: %(ink)s; font-weight: 700;
}

/* -------------------------------------------------------------------- tables */
QTableView, QTableWidget {
    background: %(surface)s;
    alternate-background-color: #fbfcfe;
    border: 1px solid %(line)s;
    border-radius: 8px;
    gridline-color: #eef2f7;
    selection-background-color: %(accent_soft)s;
    selection-color: %(ink)s;
}
QTableView::item, QTableWidget::item { padding: 4px 6px; border: none; }
QHeaderView { background: %(surface)s; border: none; }
QHeaderView::section {
    background: %(sidebar)s; color: %(muted)s;
    padding: 6px 8px; border: none;
    border-right: 1px solid %(line)s; border-bottom: 1px solid %(line)s;
    font-weight: 700; font-size: 11.5px;
}
QHeaderView::section:vertical {
    background: %(canvas)s; color: %(muted)s;
    border-right: 1px solid %(line)s; border-bottom: 1px solid %(line)s;
}
QTableCornerButton::section { background: %(sidebar)s; border: none; }

/* ----------------------------------------------- lists, trees and item views */
/* These are named explicitly as well as covered by the palette: an unstyled
   QListWidget takes its background straight from QPalette::Base, which is how a dark
   desktop theme used to turn the Graphs screen's list black. */
QListView, QListWidget, QTreeView, QTreeWidget, QColumnView {
    background: %(surface)s;
    alternate-background-color: #fbfcfe;
    color: %(body)s;
    outline: none;
    selection-background-color: %(accent_soft)s;
    selection-color: %(ink)s;
}
QListView::item, QListWidget::item, QTreeView::item {
    padding: 5px 7px; border: none; color: %(body)s;
}
QListView::item:hover, QListWidget::item:hover, QTreeView::item:hover {
    background: %(sidebar_hover)s;
}
QListView::item:selected, QListWidget::item:selected, QTreeView::item:selected {
    background: %(accent_soft)s; color: %(ink)s;
}
QListView::item:disabled, QListWidget::item:disabled { color: %(faint)s; }
QAbstractScrollArea { background: %(surface)s; }
QAbstractScrollArea::corner { background: %(surface)s; border: none; }

/* ---------------------------------------------------------------- navigation */
QWidget#rail {
    background: %(sidebar)s;
    border-right: 1px solid %(line)s;
}
QWidget#railBrand {
    background: %(sidebar)s;
    border-bottom: 1px solid %(line)s;
}
QListWidget#sidebar {
    background: transparent;
    border: none;
    outline: none;
    padding: 6px 0 12px 0;
}
QListWidget#sidebar::item {
    padding: 8px 10px 8px 12px;
    border-left: 3px solid transparent;
    color: %(body)s;
}
QListWidget#sidebar::item:hover { background: %(sidebar_hover)s; }
QListWidget#sidebar::item:selected {
    background: %(accent_soft)s;
    border-left: 3px solid %(accent)s;
    color: %(ink)s;
    font-weight: 700;
}
QLabel[role="nav-title"] { color: %(ink)s; font-size: 15px; font-weight: 700; }
QLabel[role="nav-sub"]   { color: %(muted)s; font-size: 11px; }

/* --------------------------------------------------------------- scrollbars */
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical {
    background: #c3cfdb; border-radius: 5px; min-height: 28px; margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #a9b8c7; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:horizontal {
    background: #c3cfdb; border-radius: 5px; min-width: 28px; margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: #a9b8c7; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ------------------------------------------------------------------- chrome */
QStatusBar { background: %(surface)s; border-top: 1px solid %(line)s; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: %(muted)s; font-size: 12px; }
QProgressBar {
    background: %(canvas)s; border: 1px solid %(line)s; border-radius: 6px;
    height: 12px; text-align: center; color: %(muted)s; font-size: 10px;
}
QProgressBar::chunk { background: %(accent)s; border-radius: 5px; }
QSplitter::handle { background: %(line)s; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QMenuBar { background: %(surface)s; border-bottom: 1px solid %(line)s; }
QMenuBar::item { padding: 6px 11px; background: transparent; }
QMenuBar::item:selected { background: %(accent_soft)s; border-radius: 5px; }
QMenu { background: %(surface)s; border: 1px solid %(line_strong)s; padding: 5px; }
QMenu::item { padding: 6px 22px 6px 12px; border-radius: 5px; }
QMenu::item:selected { background: %(accent_soft)s; color: %(ink)s; }
QMenu::separator { height: 1px; background: %(line)s; margin: 4px 8px; }
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title {
    background: %(sidebar)s; padding: 5px 9px; border-bottom: 1px solid %(line)s;
    font-weight: 700; color: %(muted)s;
}
QToolTip {
    background: %(ink)s; color: #ffffff; border: none; padding: 5px 8px; border-radius: 5px;
}
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""


def stylesheet() -> str:
    """The stylesheet with the palette and the generated images substituted in."""
    values = dict(PALETTE)
    values["mono"] = ", ".join(f'"{f}"' for f in MONO_FAMILIES)
    assets = _assets()
    if assets:
        values.update(assets)
        values["input_rules"] = _INPUT_RULES % values
    else:
        values["input_rules"] = _INPUT_FALLBACK % values
    return STYLESHEET % values


#: Whether this process has already pinned the widget style.  A module flag rather than
#: a check on ``app.style()``: once a stylesheet is installed, ``QApplication.style()``
#: returns Qt's ``QStyleSheetStyle`` wrapper, so asking it whether it is Fusion always
#: answers no — and re-installing the style repolishes every widget for nothing.
_STYLE_PINNED = False


def apply() -> None:
    """Install the Fusion style, the palette, the font and the stylesheet.

    Order matters: ``setStyle`` can reset the palette, so the palette goes on *after* it
    and *before* the stylesheet, which is resolved against it.
    """
    global _STYLE_PINNED

    app = QApplication.instance()
    if app is None:
        return
    if not _STYLE_PINNED:
        app.setStyle("Fusion")
        _STYLE_PINNED = True
    font = QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPointSize(10)
    app.setFont(font)
    app.setPalette(palette())
    app.setStyleSheet(stylesheet())


def tag(widget: QWidget, role: str, **properties) -> QWidget:
    """Mark a widget with a role (and any extra QSS properties), then repolish it.

    Repolishing is the part that matters: Qt caches the resolved style of a widget, so
    a property selector only takes effect after ``unpolish``/``polish``.
    """
    widget.setProperty("role", role)
    for name, value in properties.items():
        widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
    return widget


def retag(widget: QWidget, **properties) -> QWidget:
    """Change QSS properties without touching the role (used for callout tones)."""
    for name, value in properties.items():
        widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
    return widget
