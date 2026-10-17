"""The in-app user manual.

The manual is a plain ``USER_MANUAL.md`` at the project root, rendered with Qt's own
markdown engine.  Keeping it as a file rather than a pile of widgets means it can be
read on GitHub, printed, or edited by anyone — and the application only has to know
where it is.

One thing Qt does not do is turn a heading into a link target.  ``setMarkdown`` draws
the headings but creates **no anchors**, so the manual's table of contents would be
decorative and — worse — clicking an entry would hand Qt a URL it would try to *load*,
replacing the document.  :meth:`ManualDialog._jump` therefore finds the heading itself,
by name, and moves the cursor there.  The names are derived with the same rule GitHub
uses (:func:`slugify`), so the same links work in a browser and in here.
"""
from __future__ import annotations

import os
import re

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout,
)

from . import theme

MANUAL_NAME = "USER_MANUAL.md"


def manual_path(project_dir: str) -> str:
    """Absolute path of the manual that belongs to this copy of the application."""
    return os.path.join(project_dir, MANUAL_NAME)


def read_manual(project_dir: str) -> str:
    """The manual's text, or an explanation of where it should have been."""
    path = manual_path(project_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        return (f"# The user manual could not be read\n\n"
                f"Expected it at `{path}`.\n\n`{type(exc).__name__}: {exc}`\n\n"
                "It ships with the project; a copy of the application without it is "
                "incomplete.")


def slugify(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces dashed.

    ``"4.1 · Load the data"`` becomes ``"41--load-the-data"`` — the two dashes are real,
    because the middle dot is dropped and leaves *two* spaces behind, and GitHub turns
    each space into its own dash rather than collapsing the run.  The manual's table of
    contents is written against this rule, so it has to be reproduced exactly.
    """
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", text)


def headings(markdown: str) -> dict:
    """Every heading in the document, keyed by the anchor its table of contents uses."""
    found = {}
    for line in markdown.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                found.setdefault(slugify(heading), heading)
    return found


class ManualDialog(QDialog):
    """A read-only window onto the manual, with a way out to the real file."""

    def __init__(self, project_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ML Workbench — user manual")
        self.resize(1080, 820)
        self.project_dir = project_dir

        title = theme.tag(QLabel("User manual"), "page-title")
        blurb = theme.tag(QLabel(
            "A step-by-step walkthrough written for someone who has never trained a "
            "model before, using a small example dataset that ships with the app."),
            "page-subtitle")
        blurb.setWordWrap(True)

        text = read_manual(project_dir)
        self._headings = headings(text)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._anchor_clicked)
        # the manual refers to its pictures by a path relative to the project root, so
        # the browser has to be told where that root is
        self.view.setSearchPaths([project_dir])
        self.view.document().setBaseUrl(QUrl.fromLocalFile(os.path.join(project_dir, "")))
        self.view.document().setDefaultStyleSheet(
            "a { color: #2563a8; } "
            "code, pre { color: #1b4c84; } "
            "h1, h2, h3 { color: #16202b; }")
        self.view.setMarkdown(text)

        open_file = QPushButton("Open the file…")
        open_file.setToolTip("Open USER_MANUAL.md in whatever application handles markdown")
        open_file.clicked.connect(self._open_externally)
        close = QPushButton("Close")
        theme.tag(close, "primary")
        close.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addWidget(open_file)
        buttons.addStretch(1)
        buttons.addWidget(close)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(blurb)
        layout.addWidget(self.view, 1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ links
    def _jump(self, anchor: str) -> bool:
        """Scroll to the heading a table-of-contents link names.  False if it is unknown.

        Two traps here.  First, ``ensureCursorVisible`` only guarantees the heading is
        *somewhere* on screen, so a jump far down the document parks it on the last line —
        which reads as if the link went to the wrong place; the scrollbar is nudged
        afterwards to bring the heading to the top instead.  Second, a contents entry
        contains exactly the same text as the heading it points at, so matching on text
        alone finds the *list item* and every link lands in the contents.  Only blocks that
        Qt marked as headings (``headingLevel() > 0``) are considered.
        """
        wanted = self._headings.get(anchor)
        if wanted is None:
            return False
        block = self.view.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() > 0 and block.text().strip() == wanted:
                cursor = QTextCursor(block)
                self.view.setTextCursor(cursor)
                self.view.ensureCursorVisible()
                bar = self.view.verticalScrollBar()
                bar.setValue(max(0, bar.value() + self.view.cursorRect(cursor).top() - 6))
                return True
            block = block.next()
        return False

    def _anchor_clicked(self, url: QUrl):
        if url.isLocalFile():
            QDesktopServices.openUrl(url)
            return
        # a fragment-only link is the manual's own table of contents
        if url.fragment() and not url.path():
            self._jump(url.fragment())
            return
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
            return
        # a relative link to another file in the tree, e.g. DEVELOPER.md
        if url.path():
            target = os.path.normpath(os.path.join(self.project_dir, url.path()))
            if os.path.exists(target):
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def _open_externally(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(manual_path(self.project_dir)))
