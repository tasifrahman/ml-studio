"""Running long jobs without freezing the window.

Every heavy operation (fitting, cross-validating, sweeping a grid, searching a
design space, rendering the report) runs on a :class:`Task` thread.  The task
receives two callables:

* ``progress(current, total, message)`` — safe to call from the worker thread;
  it is emitted as a signal and delivered on the GUI thread.
* ``cancel`` — a ``threading.Event``.  Core code calls
  ``core.train.check_cancel(cancel)`` at safe points and raises
  ``core.train.Cancelled``, which this class reports as a cancellation rather
  than an error.

matplotlib is deliberately *not* touched from a worker except through
:meth:`Task`'s own render path, and every figure the worker builds is passed back
as a plain object; the widget that owns a canvas draws it on the GUI thread.
"""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class Task(QThread):
    """A function run on its own thread, reporting progress and result."""

    progressed = pyqtSignal(int, int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)        # message, traceback text
    cancelled = pyqtSignal()
    finished_always = pyqtSignal()

    def __init__(self, fn, *args, label: str = "", parent: QObject | None = None, **kwargs):
        super().__init__(parent)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.label = label or getattr(fn, "__name__", "task")
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ api
    def cancel(self):
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def _progress(self, current=0, total=0, message=""):
        # Emitting from the worker thread is fine; Qt queues it to the GUI thread.
        self.progressed.emit(int(current), int(total), str(message))

    def _status_only(self, current=0, total=0, message=""):
        """A message-only update: leaves the progress bar where the job put it.

        The deep-learning training loop reports once per epoch, which is far more
        granular than the job's own stages.  Sending a real ``total`` would keep
        resetting the bar (and, during a tuning sweep, would hide which candidate
        is running), so epoch lines go to the status message only.
        """
        if message:
            self.progressed.emit(-1, -1, str(message))

    # ---------------------------------------------------------------- thread
    def run(self):  # noqa: D102 - QThread entry point
        from core.dl import runtime
        from core.train import Cancelled

        try:
            # Installing the observer here rather than in each job means every
            # path that ends up inside a neural network — plain fitting, k-fold
            # cross-validation, a tuning sweep, the one-click run — gets live
            # epoch progress and a cancel button that actually interrupts training.
            with runtime.observing(progress=self._status_only, cancel=self._cancel):
                result = self.fn(self._progress, self._cancel, *self.args, **self.kwargs)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:                      # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self.succeeded.emit(result)
        finally:
            self.finished_always.emit()


class TaskRunner(QObject):
    """Owns the currently running task and keeps the window from stacking them."""

    busy_changed = pyqtSignal(bool, str)
    progressed = pyqtSignal(int, int, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.task: Task | None = None
        self._keep_alive: list[Task] = []

    @property
    def busy(self) -> bool:
        return self.task is not None and self.task.isRunning()

    def cancel(self):
        if self.busy and self.task is not None:
            self.task.cancel()
            self.progressed.emit(0, 0, "cancelling…")

    def run(self, fn, *args, label: str = "", on_done=None, on_error=None, on_cancelled=None,
            on_progress=None, **kwargs) -> Task | None:
        """Start ``fn`` in the background.  Refuses to start a second job."""
        if self.busy:
            self.progressed.emit(0, 0, "another task is still running")
            return None

        task = Task(fn, *args, label=label, parent=self, **kwargs)
        self.task = task
        self._keep_alive.append(task)

        def handle_progress(current, total, message):
            self.progressed.emit(current, total, message)
            if on_progress is not None:
                on_progress(current, total, message)

        def handle_done(result):
            if on_done is not None:
                on_done(result)

        def handle_error(message, detail):
            self.progressed.emit(0, 0, message)
            if on_error is not None:
                on_error(message, detail)

        def handle_cancelled():
            self.progressed.emit(0, 0, "cancelled")
            if on_cancelled is not None:
                on_cancelled()

        def cleanup():
            self.busy_changed.emit(False, task.label)
            if task in self._keep_alive:
                self._keep_alive.remove(task)

        task.progressed.connect(handle_progress)
        task.succeeded.connect(handle_done)
        task.failed.connect(handle_error)
        task.cancelled.connect(handle_cancelled)
        task.finished_always.connect(cleanup)

        self.busy_changed.emit(True, task.label)
        task.start()
        return task
