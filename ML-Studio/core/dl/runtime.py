"""CPU-only runtime policy shared by every deep-learning model here.

This machine is an AMD Ryzen 5 7520U: 4 physical cores / 8 threads, AVX2 (no
AVX512), no GPU, and only a few GB of RAM free.  Three consequences are enforced
here rather than left to each model to remember:

* **Threads are pinned explicitly.**  PyTorch defaults to one thread per *logical*
  core.  For the small networks this workbench trains, the synchronisation cost of
  eight threads usually outweighs the parallelism, so every fit sets the count.
* **Every fit has a wall-clock budget.**  A network with a badly-scaled learning
  rate will happily spin forever; ``max_seconds`` turns that into a finished,
  slightly-worse model instead of a frozen window.
* **Gradients are never built during prediction.**  ``predict`` runs under
  ``no_grad``, which is both faster and much lighter on memory.

Progress and cancellation travel from the GUI into the training loop through a
thread-local observer.  scikit-learn's ``fit(X, y)`` has nowhere to put a callback,
and the training loop is several frames below the code that owns the progress bar.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

OBSERVER = threading.local()

_TORCH_IMPORTED = False
_TORCH_PROBLEM = ""


def cpu_count() -> int:
    return os.cpu_count() or 1


def default_threads() -> int:
    """Physical-ish core count.  4 logical cores here, so 4 threads."""
    return max(1, min(4, cpu_count()))


def torch_available() -> bool:
    """True when torch can actually be imported (it is an optional dependency).

    The answer is cached both ways: ``is_available`` asks this for every model while
    the menus are built, and a *broken* install would otherwise re-attempt the import
    — and re-print its error — on every refresh.
    """
    global _TORCH_IMPORTED, _TORCH_PROBLEM
    if _TORCH_IMPORTED:
        return True
    if _TORCH_PROBLEM:
        return False
    try:
        import torch  # noqa: F401

        _TORCH_IMPORTED = True
        return True
    except Exception as exc:  # noqa: BLE001 - a broken torch install must not crash the app
        _TORCH_PROBLEM = f"{type(exc).__name__}: {exc}"
        return False


def torch_problem() -> str:
    return _TORCH_PROBLEM


def configure_threads(threads: int | None = None) -> int:
    """Pin torch to a sensible number of CPU threads.  Returns what it used."""
    import torch

    requested = int(threads or 0)
    count = max(1, min(requested or default_threads(), cpu_count()))
    try:
        torch.set_num_threads(count)
    except Exception:  # noqa: BLE001
        pass
    try:
        # Inter-op parallelism only helps when ops can run concurrently, which for
        # these tiny graphs they cannot.  Can only be set once per process.
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001
        pass
    return count


def describe_runtime() -> str:
    """One line for the About box / log: what the DL models will actually use."""
    if not torch_available():
        return "PyTorch is not installed — the neural-network models are unavailable"
    import torch

    return (f"PyTorch {torch.__version__} on CPU · {default_threads()} of {cpu_count()} "
            f"threads · no GPU")


# --------------------------------------------------------------------------
# progress + cancellation
# --------------------------------------------------------------------------


@contextmanager
def observing(progress=None, cancel=None):
    """Install a progress/cancel observer for the current thread only.

    ``TaskRunner`` runs each job in its own worker thread, so a thread-local is
    exactly the right scope: two jobs running at once cannot see each other's
    callbacks.
    """
    previous = (getattr(OBSERVER, "progress", None), getattr(OBSERVER, "cancel", None))
    OBSERVER.progress = progress
    OBSERVER.cancel = cancel
    try:
        yield
    finally:
        OBSERVER.progress, OBSERVER.cancel = previous


def report(done: int, total: int, message: str) -> None:
    callback = getattr(OBSERVER, "progress", None)
    if callback is None:
        return
    try:
        callback(int(done), int(total), str(message))
    except Exception:  # noqa: BLE001 - a broken callback must not kill a fit
        pass


def check_cancel() -> None:
    cancel = getattr(OBSERVER, "cancel", None)
    if cancel is not None and cancel.is_set():
        from ..train import Cancelled

        raise Cancelled("cancelled by user")


def budget_exceeded(started: float, max_seconds) -> bool:
    limit = float(max_seconds or 0)
    return bool(limit) and (time.time() - started) >= limit
