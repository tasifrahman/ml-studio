"""Project files: everything needed to rebuild a session, stored as readable JSON.

A ``.mlproj`` file holds the *recipe*, not the data: where the data came from,
which cleaning steps were applied, how the split was made, which model and
hyperparameters were chosen, and which figures were on screen.  Reopening one
and pressing Run reproduces the run from scratch, which is what makes an
interrupted session resumable.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field

SESSION_FORMAT = "ml-workbench-session"
SESSION_VERSION = 1


@dataclass
class Session:
    source_file: str = ""
    source_format: str | None = None
    source_table: str | None = None
    source_options: dict = field(default_factory=dict)
    cleaning_ops: list = field(default_factory=list)
    train_spec: dict = field(default_factory=dict)
    preprocess_spec: dict = field(default_factory=dict)
    model_key: str | None = None
    model_params: dict = field(default_factory=dict)
    tune_spec: dict = field(default_factory=dict)
    grid_text: str = ""
    select_spec: dict = field(default_factory=dict)
    design_spec: dict = field(default_factory=dict)
    plot_views: list = field(default_factory=list)     # [{"kind":..., "options":{...}}]
    bundle_path: str = ""
    report_path: str = ""
    notes: str = ""
    log: list = field(default_factory=list)
    created: str = ""
    saved: str = ""

    def to_dict(self) -> dict:
        return {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "source": {
                "file": self.source_file,
                "format": self.source_format,
                "table": self.source_table,
                "options": self.source_options,
            },
            "cleaning_ops": self.cleaning_ops,
            "train_spec": self.train_spec,
            "preprocess_spec": self.preprocess_spec,
            "model": {"key": self.model_key, "params": self.model_params},
            "tune": {"spec": self.tune_spec, "grid_text": self.grid_text},
            "select_spec": self.select_spec,
            "design_spec": self.design_spec,
            "plot_views": self.plot_views,
            "outputs": {"bundle": self.bundle_path, "report": self.report_path},
            "notes": self.notes,
            "log": self.log,
            "created": self.created or _dt.datetime.now().isoformat(timespec="seconds"),
            "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
            raise ValueError("this is not a ML Workbench project file")
        source = data.get("source") or {}
        return cls(
            source_file=source.get("file") or "",
            source_format=source.get("format"),
            source_table=source.get("table"),
            source_options=source.get("options") or {},
            cleaning_ops=data.get("cleaning_ops") or [],
            train_spec=data.get("train_spec") or {},
            preprocess_spec=data.get("preprocess_spec") or {},
            model_key=(data.get("model") or {}).get("key"),
            model_params=(data.get("model") or {}).get("params") or {},
            tune_spec=(data.get("tune") or {}).get("spec") or {},
            grid_text=(data.get("tune") or {}).get("grid_text") or "",
            select_spec=data.get("select_spec") or {},
            design_spec=data.get("design_spec") or {},
            plot_views=data.get("plot_views") or [],
            bundle_path=(data.get("outputs") or {}).get("bundle") or "",
            report_path=(data.get("outputs") or {}).get("report") or "",
            notes=data.get("notes") or "",
            log=data.get("log") or [],
            created=data.get("created") or "",
            saved=data.get("saved") or "",
        )


def save_session(session: Session, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not path.endswith(".mlproj"):
        path += ".mlproj"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(session.to_dict(), fh, indent=2, default=str)
    return path


def load_session(path: str) -> Session:
    with open(path, encoding="utf-8") as fh:
        return Session.from_dict(json.load(fh))


def recent_sessions(folder: str, limit: int = 12) -> list[dict]:
    """Newest project files first, for the welcome screen."""
    if not os.path.isdir(folder):
        return []
    entries = []
    for name in os.listdir(folder):
        if not name.endswith(".mlproj"):
            continue
        full = os.path.join(folder, name)
        try:
            stat = os.stat(full)
            data = load_session(full)
        except Exception:
            continue
        entries.append({
            "path": full,
            "name": name,
            "saved": data.saved,
            "target": data.train_spec.get("target") if data.train_spec else None,
            "model": data.model_key,
            "modified": stat.st_mtime,
        })
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return entries[:limit]


def summarise(session: Session) -> str:
    bits = []
    if session.source_file:
        bits.append(os.path.basename(session.source_file))
    target = (session.train_spec or {}).get("target")
    if target:
        bits.append(f"target {target}")
    if session.model_key:
        bits.append(session.model_key)
    if session.bundle_path:
        bits.append("has saved model")
    return " · ".join(bits) or "empty project"
