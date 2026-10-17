"""Self-contained HTML report: every number and every figure in one file.

The report is built from a saved bundle plus a list of plot requests, which
means it reproduces exactly what the GUI showed.  Figures are embedded as base64
PNG data URIs, so the file can be emailed or attached to a thesis with no
supporting folder.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import os
from base64 import b64encode as _b64encode

import numpy as np
import pandas as pd

from . import metrics as M
from . import plots as P

CSS = """
:root { --ink:#1b2430; --muted:#5b6b7c; --line:#d8e0e8; --accent:#2f5d8a; --accent2:#2e7d5b; --warn:#a4560f; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f8fa; color:var(--ink);
       font:14px/1.55 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:28px 22px 80px; }
header { border-bottom:3px solid var(--accent); padding-bottom:14px; margin-bottom:22px; }
h1 { margin:0 0 4px; font-size:26px; letter-spacing:-0.2px; }
h2 { margin:34px 0 10px; font-size:19px; border-left:4px solid var(--accent); padding-left:10px; }
h3 { margin:22px 0 8px; font-size:15px; color:var(--muted); text-transform:uppercase; letter-spacing:0.6px; }
.sub { color:var(--muted); font-size:13px; }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:12px 0;
        box-shadow:0 1px 2px rgba(16,32,48,0.04); }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px,1fr)); gap:12px; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.kpi .label { font-size:11.5px; text-transform:uppercase; letter-spacing:0.5px; color:var(--muted); }
.kpi .value { font-size:21px; font-weight:600; margin-top:3px; }
table { border-collapse:collapse; width:100%; margin:8px 0 4px; font-size:13px; }
th, td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
th { background:#eef2f6; font-weight:600; }
tr:nth-child(even) td { background:#fafcfd; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12.5px; }
pre { background:#f2f5f8; border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; }
.note { background:#fff8ec; border:1px solid #f0d9b5; color:var(--warn); border-radius:8px; padding:10px 12px; margin:10px 0; }
.ok { background:#eef8f2; border:1px solid #cbe7d8; color:#245c42; border-radius:8px; padding:10px 12px; margin:10px 0; }
figure { margin:0 0 20px; }
figure img { width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }
figcaption { color:var(--muted); font-size:12.5px; margin-top:6px; }
.muted { color:var(--muted); }
@media print { body { background:#fff; } .card, .kpi { box-shadow:none; } h2 { page-break-after:avoid; }
                figure { page-break-inside:avoid; } }
"""


def _table(frame: pd.DataFrame, max_rows: int = 60, float_format: str = "{:,.4f}") -> str:
    if frame is None or not len(frame):
        return '<p class="muted">nothing to show</p>'
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda v: "" if pd.isna(v) else float_format.format(v) if abs(v) >= 1e-4 else f"{v:.3g}")
    header = "".join(f"<th>{_html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{_html.escape('' if pd.isna(v) else str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    more = ""
    if len(frame) > max_rows:
        more = f'<p class="muted">showing {max_rows} of {len(frame)} rows</p>'
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>{more}"


def _kpi(label, value, hint="") -> str:
    return (f'<div class="kpi"><div class="label">{_html.escape(str(label))}</div>'
            f'<div class="value">{_html.escape(str(value))}</div>'
            f'<div class="sub">{_html.escape(str(hint))}</div></div>')


def _fmt(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _metrics_table(bundle: dict, which: str) -> pd.DataFrame:
    values = (bundle.get("metrics") or {}).get(which) or {}
    task = bundle.get("task") or "regression"
    rows = []
    for key, value in values.items():
        if not isinstance(value, (int, float)):
            continue
        rows.append({
            "metric": M.label(task, key),
            "value": value,
            "direction": "higher is better" if M.higher_is_better(task, key) else "lower is better",
        })
    return pd.DataFrame(rows)


def build_report(
    bundle: dict,
    context: P.PlotContext | None = None,
    plot_requests: list[tuple[str, dict]] | None = None,
    extra_sections: dict | None = None,
    extra_figures: list[tuple[str, bytes]] | None = None,
    title: str = "Machine learning report",
    dpi: int = 120,
    progress=None,
) -> tuple[str, list[str]]:
    """Assemble the HTML.  Returns ``(html, warnings)``.

    ``extra_sections`` adds tables under the Analysis heading; ``extra_figures`` adds
    already-rendered pictures (``(caption, png_bytes)``) beside them.
    """
    warnings: list[str] = list(bundle.get("_warnings") or [])
    task = bundle.get("task") or "regression"
    versions = bundle.get("versions") or {}
    primary = bundle.get("primary_metric")
    test_metrics = (bundle.get("metrics") or {}).get("test") or {}
    cv = bundle.get("cv") or {}

    parts: list[str] = []
    parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">""")

    parts.append(f"""<header><h1>{_html.escape(title)}</h1>
<div class="sub">Generated by ML Workbench on {_dt.datetime.now().strftime('%d %B %Y at %H:%M')} ·
task: <strong>{_html.escape(str(task))}</strong> ·
target: <strong>{_html.escape(str(bundle.get('target') or '—'))}</strong></div></header>""")

    # ---- headline numbers
    headline = [
        _kpi(M.label(task, primary), _fmt(test_metrics.get(primary)), "held-out test set"),
        _kpi("Model", bundle.get("model_label") or "—", bundle.get("model_key") or ""),
        _kpi("Rows used", f"{bundle.get('n_train', 0):,} train / {bundle.get('n_test', 0):,} test",
             f"{bundle.get('n_features_in')} input features"),
        _kpi("Cross-validation", _fmt((cv.get("primary") or {}).get("mean")),
             f"± {_fmt((cv.get('primary') or {}).get('std'))} over {(cv.get('primary') or {}).get('n_folds', '—')} folds"),
    ]
    if bundle.get("feature_importances"):
        top = list(bundle["feature_importances"].items())[:1]
        if top:
            headline.append(_kpi("Strongest feature", str(top[0][0]), f"score {_fmt(top[0][1])}"))
    parts.append('<div class="grid">' + "".join(headline) + "</div>")

    for message in warnings:
        parts.append(f'<div class="note"><strong>Warning:</strong> {_html.escape(message)}</div>')

    gap = bundle.get("overfit_gap")
    if gap is not None:
        direction = "worse" if float(gap) > 0 else "better"
        parts.append(
            f'<div class="card"><strong>Training versus test.</strong> The primary metric is '
            f'{_fmt(abs(float(gap)))} {direction} on the test set than on the training set. '
            + ("A large gap means the model has memorised the training rows."
               if abs(float(gap)) > 0.1 else "That gap is small, which is what you want.")
            + "</div>"
        )

    # ---- data and preparation
    parts.append("<h2>1. Data and preparation</h2>")
    data_rows = [
        ("source file", bundle.get("source_file") or "—"),
        ("rows in the modelling frame", f"{bundle.get('n_train', 0) + bundle.get('n_test', 0):,}"),
        ("input features", ", ".join(bundle.get("features") or []) or "—"),
        ("dropped rows", _json_ish(bundle.get("dropped"))),
    ]
    parts.append('<div class="card">' + _table(pd.DataFrame(data_rows, columns=["field", "value"]), 40) + "</div>")

    ops = bundle.get("cleaning_ops") or []
    if ops:
        parts.append("<h3>Cleaning steps applied</h3><pre>" + _html.escape("\n".join(map(str, ops))) + "</pre>")
    recipe = bundle.get("preprocess") or {}
    if recipe:
        parts.append("<h3>Preprocessing recipe (fitted on the training split only)</h3><pre>"
                     + _html.escape(_json_ish(recipe)) + "</pre>")

    # ---- model
    parts.append("<h2>2. Model</h2>")
    model_rows = [
        ("algorithm", f"{bundle.get('model_label')} ({bundle.get('model_key')})"),
        ("hyperparameters", ", ".join(f"{k} = {v}" for k, v in (bundle.get("model_params") or {}).items()) or "library defaults"),
        ("encoded features", str(bundle.get("n_features_out") or "—")),
        ("classes", ", ".join(map(str, bundle.get("labels") or [])) or "—"),
    ]
    parts.append('<div class="card">' + _table(pd.DataFrame(model_rows, columns=["field", "value"]), 40) + "</div>")

    tuning = bundle.get("tuning")
    if tuning and tuning.get("results") is not None and len(tuning["results"]):
        parts.append("<h3>Tuning leaderboard</h3>")
        parts.append(f'<p class="sub">method: {_html.escape(str(tuning.get("method")))} · '
                     f'{tuning.get("n_scored")} of {tuning.get("n_candidates")} candidates scored · '
                     f'{_fmt(tuning.get("elapsed"), 1)} s · best '
                     f'{_html.escape(str(tuning.get("metric_label")))} = {_fmt(tuning.get("best_score"))}</p>')
        table = tuning["results"]
        keep = [c for c in table.columns if not c.startswith("std ")][:12]
        parts.append(_table(table[keep], 15))

    # ---- results
    parts.append("<h2>3. Results</h2>")
    parts.append("<h3>Held-out test set</h3>" + _table(_metrics_table(bundle, "test"), 30))
    parts.append("<h3>Training set (same rows the model learned from)</h3>"
                 + _table(_metrics_table(bundle, "train"), 30))

    if cv.get("folds"):
        fold_rows = []
        for i, fold in enumerate(cv["folds"], 1):
            row = {"fold": i}
            for key, value in fold.items():
                if isinstance(value, (int, float)):
                    row[M.label(task, key)] = value
            fold_rows.append(row)
        parts.append("<h3>Cross-validation folds</h3>" + _table(pd.DataFrame(fold_rows), 20))

    evaluation = (context.evaluation if context else None) or {}
    if task in ("binary", "multiclass") and evaluation.get("test_truth") is not None:
        truth, pred = evaluation["test_truth"], evaluation["test_pred"]
        parts.append("<h3>Confusion matrix (test set)</h3>" + _table(M.confusion(truth, pred, bundle.get("labels"))))
        parts.append("<h3>Per-class detail</h3>" + _table(M.classification_report(truth, pred, bundle.get("labels")), 30))
    elif task == "regression" and evaluation.get("test_truth") is not None:
        frame = M.regression_residuals(evaluation["test_truth"], evaluation["test_pred"])
        parts.append("<h3>Residual summary (test set)</h3>" + _table(frame.describe().T.reset_index(names="statistic"), 12))

    importances = bundle.get("feature_importances") or {}
    if importances:
        frame = pd.DataFrame({"feature": list(importances), "importance": list(importances.values())})
        parts.append("<h3>Feature importances</h3>" + _table(frame, 25))

    # ---- figures
    if plot_requests:
        parts.append("<h2>4. Figures</h2>")
        for i, (kind, opts) in enumerate(plot_requests, 1):
            spec = P.PLOT_INDEX.get(kind)
            label = spec.label if spec else kind
            if progress:
                progress(i - 1, len(plot_requests), f"figure {i}/{len(plot_requests)}: {label}")
            try:
                import matplotlib.pyplot as plt

                with P.RENDER_LOCK:
                    figure = P.render(kind, context, opts or {})
                    uri = P.figure_data_uri(figure, dpi=dpi)
                    plt.close(figure)
                caption = _html.escape(label)
                detail = ", ".join(f"{k}={v}" for k, v in (opts or {}).items() if v not in (None, ""))
                parts.append(
                    f'<figure><img alt="{caption}" src="{uri}">'
                    f"<figcaption><strong>{caption}</strong>"
                    + (f" · {_html.escape(detail)}" if detail else "") + "</figcaption></figure>"
                )
            except Exception as exc:
                parts.append(f'<div class="note">Figure "{_html.escape(label)}" could not be drawn: '
                             f"{_html.escape(str(exc))}</div>")
        if progress:
            progress(len(plot_requests), len(plot_requests), "figures done")

    if extra_sections or extra_figures:
        parts.append("<h2>5. Analysis</h2>")
        for heading, payload in extra_sections.items() if extra_sections else []:
            parts.append(f"<h3>{_html.escape(heading)}</h3>")
            if isinstance(payload, pd.DataFrame):
                parts.append(_table(payload, 40))
            elif isinstance(payload, dict):
                parts.append("<pre>" + _html.escape(_json_ish(payload)) + "</pre>")
            else:
                parts.append(f"<p>{_html.escape(str(payload))}</p>")
        # Figures produced by the statistics, curve-fitting, time-series, design and
        # reliability screens.  They are handed over already rendered as PNG bytes,
        # because those screens draw from their own results rather than from the plot
        # registry's context, and re-running them here would mean recomputing the study.
        for caption, payload in (extra_figures or []):
            try:
                data = payload if isinstance(payload, (bytes, bytearray)) else None
                if data is None and isinstance(payload, dict):
                    data = payload.get("png")
                if not data:
                    continue
                uri = "data:image/png;base64," + _b64encode(bytes(data)).decode("ascii")
                parts.append(
                    f'<figure><img alt="{_html.escape(str(caption))}" src="{uri}">'
                    f"<figcaption><strong>{_html.escape(str(caption))}</strong></figcaption></figure>")
            except Exception as exc:  # noqa: BLE001
                parts.append(f'<div class="note">Figure "{_html.escape(str(caption))}" could not be '
                             f"added: {_html.escape(str(exc))}</div>")

    # ---- appendix
    parts.append("<h2>Appendix</h2>")
    parts.append("<h3>Reproducibility</h3><pre>" + _html.escape(_json_ish({
        "primary_metric": primary,
        "train_spec": bundle.get("train_spec"),
        "preprocess": bundle.get("preprocess"),
        "model_key": bundle.get("model_key"),
        "model_params": bundle.get("model_params"),
        "random_state": (bundle.get("train_spec") or {}).get("random_state"),
        "versions": versions,
    })) + "</pre>")
    if bundle.get("notes"):
        parts.append(f"<h3>Notes</h3><p>{_html.escape(str(bundle['notes']))}</p>")

    parts.append('<p class="sub">Every number above came from the same pipeline object that would be '
                 "used to predict on new data. Figures were produced by the same code that drew them on screen.</p>")
    parts.append("</div></body></html>")
    return "".join(parts), warnings


def _json_ish(value) -> str:
    import json

    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    try:
        return json.dumps(value, indent=2, default=default)
    except Exception:
        return str(value)


def write_report(path: str, **kwargs) -> str:
    html, warnings = build_report(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
