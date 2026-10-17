"""Generate a standalone script that reproduces exactly what was built in the GUI.

The exported file deliberately imports the workbench's own ``core`` package
rather than re-implementing preprocessing inline.  A re-implementation would
drift from the real one after the first bug fix, and a script that quietly
computes something slightly different from the GUI is worse than no script at
all.  The header says where to run it from.
"""
from __future__ import annotations

import datetime as _dt
import os

HEADER = '''#!/usr/bin/env python3
"""{title}

Reproduces the model built in ML Workbench on {created}.

The script imports the workbench's own `core` package so that preprocessing is
byte-for-byte the same as the GUI. It looks for that package in the directory it
was exported from, so it can be run from anywhere:

    {interpreter_hint} {script_name}

Options:
    --bundle PATH   load a saved .joblib bundle instead of refitting
    --output DIR    where to write outputs (default: ./exported_output)

The numbers printed here should match the report exactly; if they do not, the
data or the environment has changed since the model was fitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

WORKBENCH_ROOT = {workbench_root!r}
if WORKBENCH_ROOT and WORKBENCH_ROOT not in sys.path:
    sys.path.insert(0, WORKBENCH_ROOT)

# --- configuration captured from the GUI -------------------------------------
SOURCE_FILE = {source_file!r}
SOURCE_FORMAT = {source_format!r}
SOURCE_TABLE = {source_table!r}

CLEANING_OPS = {cleaning_ops!r}

TRAIN = {train_spec!r}

PREPROCESS = {preprocess!r}

MODEL_KEY = {model_key!r}
MODEL_PARAMS = {model_params!r}

NOTES = {notes!r}
'''

BODY = '''

def load_data(path, fmt, table):
    from core import loaders
    tables = loaders.read_all(path, fmt=fmt)
    if not tables:
        raise SystemExit(f"no tables found in {path}")
    if table and table in tables:
        return tables[table]
    return next(iter(tables.values()))


def show_predictions(pipe, split, out_dir):
    """Score a few rows of brand-new-shaped data — the point of saving a pipeline."""
    if split is None or not len(split.X_test):
        return
    import pandas as pd
    sample = split.X_test.head(5)
    predictions = pd.DataFrame({"prediction": pipe.predict(sample)})
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(sample)
            for i in range(proba.shape[1]):
                predictions[f"probability_{i}"] = proba[:, i]
        except Exception:
            pass
    combined = pd.concat([sample.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    path = os.path.join(out_dir, "example_predictions.csv")
    combined.to_csv(path, index=False)
    print("\\nwrote", path)
    print(combined.head().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="use a saved .joblib bundle instead of refitting")
    parser.add_argument("--output", default="exported_output", help="output directory")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from core import clean, explain, metrics, models, train
    from core.preprocess import PreprocessSpec

    spec = train.TrainSpec.from_dict(TRAIN)
    pre_spec = PreprocessSpec.from_dict(PREPROCESS)

    if args.bundle:
        bundle = train.load_bundle(args.bundle)
        pipe = bundle["pipeline"]
        print("loaded bundle:", os.path.basename(args.bundle))
        print("model:", bundle.get("model_label"), bundle.get("model_params"))
        print("saved test metrics:")
        print(json.dumps(bundle.get("metrics", {}).get("test", {}), indent=2))
        for message in bundle.get("_warnings") or []:
            print("warning:", message)
        return bundle, None

    df = load_data(SOURCE_FILE, SOURCE_FORMAT, SOURCE_TABLE)
    print(f"loaded {len(df):,} rows x {df.shape[1]} columns from {SOURCE_FILE}")

    if CLEANING_OPS:
        df, log = clean.apply_ops(df, CLEANING_OPS)
        for line in log:
            print("  cleaned:", line)
    else:
        print("  no cleaning steps were recorded")

    split = train.make_split(df, spec)
    print(f"train {split.n_train:,} rows / test {split.n_test:,} rows "
          f"over {len(split.feature_names)} features")

    pipe, _ = train.build_train_pipeline(pre_spec, MODEL_KEY, MODEL_PARAMS, spec.task)
    train.fit_full(pipe, split, spec)

    evaluation = train.evaluate_fitted(pipe, split, spec)
    print("\\ntest metrics")
    for key, value in evaluation["test"].items():
        if isinstance(value, (int, float)):
            print(f"  {metrics.label(spec.task, key):<28} {value: .6f}")

    importances = explain.importance_frame(pipe, split.X_train, split.y_train, spec.task)
    importance_path = os.path.join(args.output, "feature_importance.csv")
    importances.to_csv(importance_path, index=False)
    print("\\nwrote", importance_path)

    bundle = train.make_bundle(pipe, spec, split, pre_spec, MODEL_KEY, MODEL_PARAMS, evaluation,
                               source_file=SOURCE_FILE, cleaning_ops=CLEANING_OPS, notes=NOTES)
    path = train.save_bundle(bundle, os.path.join(args.output, "model.joblib"))
    print("wrote", path)

    show_predictions(pipe, split, args.output)
    return bundle, split


if __name__ == "__main__":
    main()
'''


def build_script(
    model_key: str,
    model_params: dict,
    train_spec: dict,
    preprocess_spec: dict,
    cleaning_ops: list,
    source_file: str,
    source_format: str | None = None,
    source_table: str | None = None,
    notes: str = "",
    title: str = "Reproduce this model",
    workbench_root: str | None = None,
) -> str:
    created = _dt.datetime.now().strftime("%d %B %Y at %H:%M")
    script_name = "reproduce_model.py"
    root = workbench_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    interpreter = os.path.join(root, ".venv", "bin", "python") if os.path.exists(
        os.path.join(root, ".venv", "bin", "python")) else "python3"
    head = HEADER.format(
        title=title, created=created, script_name=script_name,
        workbench_root=root, interpreter_hint=interpreter,
        source_file=source_file, source_format=source_format, source_table=source_table,
        cleaning_ops=cleaning_ops or [], train_spec=train_spec, preprocess=preprocess_spec,
        model_key=model_key, model_params=model_params, notes=notes,
    )
    return head + BODY


def write_script(path: str, **kwargs) -> str:
    script = build_script(**kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    os.chmod(path, 0o755)
    return path
