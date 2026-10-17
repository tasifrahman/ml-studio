#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=python3.14
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Need Python 3.14 on PATH." >&2
    exit 1
fi

$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dl.txt

echo "Done. Run with:  ./.venv/bin/python app.py"
