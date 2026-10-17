# ML Studio

An end-to-end machine-learning workbench for tabular data, built on PyQt6,
pandas, scikit-learn, and (optionally) CPU-only PyTorch. It walks you through
the whole workflow — **load → explore → clean → prepare → train → tune →
evaluate → save → predict → optimise → plot → report** — plus an engineering
toolkit (statistics, curve fitting, design of experiments, time series, units,
uncertainty/reliability by Monte-Carlo) and Excel/Word export.

> The project lives in a directory whose name contains a space:
> `"ML Workbench DL/"`. Quote it in every shell command.

## Run from source

Everything is run from this directory so that `import core` and `import gui` resolve:

```sh
cd "ML Workbench DL"
./.venv/bin/python app.py                 # the window
./.venv/bin/python app.py --self-test     # headless end-to-end check (73 checks)
```

Optional launch flags:

- `--data PATH` — load a file at startup (or in the self-test)
- `--target COLUMN` — pick the target column in the self-test
- `--out DIR` — output folder for self-test artefacts

A modal dialog would block forever off-screen; the harness closes any that
appear and reports them as a warning. A line reading
`(auto-dismissed a dialog: …)` in the output is worth reading.

The environment needs `QT_QPA_PLATFORM=offscreen` when there is no display.

## Install as a Debian package

A prebuilt package is provided:

- File: `ml-studio_1.0.0_amd64.deb`
- SHA-256: `1ba5b9af2d842f24546d125bef8890ee0a96d2df3f22b76f71ed730bfb795d79`

Install it on Debian/Ubuntu/Kali with:

```sh
sudo apt install ./ml-studio_1.0.0_amd64.deb
```

`apt` pulls in `python3.14` and the required system libraries. After
installation a `ML Studio` entry appears in the applications menu, and the
launcher is available at `/usr/bin/ml-studio`. The package is
self-contained: it ships the bundled Python virtual environment at
`/opt/mlstudio`, so no other Python packages are needed on the target machine.

## Environment

- Python 3.14, PyQt6 6.11, pandas 3.0, numpy 2.5, scikit-learn 1.9,
  matplotlib 3.11, xgboost, lightgbm, openpyxl …
- PyTorch 2.14.1 (CPU-only build) for the three neural networks
- Install from source in two steps, so the torch `--index-url` does not
  misdirect the base packages:

```sh
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-dl.txt
```

## Project layout

```text
core/          tabular + engineering + DL logic (Qt-free where possible)
core/dl/       TorchMLP (regression/classification) + autoencoder
gui/           PyQt6 interface, theme, widgets, main window
gui/steps/     the wizard screens (1–19)
docs/          rendered manual screenshots
examples/      three seeded example datasets + generator
USER_MANUAL.md beginner manual (opens inside the app via Help)
app.py         entry point / self-test harness
```

## Documentation

- `USER_MANUAL.md` — a complete beginner walkthrough of the app, including
  the engineering toolkit (sections 1–6 and three example projects).
- `DEVELOPER.md` — module map, invariants, findings log and run commands.
- `./.venv/bin/python docs/make_manual_shots.py` — regenerates the 19 screenshots
  the manual refers to.
- `./.venv/bin/python examples/make_examples.py` — regenerates the example CSVs.

## License

No licence file is included; this repository is a personal learning project.
Treat the code accordingly.
EOF
