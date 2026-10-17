# ML Workbench DL — developer notes

`ML Workbench` with **deep learning added**: the same PyQt6 workbench for the whole
tabular workflow (load → explore → clean → prepare → train → tune → evaluate → save →
predict → optimise → plot → report), plus three CPU-only PyTorch models that plug into
every part of it, plus an **engineering toolkit** (screens 13–18: tests and statistics,
curve fitting, design of experiments, time series and control charts, units, uncertainty
and reliability) and Excel/Word export.

**This is a copy.** `../ML Workbench` is the original, it is the reference
implementation, and it must stay untouched. If you need to know how the *shared*
machinery works — the preprocessor, the hand-written CV loop, the plot registry, the
report, the bundle format, the screens — read `../ML Workbench/DEVELOPER.md`
first. This file documents only what is different here, plus the state a fresh
session cannot reconstruct.

**Read this file first when resuming.** Update it before you stop.

---

## 1. Run it

Everything is run from this directory (`.venv/bin/python` + cwd here, so `import core`
and `import gui` resolve). The folder name contains a space — always quote it.

```sh
cd "/home/tasifrari/Application/opencode/ML Workbench DL"

./.venv/bin/python app.py                     # the real window
./.venv/bin/python app.py --data file.csv --target column
./.venv/bin/python app.py --self-test         # headless end-to-end check, 73 checks
./.venv/bin/python app.py --self-test --data /tmp/opencode/churn.csv --target churned
./.venv/bin/python app.py --self-test --data ... --target ... --out /tmp/opencode/out

./.venv/bin/python docs/make_manual_shots.py  # regenerate the manual's 19 pictures
./.venv/bin/python examples/make_examples.py  # regenerate all three example datasets
```

`app.py --self-test` is **the regression test for this project** — there is no pytest
suite. It is the v1 harness (29 checks), 6 deep-learning checks, 6 checks for the
interface and the manual, and 31 more covering the engineering toolkit, the two new
example files and the cleaning-recipe rules (they drive the real screens on the real
example files and assert the numbers the manual quotes); see §7. Both scripts above render
offscreen and need no display.

Headless notes are identical to v1: `QT_QPA_PLATFORM=offscreen` is required with no
display, `This plugin does not support propagateSizeHints()` and
`findfont: Failed to find font weight semibold for DejaVu Sans` are benign, and a line
reading `(auto-dismissed a dialog: …)` is worth reading — most of them are the
"Added … to the dataset" confirmations the engineering screens raise after they have
already done the thing, but anything else is a bug even when the check still passes.
**A modal dialog under `offscreen` blocks the event loop forever**; the harness closes any
that appears with a `QTimer` (60 ms) rather than hanging, and so does
`docs/make_manual_shots.py`.

## 2. Environment

In-tree venv at `.venv/` (**2.2 GB** — 0.9 GB of it is PyTorch), Python **3.14.7**.
Everything from v1 is the same version (pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1,
PyQt6 6.11.0, matplotlib 3.11.2, seaborn 0.13.2, xgboost 3.4.1, lightgbm 4.7.0), plus:

| package | version | note |
|---|---|---|
| torch | 2.14.1+cpu | CPU-only build, from the PyTorch CPU index |

Install it in **two steps** — the second file carries `--index-url`, which would
misdirect the base packages if the two were combined:

```sh
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dl.txt
```

### Why PyTorch and not TensorFlow

Measured, not assumed: **TensorFlow has no cp314 wheels** (it stops at cp313) and this
machine has only `python3.14` — 3.11/3.12/3.13 are all absent. PyTorch 2.14.1 ships a
cp314 wheel. That decided it.

### Why the CPU index

The default PyPI Linux `torch` wheel is 555 MB and drags in `nvidia-cudnn`,
`nvidia-nccl` and `triton` — several GB of GPU libraries that are dead weight on a
Ryzen 5 7520U with no GPU. `requirements-dl.txt` pins `torch==2.14.1+cpu` from
`https://download.pytorch.org/whl/cpu`.

## 3. Layout

Identical to v1 except for the deep-learning package, the interface layer, the manual, the
engineering toolkit and the example projects. Nothing else moved.

```
core/dl/runtime.py          thread pinning, the time budget, the thread-local observer
core/dl/nets.py             TabularMLP, AutoencoderNet, embedding_dim_for, parse_hidden
core/dl/estimators.py       TorchMLPRegressor, TorchMLPClassifier, TorchAutoencoder
core/dl/__init__.py         lazy re-exports — importing this must NOT import torch

core/units.py               quantities, factors, affine scales, constants
core/stats.py               describe, compare, chi-square, power, sample size, post-hoc
core/curvefit.py           13 engineering forms, ranking, solve, interpolate, residuals
core/timeseries.py          trend/Kendall, ACF/PACF, resample, decompose, control charts,
                            capability, gaps
core/doe.py                 factorial/fractional/RSM/space-filling designs, effects
core/montecarlo.py          distributions, sampling, copulas, reliability, sensitivity
core/export_office.py       hand-written .xlsx (openpyxl) and .docx (raw OOXML) writers
core/plots_engineering.py   every figure the six engineering screens draw

gui/theme.py                the whole visual language: palette, stylesheet, tag()
gui/manual.py               the in-app viewer for USER_MANUAL.md
gui/steps/start_steps.py    StartPage — the screen the window opens on
gui/steps/engineering_steps.py   screens 13–17 (statistics, curves, DOE, series, units)
gui/steps/reliability_steps.py   screen 18 (Monte-Carlo, reliability, sensitivity)

USER_MANUAL.md              the beginner's manual, rendered inside the app
examples/make_examples.py   seeded generator for all three example files
examples/customer_churn.csv the example the manual walks through (4,000 rows)
examples/concrete_mix.csv   1,200 trial mixes, with a C30/37 pass/fail column
examples/pile_settlement_monitoring.csv   157 weekly readings of one pile
docs/make_manual_shots.py   drives the real window and photographs each screen
docs/manual/*.png           those photographs, referred to by USER_MANUAL.md

requirements-dl.txt         the CPU-only torch install
```

`core/dl/` sits beside the v1 core modules and follows the same rule: **no Qt in it** —
and the six engineering core modules obey it too, which is why every one of them is
independently runnable and why the self-test can assert their numbers without a window.

Files changed from v1 — **the code files that differ** (this document also differs, as it
must). Verified with `cmp`-style comparison; the list has grown in three passes:

| file | change |
|---|---|
| `core/models.py` | 3 `ModelSpec`s, 3 lazily-importing constructors, DL param helpers |
| `core/train.py` | `torch` in `VERSION_KEYS`; `library_versions()` records torch + threads |
| `core/plots.py` | `training_curve` and `reconstruction_error` plots, `_model_step` helper, `_has_projection` for PCA/t-SNE |
| `core/clean.py` | the `derive_column` operation — a new column from a formula, through the cleaning pipeline |
| `core/formula.py` | **quoted column names** — `"Blast furnace slag" / Cement` now parses |
| `core/report.py` | `build_report(..., extra_sections=..., extra_figures=...)` for the engineering screens |
| `gui/workers.py` | `Task.run` installs the DL observer; message-only status channel |
| `gui/main_window.py` | theme, branded sidebar, Help menu, prerequisite signposts, View → Full screen (F11), a screen-clamped opening size |
| `gui/widgets.py` | `Callout`, themed helpers, `StepPage` bodies in a scroll area, a width floor on scrollable panels |
| `gui/state.py` | engineering fields, `remember_figure`/`remember_section`, `set_raw` clears the cleaning recipe |
| `gui/steps/__init__.py` | `StartPage`; the **Engineering** group and the 13→19 renumber |
| `gui/steps/data_steps.py` | inline styles → theme roles; `requires`; the shipped-example picker |
| `gui/steps/model_steps.py` | same, plus `training_curve` in the default report set, Excel/Word export |
| `gui/steps/analysis_steps.py` | same, plus silhouette measured in the latent space |
| `app.py` | 6 deep-learning checks + 7 interface/manual checks + 31 engineering checks |

(`workbench_output/*` also differs, but those are generated artefacts from each project's
own runs, not source: never copy one over the other.)

New in this copy, i.e. not in v1 at all: the 12 files in `core/` (six engineering modules,
`export_office.py`, `plots_engineering.py`, `core/dl/*`), `gui/theme.py`, `gui/manual.py`,
`gui/steps/{start,engineering,reliability}_steps.py`, `docs/`, `examples/`,
`USER_MANUAL.md` and `requirements-dl.txt`.

## 3b. The interface layer

Four files, and the point of all of them is that a screen never names a colour.

- **`gui/theme.py`** holds one palette, one stylesheet, and `tag(widget, role)`.
  `theme.apply()` is called once, at the top of `MainWindow.__init__`, and pins the app to
  Qt's **Fusion** style — the native Linux style is drawn by the desktop theme and ignores
  large parts of a stylesheet, so the look would otherwise differ per machine. It also
  installs an **explicit palette**; see trap 3 below for why that is not optional.
- **`gui/widgets.py`** gained `Callout`, and the shared helpers (`hint`, `section_title`,
  `separator`) now return themed widgets instead of styled ones.
- **`gui/main_window.py`** builds the navigation rail, the Help menu, and drives
  `page.update_requirements()` on every refresh *and* on every screen change.
- **`gui/steps/start_steps.py`** is the new first screen: a live checklist read from
  `AppState`, two one-click ways to get data in, and a map of the other screens.
  It is registered as the `"Start"` group in `gui/steps/__init__.py`.

A page declares what it needs with a class attribute, and the window does the rest:

```python
class EvaluatePage(StepPage):
    title = "7 · Evaluate"
    requires = "model"          # None | "data" | "model"
```

`StepPage.update_requirements()` shows a warn `Callout` at the top of the screen with a
button that jumps to the screen which supplies the missing thing. It is called from
`MainWindow._refresh_pages` and `_sidebar_changed`, **not** from subclass `refresh()`
overrides — putting it in the base class is what stops a new screen from forgetting it,
and the self-test check "ui: every screen explains what it needs" fails if a screen
declares a prerequisite and then does not show its signpost.

### Four Qt traps that cost real time here

1. **A property selector does nothing until the widget is repolished.** Qt caches a
   widget's resolved style, so setting `role` after the widget exists has no visible
   effect unless you `unpolish`/`polish` — which is the entire reason `theme.tag` and
   `theme.retag` exist. Use them; do not set properties by hand.
2. **A `QComboBox`, `QSpinBox` or `QCheckBox` loses its indicator the moment the
   stylesheet gives it a custom border.** Qt clips the sub-control to a sliver, so you get
   an empty bracket where the arrow was — and styling `::drop-down`/`::up-button` makes it
   worse, not better (measured: four variants, only the image one worked). The fix is to
   supply the indicator as an image, so `theme._assets()` draws the three needed arrows
   and the checkmark **from the palette** into a hash-named folder in the temp directory
   and substitutes their paths into the stylesheet. If that write fails — a read-only temp
   directory is the only way — `stylesheet()` falls back to `_INPUT_FALLBACK`, which drops
   the rounded border rather than shipping a broken arrow.
3. **Pinning the style does not pin the palette.** `setStyle("Fusion")` leaves the
   application palette alone, and Qt seeds that palette from the desktop's colour scheme.
   On a desktop set to a dark theme — this machine's GNOME session is one — every widget
   the stylesheet does not name is then drawn **dark on a light window**: a plain
   `QListWidget` viewport takes `QPalette::Base`, a table's header takes `QPalette::Button`,
   and the standard icons Qt draws itself (the sidebar's `QStyle.StandardPixmap` icons)
   come out light-on-light. Measured on this machine: an unstyled list rendered `#1e1e1e`
   and a table header `#333333`, which is exactly the "text and icons I cannot read"
   report. **The fix is `theme.palette()` + `app.setPalette(...)` in `apply()`, after
   `setStyle`** (which can reset the palette) and before the stylesheet. That one call
   covers every widget the stylesheet never mentions, including the disabled colour group
   and the icons Qt draws natively. `stylesheet()` also names `QListView`/`QListWidget`/
   `QTreeView`/`QHeaderView` explicitly as a second line of defence. Self-test check
   "ui: the theme is installed" forces a dark palette, calls `apply()` again, and fails if
   the window/base/button colours or a fresh list's viewport come back dark.
4. **A resizable `QScrollArea` sizes its widget to the widget's *minimum*, not to what it
   wants — unless the size policy says otherwise.** It uses
   `max(viewport, qSmartMinSize(widget))`, and `qSmartMinSize` answers with the minimum
   size *hint* when the policy carries the shrink flag, which every default policy does
   (`Preferred` = 5, `Expanding` = 7; only `MinimumExpanding` = 3 and `Fixed` = 0 do not).
   A page's minimum hint is smaller than the height its form actually needs, because a box
   layout overrules an individual minimum once it is over-constrained, so the body was
   sized to that minimum and every row of the form was clipped mid-glyph. `StepPage`'s body
   therefore sets `setSizePolicy(Preferred, MinimumExpanding)`: scroll, never squeeze.
   `scrollable()`'s panels are the deliberate exception — they are allowed to be narrower
   than they would like, and show a scrollbar when they are.

## 4. How the deep models plug in

**The DL models are ordinary scikit-learn estimators.** They subclass
`BaseEstimator`/`RegressorMixin`/`ClassifierMixin`/`TransformerMixin` and implement
`fit`/`predict`/`predict_proba`/`transform`. There is deliberately **no parallel DL
path**: no second preprocessor, no separate tuning loop, no new screens. Because they
are sklearn estimators, they inherit, unchanged:

- the fitted `WorkbenchPreprocessor` (impute → encode → scale) inside the pipeline
- `cross_validate_pipeline` and the hand-written tuning loop, with per-candidate
  progress and cancellation
- permutation importance (they expose **no** `feature_importances_`)
- the HTML report, the `.joblib` bundle, and the exported reproduction script
- the Predict page and the Optimize/selection tools

### Embeddings are recovered from the one-hot block

`WorkbenchPreprocessor` names its one-hot columns `column=category`. A run of ≥2 such
columns that are all 0/1 is recognised as **one categorical variable** and replaced by
a learned `nn.Embedding`. The preprocessor was not modified — the column names are the
contract, and they stay meaningful in plots and importances.

- Embedding width: `min(cap=16, round(1.6 * cardinality**0.56))`, or a fixed
  `embedding_dim` if the user sets one.
- The table has `len(one-hot group) + 1` rows: the extra row is the **"unseen
  category"** bucket, which is where `handle_unknown` (all-zeros) lands at predict time.
- A group needs ≥2 columns **and** all-0/1 values — this guards against a numeric
  column whose name happens to contain `=`.
- `use_embeddings=False` turns the grouping off and feeds every prepared column in as a
  plain numeric input, which is the honest way to show the difference.

### Weights are stored, never pickled as a module

The fitted estimator keeps `state_dict_` (plain tensors) and `arch_` (a dict describing
the schema), and **drops the live `nn.Module` in `__getstate__`**, rebuilding it on
demand. That is why a bundle with a network in it is ~14 KB instead of megabytes, and
why the bundle stays portable.

### Progress and cancellation reach the training loop

sklearn's `fit(X, y)` has nowhere to put a callback, and `TaskRunner` runs jobs in
worker threads. So the training loop reports through a **thread-local observer** in
`core/dl/runtime.py`: `report()` for per-epoch lines, `check_cancel()` for cancellation.

The observer is installed **once, in `Task.run`** (`gui/workers.py`), not in each job —
so every path that ends up inside a network (plain fitting, k-fold CV, a tuning sweep,
the one-click run) gets live epoch progress and a working Cancel button. Epoch lines go
out on a **message-only** channel (`progressed.emit(-1, -1, msg)`, handled by
`MainWindow._progressed`) so they do not keep resetting the job's progress bar or hide
which tuning candidate is running.

## 5. The CPU policy

This is a 4-core/8-thread machine with no GPU, so every default in `core/dl/runtime.py`
and the model registry is chosen for that:

| knob | default | why |
|---|---|---|
| `threads` | 4 | explicit `torch.set_num_threads`; small nets are often *faster* with fewer |
| inter-op threads | 1 | `set_num_interop_threads(1)` — more only adds contention |
| `max_seconds` | 120 | hard wall-clock budget per fit; keeps the best weights so far |
| `epochs` | 120 | upper bound; early stopping normally ends far sooner |
| `patience` | 12 | epochs without validation improvement before stopping |
| hidden layers | `"64, 32"` | deliberately small |
| `batch_size` | 64 | |
| prediction | `torch.no_grad()` | no graph, no wasted memory |

A fit that hits `max_seconds` returns the best weights so far rather than hanging. The
tuning grid is deliberately tiny — only `hidden_layers` (3 values) and `learning_rate`
(2) carry a `grid`, i.e. **6 candidates**, because 6 candidates × 5 folds is already 30
network fits. Everything else is left out of the default sweep on purpose.

## 5b. The engineering toolkit (screens 13–18)

The point of the whole feature: a civil engineer should never need a private Python script.
Each screen answers one question that arrives as a number, and every one of them is
**read-only with respect to the model**.

| screen | core module | question |
|---|---|---|
| 13 · Tests & statistics | `core/stats.py` | is that difference real, or scatter? |
| 14 · Curve fitting | `core/curvefit.py` | what is the equation, and what does it predict? |
| 15 · Design of experiments | `core/doe.py` | how few runs will tell me which factors matter? |
| 16 · Time series & control | `core/timeseries.py` | is it moving, is it stable, is it capable? |
| 17 · Units & constants | `core/units.py` | what is that in the units the client uses? |
| 18 · Uncertainty & reliability | `core/montecarlo.py` | how likely is it to fail? |
| (8 · Save & export) | `core/export_office.py` | Excel and Word for the client |

**No new dependencies.** `scipy` (1.18.1) and `openpyxl` (3.1.5) were already there,
which covers hypothesis tests, distributions and `.xlsx`. Everything else that would
normally be imported is **absent**, so it is implemented here:

- **Word export** is a hand-written minimal OOXML package in `core/export_office.py` —
  `[Content_Types].xml`, `_rels`, `document.xml` with real `w:p`/`w:tbl`/`w:drawing`
  blocks. It is deliberately *not* a general docx library: it writes what the report
  screen has, and `verify_docx()` re-reads it to prove it is not corrupt (self-test).
- **PACF** uses the Durbin–Levinson recursion rather than `statsmodels`.
- **Dunn's test** (post-hoc after Kruskal–Wallis) and **Welch's ANOVA** are implemented
  directly, with a Bonferroni adjustment stated on the screen.
- **Design of experiments** generates the designs itself (factorial, fractional with
  alias tables, CCD, Box–Behnken, LHS, Sobol, Halton).

**One place computes, one place draws.** Every core module is pure Python + numpy/pandas/
scipy with **no Qt**, so the self-test can assert their numbers headlessly; the figures
live in `core/plots_engineering.py`, which takes frames the core modules produced.

**The pages never renumber v1's.** Screens 1–12 keep their numbers exactly (the manual,
the report and the Start screen all refer to them), the six new ones are 13–18 in a new
sidebar group **Engineering**, and v1's "Run everything" moved 13 → **19**.

**How the results reach the report.** Not by coupling a screen to `core/report.py`, and
not by polluting `PlotContext` (which belongs to the *model*). The screens push into
`AppState` with `remember_section(heading, frame)` / `remember_figure(caption, png_bytes)`,
`SavePage._extra_sections()` and `extra_figures` hand them to
`build_report(..., extra_sections=..., extra_figures=...)`. Figures are passed as
**already-rendered PNG bytes**, because these screens draw from their own results and
re-running them inside the report would mean recomputing the study.

**Exactly two engineering controls change the dataset**, and both say so on the button
and go through the cleaning pipeline so the change is reproducible:

- time series → **Add these lag and rolling columns to the dataset** (lag, rolling mean,
  change per step), and
- units → **Convert the column and add it to the dataset**.

They append a `derive_column` op to `state.clean_ops` and call `state.apply_cleaning()`.
That is what makes them survive saving and reopening the project, and it is why the
formula engine (below) had to learn quoted names: the columns of the real examples are
called `Age (days)` and `settlement_mm`, and the derived columns are named readably —
`settlement_mm (lag 1)`, `Cement (lb)` — rather than in an identifier-safe dialect.

**`core/formula.py` understands quoted column names.** A screen emitting a formula writes
`FORM.quote(column)`, so `("Blast furnace slag") / ("Cement")` round-trips. Bare names
still work for single-word headings, and both are covered by the self-test.

**The reliability screen is dtype-transparent.** A column of *names* (a curing method, a
supplier) has no curve to fit, so `MC.choose_distribution` returns `constant` for it, the
grid unticks its "vary" box and shows the mode, and the screen states which inputs it held
rather than sampled. A held category is *not* a missing answer: it is "given everything
else is normal, what does this one thing do?". If the user ticks such a row with
`constant`, the value typed in the parameter box is kept as text (not coerced to `None`),
so "hold the curing method at steam" actually simulates that.

**Nothing on these screens is silent about failure.** A quantity that cannot be converted
affinely (temperature) is refused with a reason; a control chart on a rising series says
how many signals it raised and why that is the chart complaining about the trend;
extrapolated interpolations are labelled; a design that cannot separate two effects says
so through the alias table.

## 6. Invariants worth not breaking

All of v1's invariants still hold (see `../ML Workbench/DEVELOPER.md` §4). These are
the additional ones:

- **`import core.dl` must not import torch.** `core/models.py` imports the estimator
  classes *lazily, inside the constructor functions*, and `is_available` only offers the
  networks when torch can actually be **imported** — not merely found by `find_spec`,
  because a broken torch install would otherwise look available and fail mid-fit (see
  finding 9). The app must keep working with torch absent *or* broken.
- **A fitted network never holds a live `nn.Module`.** `state_dict_` + `arch_` only;
  `__getstate__` drops `module_`. Do not "simplify" this into pickling the module.
- **`feature_importances_` on the classifier is a property that deliberately raises
  `AttributeError`**, so `explain` falls back to permutation importance instead of
  reporting meaningless numbers.
- **`TorchAutoencoder.transform`/`reconstruction_error` take the *prepared* matrix**,
  not a raw frame — by the time the app calls them the pipeline has already prepared.
  `_prepared()` validates the columns and raises a message naming the `prepare` step,
  because pandas' own error ("could not convert string to float") says nothing useful.
- **The validation loss must use the same `sample_weight` as the training loss** — see
  finding 6. The epoch restored at the end is chosen by that number.
- **`_dl_architecture_params()` / `_dl_training_params()` build fresh `Param` lists per
  spec**; do not hoist them to module-level constants and share the list objects.
- **Only `hidden_layers` and `learning_rate` carry a `grid`** for the supervised nets.
  Adding a grid to `activation`/`optimizer`/`batch_size` multiplies the sweep by their
  choice counts and turns a 6-candidate search into hundreds of fits.

And the interface ones, which are just as easy to break:

- **No screen may hard-code a colour or a font.** Cosmetics live in `gui/theme.py`; a
  screen says `theme.tag(widget, "primary")`. The conversion of the ~40 scattered inline
  stylesheets was done once — do not reintroduce `setStyleSheet` outside `theme.py`.
- **A widget styled with `role` must be styled through `theme.tag`/`retag`**, because of
  the repolish trap in §3b. Setting the property directly appears to do nothing.
- **`Callout.set_text`/`set_action` are the only way to change a callout**, and the
  requirement signpost is driven from `StepPage.update_requirements()` — never from a
  subclass's `refresh()`, which would let a new screen forget it.
- **`gui/widgets.py` must not import `gui.steps` at module level.** `gui.steps` imports
  `gui.widgets`, so the `LoadPage`/`ModelPage` lookup inside `StepPage._goto` is a *late*
  import and has to stay one.
- **The manual and its pictures are part of the build.** `USER_MANUAL.md` refers to
  `docs/manual/*.png` by project-relative path; `ManualDialog` resolves them with
  `setSearchPaths` plus a `baseUrl`. Renaming one without the other fails the self-test.
- **The manual's table of contents is matched by name, not by anchor.** Qt's markdown
  renderer creates *no* anchors for headings, so `ManualDialog._jump` looks the heading up
  by its text, using `slugify()` — the same rule GitHub uses, where *every* space becomes
  its own dash (`4.1 · Load the data` → `41--load-the-data`). The self-test walks all 35
  contents links and fails on a dangling one, which is what a renamed heading produces.
  Only the entries inside the *contents list* are also checked for wording (an inline
  sentence such as "the questions in [the engineering toolkit](#…)" is prose, and the
  label need not be a substring of the heading); every link, anywhere, must resolve.
- **The example dataset is load-bearing.** The manual quotes its shape, its two-class
  text target and its missing values; `examples/make_examples.py` is seeded, and the
  self-test check "example: the manual's project loads" fails if the CSV stops matching.
  The same is now true of `concrete_mix.csv` and `pile_settlement_monitoring.csv`: the
  manual quotes their shape, their mean strength, their missing readings, the fitted R²,
  the settlement rate and the reliability probability, and a self-test check asserts each
  one.

And the engineering ones:

- **Every engineering screen is read-only with respect to the model.** They may *need* a
  model (18 does), but none of them changes it. The only two controls on the six screens
  that touch the data are named as such and go through `state.clean_ops` +
  `apply_cleaning()`; never write to `state.df` directly, or the change will vanish on the
  next `apply_cleaning()` and will not survive reopening the project.
- **The engineering core modules must not import Qt.** They are the part the self-test
  exercises headlessly, and the reason a number can be asserted without a window.
- **A screen must not reach into `core/report.py`.** Queue with `remember_section` /
  `remember_figure`; the report screen assembles. Adding a new engineering screen should
  need no change to `report.py` at all.
- **`set_raw()` clears the cleaning recipe** (and the engineering extras). A recipe names
  the *previous* file's columns — loading a second file with the first file's ops still in
  place fails much later with `operation 1 failed: unknown name(s): Water`. Do not
  "optimise" that clear away; self-test check "clean: a new file does not inherit the old
  recipe" exists because the failure is so far from its cause.
- **Two screens force a distinction the UI must keep:** a *held* input is not a *missing*
  one (screen 18 says which inputs it held, and why); and a signal on a control chart
  names the **date** it happened on, not "point 112".
- **Anything the manual quotes must be asserted.** The manual is a deliverable and its
  numbers are checked by the self-test against the real example files, so an edit that
  moves a number fails the build rather than quietly making the manual wrong.
- **No screen may dictate the size of the window.** A `QStackedWidget` takes the maximum
  over its pages, so one tall or wide screen sets the floor for the whole application — and
  a window that cannot shrink to the work area cannot be maximized either (finding 24).
  Every page's body therefore lives in `StepPage._scroll` with a `MinimumExpanding` vertical
  policy (finding 25), and the window's own floor is asserted to stay under 1000 × 700.
  Consequences for new code: a screen must look right at any size, a panel that must not be
  squeezed goes in `scrollable()`, and a widget whose size hint follows *content* (a combo
  sized to its longest item is the easy mistake) needs its own cap. Self-test check 13.

## 7. Verification status (last full run: 17 Oct 2026)

| check | result |
|---|---|
| `--self-test` on the built-in 900-row example (regression) | **73/73**, 18/18 models, 35 plots, 0 failed, linear R² = 0.8219 |
| `--self-test --data Concrete_Data.csv` (acceptance set, 1030×9) | **73/73**, 18/18 models, 35 plots, linear R² = **0.5849** (identical to v1) |
| `--self-test --data /tmp/opencode/churn.csv --target churned` (binary, 1,500×10, with an id and a date column) | **73/73**, 16/16 models, 36 plots, 0 failed |
| `--self-test --data /tmp/opencode/grades.csv --target grade` (3-class, text labels, 600×8) | **73/73**, 16/16 models, 38 plots, 0 failed |
| `../ML Workbench` original, unchanged | **29/29** — the copy did not disturb it |
| the real window on this 1920 × 1080 Wayland session | minimum size hint **538 × 341**; `showMaximized()` → **1920 × 1013** (the full work area) and `showFullScreen()` → **1920 × 1080**; before the fix the same calls gave a "maximized" 1484 × 1059 with a 436 px gutter |
| torch absent, and torch installed-but-broken | networks grey out with "(needs torch)", `build_model` raises a clear error, linear R² = 0.9596, the rest of the app unaffected |
| the stylesheet's fallback path (unwritable temp directory) | still returns a usable sheet (7,859 chars, no image URLs) and recovers on the next call |
| a dark desktop palette (the readability bug) | reproduced first (`QListWidget` viewport `#1e1e1e`, table header `#333333`); after the fix, screenshots of Clean / Features / Graphs / Explore under a forced dark palette are light throughout |
| shared files differing between the two projects (`diff -rq`) | the 15 code files listed in §3, plus this document and each project's own `workbench_output/` |
| the manual's 35 contents links | all resolve to a heading, all scroll, each contents entry reads like its heading |
| `docs/make_manual_shots.py` | writes all **19** pictures; every one of them was read back and checked for real content (a filled result panel, a chart, a table), not just for existing. A screen too tall for the window is photographed in a taller window, so 4 of the 19 are 680–903 px tall rather than 652 |

The four self-test runs are the ones that matter: the same 73 checks pass on a
regression table, an acceptance set, a two-class table and a three-class table with text
labels. The deep-learning checks print their evidence, e.g. `PyTorch 2.14.1+cpu on CPU ·
4 of 8 threads · no GPU | regression 40ep; binary 40ep; multiclass 40ep; unsupervised …`.

The two synthetic configurations are the sharp ones: their *first* column is an id, which
is what caught finding 26, and the feature-combination search on the churn frame settles on
a single column, which is what caught finding 27 — the binary run draws **36** plots where
it used to report 38, because PCA and t-SNE are now correctly listed as unavailable in that
context instead of being drawn into a scikit-learn error.

The engineering screens were also exercised end-to-end on their own (a scratch harness in
`/tmp/opencode/eng_e2e.py`, 37 checks): concrete trial report → random-forest test
r² **0.9471**; every statistics tab; all 13 curve forms (best = offset power law,
R² **0.7901**) plus solve/interpolate/residuals/ranking; the eight-run design plus its
analysis and three figures; units (30 MPa = **4,351.13 psi**, column conversion, affine
refusal); reliability (P(strength < 37 MPa) = **0.138**, β = 1.14, top input `Age (days)`,
all five charts, JSON round-trip); the HTML report (**1,186 KB**, 5 engineering sections,
5 figures, **0 warnings**); and both office exports re-read and validated. Those numbers
are what the manual quotes, and the self-test now asserts each of them.

The six deep-learning checks are:

1. **every network fits its tasks** — regression / binary / multiclass / unsupervised on
   a self-contained synthetic frame; asserts a history, epochs, an architecture, stored
   weights, probabilities that sum to one, labels drawn only from the training
   vocabulary, a latent code of the declared width, and that the training curve renders.
   Also prints the runtime line (`PyTorch 2.14.1+cpu on CPU · 4 of 8 threads · no GPU`).
2. **categoricals become embeddings** — asserts the preprocessor produced `col=cat`
   columns, that the model grouped them, that it built a table, that the numeric block
   shrank, and that `use_embeddings=False` feeds every column in plain.
3. **row weights reach the loss** — the regression test for finding 5. Down-weights half
   of a two-regime dataset by `1e-7` and asserts the error on that half gets *much*
   worse, which is only true if the weights are actually applied.
4. **cancel interrupts training** — a pre-set cancel event must raise `Cancelled`
   instead of running 5000 epochs.
5. **bundle round-trip** — save/reload a network, assert identical predictions and that
   the bundle records the torch version.
6. **the overfit gap points the right way** — the regression test for finding 14. Fits an
   unpruned decision tree on noisy data so the training score is genuinely higher than the
   test score, then asserts `overfit_gap > 0` and equals `train - test`.

The seven interface checks are:

7. **every screen explains what it needs** — walks all 20 pages and asserts that each one
   declares `requires` from `{None, "data", "model"}` and that its signpost is shown
   exactly when the prerequisite is missing, with an action button that goes somewhere.
8. **the theme is installed** — the stylesheet is non-empty and carries the palette, the
   *base* style under it is `QFusionStyle`, a role set at runtime survives, and the
   palette stays pinned: it sets a dark desktop palette, calls `theme.apply()` again, and
   asserts the window/base/button colours and a fresh `QListWidget` viewport all come back
   light (the regression test for finding 15).
9. **the start screen tracks progress** — the checklist ticks itself off from the shared
   state (`data=✔` once a frame is loaded, `model=○` before anything is trained).
10. **the user manual is present and renders** — `USER_MANUAL.md` exists, every
    `docs/manual/*.png` it refers to exists, `ManualDialog` actually renders that many
    pictures, and all 35 links resolve to a heading (only the contents entries must also
    *read* like their heading), scroll the view *and* land that heading at the top of it.
11. **the manual's example project loads** — `examples/customer_churn.csv` still has 10
    columns, a `yes`/`no` target and missing values, which is what the manual claims.
12. **every shipped example is in the loader's list** — the manual sends the reader to the
    picker on **1 · Load data**, so a file in `examples/` the picker does not offer would
    make the manual wrong. It also fails if the picker offers a file that is not there.
13. **the window can be filled, and a screen scrolls** — the regression test for findings
    24 and 25. Walks every page and asserts its body is inside a `QScrollArea`, that the
    body's vertical policy carries no shrink flag (so a form is scrolled, never squeezed),
    and that no screen asks the window for more than 900 × 700. Then it asserts the View
    menu's Full screen action is bound to `F11` and that ticking it sets
    `isFullScreen()` and unticking it clears it. Prints e.g.
    `widest screen 330 px, tallest 274 px, window floor 577 x 338`.

The two plot checks are:

14. **every available graph draws** — the regression test for finding 23's class of bug:
    walks both families, draws each plot the screen reports as *available*, and fails on the
    first exception, printing the sidebar's list and what was unavailable in this context.
    It also goes through the GUI path (`_draw()`), which is where a failure would surface as
    a dialog.
15. **a projection needs two columns** — the regression test for finding 27. A one-column
    model must not list PCA or t-SNE at all, a direct call must fail with a `PlotError` that
    says how many features are needed, and a two-column frame must still draw. Asserted
    directly because the check above only caught it on one of the four datasets.

The engineering checks (the bulk of them) drive the real screens on the real example
files and assert the numbers the manual quotes, so a change that moves a number fails the
build instead of quietly making the manual lie:

- **the two new examples load** — `concrete_mix.csv` is 1,200 × 13 with 74 missing values
  and a 67%-show `meets_C30_37` column; `pile_settlement_monitoring.csv` has 157 readings,
  3 missed weeks and a maximum of 13.07 mm.
- **statistics** — describe one column; the test is *chosen* (Welch/t-test/Mann–Whitney);
  the curing-method comparison reports p = 0.1091 (no significant difference) and the
  headline says so; `Cement` vs strength is r = 0.4352 over 1,200 batches; the sample-size
  tab returns 32 per group for d = 0.5 at 80% power; the correlation matrix ranks six
  pairs; the figures reach the report queue.
- **curve fitting** — all 13 forms rank, best = offset power law at R² 0.7901;
  `y = −79.005 + 94.935·x^0.095036`; solve and interpolate work; 1,192 repeated ages are
  averaged (finding: duplicate x).
- **design of experiments** — an eight-run two-level design with six columns in the run
  sheet, randomised against standard order; the analysis names `Cement` as the strongest
  term (+9.849).
- **units** — 30 MPa = 4,351.13 psi, 24 quantities listed; a column converts through the
  cleaning pipeline; an affine quantity (temperature) is refused.
- **reliability** — the input grid fills from the model's features, a categorical column
  is **held, not sampled**; P(strength < 37 MPa) = 0.138, β = 1.14, top input
  `Age (days)`; all five charts; the setup round-trips through JSON.
- **time series** — the settlement trend is +3.6455 mm/yr (CI 3.440–3.851); `analysis_combo`
  runs every analysis without raising; the ACF/PACF bands are right; lag/rolling/change
  columns go through `derive_column` and survive `apply_cleaning()`; the control chart
  finds the week ending 2025-03-31 (+5.28σ) and names the *date*; capability against
  ±0.5 mm/week gives Cp 0.6978 / Cpk 0.5830, "not capable".
- **report and office** — the engineering sections and figures appear in the HTML report,
  and both the `.xlsx` and `.docx` exports are written and re-read.
- **the six core modules** — a direct headless exercise of `units`, `stats`, `curvefit`,
  `timeseries`, `doe`, `montecarlo` (shapes, invariants and edge cases), which is why
  those modules must stay Qt-free.
- **a new file does not inherit the old recipe** — the regression test for finding 16:
  loading a second file must clear `clean_ops`, or the old recipe's column names break the
  new dataset many steps later.
- **the cleaning recipe is applied to a numeric column** — the regression test for finding
  26: `do_clean` used to cast and filter `columns[0]`, which is an id in a real export, so
  the frame came back empty and fourteen downstream checks failed with messages that pointed
  nowhere near the cause. It now names the first numeric column and says which one it used.

Datasets: `/tmp/opencode/churn.csv` (binary, has NaN + a date + an id column) and
`/tmp/opencode/grades.csv` (3 classes, with categorical columns) are synthetic and were
regenerated for this project; regenerate them if `/tmp` is wiped. The acceptance set is
`../Learn Machine Learning/Concrete_Data.csv`. `LoadPage.example_frame()` is the
built-in 900-row concrete dataset, so the app is demoable with no files at all. The three
files in `examples/` are load-bearing (the manual quotes them and the self-test asserts
them) and are regenerated by `examples/make_examples.py`, seeded so they are stable.

Scratch tests from this build (not in the project): `/tmp/opencode/dl_smoke{1,2,3}.py`,
`dl_registry.py`, `dl_integration{,2}.py`, `dl_wfix.py`, `dl_plots.py`, `dl_export{,2}.py`,
`dl_weights.py`, `dl_selftest_dbg.py`, `dl_notorch2.py`, `dl_absent.py`, and for the
engineering toolkit `eng_e2e.py` (a 37-check harness for the six screens, written before
the checks were folded into `app.py --self-test`). `dl_wfix.py` is the clearest
demonstration of findings 5 and 6; `dl_export2.py` runs the exported script as a
subprocess for both a regression and a multiclass network; `dl_absent.py` is meant to be
run with `site-packages/torch` temporarily moved aside. For the window-size work:
`fs_real2.py` (shows/ maximizes/ full-screens a real window inside a real event loop and
prints the geometry at each step — the only way to see findings 24/25, because the offscreen
platform answers `showMaximized()` with 796 × 796), `fs_f11.py` (drives the F11 action both
ways) and `fs_after.py` (walks a page list at two window sizes and reports which scrollbars
appear). `make_datasets.py` regenerates `/tmp/opencode/{churn,grades}.csv` if `/tmp` is
wiped; those two files are only needed for the four-configuration verification, not by the
program.

## 8. Findings log — bugs already found and fixed

Do not re-fix these; do check that a change has not undone them. (v1's findings 1–14
also apply and are not repeated here.) Findings 1–9 came out of the deep-learning work,
10–15 out of the interface and documentation pass, 16–23 out of the engineering
toolkit, and 24–27 out of the window-size work and the re-verification it forced. Where a finding has a regression test, the number of that check is given.

1. **`_finalise` misread the autoencoder's attributes.** It inferred the output width
   from attributes that only make sense for a supervised net, so an autoencoder ended up
   with the wrong head. `_finalise` now takes an explicit `arch` dict.
2. **`AutoencoderNet` lost its dropout / activation / batch-norm settings.** The
   encoder and decoder were built without them. It now stores and applies them, and its
   `forward` accepts and ignores `x_cat` so it has the same signature as `TabularMLP`.
3. **A stray `torch.set_grad_enabled(True)`** left global autograd state modified after
   `predict`, and an artifact string was being concatenated into `architecture_note`.
   Both removed.
4. **The classifier stringified its labels.** `fit` called `str()` on the target, so a
   numeric 0/1 column produced classes `'0'`/`'1'` and would have scored **0.0000**
   accuracy against the app's integer target — while looking perfectly healthy. Now
   `pd.factorize(y, sort=True)` preserves the label dtype, NaN becomes a `"(missing)"`
   class, and fewer than 2 distinct labels is a `ValueError`. Binary was the only
   affected case, which is why it hid.
5. **Row weights were silently ignored for every regression fit.** The regression loss
   comes back as `(batch, 1)` while the weight vector is `(batch,)`, so `loss * w`
   broadcast into a `(batch, batch)` **outer product** — and then `.sum() / w.sum()`
   cancelled the weights back out of the mean exactly. Weights were accepted, reported
   as accepted, and had *no effect whatsoever*. Fixed by `loss.reshape(-1)` before
   weighting. Classification was unaffected because `CrossEntropyLoss` already returns
   `(batch,)`, which is how it went unnoticed. This is the same *class* of bug as v1's
   finding 7, and it is the reason self-test check 3 exists.
6. **The validation loss ignored `sample_weight`.** Training was weighted but the
   per-epoch validation loss — and therefore the epoch whose weights get restored at the
   end — was not, so the "best" epoch was the one that fitted the **down-weighted** rows
   best: the exact opposite of what the weights asked for. `_train_module` now takes
   `valid_weights` and weights the validation loss the same way. Found only because
   fixing 5 made the weights strong enough for the wrong best-epoch to be visible.
7. **`reconstruction_error`/`transform` on a raw frame gave a useless error.** Passing
   the raw frame instead of the prepared matrix raised
   `ValueError: could not convert string to float: 'alpha'` from deep inside pandas,
   which says nothing about the real mistake. Added `_prepared()`, which checks the
   columns against `feature_names_in_` and names the `prepare` step in the message. (The
   self-test's own first draft made this mistake, which is what surfaced it.)
8. **`param_default_grid` skipped `str` params that have a grid.** `hidden_layers` is a
   free-text string, so the auto-validation plot fell through to the first *choice*
   param — `activation`, i.e. five full network fits — instead of sweeping
   `hidden_layers`, i.e. three. `param_default_grid` now includes `int`/`float`/`str`
   params that carry a grid. No existing model was affected (none used a `str` param).
9. **`is_available` trusted `find_spec` for torch.** A torch that is *installed but
   broken* — a missing `libtorch.so`, a wheel built for another Python — passes
   `find_spec` and then raises inside `fit`, so the user gets a menu entry that looks
   available and fails mid-job. `is_available` now calls `runtime.torch_available()`
   (a real import) for torch, so the entry is greyed out with "(needs torch)" and the
   About box explains why. `_has` is exception-safe too: `find_spec` *raises* rather
   than returning `None` for a module sitting in `sys.modules` half-initialised.
   Verified both ways — moving `site-packages/torch` aside (genuine absence) and a
   loader that raises on import (broken install): the networks grey out, `build_model`
   raises a clear error, and linear regression still scores R² 0.9596. `torch_available()`
   caches the answer in both directions so a broken install does not re-print its error
   on every menu refresh.
10. **A stylesheet property selector did nothing until the widget was repolished.** Qt
    resolves a widget's style once and caches it, so setting `role` after construction
    had no visible effect at all — the widget simply kept its old look, with no warning.
    This is why `theme.tag`/`retag` exist and why no screen sets a dynamic property by
    hand. Cost: an hour of "the stylesheet is obviously correct, so why is nothing
    styled".
11. **A custom border on a `QComboBox`/`QSpinBox`/`QCheckBox` clips its indicator to a
    sliver.** The arrow and the tick vanished, leaving an empty bracket. Styling
    `::drop-down`/`::up-button`/`::indicator` made it *worse*, not better. Measured four
    variants; only supplying the indicator as an **image** worked, so `theme._assets()`
    now draws the three arrows and the tick from the palette into a temp directory and
    substitutes their paths into the stylesheet, with `_INPUT_FALLBACK` (square inputs,
    no images) as the read-only-temp-dir fallback.
12. **Qt's markdown renderer creates no anchors for headings, and a contents link is a
    URL it will try to load.** The manual's table of contents looked perfect and did
    nothing; worse, the original `_anchor_clicked` passed the fragment to
    `QTextBrowser.setSource`, which *replaced the whole document*. Fixed by looking the
    heading up by name with GitHub's slug rule and moving the cursor — see §6 and
    self-test check 10. Two follow-on traps, both found by the check being tightened:
    a contents entry has **the same text as its heading**, so matching on text alone
    finds the list item and every link "works" while landing in the contents itself
    (fixed by requiring `blockFormat().headingLevel() > 0`); and `ensureCursorVisible`
    only makes the heading *visible*, which for a jump down the document parks it on the
    last line — the scrollbar is nudged afterwards so it lands at the top.
13. **`QApplication.style()` stops being the base style once a stylesheet is installed.**
    It returns Qt's `QStyleSheetStyle` wrapper, so an "am I already on Fusion?" guard
    written as `style().objectName() == "fusion"` could never fire. The pin is a module
    flag (`theme._STYLE_PINNED`) instead. Found by the self-test check that asserts the
    *base* style under the sheet is `QFusionStyle` — a check that would have been written
    the other way round and passed. `QApplication.setStyle` also has to happen **before**
    the stylesheet, or the sheet is re-resolved against the wrong base.
14. **The overfit gap had its sign inverted, so the app confidently said the opposite of
    the truth.** `_gap` computed `test - train` for a higher-is-better metric, but every
    consumer reads a positive gap as *overfitting*: the Model screen's card is labelled
    "large positive = memorising", and both the card and the report print
    `"worse" if gap > 0 else "better"`. For ROC AUC that produced, on the manual's own
    example, **"0.0487 better on the test set"** when the test score was in fact 0.0487
    *worse* — with the model's own numbers printed next to it (test 0.7016, train 0.7503).
    It is inherited verbatim from v1 (same code in `../ML Workbench/core/train.py`), it is
    a plain sentence rather than an exception, and it survived 29 v1 self-tests. Fixed by
    negating the higher-is-better branch; self-test check 6 is the guard. **If you port
    anything back to v1, this is the fix to port.**
15. **The theme pinned the style but not the palette, so a dark desktop theme made text
    and icons unreadable.** Reported from real screenshots: list items, table headers and
    the sidebar icons were dark-on-dark. Cause: `setStyle("Fusion")` does not touch
    `QApplication.palette()`, and Qt seeds that palette from the desktop's colour scheme —
    so on this GNOME session (dark) every widget the stylesheet does not name was drawn
    from a dark palette on a light window. Reproduced exactly by forcing a dark palette
    before `theme.apply()`: a plain `QListWidget` viewport came out `#1e1e1e` and a
    `QTableWidget` header `#333333`. The offscreen screenshots used for the manual never
    caught it because offscreen has no platform theme, so Qt's default palette is light —
    **the bug was invisible in every artefact produced while developing, and only a real
    desktop showed it.** Fixed by `theme.palette()` + `app.setPalette()` in `apply()`
    (after `setStyle`, before the stylesheet), plus explicit `QListView`/`QListWidget`/
    `QTreeView`/`QHeaderView` rules. Guarded by self-test check 8, which now forces a dark
    palette and asserts the app palette and a fresh list come back light.

Findings 16–23 are from the engineering toolkit. As a group they share one shape: the
program was *right*, and the sentence it printed about itself was wrong — a chart that
flagged 163 points out of 154, a ranking table that ranked two unnamed columns, a
probability computed on a model nobody would have built. Every one of them was found by
running the real screens on the real example files and reading the output as a person
would, which is exactly what a self-test that only checks exit codes cannot do.

16. **A derived column that only existed until the next `apply_cleaning()`.** The time
    series screen built `settlement_mm (change per step)` by assigning to `state.df`
    directly, so the column was real, charted correctly, and then **silently vanished** the
    moment anything re-applied the recipe — and was gone if the project was saved and
    reopened. Fixed by adding the `derive_column` operation to `core/clean.py` and routing
    both of the data-changing engineering controls through `state.clean_ops` +
    `apply_cleaning()`. The rule this bought is in §6: never write to `state.df` directly
    from a screen.
17. **A confirmation dialog made the button do nothing.** The "add these columns" button
    asked `QMessageBox.question` first. That is fine for a person — except the self-test's
    dismisser *closes* dialogs and does not click Yes, so the action was skipped and the
    check that followed it failed with no hint of why. The button now does the work and
    *informs* afterwards (matching what the sibling screens already did), naming the
    columns it added. Lesson recorded because it generalises: **a confirmation on a
    destructive action is right; a confirmation on an additive one breaks the harness and
    annoys the user.**
18. **The autocorrelation capped its lag at 24, which hid the signal.** The pile record is
    weekly and its real cycle is annual, so the interesting lag is 52 — outside a fixed cap
    of 24, so the ACF looked like white noise on a series that plainly had a season. Now
    adaptive: `min(len - 2, max(24, min(104, round(len / 3))))`, i.e. a third of the
    record, floored at 24 and capped at 104. A cap that is right for a 200-row daily record
    is wrong for a 157-row weekly one.
19. **A ranked table of correlations that did not say which columns it ranked.**
    `core/stats.py::correlation_p` only ever sees two anonymous series, so the "strongest
    pairs" table listed numbers with no names — unusable. The *page* now prepends the two
    column names it fed in, which is the only place that information exists. Core stays
    generic; the screen supplies the labels.
20. **A control chart that said "point 112".** True, and useless: the reader needs the
    date. The page now maps the violation's positional index back through the cleaned
    frame it charted and inserts a `when` column, so a signal reads *2025-03-31*. (The
    index, not the timestamp, is what the chart core has; converting it is the screen's
    job.)
21. **A categorical input silently became `None`.** The reliability grid's `constant`
    distribution stores its parameter as a number — so "hold the curing method at *steam*"
    turned the text into `None` and the simulation ran with a missing category, producing a
    real-looking distribution for a batch with no curing method. Fixed with `_param_value`,
    which keeps the text when the column is non-numeric; `core/montecarlo.py` is now
    dtype-transparent throughout `sample_inputs`/`influence`/`tornado`, and the screen
    states which inputs it **held** rather than sampled.
22. **Two invisible-to-the-tests sentences in the time-series screen.** The trend
    "sentence" and the summary cards were the only evidence the screen produced, and both
    were read from the wrong key: `state.timeseries` stored the analysis *key* where the
    label was expected, and the cards were read from a `MetricCards` attribute that does
    not exist (they are in `state.timeseries["summary"]`). Neither raised anything — the
    screen simply said less than it knew. Found by writing a check that asserted the words,
    not the exit code.
23. **Three tables and a chart in one column means the chart is invisible.** On the DOE
    *read the results* tab the three tables (terms, coefficients, main effects) consumed
    the whole column at a 900 px window height, so pressing "Plot the Pareto" drew into a
    sliver about 40 px tall. Nobody had noticed because the tables are what the eye lands
    on. Fixed by putting the tables and the canvas in a vertical `QSplitter`
    (`setSizes([340, 300])`) so the chart gets real room and the tables scroll. The manual
    screenshot is what caught it: a picture of a screen is a much harder test than a check
    that the canvas exists.
24. **The window could not be maximized, because its minimum size was bigger than the
    screen.** Reported as "the fullscreen doesn't make the program full screen, it looks
    unpleasing". On this 1920 × 1080 display, `showMaximized()` produced a window of
    **1484 × 1059** — nominally maximized (`isMaximized()` was `True`) but parked at its
    minimum size with a 436 px gutter beside it, and it could not be made smaller either.
    The cause was the *measured* minimum size hint: **1478 × 1060**. A `QStackedWidget`
    takes the maximum over its pages, so the tallest and widest screens set the floor for
    every screen — 891 px for the feature form, 1235 px for the reliability grid — and
    chrome added ~150 px. Mutter will not both honour a client minimum taller than the work
    area and fill it, so it gave up. Fixed by making each page's body scrollable
    (`StepPage._scroll`, in `gui/widgets.py`), which took the window's floor to
    **538 × 341**; the honest behaviour is that the longest screens scroll a little.
    `showFullScreen()` always worked, which is why the bug looked like "fullscreen does
    nothing". Regression test: interface check 13. Two follow-on details were needed to
    stop this being a cure worse than the disease: `scrollable()`'s width floor came down
    from 560 to 440, and the loader's example picker was sized to a 560 px item string
    (`AdjustToMinimumContentsLengthWithIcon`) instead of to its longest entry.
    **Modal-dialog lesson:** none of this was reachable offscreen — `showMaximized()` on the
    offscreen platform returns a nonsense 796 × 796 — so it had to be measured inside a
    real event loop on the real session.
25. **Making the body scrollable squeezed the forms instead of scrolling them.** The first
    version of finding 24's fix looked *worse* than the bug: every row of the
    feature form was clipped mid-glyph. A resizable `QScrollArea` sizes its widget to
    `max(viewport, qSmartMinSize(widget))`, and `qSmartMinSize` returns the *minimum* size
    hint when the widget's size policy carries the shrink flag — which every default policy
    does (`Preferred` = 5, `Expanding` = 7). A page's minimum hint (805 px for that screen)
    is smaller than the height its form actually needs (991 px), because a box layout
    overrules an individual minimum when it is over-constrained, so the body was sized to
    805 and the layout squeezed every row. Fixed with
    `setSizePolicy(Preferred, MinimumExpanding)` — `MinimumExpanding` is 3, the one policy
    without the shrink flag, so the floor becomes the size hint and the page scrolls.
    Horizontally the default is kept, so a table may still be narrowed rather than forcing
    a sideways scrollbar. The self-test now asserts the policy, because the failure is a
    pixels-only difference no value check can see.
26. **The self-test's cleaning check assumed the first column was numeric.** `do_clean`
    applied its four-step recipe to `columns[0]` — *cast to float, keep rows > 0* — which is
    an **id** column in any real export. Casting text to a float gives NaN, the `> 0` filter
    then drops every row, and the frame came back with 0 rows. It reported "ok, 0 rows after
    4 steps", and then **fourteen unrelated checks failed** with
    `ValueError: every row is missing the target value`, `KeyError: 'unknown model None'`
    and `FileNotFoundError: ''` — the misleading-distance problem again, and the reason it
    had gone unnoticed is that the built-in example's first column (`Cement`) is numeric, so
    the order of the columns was doing the test's job for it. Found by pointing the
    self-test at `/tmp/opencode/{churn,grades}.csv`, whose first column is an id: it also
    means the two extra configurations were never really testing a text-target frame, since
    only the shipped examples (first column `customer_id`, also an id) were ever loaded that
    way. Fixed by choosing the first **numeric** column, and the check now names the column
    it used.
27. **PCA and t-SNE were offered on a one-column model and died inside scikit-learn.**
    Their availability predicate was `ctx.X is not None`, but `_numeric_matrix` prefers the
    *encoded* matrix the pipeline produced, so a model whose best feature combination was a
    single column made `PCA(n_components=2)` raise
    `n_components=2 must be between 0 and min(n_samples, n_features)=1`. The Graphs screen
    therefore listed a plot that could not be drawn and reported a raw scikit-learn message
    instead of its own wording. Two gaps in one shape: the predicate did not ask the
    question the plot actually has (how many columns are there *to project*), and
    `_numeric_matrix` applied its `min_columns` guard on the raw-frame path but not on the
    prepared path, so it could return a one-column matrix after promising not to. Fixed with
    `_projection_width()` / `_has_projection()`, and by moving the guard onto the prepared
    path too. Found by finding 26's fix: with the frame no longer emptied, the search was
    free to choose one feature and the plots check finally reached that state.

## 9. Adding things

Everything in `../ML Workbench/DEVELOPER.md` §7 still applies. For this project:

- **A network**: add a `ModelSpec` to `core/models.py` with `optional="torch"`, and a
  constructor in `_CONSTRUCTORS` that imports the class **inside the function**. Build
  its param list from `_dl_architecture_params()` / `_dl_training_params()`, and give a
  `grid` only to the one or two knobs genuinely worth searching.
- **A per-epoch diagnostic**: the history is a dict of lists on the fitted estimator
  (`history_["epoch"]`, `["train_loss"]`, `["val_loss"]`), plus `n_epochs_`,
  `best_epoch_`, `stopped_reason_`, `architecture_`, `arch_`, `state_dict_`. A plot only
  needs a `_has_*` predicate over `_model_step(ctx)` and a `PlotSpec` entry — see
  `training_curve`.
- Keep the time budget honest: a new model that can run long must honour
  `runtime.budget_exceeded()` and call `runtime.check_cancel()` at least once per epoch,
  or the Cancel button stops working for it.
- **A screen**: subclass `StepPage`, give it `title`, `subtitle` and — if it cannot work
  on an empty session — `requires = "data"` or `"model"`. Register it in
  `STEP_GROUPS` in `gui/steps/__init__.py`, and add its class name to `SIDEBAR_ICONS` in
  `gui/main_window.py` (the lookup is by name and falls back to no icon, so a missing
  entry is harmless rather than broken). Style everything with `theme.tag`; the
  self-test's "ui: every screen explains what it needs" check will pick the new page up
  automatically and fail if the signpost does not follow the declaration.
- **An engineering screen**: put the arithmetic in a new Qt-free module in `core/` (or in
  an existing one — `stats.py` for tests, `timeseries.py` for anything with a clock), the
  figure in `core/plots_engineering.py`, and only the widgets in
  `gui/steps/engineering_steps.py`. Screens 13–18 are numbered and read-only with respect
  to the model; if your screen must change the data, route it through `clean_ops`
  (`derive_column` already exists for a formula-built column) and label the button as the
  thing that changes the dataset. Queue evidence for the report with
  `state.remember_section(...)` / `state.remember_figure(...)` — never by importing the
  report module.
- **A formula column for a screen**: build the expression with `FORM.quote(column)`, not by
  pasting the name in. Real column headings contain spaces, brackets and slashes.
- **A manual picture**: add a `shoot(...)` line to `docs/make_manual_shots.py`, re-run it,
  then reference `docs/manual/<name>.png` from `USER_MANUAL.md`. The self-test fails if
  the manual refers to a picture that is not there. The script runs the real screens on the
  real example files, so a new shot usually means loading a dataset in it; it installs a
  `QTimer` that closes message boxes, or the run hangs (see the header of that file).

## 10. Known limits

- **No GPU, no images, no text.** Tabular only — that was the agreed scope.
- **The autoencoder clusters with k-means on the latent code**, not with a deep
  clustering objective. `n_clusters=0` disables clustering and leaves the embedding.
- **Early stopping uses a random `val_fraction` split, not cross-validation**, so a fit
  is not fully deterministic across `random_state` changes — though the bundle records
  the torch version, device and thread count precisely because a network result is only
  reproducible with all three.
- **Tuning a network is slow by construction.** `max_seconds` bounds *one* fit, so a
  sweep of 6 candidates × 5 folds on a large dataset can still take many minutes. The
  progress bar and Cancel work throughout.
- `train_job`'s "compare a handful of algorithms" list and the one-click Auto run only
  reach the networks when the user picks one; they are not auto-selected, because on
  small tabular data a boosted tree usually wins and is far cheaper.
- **A screen may have to scroll.** At a 1380 × 900 window, screens 3 (Clean), 4 (Prepare
  features) and 13 (Tests & statistics) are taller than the body area; at a *maximized*
  window only 4 and 13 still need about 150 px. There are no horizontal scrollbars at
  either size. This is the price of a window that can be maximized at all (findings 24/25)
  and is the honest behaviour — the alternative was a window larger than the screen.
  `docs/make_manual_shots.py` makes the window taller for a picture of such a screen, so the
  manual still shows the whole thing.
- **The manual is English-only and lives beside the code.** It is a file, not a
  translation layer; there is no per-user help content beyond it and the shortcut list.

The engineering toolkit's limits are worth stating plainly, because a screen that looks
authoritative is easy to over-trust:

- **No statsmodels, no SALib, no python-docx, no xlsxwriter** — they are not installed and
  must not be added. The consequence is that `core/export_office.py` writes a *minimal*
  docx (text, tables, images), not an arbitrary one: styles are defaults, and there is no
  support for headers/footers, footnotes or fields. The `.xlsx` writer is openpyxl, which
  is a real library and handles formatting, so Excel export is the richer of the two.
- **The reliability numbers are only as good as the model behind them.** The screen draws
  each input independently from its observed spread, or — with the correlation option —
  through a **Gaussian copula with rank reordering**, which imposes the observed
  correlation without distorting any single input's marginal. A held category is left out
  of the copula entirely. And it samples the *inputs*, not the residuals, so read the
  answer as "given this model and these marginal distributions", which is what the screen
  says.
- **The control charts implement Nelson's eight rules** (one entry per episode, not per
  window) on individuals/moving-range and X-bar/R charts. CUSUM and EWMA are judged against
  their own thresholds instead, deliberately: a smoothed series is autocorrelated, so run
  rules would fire on the smoothing rather than on the process. The violation table names
  the rule that fired and what was seen, not a tutorial.
- **Process capability's expected-outside-% assumes normality** (`sps.norm.cdf`), and the
  verdict is the usual Cp/Cpk ladder (≥ 1.33 capable, ≥ 1.00 marginal). On a strongly
  skewed column both are optimistic, which is why the screen shows the *observed* share
  outside the limits beside them, and why the trend work happens first: fitting a
  distribution to a series that is still moving is the classic way to get a confident
  wrong answer.
- **Design of experiments fits main effects and two-way interactions by OLS** — no
  blocking, no split plots, no unbalanced designs. It is a screen for 2–6 factors and
  8–50 runs, which is the size of experiment a single engineer runs.
- **The engineering screens have no undo of their own.** The two that change the data go
  through the cleaning recipe, so removing the step on **3 · Clean** is the undo.

## 11. Change log

- **17 Oct 2026** — created the project: copied v1 (`rsync`, excluding `.venv`,
  `__pycache__`, `exported_output`) so the original stayed byte-for-byte intact; wrote
  `requirements-dl.txt`; built the 2.2 GB in-tree venv with the CPU-only torch wheel.
  Wrote `core/dl/{runtime,nets,estimators,__init__}.py` and wired the three networks into
  the registry, the version record, the plot list, the report set and the cluster job.
  Installed the progress/cancel observer once in `Task.run` and added the message-only
  status channel. Added 5 self-test checks (29 → **34**). Fixed findings 1–8, all found by
  running the thing. Verified 34/34 on four configurations, 34/34 on the acceptance set
  reproducing v1's linear R² of 0.5849, the exported script as a subprocess for a
  regression and a multiclass network, and 29/29 on the untouched original. Wrote this
  file.
- **17 Oct 2026 (later)** — interface and documentation pass, on the same day:
  - Added `gui/theme.py` (palette, complete stylesheet, `apply`/`tag`/`retag`) and pinned
    the app to the Fusion style, so the look no longer depends on the desktop theme.
  - Added `Callout` to `gui/widgets.py`, `StepPage.requires` + `update_requirements()`,
    and a `requires` declaration on all 12 dependent screens — a screen that cannot work
    now says so and offers a button to the screen that fixes it.
  - Rebuilt the navigation rail (brand block, icons, accent bar on the selected item) and
    added a **Help** menu with the manual, a quick-start pointer, a shortcut list and an
    About box.
  - Added **Start here** (`gui/steps/start_steps.py`) as the screen the window opens on:
    a live progress checklist read from `AppState`, one-click loading of the built-in
    example and of the example project, and a map of the other thirteen screens.
  - Replaced every inline `setStyleSheet` in `gui/` and `gui/steps/` with theme roles
    (≈40 sites) and gave scrollable panels a width floor — this fixed a long-standing
    bug in **both** projects where a splitter squeezed a form until its spin boxes were
    clipped.
  - Wrote `USER_MANUAL.md` (a beginner's walkthrough of the whole workflow, using a new
    example project), `examples/make_examples.py`, `examples/customer_churn.csv` (4,000
    rows, seeded), `docs/make_manual_shots.py` and the 13 screenshots it produces (a
    separate one with the network selected, because the deep-learning section was
    otherwise illustrated with a picture of logistic regression).
  - Added 6 self-test checks (34 → **40**). Four real bugs were found by them: the
    "already Fusion?" guard in `theme.apply()` could never fire, because
    `QApplication.style()` returns Qt's `QStyleSheetStyle` wrapper once a stylesheet is
    installed; the spin-box/combo-box indicators were being clipped by the custom border
    (see §3b); the manual's table of contents did nothing at all — Qt's markdown
    renderer creates **no anchors for headings**, so clicking an entry handed
    `QTextBrowser` a URL it tried to *load*, replacing the document (`ManualDialog` now
    finds the heading by name — `slugify`, GitHub's rule — and moves the cursor, and the
    self-test walks all 22 entries and fails on a dangling or mis-worded one); and the
    overfit gap's sign was inverted, which made the app state the opposite of the truth
    about whether a model was overfitting (finding 14).
  - The manual's quoted numbers were checked against the program rather than remembered,
    and three were wrong: the cross-validation card (0.73, actually 0.7468), the model
    comparison table (it quoted *held-out test* scores for a tab that cross-validates, so
    0.70/0.73/0.73/0.73/0.72 became 0.75/0.77/0.76/0.76/0.78), and the illustration in §1
    (invented rows, now the real first three). The graph count was 45, not 46, and the
    reader-format list was missing fixed-width text and SQL databases.
  - The example dataset took four attempts to get right. The signal has to be *threshold
    and interaction shaped*, not additive: with a purely logistic generator, regularised
    logistic regression beat every boosted tree by 0.06 ROC AUC and the example taught the
    wrong lesson. With strong conjunctions and 4,000 rows the trees edge ahead
    (0.77 vs 0.75 cross-validated), which is both true and useful — and the network, at
    0.78, edges ahead of *all* of them, which is an honest reminder not to assume it.
- **17 Oct 2026 (readability fix)** — the user reported from real screenshots that list
  items, table headers and the navigation icons were almost invisible, dark on dark. The
  cause was that `theme.apply()` pinned the *style* but never the *palette*: Qt seeds the
  application palette from the desktop's colour scheme, so on this machine's dark GNOME
  session every widget the stylesheet did not name was drawn from a dark palette on a
  light window. It was invisible in every offscreen screenshot taken while building,
  because offscreen has no platform theme and Qt's default palette is light — the bug
  only existed on a real desktop. Fixed by adding `theme.palette()` and calling
  `app.setPalette(...)` in `apply()` (after `setStyle`, which can reset the palette, and
  before the stylesheet), and by naming `QListView`/`QListWidget`/`QTreeView`/`QHeaderView`
  explicitly in the stylesheet as well. Self-test check 8 now forces a dark palette, calls
  `apply()` again, and fails if the app palette or a fresh list comes back dark, so the
  regression cannot return silently. The 13 manual screenshots were regenerated against
  the corrected theme. See finding 15.
- **17 Oct 2026 (engineering toolkit)** — the civil-engineering pass, so that no user needs
  a private Python script for the questions that arrive as a number. Added six Qt-free core
  modules (`core/units.py`, `stats.py`, `curvefit.py`, `timeseries.py`, `doe.py`,
  `montecarlo.py`), the figure layer (`core/plots_engineering.py`), the office writers
  (`core/export_office.py`: openpyxl for `.xlsx`, hand-written OOXML for `.docx`), the
  `derive_column` cleaning operation in `core/clean.py`, quoted column names in
  `core/formula.py`, and `extra_sections`/`extra_figures` on `core/report.py`.
  - Six new screens in a new **Engineering** group — 13 tests & statistics, 14 curve
    fitting, 15 design of experiments, 16 time series & control, 17 units & constants,
    18 uncertainty & reliability — with v1's "Run everything" moved 13 → **19**. Every
    other screen keeps its number, because the manual, the report and the Start screen all
    refer to them.
  - **No new dependencies.** Word export, PACF (Durbin–Levinson), Dunn's test and Welch's
    ANOVA are implemented here, because `statsmodels`/`python-docx`/`xlsxwriter`/`SALib`
    are not installed and must not be added.
  - Two example projects added and made load-bearing: `examples/concrete_mix.csv` (1,200
    trial mixes) and `examples/pile_settlement_monitoring.csv` (157 weekly readings of one
    pile, with a real settlement trend and one abrupt step). `examples/make_examples.py`
    regenerates all three; the settlement "spike" was a one-week blip and is now a
    **permanent step**, which is what a pile that has moved actually looks like.
  - The loader gained a shipped-example picker, and `gui/state.py` grew the extra
    result/ figure queues and clears the cleaning recipe on `set_raw`.
  - `USER_MANUAL.md` grew **section 6, "The engineering toolkit"** (6.1–6.8), the tail
    sections were renumbered 7–10, and six new pictures were added by
    `docs/make_manual_shots.py` (**19 pictures** in total).
  - Added **31 self-test checks** (40 → **71**) and found the eight bugs recorded as
    findings 16–23 — every one of them a case where the program computed the right thing
    and said the wrong thing about it. Verified 71/71 on all four configurations (default
    example, acceptance set at linear R² 0.5849, churn, grades) and 29/29 on the untouched
    original. All 19 manual pictures were regenerated and then read back to check they
    show real content, not an empty panel.
- **17 Oct 2026 (the window could not be made full screen)** — the user reported "the
  fullscreen doesn't make the program full screen, it looks unpleasing". It was not a
  missing feature: the window's *minimum* size was **1478 × 1060**, larger than the work
  area of this 1920 × 1080 display, so `showMaximized()` left a window of 1484 × 1059
  parked in the middle of the screen with a 436 px gutter beside it (`isMaximized()` was
  `True` throughout, which is why it looked like nothing happened). Diagnosed by measuring
  a real window inside a real event loop, because the offscreen platform answers
  `showMaximized()` with a nonsense 796 × 796.
  - Fixed in `gui/widgets.py`: `StepPage` puts its body in a `QScrollArea` whose widget is
    `MinimumExpanding` vertically, so the window's floor drops to **538 × 341** and the three
    genuinely long screens (Clean, Prepare features, Tests & statistics) scroll a little
    instead of dictating the size of every screen. Both halves of that change are needed —
    see finding 25 for what happens without the size policy.
  - `gui/main_window.py` no longer opens at a fixed size: it clamps its opening size to the
    available screen minus 60 px, and got **View → Full screen**, bound to **F11** (which
    was the affordance actually missing). `showFullScreen()` is used rather than
    `showMaximized()` because it is a request the client can always satisfy.
  - Two follow-on layout fixes so the cure was not worse than the disease: `scrollable()`'s
    width floor 560 → 440 in `gui/widgets.py`, and the loader's example picker no longer
    sizes itself to a 560 px item string (`gui/steps/data_steps.py`); the Load screen's two
    panels also went through `scrollable()` like every other split panel.
  - Added self-test check 13 (interface check 7), which asserts the whole chain: every page
    body scrollable, no shrink flag on the vertical policy, no screen asking for more than
    900 × 700, and F11 toggling `isFullScreen()` both ways. **73/73** on the default
    example. `docs/make_manual_shots.py` now photographs a screen too tall for the window in
    a taller window, so 4 of the 19 pictures are 680–903 px tall instead of 652 — the
    feature form's shot shows the whole form again, including the button the manual tells
    the reader to press. All 19 were regenerated and read back. `USER_MANUAL.md` §2 "The
    window" documents F11 and that a long screen scrolls.
  - `../ML Workbench` was not touched.
  - Re-verifying the four configurations turned up two more bugs in the checks and in the
    plots, both recorded as findings 26 and 27. The first: the self-test's
    `clean: apply a four-step recipe` check cast and filtered `columns[0]`, which is an
    **id** column in a real export, so it emptied the frame and fourteen unrelated checks
    failed with misleading errors; it now uses the first numeric column. The second: PCA and
    t-SNE were listed as available for a one-column model and then died inside scikit-learn;
    the availability predicate now asks how many columns there are to project, and
    `_numeric_matrix` applies its `min_columns` guard on the prepared path too.
    Both only surfaced because the two synthetic datasets were regenerated after `/tmp` was
    wiped (`/tmp/opencode/make_datasets.py`, now noted in §7). Added a check
    (72 → **73**) that asserts the projection rule directly rather than relying on a dataset
    happening to be narrow.
