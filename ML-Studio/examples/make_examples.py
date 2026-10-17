"""Generate the small example dataset that ships with the workbench.

Run it with the project's own interpreter, from the project root::

    ./.venv/bin/python examples/make_examples.py

The CSV it writes is committed alongside this script, so you only need to run it if
you want to change the data.  Everything is seeded, so the output is reproducible.

``customer_churn.csv`` is deliberately awkward in the ways real tabular data is
awkward, because the point of the example is to exercise the workbench rather than to
be easy:

* an identifier column (``customer_id``) and a date column (``signup_date``) that are
  useless to a model and have to be got rid of,
* a mix of numeric columns and categorical ones (``region``, ``plan``),
* missing values in three columns,
* a binary target stored as text (``yes``/``no``) rather than 0/1,
* and a signal that is real but noisy, so the model lands somewhere believable
  (around 0.80 ROC AUC) instead of perfect.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261017
ROWS = 4000


def customer_churn(n: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    """A synthetic subscription-churn table with an interpretable signal."""
    rng = np.random.default_rng(seed)

    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.45, 0.38, 0.17])
    base_charge = {"Basic": 29.0, "Standard": 59.0, "Premium": 99.0}
    monthly_charge = np.round(np.clip(
        np.array([base_charge[p] for p in plan]) + rng.normal(0, 6.0, n), 15, 160), 2)

    tenure_months = rng.integers(1, 73, n)
    age = np.clip(np.round(rng.normal(41, 12, n)), 18, 82)
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.30, 0.26, 0.24, 0.20])
    support_calls = rng.poisson(1.4, n)
    satisfaction = np.clip(np.round(rng.normal(3.6, 1.1, n)), 1, 5)

    # The signal the model is supposed to find.  It is deliberately *not* additive:
    # churn is driven by a handful of thresholds and conjunctions, which is how real
    # churn behaves (the risk jumps at the end of a contract, and it is the combination
    # of an unhappy customer who keeps ringing that really walks).  A signal this shape
    # is what makes gradient boosting beat plain logistic regression here, and it is
    # also why the interaction matters more than any single column.
    logit = (
        -1.80
        + 2.60 * ((support_calls >= 3) & (satisfaction <= 2))   # unhappy *and* ringing
        + 2.00 * (tenure_months <= 6)                            # still in the trial
        - 1.80 * ((tenure_months >= 30) & (satisfaction >= 4))   # long and content
        + 1.20 * (monthly_charge > 85.0)                         # on the expensive plan
        + 0.60 * (plan == "Basic")
        + 0.50 * (region == "South")
        + 0.25 * support_calls
        - 0.010 * (age - 41)
    )
    churned = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))

    start = np.datetime64("2023-01-01")
    signup_date = start + rng.integers(0, 365 * 3, n).astype("timedelta64[D]")

    frame = pd.DataFrame({
        "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
        "signup_date": pd.to_datetime(signup_date).strftime("%Y-%m-%d"),
        "region": region,
        "plan": plan,
        "age": age.astype(float),
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "support_calls": support_calls.astype(float),
        "satisfaction": satisfaction.astype(float),
        "churned": np.where(churned, "yes", "no"),
    })

    # Scatter a few holes in it: missing values are normal, and the Clean screen exists.
    for column, share in (("age", 0.030), ("satisfaction", 0.050), ("support_calls", 0.020)):
        holes = rng.random(n) < share
        frame.loc[holes, column] = np.nan
    return frame


# --------------------------------------------------------------------------
# the civil-engineering examples
# --------------------------------------------------------------------------
#
# The two files below exist because "a civil engineer never needs a custom Python
# script" is only credible against the kind of data a civil engineer actually has:
# a concrete-mix trial report with a specification limit to meet, and a pile that is
# being watched settle.  Both are seeded, both are committed, and the self-test
# asserts the shape of each so the manual can quote real numbers.

CONCRETE_ROWS = 1200
CONCRETE_SEED = 20261018
#: C30/37 is a 30 MPa cylinder / 37 MPa cube characteristic strength, so 37 MPa is the
#: number the trial report is judged against — which is exactly what makes the second
#: target (a yes/no "does this mix pass?") worth having.
CUBE_SPECIFICATION = 37.0

SETTLEMENT_WEEKS = 157
SETTLEMENT_SEED = 20261019


def concrete_mix(n: int = CONCRETE_ROWS, seed: int = CONCRETE_SEED) -> pd.DataFrame:
    """A concrete trial-mix table, with both a strength and a pass/fail target."""
    rng = np.random.default_rng(seed)

    # The ranges are the ordinary working range for a structural mix, and the column
    # headings match Webster-style mix-design reports so a real file can be dropped in
    # without renaming anything.
    cement = rng.uniform(150, 500, n)
    slag = np.clip(rng.normal(70, 65, n), 0, 300)
    ash = np.clip(rng.normal(45, 55, n), 0, 190)
    water = np.clip(130 + 0.09 * cement + rng.normal(0, 14, n), 120, 235)
    superplasticiser = np.clip(rng.normal(6.0, 4.0, n), 0, 26)
    fine = rng.uniform(600, 940, n)
    coarse = rng.uniform(810, 1140, n)
    age = rng.choice([1, 3, 7, 14, 28, 56, 90, 180], n).astype(float)
    curing = rng.choice(["air", "water", "steam"], n, p=[0.46, 0.39, 0.15])

    # Water/cement ratio governs strength more than anything else, the slag and ash
    # contribute late, and every real strength gain curve is a curve in log(age) — so
    # the "strength age" fit on screen 14 has something true to find.
    ratio = water / cement
    curing_bonus = np.where(curing == "water", 1.6, np.where(curing == "steam", 2.4, 0.0))
    strength = (
        17.0
        + 0.058 * cement
        + 0.030 * slag
        + 0.022 * ash
        - 0.140 * water
        + 0.0011 * fine
        + 0.0014 * coarse
        + 0.30 * superplasticiser
        + 13.0 * np.log1p(age)
        - 26.0 * ratio
        + curing_bonus
        + rng.normal(0, 3.4, n)
    )
    strength = np.clip(strength, 4.0, None)

    start = np.datetime64("2024-01-08")
    cast = start + rng.integers(0, 365 * 2, n).astype("timedelta64[D]")
    order = np.argsort(cast)

    frame = pd.DataFrame({
        "batch_id": [f"B-{i:05d}" for i in range(1, n + 1)],
        "cast_date": pd.to_datetime(cast).strftime("%Y-%m-%d"),
        "Cement": cement,
        "Blast furnace slag": slag,
        "Fly ash": ash,
        "Water": water,
        "Superplasticiser": superplasticiser,
        "Fine aggregate": fine,
        "Coarse aggregate": coarse,
        "Age (days)": age,
        "Curing method": curing,
        "Compressive strength": strength,
        "meets_C30_37": np.where(strength >= CUBE_SPECIFICATION, "yes", "no"),
    }).iloc[order].reset_index(drop=True)
    frame["batch_id"] = [f"B-{i:05d}" for i in range(1, n + 1)]
    frame = frame.round({"Cement": 1, "Blast furnace slag": 1, "Fly ash": 1, "Water": 1,
                         "Superplasticiser": 2, "Fine aggregate": 1, "Coarse aggregate": 1,
                         "Compressive strength": 2})
    for column, share in (("Water", 0.032), ("Fine aggregate", 0.015),
                          ("Curing method", 0.010), ("Blast furnace slag", 0.008)):
        frame.loc[rng.random(n) < share, column] = np.nan
    return frame


def pile_settlement(weeks: int = SETTLEMENT_WEEKS,
                    seed: int = SETTLEMENT_SEED) -> pd.DataFrame:
    """One instrumented pile watched weekly for three years.

    The story in the numbers is the one the monitoring screens are for.  A surcharge was
    placed in week 53, settlement moved as the load came on, the annual thermal cycle is
    visible on top of the long-term consolidation, and in week 118 something gave a
    sudden extra millimetre — which is the reading the control chart has to catch.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, weeks + 1)

    # The load history: nothing, embankment, surcharge, then surcharge removed back to
    # the service level.
    load = np.full(weeks, 2500.0)
    load[t < 13] = 0.0
    load[(t >= 13) & (t < 53)] = 1900.0
    load[(t >= 53) & (t < 131)] = 3300.0

    # Settlement as the sum of four consolidation curves, each starting when its load
    # stage starts: monotone, bounded, and slower the further along it is.
    def consolidate(start: int, amplitude: float, time_constant: float) -> np.ndarray:
        elapsed = np.clip(t - start, 0, None).astype(float)
        return amplitude * (1.0 - np.exp(-elapsed / time_constant))

    settlement = (
        consolidate(1, 4.5, 9.0)
        + consolidate(13, 3.0, 14.0)
        + consolidate(53, 4.0, 18.0)
        + consolidate(131, 0.6, 10.0)
        + 0.35 * np.sin(2.0 * np.pi * (t - 10) / 52.18)      # thermal movement
        + rng.normal(0, 0.12, weeks)
    )
    # Week 118: something gave, and the pile settled an extra 1.1 mm and *stayed* there.  A
    # permanent step, not a single bad reading — because that is what a control chart of the
    # weekly movement has to be able to tell apart from a misprint.
    settlement[117:] += 1.1

    pore = (48.0 + 0.030 * load + 6.0 * np.sin(2.0 * np.pi * (t - 16) / 52.18)
            + rng.normal(0, 2.5, weeks))
    groundwater = (3.15 + 0.85 * np.sin(2.0 * np.pi * (t + 20) / 52.18)
                   + rng.normal(0, 0.14, weeks))

    stage = np.where(t < 13, "instrumentation",
                     np.where(t < 53, "embankment",
                              np.where(t < 131, "surcharge", "service")))
    frame = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=weeks, freq="W-MON").strftime("%Y-%m-%d"),
        "settlement_mm": settlement,
        "load_kN": load,
        "pore_pressure_kPa": pore,
        "groundwater_level_m": groundwater,
        "stage": stage,
    }).round(2)
    # Three weeks when the settlement gauge was down, so the gap logic has real gaps in it.
    frame.loc[[64, 65, 111], "settlement_mm"] = np.nan
    return frame


def main():
    wrote = []

    churn = customer_churn()
    wrote.append(("customer_churn.csv", churn,
                  f"{(churn['churned'] == 'yes').mean():.1%} churned"))

    concrete = concrete_mix()
    wrote.append(("concrete_mix.csv", concrete,
                  f"{(concrete['meets_C30_37'] == 'yes').mean():.1%} meet C30/37, "
                  f"mean strength {concrete['Compressive strength'].mean():.2f} MPa"))

    settlement = pile_settlement()
    wrote.append(("pile_settlement_monitoring.csv", settlement,
                  f"settlement {settlement['settlement_mm'].min():.2f} to "
                  f"{settlement['settlement_mm'].max():.2f} mm, "
                  f"{settlement['settlement_mm'].isna().sum()} missed readings"))

    for name, frame, note in wrote:
        path = os.path.join(HERE, name)
        frame.to_csv(path, index=False)
        print(f"wrote {path}")
        print(f"  {len(frame):,} rows x {frame.shape[1]} columns, {note}, "
              f"{int(frame.isna().sum().sum())} missing values")


if __name__ == "__main__":
    main()
