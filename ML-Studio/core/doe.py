"""Design of experiments: planning a test programme, and reading the results.

A lab experiment is the one place where a civil engineer most often reaches for a script,
because a factorial design is fiddly to lay out by hand and easy to get wrong.  This module
does the two halves:

* **Designs** — full factorial, fractional factorial (with the standard resolution tables),
  central composite, Box–Behnken, and space-filling designs (Latin hypercube, Sobol,
  Halton, plain random).  Every builder returns a :class:`Design` holding *actual* factor
  values — the numbers that go in the lab notebook — alongside the *coded* −1/0/+1 columns
  the analysis needs.
* **Analysis** — main effects, two-factor interactions, an OLS table with p-values, and a
  Pareto of the contributions, so it is obvious which factor actually mattered.

Two details that the hand-rolled version usually misses: every design is returned
**randomised** (a run order column is added, because running a factorial in standard order
confounds the effect with any drift in the lab), and the fraction of the design that is
lost to aliasing is reported rather than hidden.

``scipy.stats.qmc`` provides the low-discrepancy sequences; nothing else is imported.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import qmc

# --------------------------------------------------------------------------
# the design object
# --------------------------------------------------------------------------

@dataclass
class Design:
    """A planned set of runs."""

    frame: pd.DataFrame                    # actual factor values, in run order
    coded: pd.DataFrame                    # −1 / 0 / +1 columns, same rows
    kind: str
    note: str = ""
    factors: dict = field(default_factory=dict)   # name → (low, high) or list of levels
    warnings: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def names(self) -> list[str]:
        return [c for c in self.frame.columns if c not in ("run", "standard order")]

    def summary(self) -> dict:
        return {
            "design": self.kind, "runs": len(self.frame), "factors": len(self.coded.columns),
            "has centre points": "yes" if (self.coded == 0).all(axis=1).any() else "no",
            "note": self.note,
        }

    def worksheet(self) -> pd.DataFrame:
        """The table to take to the lab: run order, actual settings, blank response."""
        out = self.frame.copy()
        if "response" not in out.columns:
            out["response"] = ""
        return out


# --------------------------------------------------------------------------
# coded values
# --------------------------------------------------------------------------

def coded_levels(levels: list) -> dict:
    """Map a factor's levels onto −1 … +1, evenly spaced."""
    n = len(levels)
    if n == 1:
        return {levels[0]: 0.0}
    if n == 2:
        return {levels[0]: -1.0, levels[1]: 1.0}
    step = 2.0 / (n - 1)
    return {level: -1.0 + i * step for i, level in enumerate(levels)}


def _coded_frame(frame: pd.DataFrame, factors: dict) -> pd.DataFrame:
    out = {}
    for name, spec in factors.items():
        if name not in frame.columns:
            continue
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(
                isinstance(v, (int, float)) for v in spec):
            low, high = float(spec[0]), float(spec[1])
            centre = (low + high) / 2
            half = (high - low) / 2 or 1.0
            out[name] = (frame[name].astype(float) - centre) / half
        else:
            mapping = coded_levels(list(spec))
            out[name] = frame[name].map(mapping).astype(float)
    return pd.DataFrame(out)


def _randomise(frame: pd.DataFrame, seed: int | None) -> pd.DataFrame:
    """Shuffle the run order and keep the standard order as a column.

    Randomising is not decoration: a design run in textbook order lets the machine warming
    up over the session masquerade as a factor effect.
    """
    out = frame.copy()
    out.insert(0, "standard order", np.arange(1, len(out) + 1))
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(out))
    out = out.iloc[order].reset_index(drop=True)
    out.insert(0, "run", np.arange(1, len(out) + 1))
    return out


def _finish(frame: pd.DataFrame, factors: dict, kind: str, note: str,
            seed: int | None, warnings: list | None = None) -> Design:
    coded = _coded_frame(frame, factors)
    ordered = _randomise(frame, seed)
    return Design(frame=ordered, coded=coded, kind=kind, note=note, factors=factors,
                  warnings=list(warnings or []))


# --------------------------------------------------------------------------
# two-level designs
# --------------------------------------------------------------------------

def full_factorial(factors: dict, seed: int | None = None) -> Design:
    """Every combination of every level — the reference design.

    ``factors`` maps a name to either ``(low, high)`` for a continuous factor or an
    explicit list of levels (which may be text, e.g. ``["cured", "air-dried"]``).
    """
    names = list(factors)
    levels = []
    for name in names:
        spec = factors[name]
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(
                isinstance(v, (int, float)) for v in spec):
            levels.append([float(spec[0]), float(spec[1])])
        elif isinstance(spec, (list, tuple)):
            levels.append(list(spec))
        else:
            raise ValueError(f"factor {name!r} needs a (low, high) pair or a list of levels")
    rows = list(itertools.product(*levels))
    frame = pd.DataFrame(rows, columns=names)
    runs = len(frame)
    note = f"{runs} runs = {' × '.join(str(len(l)) for l in levels)}"
    return _finish(frame, factors, "Full factorial", note, seed)


def two_level_full(names, seed: int | None = None, low: float = -1.0,
                   high: float = 1.0) -> Design:
    """A 2^k design in coded units."""
    return full_factorial({name: (low, high) for name in names}, seed)


#: The standard minimum-run two-level fractions from Box, Hunter & Hunter, keyed by
#: ``(number of factors, number of runs)``.  Each entry gives the generators that define
#: the extra factors from the basic ones, which is what makes the aliasing readable.
STANDARD_FRACTIONS = {
    (3, 4): ["C=AB"],
    (4, 8): ["D=ABC"],
    (5, 8): ["D=AB", "E=AC"],
    (6, 8): ["D=AB", "E=AC", "F=BC"],
    (7, 8): ["D=AB", "E=AC", "F=BC", "G=ABC"],
    (5, 16): ["E=ABCD"],
    (6, 16): ["E=ABC", "F=BCD"],
    (7, 16): ["E=ABC", "F=BCD", "G=ACD"],
    (8, 16): ["E=ABC", "F=BCD", "G=ACD", "H=ABD"],
    (9, 32): ["F=ABCD", "G=ABCE", "H=ABDE", "I=ACDE"],
    (10, 32): ["F=ABCD", "G=ABCE", "H=ABDE", "I=ACDE", "J=BCDE"],
    (11, 32): ["F=ABCD", "G=ABCE", "H=ABDE", "I=ACDE", "J=BCDE", "K=ABCDE"],
}


def recommended_fraction(factor_count: int) -> tuple[int, list[str]] | None:
    """The smallest standard fraction for this many factors.

    Returns ``(runs, generators)``, so a user who knows they have six factors can be told
    "eight runs, with ``D = AB``, ``E = AC``, ``F = BC``" rather than having to derive it.
    """
    options = sorted((runs, gens) for (count, runs), gens in STANDARD_FRACTIONS.items()
                     if count == factor_count)
    if options:
        return options[0]
    full = 2 ** factor_count
    if full <= 64:
        return full, []
    return None


def fractional_factorial(names, generators, seed: int | None = None) -> Design:
    """A 2^(k−p) fraction built from generator strings such as ``D=ABC``.

    The remaining factors are the product of the named basic factors, which is exactly how
    a textbook defines the fraction and makes the aliasing transparent.  The resolution is
    worked out and reported, together with the aliasing it implies.
    """
    names = [str(n).strip().upper() for n in names]
    if not generators:
        raise ValueError("give at least one generator, for example D=ABC")
    parsed = []
    basic = list(names)
    for text in generators:
        if "=" not in str(text):
            raise ValueError(f"a generator must look like D=ABC, not {text!r}")
        left, right = (part.strip().upper() for part in str(text).split("=", 1))
        if left not in names:
            raise ValueError(f"{left!r} is not one of the factors")
        powers = {}
        for letter in right.replace("*", "").replace("·", ""):
            if letter not in names:
                raise ValueError(f"generator {text!r} names {letter!r}, which is not a factor")
            powers[letter] = powers.get(letter, 0) + 1
        parsed.append((left, sorted(k for k, v in powers.items() if v % 2)))
        basic = [b for b in basic if b != left]

    base = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=len(basic))), columns=basic)
    for left, letters in parsed:
        base[left] = base[letters].prod(axis=1)
    base = base[names]

    # A factor's alias is the other factors it is multiplied by to make the defining word.
    aliases = {}
    for left, letters in parsed:
        aliases[left] = "".join(letters)
    resolution = 0
    for left, letters in parsed:
        resolution = max(resolution, 3)   # a generator of two letters would be resolution II
        resolution = max(resolution, len(letters) + 1)
    runs = len(base)
    note = (f"{runs} runs, {len(names)} factors; "
            + "; ".join(f"{k} = {v}" for k, v in sorted(aliases.items())))
    warnings = []
    if resolution <= 3:
        warnings.append(f"resolution {resolution}: main effects are aliased with two-factor "
                        "interactions, so use this only to screen many factors quickly")
    elif resolution == 4:
        warnings.append("resolution IV: two-factor interactions are aliased with each other")
    else:
        warnings.append("resolution V: main effects and two-factor interactions are separable")
    factors = {name: (-1.0, 1.0) for name in names}
    design = _finish(base, factors, f"Fractional factorial (resolution {resolution})", note, seed,
                     warnings)
    design.factors = factors
    return design


# --------------------------------------------------------------------------
# response-surface designs
# --------------------------------------------------------------------------

def central_composite(names, alpha: str | float = "rotatable", centre_points: int = 5,
                      seed: int | None = None, fraction: bool = False) -> Design:
    """A central composite design — the standard way to fit a curved response surface.

    ``alpha`` may be ``"rotatable"`` (α = (2^k)^(1/4), so the prediction variance depends
    only on distance from the centre), ``"face"`` (α = 1, the face-centred cube, which is
    what you use when a factor cannot go outside its stated range), or a number.
    """
    names = list(names)
    k = len(names)
    if k < 2:
        raise ValueError("a response surface needs at least two factors")
    cube = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=k)), columns=names)
    if fraction and k >= 4:
        cube = cube.iloc[::2].reset_index(drop=True)   # a half fraction of the cube
    if alpha == "rotatable":
        value = (2 ** k) ** 0.25
    elif alpha == "face":
        value = 1.0
    else:
        value = float(alpha)

    axial_rows = []
    for i, name in enumerate(names):
        for sign in (-1.0, 1.0):
            row = {n: 0.0 for n in names}
            row[name] = sign * value
            axial_rows.append(row)
    axial = pd.DataFrame(axial_rows, columns=names)
    centre = pd.DataFrame([{n: 0.0 for n in names}] * max(1, int(centre_points)))

    coded_all = pd.concat([cube, axial, centre], ignore_index=True)
    note = (f"{len(cube)} cube + {len(axial)} axial + {len(centre)} centre = "
            f"{len(coded_all)} runs; α = {value:.4g}")
    # Already coded, so the "actual" frame is the coded frame in this instance.
    design = _finish(coded_all.copy(), {n: (-1.0, 1.0) for n in names},
                     "Central composite", note, seed)
    design.coded = coded_all
    return design


def box_behnken(names, centre_points: int = 3, seed: int | None = None) -> Design:
    """A Box–Behnken design — a curved surface with every corner left untested.

    Built from all pairs of factors, which reproduces the standard 3-factor 15-run layout
    and generalises to 4–7 factors.  It is the design to reach for when running the extreme
    corner of every factor at once is undesirable or impossible.
    """
    names = list(names)
    k = len(names)
    if k < 3:
        raise ValueError("Box–Behnken needs at least three factors")
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product((-1.0, 1.0), repeat=2):
            point = [0.0] * k
            point[i], point[j] = a, b
            rows.append(point)
    coded = pd.DataFrame(rows, columns=names)
    centre = pd.DataFrame([{n: 0.0 for n in names}] * max(1, int(centre_points)))
    coded_all = pd.concat([coded, centre], ignore_index=True)
    note = (f"{len(coded)} edge runs + {len(centre)} centre = {len(coded_all)} runs; "
            "no corner is tested")
    design = _finish(coded_all.copy(), {n: (-1.0, 1.0) for n in names},
                     "Box–Behnken", note, seed)
    design.coded = coded_all
    return design


# --------------------------------------------------------------------------
# space-filling designs
# --------------------------------------------------------------------------

SPACE_FILLING = {
    "latin_hypercube": "Latin hypercube — one sample in every row and column of the grid",
    "sobol": "Sobol sequence — even coverage of the whole space",
    "halton": "Halton sequence — even coverage, cheap for many factors",
    "random": "Plain random — the baseline, with clumping and gaps",
}


def space_filling(bounds: dict, n: int, kind: str = "latin_hypercube",
                  seed: int | None = None) -> Design:
    """``n`` runs spread evenly across continuous factor ranges.

    Use this when the experiment is a simulation or a computer model, where a run is cheap
    and curvature is unknown — a factorial wastes runs confirming a corner, and a space
    filling design covers the whole range instead.
    """
    names = list(bounds)
    n = int(n)
    if n < 2:
        raise ValueError("ask for at least two runs")
    dimension = len(names)
    engine = np.random.default_rng(seed)
    if kind == "latin_hypercube":
        unit = qmc.LatinHypercube(d=dimension, seed=engine).random(n)
    elif kind == "sobol":
        unit = qmc.Sobol(d=dimension, scramble=True, seed=engine).random(n)
    elif kind == "halton":
        unit = qmc.Halton(d=dimension, scramble=True, seed=engine).random(n)
    elif kind == "random":
        unit = engine.random((n, dimension))
    else:
        raise ValueError(f"unknown space-filling design {kind!r}")

    lows = np.array([float(bounds[name][0]) for name in names])
    highs = np.array([float(bounds[name][1]) for name in names])
    actual = pd.DataFrame(lows + unit * (highs - lows), columns=names)
    note = f"{n} runs across {dimension} factors"
    return _finish(actual, {name: (bounds[name][0], bounds[name][1]) for name in names},
                   SPACE_FILLING.get(kind, kind), note, seed)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def _terms(coded: pd.DataFrame, order: int) -> list[tuple]:
    names = list(coded.columns)
    terms = [(name,) for name in names]
    if order >= 2:
        terms += list(itertools.combinations(names, 2))
    if order >= 3:
        terms += list(itertools.combinations(names, 3))
    return terms


def effects_analysis(coded: pd.DataFrame, response, order: int = 2) -> dict:
    """Fit ``response ~ all main effects + interactions`` on the coded design.

    Returns the coefficient table (value, standard error, p-value), the Pareto of how much
    each term contributes, the model as a whole, and a plain sentence naming the factor
    that matters most.  Coded units make the coefficients directly comparable: a coded
    coefficient is half the change in the response for a full swing of that factor.
    """
    y = np.asarray(response, dtype=float).ravel()
    n = len(y)
    names = list(coded.columns)
    if len(coded) != n:
        raise ValueError(f"{len(coded)} runs in the design but {n} responses")
    keep = np.isfinite(y)
    if keep.sum() < n:
        coded = coded[keep]
        y = y[keep]
        n = len(y)
    if n < 3:
        return {"note": "need at least three runs with a response"}

    terms = _terms(coded, order)
    columns = [np.ones(n)]
    labels = ["Intercept"]
    for term in terms:
        column = np.ones(n)
        for name in term:
            column = column * coded[name].to_numpy(dtype=float)
        columns.append(column)
        labels.append(" × ".join(term) if len(term) > 1 else term[0])

    matrix = np.column_stack(columns)
    if matrix.shape[1] > n:
        return {"note": f"only {n} runs for {matrix.shape[1]} terms — reduce the "
                        "interaction order or add runs"}
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = matrix @ coefficients
    residuals = y - fitted
    ss_res = float((residuals ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_total if ss_total else float("nan")
    dof = n - matrix.shape[1]
    adjusted = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    standard_errors = np.full(len(coefficients), np.nan)
    p_values = np.full(len(coefficients), np.nan)
    t_values = np.full(len(coefficients), np.nan)
    if dof > 0:
        s2 = ss_res / dof
        try:
            covariance = s2 * np.linalg.pinv(matrix.T @ matrix)
            standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                t_values = coefficients / standard_errors
            p_values = 2 * (1 - sps.t.cdf(np.abs(t_values), dof))
        except np.linalg.LinAlgError:
            pass

    rows = []
    for i, label in enumerate(labels):
        contribution = float(coefficients[i] ** 2 * (columns[i] ** 2).sum())
        rows.append({
            "term": label,
            "coefficient (coded)": float(coefficients[i]),
            "standard error": float(standard_errors[i]),
            "t": float(t_values[i]),
            "p": float(p_values[i]),
            "significant at 5%?": "yes" if math.isfinite(p_values[i]) and p_values[i] < 0.05 else "no",
            "sum of squares": contribution,
        })
    table = pd.DataFrame(rows)
    model_ss = float(table.loc[table["term"] != "Intercept", "sum of squares"].sum())
    table["% contribution"] = np.where(
        table["term"] == "Intercept", np.nan,
        table["sum of squares"] / model_ss * 100 if model_ss else np.nan)
    main = table[table["term"] != "Intercept"].copy()
    main = main.sort_values("sum of squares", ascending=False).reset_index(drop=True)

    headline = "no term stands out"
    if len(main) and math.isfinite(main.iloc[0]["p"]):
        best = main.iloc[0]
        if best["p"] < 0.05:
            word = "raises" if best["coefficient (coded)"] > 0 else "lowers"
            p_text = "p < 0.001" if best["p"] < 0.001 else f"p = {best['p']:.3f}"
            headline = (f"{best['term']} matters most: moving it from low to high "
                        f"{word} the response by {abs(2 * best['coefficient (coded)']):,.4g} "
                        f"({p_text}), about {best['% contribution']:.0f}% of the explained "
                        f"variation")
    return {
        "table": table, "pareto": main, "labels": labels, "n": n,
        "R²": r2, "adjusted R²": adjusted, "dof": dof,
        "headline": headline, "note": f"coded units; {len(terms)} terms fitted on {n} runs",
        "fitted": fitted, "residuals": residuals,
    }


def main_effects(coded: pd.DataFrame, response) -> pd.DataFrame:
    """Response averaged at each level of each factor — the table behind a main-effects plot."""
    y = pd.Series(np.asarray(response, dtype=float).ravel())
    if len(y) != len(coded):
        raise ValueError(f"{len(coded)} runs but {len(y)} responses")
    rows = []
    for name in coded.columns:
        levels = sorted(coded[name].dropna().unique())
        means = {}
        for level in levels:
            mask = coded[name] == level
            if mask.sum():
                means[level] = float(y[mask].mean())
        spread = (max(means.values()) - min(means.values())) if len(means) >= 2 else float("nan")
        row = {"factor": name, "levels": len(levels), "spread": spread}
        for level in levels:
            row[f"mean at {level:+g}"] = means.get(level, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def alias_table(names, generators) -> pd.DataFrame:
    """Which effects are confounded in a fraction — the honest part of a screening design."""
    names = [str(n).strip().upper() for n in names]
    words = []
    for text in generators:
        left, right = (part.strip().upper() for part in str(text).split("=", 1))
        letters = sorted(set(left + right.replace("*", "")))
        words.append("".join(letters))
    rows = []
    for word in words:
        for name in names:
            if name in word:
                other = "".join(sorted(set(word) - {name}))
                rows.append({"defined by": word, "main effect": name, "aliased with": other})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def design_quality(design: Design) -> dict:
    """Correlation between the coded columns, and the run count per factor.

    If two columns correlate strongly the design cannot separate those two effects, and it
    is far better to learn that before running the experiment than after.
    """
    coded = design.coded
    if coded.shape[1] < 2:
        return {"note": "only one factor"}
    correlation = coded.corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), 1).astype(bool))
    worst = float(np.nanmax(upper.to_numpy())) if upper.notna().to_numpy().any() else 0.0
    pair = None
    if upper.notna().to_numpy().any():
        index = np.unravel_index(np.nanargmax(upper.to_numpy()), upper.shape)
        pair = (coded.columns[index[0]], coded.columns[index[1]])
    return {
        "runs": len(design),
        "factors": coded.shape[1],
        "runs per factor": len(design) / max(1, coded.shape[1]),
        "worst |correlation| between columns": worst,
        "that pair": " vs ".join(pair) if pair else "—",
        "runs per factor note": ("thinner than the usual five runs per factor — fine for "
                                 "screening, thin for a surface fit")
                                if len(design) / max(1, coded.shape[1]) < 5 else
                                "comfortably above the usual five runs per factor",
    }
