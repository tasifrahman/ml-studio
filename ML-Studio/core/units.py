"""Unit conversion for the quantities a civil engineer actually works in.

Every quantity has a base unit and a map of unit → factor-to-base, so a conversion is
``value * factor[from] / factor[to]``.  Temperature is the one exception and is handled
separately, because degrees Celsius and Fahrenheit are related by an offset as well as a
scale — getting that wrong is the classic silent error, so it is kept out of the
multiplicative table on purpose.  ``temperature_difference`` *is* linear and stays in the
table.

Nothing here imports Qt or anything heavy, so the module is usable on its own.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# quantities — unit name → factor to the base unit
# --------------------------------------------------------------------------

QUANTITIES: dict[str, dict] = {
    "length": {
        "base": "m", "label": "Length",
        "units": {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "µm": 1e-6,
                  "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mile": 1609.344,
                  "nautical mile": 1852.0},
    },
    "area": {
        "base": "m²", "label": "Area",
        "units": {"m²": 1.0, "mm²": 1e-6, "cm²": 1e-4, "ha": 1e4, "km²": 1e6,
                  "in²": 0.00064516, "ft²": 0.09290304, "yd²": 0.83612736,
                  "acre": 4046.8564224, "hectare": 1e4},
    },
    "volume": {
        "base": "m³", "label": "Volume",
        "units": {"m³": 1.0, "L": 1e-3, "mL": 1e-6, "cm³": 1e-6, "mm³": 1e-9,
                  "ft³": 0.028316846592, "in³": 1.6387064e-5, "yd³": 0.764554857984,
                  "gal (US)": 0.003785411784, "gal (UK)": 0.00454609,
                  "MGal (US)": 3785.411784},
    },
    "mass": {
        "base": "kg", "label": "Mass",
        "units": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "t (tonne)": 1000.0,
                  "lb": 0.45359237, "oz": 0.028349523125, "ton (US short)": 907.18474,
                  "ton (UK long)": 1016.0469088},
    },
    "force": {
        "base": "N", "label": "Force",
        "units": {"N": 1.0, "kN": 1e3, "MN": 1e6, "kgf": 9.80665, "tonnef": 9806.65,
                  "lbf": 4.4482216152605, "kip": 4448.2216152605, "dyn": 1e-5},
    },
    "pressure": {
        "base": "Pa", "label": "Pressure / stress / modulus",
        "units": {"Pa": 1.0, "hPa": 100.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
                  "N/mm²": 1e6, "N/m²": 1.0, "bar": 1e5, "mbar": 100.0,
                  "atm": 101325.0, "torr": 133.322368421, "mmHg": 133.322387415,
                  "psi": 6894.757293168, "ksi": 6.894757293168e6,
                  "ksf (kip/ft²)": 47880.2588889, "psf": 47.8802588889,
                  "inHg": 3386.389, "m H₂O": 9806.65, "mm H₂O": 9.80665,
                  "ft H₂O": 2989.06692},
    },
    "moment": {
        "base": "N·m", "label": "Moment / torque",
        "units": {"N·m": 1.0, "kN·m": 1e3, "MN·m": 1e6, "N·mm": 1e-3,
                  "kgf·m": 9.80665, "lbf·ft": 1.3558179483314,
                  "lbf·in": 0.11298482902762, "kip·ft": 1355.8179483314,
                  "kip·in": 112.98482902762, "tonf·m": 9806.65},
    },
    "unit_weight": {
        "base": "kN/m³", "label": "Unit weight / density",
        "units": {"kN/m³": 1.0, "N/m³": 1e-3, "kg/m³": 9.80665e-3,
                  "g/cm³": 9.80665, "t/m³": 9.80665, "Mg/m³": 9.80665,
                  "pcf (lb/ft³)": 0.157087464, "lb/yd³": 0.005817313,
                  "kip/ft³": 157.087464},
    },
    "mass_per_length": {
        "base": "kg/m", "label": "Mass per length",
        "units": {"kg/m": 1.0, "g/m": 1e-3, "t/m": 1e3, "lb/ft": 1.48816394357,
                  "lb/yd": 0.496054647856, "plf": 1.48816394357},
    },
    "energy": {
        "base": "J", "label": "Energy / work",
        "units": {"J": 1.0, "kJ": 1e3, "MJ": 1e6, "GJ": 1e9, "N·m": 1.0,
                  "kWh": 3.6e6, "cal": 4.184, "kcal": 4184.0,
                  "BTU": 1055.05585262, "ft·lbf": 1.3558179483314},
    },
    "power": {
        "base": "W", "label": "Power",
        "units": {"W": 1.0, "kW": 1e3, "MW": 1e6, "GW": 1e9, "hp (metric)": 735.49875,
                  "hp (mech)": 745.6998715823, "BTU/h": 0.29307107017},
    },
    "velocity": {
        "base": "m/s", "label": "Velocity",
        "units": {"m/s": 1.0, "mm/s": 1e-3, "cm/s": 1e-2, "km/h": 1 / 3.6,
                  "mph": 0.44704, "ft/s": 0.3048, "ft/min": 0.00508,
                  "in/min": 4.2333333333e-4, "km/day": 1000 / 86400,
                  "knot": 0.51444444444},
    },
    "acceleration": {
        "base": "m/s²", "label": "Acceleration",
        "units": {"m/s²": 1.0, "cm/s²": 0.01, "g (gravity)": 9.80665,
                  "ft/s²": 0.3048, "in/s²": 0.0254},
    },
    "volumetric_flow": {
        "base": "m³/s", "label": "Volumetric flow",
        "units": {"m³/s": 1.0, "m³/h": 1 / 3600, "m³/day": 1 / 86400,
                  "L/s": 1e-3, "L/min": 1e-3 / 60, "L/h": 1e-3 / 3600,
                  "ML/day": 1000 / 86400, "MGD": 0.0438126363889,
                  "ft³/s (cfs)": 0.028316846592, "gpm (US)": 6.30901964e-5,
                  "AF/day": 1233.48183755 / 86400},
    },
    "rainfall_intensity": {
        "base": "mm/h", "label": "Rainfall / infiltration intensity",
        "units": {"mm/h": 1.0, "mm/day": 1 / 24, "mm/min": 60.0,
                  "in/h": 25.4, "in/day": 25.4 / 24, "cm/h": 10.0,
                  "L/s/ha": 0.36},
    },
    "permeability": {
        "base": "m/s", "label": "Permeability / hydraulic conductivity",
        "units": {"m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3, "m/day": 1 / 86400,
                  "ft/day": 0.3048 / 86400, "ft/s": 0.3048,
                  "in/h": 0.0254 / 3600, "gal/day/ft²": 4.716e-7},
    },
    "inertia": {
        "base": "m⁴", "label": "Second moment of area (I)",
        "units": {"m⁴": 1.0, "cm⁴": 1e-8, "mm⁴": 1e-12,
                  "in⁴": 4.162314256e-7, "ft⁴": 0.0086309748412416},
    },
    "section_modulus": {
        "base": "m³", "label": "Section modulus (S = I/y)",
        "units": {"m³": 1.0, "cm³": 1e-6, "mm³": 1e-9, "in³": 1.6387064e-5,
                  "ft³": 0.028316846592},
    },
    "angle": {
        "base": "rad", "label": "Angle",
        "units": {"rad": 1.0, "deg": math.pi / 180, "grad": math.pi / 200,
                  "minute (arc)": math.pi / 10800, "second (arc)": math.pi / 648000,
                  "revolution": 2 * math.pi},
    },
    "slope": {
        "base": "ratio (V/H)", "label": "Slope / gradient",
        "units": {"ratio (V/H)": 1.0, "percent (%)": 0.01,
                  "permille (‰)": 0.001, "1 in n": 1.0,
                  "degree": math.tan(math.pi / 180)},
    },
    "time": {
        "base": "s", "label": "Time",
        "units": {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0,
                  "day": 86400.0, "week": 604800.0, "year (365 d)": 31536000.0,
                  "month (30 d)": 2592000.0},
    },
    "concentration": {
        "base": "mg/L", "label": "Concentration (water quality)",
        "units": {"mg/L": 1.0, "µg/L": 1e-3, "g/L": 1e3, "g/m³": 1.0,
                  "mg/m³": 1e-3, "ppm": 1.0, "ppb": 1e-3, "ng/L": 1e-6,
                  "kg/m³": 1e3, "mol/L": 1.0},
    },
    "temperature_difference": {
        "base": "°C", "label": "Temperature difference (linear)",
        "units": {"°C": 1.0, "K": 1.0, "°F": 5 / 9, "R": 5 / 9},
    },
}

#: Affine quantities: ``to_base(v) = v * a + b``.  Kept out of the table above.
AFFINE: dict[str, dict] = {
    "temperature": {
        "base": "°C", "label": "Temperature",
        "units": {"°C": (1.0, 0.0), "K": (1.0, -273.15),
                  "°F": (5 / 9, -160 / 9), "R": (5 / 9, -273.15)},
    },
}

#: Physical constants the workbench quotes where relevant.
CONSTANTS: list[tuple[str, float, str]] = [
    ("Standard gravity, g", 9.80665, "m/s²"),
    ("Unit weight of water, γw (4 °C)", 9.80665, "kN/m³"),
    ("Density of water, ρw (4 °C)", 1000.0, "kg/m³"),
    ("Standard atmosphere", 101325.0, "Pa"),
    ("Stefan–Boltzmann constant, σ", 5.670374419e-8, "W/(m²·K⁴)"),
    ("Coefficient of thermal expansion, concrete (typical)", 1.0e-5, "1/°C"),
    ("Coefficient of thermal expansion, steel", 1.2e-5, "1/°C"),
    ("Young's modulus, steel (typical)", 200000.0, "MPa"),
    ("Young's modulus, normal concrete (typical)", 30000.0, "MPa"),
    ("Poisson's ratio, concrete (typical)", 0.20, "–"),
    ("Poisson's ratio, steel (typical)", 0.30, "–"),
    ("Specific heat capacity of water", 4186.0, "J/(kg·K)"),
    ("Latent heat of fusion of water", 334000.0, "J/kg"),
    ("Speed of sound in air (20 °C)", 343.0, "m/s"),
    ("Earth radius (mean)", 6371000.0, "m"),
]


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------

def quantities() -> list[str]:
    """Every quantity name, temperature included, in a stable order."""
    names = list(QUANTITIES)
    names.extend(AFFINE)
    return sorted(names)


def is_affine(quantity: str) -> bool:
    return quantity in AFFINE


def _table(quantity: str) -> dict:
    if quantity in AFFINE:
        return AFFINE[quantity]
    if quantity in QUANTITIES:
        return QUANTITIES[quantity]
    raise KeyError(f"unknown quantity: {quantity!r}")


def units_for(quantity: str) -> list[str]:
    """The unit names available for a quantity, base unit first."""
    table = _table(quantity)
    names = list(table["units"])
    base = table["base"]
    if base in names:
        names.remove(base)
        names.insert(0, base)
    return names


def base_unit(quantity: str) -> str:
    return _table(quantity)["base"]


def label(quantity: str) -> str:
    return _table(quantity).get("label", quantity)


def convert(value: float, from_unit: str, to_unit: str, quantity: str) -> float:
    """Convert ``value`` between two units of the same quantity.

    Raises ``KeyError`` for an unknown quantity or unit, and ``ValueError`` for a
    non-finite value — a silent NaN here would travel a long way before anyone noticed.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot convert a non-finite value ({value})")

    if quantity in AFFINE:
        units = AFFINE[quantity]["units"]
        if from_unit not in units or to_unit not in units:
            raise KeyError(f"{from_unit!r} or {to_unit!r} is not a unit of {quantity}")
        a_from, b_from = units[from_unit]
        a_to, b_to = units[to_unit]
        base = value * a_from + b_from
        return (base - b_to) / a_to

    table = _table(quantity)
    units = table["units"]
    if from_unit not in units or to_unit not in units:
        raise KeyError(f"{from_unit!r} or {to_unit!r} is not a unit of {quantity}")
    return value * units[from_unit] / units[to_unit]


def convert_frame(frame, columns: dict, quantity: str, to_unit: str):
    """Convert several columns of a frame, and rename them to the target unit.

    ``columns`` maps an existing column to the unit it is currently in, so a client can
    declare "this column is in psi" without renaming anything first.  Returns a new
    frame; the original is untouched.
    """
    out = frame.copy()
    for column, from_unit in columns.items():
        if column not in out.columns:
            continue
        out[column] = [convert(v, from_unit, to_unit, quantity) if v == v else v
                       for v in out[column]]
        target = f"{column} ({to_unit})"
        if target != column:
            out = out.rename(columns={column: target})
    return out


def factors_frame(quantity: str):
    """A tidy table of every unit in a quantity and its factor to the base unit.

    This is what the Units screen shows, so the arithmetic is visible rather than
    hidden behind a box that says "trust me".
    """
    import pandas as pd

    table = _table(quantity)
    rows = []
    if quantity in AFFINE:
        for name, (a, b) in table["units"].items():
            rows.append({"unit": name, "to base": f"v × {a:g} + ({b:g})" if b else f"v × {a:g}",
                         "base unit": table["base"]})
    else:
        for name, factor in table["units"].items():
            rows.append({"unit": name, "factor to base": factor, "base unit": table["base"]})
    order = units_for(quantity)
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["unit"].map({u: i for i, u in enumerate(order)})
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def format_value(value: float, digits: int = 6) -> str:
    """A short, readable rendering: no trailing zeros, scientific only when needed."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e7):
        return f"{value:.{max(1, digits - 3)}e}"
    text = f"{value:.{digits}g}"
    return text
