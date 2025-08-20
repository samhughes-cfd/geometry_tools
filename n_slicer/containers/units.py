# n_slicer/containers/units.py
from dataclasses import dataclass

@dataclass
class SectionUnits:
    rR: str = "-"        # r/R (dimensionless)
    zL: str = "-"        # z/L (dimensionless)
    c: str = "m"         # chord length unit
    L: str = "m"         # blade length unit
    R: str = "m"         # blade radius unit
    XY: str = "m"        # output geometry unit (DXF coordinates)
    beta_deg: str = "deg"  # "deg" or "rad" for input twist

# --- simple converters --------------------------------------------------------

# Base is meters for lengths, degrees for angles
_LENGTH_TO_METERS = {
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "µm": 1e-6,
    "um": 1e-6,   # alias
    "km": 1e3,
    "in": 0.0254,
    "ft": 0.3048,
}

def length_scale(from_unit: str, to_unit: str) -> float:
    """
    Return the multiplicative factor to convert a value in `from_unit`
    into `to_unit` via meters.

    Example
    -------
    length_scale("m", "mm") -> 1000.0
    length_scale("mm", "m") -> 0.001
    """
    try:
        fm = _LENGTH_TO_METERS[from_unit]
        tm = _LENGTH_TO_METERS[to_unit]
    except KeyError as e:
        raise ValueError(f"Unknown length unit: {e.args[0]}")
    return fm / tm

_ANGLE_TO_DEGREES = {
    "deg": 1.0,
    "rad": 180.0 / 3.141592653589793,
}

def angle_convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert an angle between 'deg' and 'rad'."""
    if from_unit == to_unit:
        return float(value)
    if from_unit not in _ANGLE_TO_DEGREES or to_unit not in _ANGLE_TO_DEGREES:
        raise ValueError(f"Unknown angle unit: {from_unit} -> {to_unit}")
    # convert to degrees first
    in_deg = float(value) * _ANGLE_TO_DEGREES[from_unit]
    # then from degrees to target
    return in_deg / _ANGLE_TO_DEGREES[to_unit]
