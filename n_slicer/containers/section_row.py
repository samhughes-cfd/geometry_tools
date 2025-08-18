# n_slicer/containers/section_row.py
"""
SectionRow
----------
One fully self-descriptive blade section.

Stores:
- Spanwise coordinates (rR/xL) and local properties (c, beta_deg).
- Transform settings that produced XY from XY_in.
- Geometry arrays (XY_in normalised, XY transformed).
- Properties:
    - 'norm'   : NormalisedProperties computed once on XY_in.
    - 'scaled' : ScaledProperties with both analytic-from-normalised and exact-on-XY sets.
- Centre of Pressure (CoP): set via a chordwise fraction (0..1).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, Any

from .units import SectionUnits
from .properties import NormalisedProperties, ScaledProperties

SourceType = Literal["airfoil_points", "existing_dxf"]

@dataclass
class SectionRow:
    # ---------- required (non-default) ----------
    station_idx: int
    rR: float
    c: float
    beta_deg: float

    # ---------- optional (defaulted) ----------
    name: Optional[str] = None
    xL: Optional[float] = None

    # blade context & units
    L: Optional[float] = None
    units: SectionUnits = field(default_factory=SectionUnits)

    # provenance (how XY was produced)
    source: SourceType = "airfoil_points"
    units_scale: float = 1.0
    pivot_xc: float = 0.25
    pivot_yc: float = 0.0
    twist_sign: int = 1
    keep_pivot_in_place: bool = False

    # geometry
    XY_in: Optional["np.ndarray"] = field(default=None, repr=False)  # type: ignore[name-defined]
    XY:     Optional["np.ndarray"] = field(default=None, repr=False)  # type: ignore[name-defined]
    dxf_path: Optional[str] = None

    # properties
    norm: Optional[NormalisedProperties] = None
    scaled: Optional[ScaledProperties] = None

    # convenience
    def compute_normalised_props(self) -> None:
        """Compute NormalisedProperties on XY_in (normalised airfoil)."""
        if self.XY_in is None or self.XY_in.size == 0:
            raise ValueError("XY_in is empty; cannot compute normalised properties.")
        self.norm = NormalisedProperties.compute(self.XY_in)

    def meta(self) -> Dict[str, Any]:
        """Manifest-friendly dict (arrays omitted)."""
        d = asdict(self)
        d.pop("XY_in", None); d.pop("XY", None)
        return d
