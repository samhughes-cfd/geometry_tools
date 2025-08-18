# n_slicer/containers/sampling.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any
import numpy as np

Scheme = Literal["uniform", "cosine", "power_root", "power_tip", "custom"]
FitKind = Literal["pchip", "akima", "spline", "linear"]

@dataclass
class DiscretisationSpec:
    """How r/R was discretised."""
    n: int
    scheme: Scheme = "uniform"
    rR_min: float = 0.0
    rR_max: float = 1.0
    power: Optional[float] = None              # used by power_* schemes
    mapping_name: Optional[str] = None         # name/notes when a custom mapping is used
    rR_grid: np.ndarray = field(default_factory=lambda: np.empty(0))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rR_grid"] = d["rR_grid"].tolist()
        return d

@dataclass
class FittingSpec:
    """How chord(r/R) and twist(r/R) were fitted prior to sampling."""
    chord_fit: FitKind = "pchip"
    twist_fit: FitKind = "pchip"
    spline_smoothing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)