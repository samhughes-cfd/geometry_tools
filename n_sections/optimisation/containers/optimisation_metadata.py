# containers/optimisation/optimisation_metadata.py

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OptimisationMetadata:
    label: str
    r: float
    r_over_R: float
    B_r: float
    Cx: float
    Cy: float
    mesh_h: float
    scale_factors: list[float]
    target_Jt_mm4: float
    target_Iz_mm4: float
    dxf_path: Path
