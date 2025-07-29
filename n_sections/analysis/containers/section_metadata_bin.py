# containers/analysis/section_metadata_bin.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class MetadataBin:
    """
    Stores metadata associated with a single section (blade station) analysis run.

    Attributes:
        label (str): Section label (e.g., 'station_05').
        r (float): Absolute radial position of the section [mm or m].
        r_over_R (float): Non-dimensional radial position (r/R).
        B_r (float): Local twist at this section [degrees].
        Cx (float): X-coordinate of the section centroid [mm].
        Cy (float): Y-coordinate of the section centroid [mm].
        hs (List[float]): Mesh sizes used for this section in the convergence study [mm].
        dxf_path (Path): Absolute path to the section geometry DXF file.
    """
    label: str
    r: float
    r_over_R: float
    B_r: float
    Cx: float
    Cy: float
    hs: List[float]
    dxf_path: Path
