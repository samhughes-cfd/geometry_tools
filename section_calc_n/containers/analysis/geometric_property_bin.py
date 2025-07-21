# containers\analysis\geometric_property_bin.py

from dataclasses import dataclass, field
from typing import List
from .geometric_property_analysis_bin import GeometricPropertyAnalysisBin

@dataclass
class GeometricPropertyBin:
    rows: List[GeometricPropertyAnalysisBin] = field(default_factory=list)
