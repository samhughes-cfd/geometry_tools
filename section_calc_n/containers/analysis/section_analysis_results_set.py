# containers\analysis\section_analysis_results_set.py

from dataclasses import dataclass
from typing import Optional
from containers.analysis.section_metadata_bin import MetadataBin
from containers.analysis.geometric_property_bin import GeometricPropertyBin

@dataclass
class SectionAnalysisResultSet:
    """
    Stores all result data from a single analysis pass (no optimisation).
    - MetadataBin: setup/geometry context
    - GeometricPropertyBin: result values from the section property calculator
    """
    metadata: Optional[MetadataBin] = None
    convergence: Optional[GeometricPropertyBin] = None