# containers\analysis\section_analysis_results_set.py

from dataclasses import dataclass
from typing import Optional
from n_sections.analysis.containers.section_metadata_bin import MetadataBin
from n_sections.analysis.containers.geometric_property_bin import GeometricPropertyBin

@dataclass
class SectionAnalysisResultSet:
    """
    Stores all result data from a single analysis pass (no optimisation).
    - MetadataBin: setup/geometry context
    - GeometricPropertyBin: result values from the section property calculator
    """
    metadata: Optional[MetadataBin] = None
    convergence: Optional[GeometricPropertyBin] = None