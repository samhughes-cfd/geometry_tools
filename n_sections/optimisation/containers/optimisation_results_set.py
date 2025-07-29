# containers/optimisation/optimisation_result_set.py

from dataclasses import dataclass
from n_sections.optimisation.containers.optimisation_metadata import OptimisationMetadata
from n_sections.optimisation.containers.optimisation_result_table import OptimisationResultTable

@dataclass
class OptimisationResultSet:
    metadata: OptimisationMetadata
    result_table: OptimisationResultTable
