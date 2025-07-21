# containers/optimisation/optimisation_result_set.py

from dataclasses import dataclass
from containers.optimisation.optimisation_metadata import OptimisationMetadata
from containers.optimisation.optimisation_result_table import OptimisationResultTable

@dataclass
class OptimisationResultSet:
    metadata: OptimisationMetadata
    result_table: OptimisationResultTable
