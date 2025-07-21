# containers/optimisation/optimisation_result_table.py

from dataclasses import dataclass, field
from typing import List
from containers.optimisation.optimisation_result_row import OptimisationResultRow


@dataclass
class OptimisationResultTable:
    rows: List[OptimisationResultRow] = field(default_factory=list)

    def compute_relative_changes(self):
        if not self.rows:
            return

        base_area = self.rows[0].area_mm2
        for row in self.rows:
            row.Jt_ratio = row.Jt_mm4 / row.target_Jt_mm4 if row.target_Jt_mm4 else None
            row.Iz_ratio = row.Iz_mm4 / row.target_Iz_mm4 if row.target_Iz_mm4 else None

            row.Area_reduction_pct = 100 * (1 - row.area_mm2 / base_area) if base_area else None

            row.Jt_efficiency = row.Jt_mm4 / row.area_mm2 if row.area_mm2 else None
            row.Iz_efficiency = row.Iz_mm4 / row.area_mm2 if row.area_mm2 else None

            row.Slack_Jt = max(0.0, row.Jt_mm4 - row.target_Jt_mm4)
            row.Excess_ratio_Jt = (
                row.Slack_Jt / row.target_Jt_mm4 if row.target_Jt_mm4 else None
            )

            row.Slack_Iz = max(0.0, row.Iz_mm4 - row.target_Iz_mm4)
            row.Excess_ratio_Iz = (
                row.Slack_Iz / row.target_Iz_mm4 if row.target_Iz_mm4 else None
            )

    def get_best_passed(self) -> OptimisationResultRow | None:
        """Return the first result that passes constraints with minimal area."""
        passed_rows = [r for r in self.rows if r.passed]
        if not passed_rows:
            return None
        return min(passed_rows, key=lambda r: r.area_mm2)
