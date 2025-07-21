# section_calc_n\containers\optimisation\save_optimisation_results.py

import pickle
import csv
from pathlib import Path
from containers.optimisation.optimisation_results_set import OptimisationResultSet

class SaveOptimisationResults:
    """
    Saves the full results of a section optimisation to disk after one or more scale factor runs.

    Outputs:
        - optimisation_result.pkl : full pickled result set
        - optimisation_summary.csv : result table + metadata
    """

    def __init__(self, result_set: OptimisationResultSet, output_dir: Path):
        self.result_set = result_set
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        self._save_as_pickle()
        self._save_as_csv()

    def _save_as_pickle(self):
        pkl_path = self.output_dir / "optimisation_result.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.result_set, f)
        print(f"[✅] Full optimisation result set pickled to: {pkl_path}")

    def _save_as_csv(self):
        csv_path = self.output_dir / "optimisation_summary.csv"
        metadata = self.result_set.metadata
        table = self.result_set.result_table  # <-- FIXED LINE

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # 1. Write metadata as a header block
            writer.writerow(["# Metadata"])
            for key, value in vars(metadata).items():
                writer.writerow([key, value])
            writer.writerow([])

            # 2. Optimisation Table
            if not table.rows:
                writer.writerow(["# WARNING: No optimisation results found."])
                print(f"[⚠️] Metadata-only CSV saved (no optimisation results): {csv_path}")
                return

            writer.writerow(["# Optimisation Results"])
            fieldnames = list(vars(table.rows[0]).keys())
            writer.writerow(fieldnames)

            for row in table.rows:
                writer.writerow([getattr(row, field) for field in fieldnames])

        print(f"[✅] Optimisation results + metadata saved to CSV: {csv_path}")
