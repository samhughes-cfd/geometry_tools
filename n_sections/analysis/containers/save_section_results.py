# section_calc_n\containers\analysis\save_section_results.py

import pickle
import csv
from pathlib import Path
from n_sections.analysis.containers.section_analysis_results_set import SectionAnalysisResultSet


class SaveSectionResults:
    """
    Saves the full results of a section analysis to disk after convergence is complete.

    Outputs:
        - section_analysis_result.pkl : full pickled result set for later recovery
        - section_analysis_summary.csv : convergence results as a tabular CSV,
          with metadata included as a header block
    """

    def __init__(self, result_set: SectionAnalysisResultSet, output_dir: Path):
        self.result_set = result_set
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        self._save_as_pickle()
        self._save_as_csv()

    def _save_as_pickle(self):
        pkl_path = self.output_dir / "section_analysis_result.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.result_set, f)
        print(f"[✅] Full result set pickled to: {pkl_path}")

    def _save_as_csv(self):
        csv_path = self.output_dir / "section_analysis_summary.csv"
        convergence = self.result_set.convergence
        metadata = self.result_set.metadata

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # 1. Write metadata as a header block
            writer.writerow(["# Metadata"])
            for key, value in vars(metadata).items():
                writer.writerow([key, value])
            writer.writerow([])  # Blank line

            # 2. Check if we have convergence results
            if not convergence or not convergence.rows:
                writer.writerow(["# WARNING: No successful mesh results found — convergence data unavailable."])
                print(f"[⚠️] No convergence data. Metadata-only CSV saved to: {csv_path}")
                return

            # 3. Write convergence results as a table
            writer.writerow(["# Convergence Results"])
            fieldnames = list(vars(convergence.rows[0]).keys())
            writer.writerow(fieldnames)

            for row in convergence.rows:
                writer.writerow([getattr(row, field) for field in fieldnames])

        print(f"[✅] Convergence + metadata saved to CSV: {csv_path}")