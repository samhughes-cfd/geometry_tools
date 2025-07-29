# process_section_analysis.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Geometry utilities
from n_sections.geometry_utils.raw_geometry import RawGeometry
from n_sections.geometry_utils.processed_geometry import ProcessedGeometry

# Visualisation
from n_sections.visualisation.raw_geometry_visualisation import RawGeometryVisualisation
from n_sections.visualisation.processed_geometry_visualisation import ProcessedGeometryVisualisation
from n_sections.visualisation.convergence_visualisation import ConvergenceVisualisation

# Result containers
from n_sections.analysis.convergence.convergence_study import ConvergenceStudy
from n_sections.analysis.containers.section_analysis_results_set import SectionAnalysisResultSet
from n_sections.analysis.containers.section_metadata_bin import MetadataBin
from n_sections.analysis.containers.save_section_results import SaveSectionResults

class ProcessSectionAnalysis:
    def __init__(
        self,
        dxf: Path,
        label: str,
        r: float,
        r_over_R: float,
        B_r: float,
        Cx: float,
        Cy: float,
        hs: np.ndarray,
        results_dir: Path,
        logs_dir: Path,
    ):
        self.dxf = dxf
        self.label = label
        self.r = r
        self.r_over_R = r_over_R
        self.B_r = B_r
        self.Cx = Cx
        self.Cy = Cy
        self.hs = hs

        self.section_dir = results_dir / label
        self.section_log_dir = logs_dir / label
        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.section_log_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> SectionAnalysisResultSet:
        logging.info(
            "Station %s [Analysis]: DXF=%s, twist=%.1f°, centroid=(%.2f, %.2f)",
            self.label, self.dxf, self.B_r, self.Cx, self.Cy
        )

        self._process_raw_geometry()
        geom_aligned = self._process_and_preview_geometry()
        return self._grid_convergence_section_analysis(geom_aligned)

    # ────────────────────────────────────────────────────────────────────────
    # 1) RAW GEOMETRY OPERATIONS
    # ────────────────────────────────────────────────────────────────────────
    def _process_raw_geometry(self) -> None:
        try:
            raw_geometry = RawGeometry(self.dxf, self.label, logs_dir=self.section_log_dir)
        except Exception as e:
            logging.error("Failed to initialise RawGeometry: %s", str(e), exc_info=True)
            raise

        try:
            RawGeometryVisualisation(raw_geometry, self.section_dir).plot()
        except Exception as e:
            logging.error("Failed to plot raw geometry visualisation: %s", str(e), exc_info=True)
            raise

    # ────────────────────────────────────────────────────────────────────────
    # 2) PROCESSED GEOMETRY OPERATIONS
    # ────────────────────────────────────────────────────────────────────────
    def _process_and_preview_geometry(self):
        try:
            proc = ProcessedGeometry(
                filepath=self.dxf,
                label=self.label,
                logs_dir=self.section_log_dir,
                spline_delta=0.05,
                degrees_per_segment=0.5,
                exterior_nodes=400,
            )
        except Exception as e:
            logging.error("Failed to initialise ProcessedGeometry: %s", str(e), exc_info=True)
            raise

        try:
            geom_aligned = proc.extract_and_transform(twist_deg=self.B_r, cx=self.Cx, cy=self.Cy)
            logging.info("Processed and aligned geometry extracted for %s", self.label)
        except Exception as e:
            logging.error("Failed to extract and transform processed geometry: %s", str(e), exc_info=True)
            raise

        try:
            fig, _ = ProcessedGeometryVisualisation(geometry=geom_aligned, label=self.label).plot_te_zoom(
                te_span_pct=8, figsize=(7, 6), outline_lw=1.0, cp_size=10, legend_loc="upper right"
            )
            preview_path = self.section_dir / f"processed_{self.label}.png"
            fig.savefig(preview_path, dpi=300)
            plt.close(fig)
            logging.info("Processed geometry preview saved -> %s", preview_path)
        except Exception as e:
            logging.error("Failed to visualise or save processed geometry preview: %s", str(e), exc_info=True)
            raise

        return geom_aligned

    # ────────────────────────────────────────────────────────────────────────
    # 3) SECTION ANALYSIS CONVERGENCE STUDY OPERATIONS
    # ────────────────────────────────────────────────────────────────────────
    def _grid_convergence_section_analysis(self, geom_aligned) -> SectionAnalysisResultSet:
        logging.info("Starting section convergence study …")

        try:
            study = ConvergenceStudy(
                geom_aligned=geom_aligned,
                label=self.label,
                hs=self.hs,
                output_dir=self.section_dir,
                logs_dir=self.section_log_dir
            )
            property_bin, mesh_previews = study.results()
        except Exception as e:
            logging.error("ConvergenceStudy execution failed: %s", str(e), exc_info=True)
            raise

        try:
            ConvergenceVisualisation(
                label=self.label,
                output_dir=self.section_dir
            ).plot(mesh_previews)
        except Exception as e:
            logging.error("ConvergenceVisualisation failed: %s", str(e), exc_info=True)
            raise

        try:
            metadata = MetadataBin(
                label=self.label,
                r=self.r,
                r_over_R=self.r_over_R,
                B_r=self.B_r,
                Cx=self.Cx,
                Cy=self.Cy,
                hs=self.hs.tolist(),
                dxf_path=self.dxf
            )
        except Exception as e:
            logging.error("MetadataBin construction failed: %s", str(e), exc_info=True)
            raise

        try:
            results = SectionAnalysisResultSet(
                metadata=metadata,
                convergence=property_bin
            )
        except Exception as e:
            logging.error("SectionAnalysisResultSet construction failed: %s", str(e), exc_info=True)
            raise

        try:
            SaveSectionResults(
                result_set=results,
                output_dir=self.section_dir
            ).save()

            logging.info("Section analysis result set saved")
        except Exception as e:
            logging.error("Failed to save results: %s", str(e), exc_info=True)
            raise

        return results