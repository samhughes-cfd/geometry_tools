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
        material: dict,
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
        self.material = material
        self.hs = hs

        self.section_dir = results_dir / label
        self.section_log_dir = logs_dir / label
        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.section_log_dir.mkdir(parents=True, exist_ok=True)

        self._init_logging()

    def _init_logging(self) -> None:
        log_path = self.section_log_dir / "ProcessSectionAnalysis.log"
        self.logger = logging.getLogger(f"ProcessSectionAnalysis.{self.label}")
        self.logger.propagate = False

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
                   for h in self.logger.handlers):
            handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)
        self.logger.info("Logging initialized for ProcessSectionAnalysis")

    def run(self) -> SectionAnalysisResultSet:
        self.logger.info(
            "Station %s [Analysis]: DXF=%s, twist=%.1f°, centroid=(%.2f, %.2f)",
            self.label, self.dxf, self.B_r, self.Cx, self.Cy
        )

        self.logger.info("Starting raw geometry processing …")
        self._process_raw_geometry()

        self.logger.info("Processing and visualising aligned geometry …")
        geom_aligned = self._process_and_preview_geometry()

        self.logger.info("Launching convergence study pipeline …")
        return self._grid_convergence_section_analysis(geom_aligned)

    def _process_raw_geometry(self) -> None:
        try:
            self.logger.info("Instantiating RawGeometry for DXF: %s", self.dxf)
            raw_geometry = RawGeometry(self.dxf, self.label, logs_dir=self.section_log_dir)
            self.logger.info("RawGeometry created successfully.")
        except Exception as e:
            self.logger.error("Failed to initialise RawGeometry: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Generating raw geometry plot …")
            RawGeometryVisualisation(raw_geometry, self.section_dir).plot()
            self.logger.info("Raw geometry visualisation completed and saved.")
        except Exception as e:
            self.logger.error("Failed to plot raw geometry visualisation: %s", str(e), exc_info=True)
            raise

    def _process_and_preview_geometry(self):
        try:
            self.logger.info("Instantiating ProcessedGeometry with spline/segment settings …")
            proc = ProcessedGeometry(
                filepath=self.dxf,
                label=self.label,
                logs_dir=self.section_log_dir,
                spline_delta=0.05,
                degrees_per_segment=0.5,
                exterior_nodes=400,
            )
            self.logger.info("ProcessedGeometry successfully instantiated.")
        except Exception as e:
            self.logger.error("Failed to initialise ProcessedGeometry: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Extracting and aligning geometry with twist=%.1f°, cx=%.2f, cy=%.2f", self.B_r, self.Cx, self.Cy)
            geom_aligned = proc.extract_and_transform(twist_deg=self.B_r, cx=self.Cx, cy=self.Cy)
            self.logger.info("Geometry aligned and transformed successfully.")
        except Exception as e:
            self.logger.error("Failed to extract and transform processed geometry: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Generating processed geometry visualisation with trailing edge zoom …")
            fig, _ = ProcessedGeometryVisualisation(geometry=geom_aligned, label=self.label).plot_te_zoom(
                te_span_pct=8, figsize=(7, 6), outline_lw=1.0, cp_size=10, legend_loc="upper right"
            )
            preview_path = self.section_dir / f"processed_{self.label}.png"
            fig.savefig(preview_path, dpi=300)
            plt.close(fig)
            self.logger.info("Processed geometry preview saved -> %s", preview_path)
        except Exception as e:
            self.logger.error("Failed to visualise or save processed geometry preview: %s", str(e), exc_info=True)
            raise

        return geom_aligned

    def _grid_convergence_section_analysis(self, geom_aligned) -> SectionAnalysisResultSet:
        self.logger.info("Starting section convergence study using %d mesh sizes: %s", len(self.hs), self.hs)

        try:
            study = ConvergenceStudy(
                geom_aligned=geom_aligned,
                label=self.label,
                hs=self.hs,
                output_dir=self.section_dir,
                logs_dir=self.section_log_dir
            )
            self.logger.info("ConvergenceStudy instantiated. Running results computation …")
            property_bin, mesh_previews = study.results()
            self.logger.info("Convergence study completed successfully.")
        except Exception as e:
            self.logger.error("ConvergenceStudy execution failed: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Creating convergence visualisation …")
            ConvergenceVisualisation(
                label=self.label,
                output_dir=self.section_dir
            ).plot(mesh_previews)
            self.logger.info("Convergence visualisation saved.")
        except Exception as e:
            self.logger.error("ConvergenceVisualisation failed: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Building metadata container for section %s …", self.label)
            metadata = MetadataBin(
                label=self.label,
                r=self.r,
                r_over_R=self.r_over_R,
                B_r=self.B_r,
                Cx=self.Cx,
                Cy=self.Cy,
                hs=self.hs.tolist(),
                dxf_path=self.dxf,
                material_name=self.material["name"]
            )
            self.logger.info("MetadataBin created.")
        except Exception as e:
            self.logger.error("MetadataBin construction failed: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Composing SectionAnalysisResultSet …")
            results = SectionAnalysisResultSet(
                metadata=metadata,
                convergence=property_bin
            )
            self.logger.info("SectionAnalysisResultSet successfully built.")
        except Exception as e:
            self.logger.error("SectionAnalysisResultSet construction failed: %s", str(e), exc_info=True)
            raise

        try:
            self.logger.info("Saving results to directory: %s", self.section_dir)
            SaveSectionResults(
                result_set=results,
                output_dir=self.section_dir
            ).save()
            self.logger.info("Section analysis results successfully saved.")
        except Exception as e:
            self.logger.error("Failed to save results: %s", str(e), exc_info=True)
            raise

        return results