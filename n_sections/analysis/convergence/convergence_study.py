# convergence/convergence_study.py

import logging
from typing import Tuple, List
from pathlib import Path
import copy
from sectionproperties.analysis.section import Section

from n_sections.analysis.geom_utils.mesh_dxf import MeshDXF
from n_sections.analysis.geom_utils.section_dxf import SectionDXF
from n_sections.visualisation.convergence_visualisation import ConvergenceVisualisation

from n_sections.analysis.containers.geometric_property_bin import GeometricPropertyBin
from n_sections.analysis.containers.geometric_property_analysis_bin import GeometricPropertyAnalysisBin


class ConvergenceStudy:
    """
    Orchestrates the section property convergence study across multiple mesh sizes.

    Executes upon instantiation.

    Attributes:
        property_bin (GeometricPropertyBin): Convergence data across mesh sizes.
        meshes_for_plot (List[Tuple[str, MeshDXF]]): Preview meshes for visualisation.
    """

    def __init__(
        self,
        geom_aligned,
        label: str,
        hs: List[float],
        output_dir: Path,
        logs_dir: Path,
        auto_plot: bool = True
    ):
        self.geom_aligned = geom_aligned
        self.label = label
        self.hs = hs
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.auto_plot = auto_plot

        self.property_bin = GeometricPropertyBin()
        self.meshes_for_plot: List[Tuple[str, MeshDXF]] = []

        self._setup_logger()
        self._run_study()

    def _setup_logger(self):
        self.logger = logging.getLogger(f"convergence_study.{self.label}")
        self.logger.setLevel(logging.DEBUG)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"convergence_study_{self.label.replace(' ', '_')}.log"

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="w")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info("Logger initialized for ConvergenceStudy [%s]", self.label)

    def _run_study(self) -> None:
        self.logger.info("Starting convergence study over %d mesh sizes: %s", len(self.hs), self.hs)

        for i, h in enumerate(self.hs, 1):
            run_label = f"{self.label}_h{h:.4g}"
            self.logger.info("[%d/%d] Processing mesh size h = %.4g", i, len(self.hs), h)

            mesh = self._generate_mesh(run_label, h)
            if mesh is None:
                self.logger.warning("[%s] Skipped due to mesh generation failure.", run_label)
                continue

            row_bin = self._analyse_section(run_label, h, mesh)
            if row_bin is not None:
                self.property_bin.rows.append(row_bin)
                self.meshes_for_plot.append((run_label, mesh))
                self.logger.info("[%s] Section analysis completed successfully.", run_label)
            else:
                self.logger.warning("[%s] Skipped due to section analysis failure.", run_label)

        self.logger.info("Convergence study completed: %d successful, %d attempted",
                         len(self.property_bin.rows), len(self.hs))

        if self.auto_plot and self.meshes_for_plot:
            self._visualise()

    def _generate_mesh(self, run_label: str, h: float) -> MeshDXF | None:
        try:
            self.logger.debug("[%s] Copying geometry and initiating meshing …", run_label)
            geometry_copy = copy.deepcopy(self.geom_aligned)

            mesh = MeshDXF(
                geometry=geometry_copy,
                label=run_label,
                mesh_h=h,
                logs_dir=self.logs_dir
            )

            if mesh.geometry_meshed is None or not hasattr(mesh.geometry_meshed, "mesh"):
                self.logger.error("[%s] Mesh is missing or invalid — skipping", run_label)
                return None

            self.logger.info("[%s] Mesh created successfully", run_label)
            return mesh

        except Exception as e:
            self.logger.exception("[%s] Mesh generation failed: %s", run_label, str(e), exc_info=True)
            return None

    def _analyse_section(self, run_label: str, h: float, mesh: MeshDXF) -> GeometricPropertyAnalysisBin | None:
        try:
            self.logger.debug("[%s] Starting section property calculation …", run_label)

            section_obj = Section(geometry=mesh.geometry_meshed)
            dxf_calc = SectionDXF(
                run_label=run_label,
                mesh_h=h,
                section=section_obj,
                logs_dir=self.logs_dir
            )

            if not hasattr(dxf_calc, "row") or dxf_calc.row is None:
                self.logger.error("[%s] SectionDXF returned no data", run_label)
                return None

            self.logger.debug("[%s] Section analysis row returned", run_label)
            return GeometricPropertyAnalysisBin.from_row(dxf_calc.row)

        except Exception as e:
            self.logger.exception("[%s] Section analysis failed: %s", run_label, str(e), exc_info=True)
            return None

    def _visualise(self) -> None:
        try:
            self.logger.info("Generating convergence visualisation …")
            ConvergenceVisualisation(
                label=self.label,
                output_dir=self.output_dir
            ).plot(self.meshes_for_plot)
            self.logger.info("Convergence visualisation completed.")
        except Exception as e:
            self.logger.warning("Convergence visualisation failed: %s", str(e), exc_info=True)

    def results(self) -> Tuple[GeometricPropertyBin, List[Tuple[str, MeshDXF]]]:
        """
        Returns the convergence study results.

        Returns:
            Tuple containing the property bin and preview meshes.
        """
        return self.property_bin, self.meshes_for_plot