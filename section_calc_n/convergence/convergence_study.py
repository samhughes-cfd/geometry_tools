# convergence/convergence_study.py

import logging
from typing import Tuple, List
from pathlib import Path
from sectionproperties.analysis.section import Section

from geometry_utils.mesh_dxf import MeshDXF
from geometry_utils.section_dxf import SectionDXF
from visualisation.convergence_visualisation import ConvergenceVisualisation

from containers.analysis.geometric_property_bin import GeometricPropertyBin
from containers.analysis.geometric_property_analysis_bin import GeometricPropertyAnalysisBin


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
        logs_dir: Path
    ):
        self.geom_aligned = geom_aligned
        self.label = label
        self.hs = hs
        self.output_dir = output_dir
        self.logs_dir = logs_dir

        self.property_bin = GeometricPropertyBin()
        self.meshes_for_plot: List[Tuple[str, MeshDXF]] = []

        self._run_study()

    def _run_study(self) -> None:
        for i, h in enumerate(self.hs, 1):
            run_label = f"{self.label}_h{h:.4g}"
            logging.info("[%d/%d] Meshing %s with h = %.4g", i, len(self.hs), self.label, h)

            try:
                mesh = MeshDXF(
                    geometry=self.geom_aligned,
                    label=run_label,
                    mesh_h=h,
                    logs_dir=self.logs_dir
                )

                if mesh.geometry_meshed is None or not hasattr(mesh.geometry_meshed, "mesh"):
                    logging.error("[%s] Mesh missing or invalid — skipping", run_label)
                    continue

                section_obj = Section(geometry=mesh.geometry_meshed)

            except Exception as e:
                logging.error("[%s] Mesh generation failed — skipping: %s", run_label, str(e), exc_info=True)
                continue

            try:
                dxf_calc = SectionDXF(
                    run_label=run_label,
                    mesh_h=h,
                    section=section_obj,
                    logs_dir=self.logs_dir
                )

                if not hasattr(dxf_calc, "row") or dxf_calc.row is None:
                    logging.error("[%s] No section results returned — skipping", run_label)
                    continue

                # Convert row to GeometricPropertyAnalysisBin
                row = GeometricPropertyAnalysisBin.from_row(dxf_calc.row)
                self.property_bin.rows.append(row)
                self.meshes_for_plot.append((run_label, mesh))

            except Exception as e:
                logging.error("[%s] Section analysis failed — skipping: %s", run_label, str(e), exc_info=True)
                continue

        if self.meshes_for_plot:
            try:
                ConvergenceVisualisation(
                    label=self.label,
                    output_dir=self.output_dir
                ).plot(self.meshes_for_plot)
            except Exception as e:
                logging.warning("Visualisation failed: %s", str(e), exc_info=True)

    def results(self) -> Tuple[GeometricPropertyBin, List[Tuple[str, MeshDXF]]]:
        """
        Returns the convergence study results.

        Returns:
            Tuple containing the property bin and preview meshes.
        """
        return self.property_bin, self.meshes_for_plot