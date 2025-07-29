# section_calc_n\optimisation\area_optimisation_study.py

from typing import Tuple
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from n_sections.analysis.geom_utils.mesh_dxf import MeshDXF
from n_sections.analysis.geom_utils.section_dxf import SectionDXF
from n_sections.analysis.containers.geometric_property_analysis_bin import GeometricPropertyAnalysisBin
from n_sections.optimisation.containers.optimisation_result_row import OptimisationResultRow
from sectionproperties.analysis import Section


class AreaOptimisationStudy:
    def __init__(
        self,
        geom_aligned,
        label: str,
        mesh_h: float,
        scale_factor: float,
        target_Jt: float,
        target_Iz: float,
        logs_dir: Path,
        results_dir: Path
    ):
        self.geom_aligned = geom_aligned
        self.label = label
        self.mesh_h = mesh_h
        self.scale_factor = scale_factor
        self.target_Jt = target_Jt
        self.target_Iz = target_Iz
        self.logs_dir = logs_dir
        self.results_dir = results_dir

        # ───── Set up dedicated logger ─────
        self.logger = logging.getLogger(f"AreaOptimisation.{label}")
        self.logger.setLevel(logging.INFO)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / "AreaOptimisation.log"

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(sh)

    def run(self) -> Tuple[GeometricPropertyAnalysisBin, OptimisationResultRow]:
        run_label = f"{self.label}_s{self.scale_factor:.3f}_h{self.mesh_h:.3f}"
        self.logger.info("Starting optimisation run: %s", run_label)

        # ───── Mesh the geometry ─────
        mesh = MeshDXF(
            geometry=self.geom_aligned,
            label=run_label,
            mesh_h=self.mesh_h,
            logs_dir=self.logs_dir
        )

        if mesh.geometry_meshed is None or not hasattr(mesh.geometry_meshed, "mesh"):
            self.logger.error("[%s] Mesh missing or invalid — skipping", run_label)
            raise RuntimeError(f"[{run_label}] Mesh generation failed.")

        section_obj = Section(geometry=mesh.geometry_meshed)

        # ───── SAVE mesh plot ─────
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            mesh.plot(ax=ax)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            fig_path = self.results_dir / f"mesh_{run_label}.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            self.logger.info("[%s] Mesh plot saved to %s", run_label, fig_path)
        except Exception as e:
            self.logger.warning("[%s] Mesh plot save failed: %s", run_label, str(e))

        # ───── Compute section properties ─────
        try:
            dxf_calc = SectionDXF(
                run_label=run_label,
                mesh_h=self.mesh_h,
                section=section_obj,
                logs_dir=self.logs_dir
            )
        except Exception as e:
            self.logger.error("[%s] Section property analysis failed: %s", run_label, str(e), exc_info=True)
            raise

        if dxf_calc.row is None:
            self.logger.error("[%s] No section results returned", run_label)
            raise RuntimeError(f"[{run_label}] No section results returned.")

        # ───── Raw section property container ─────
        analysis_bin = GeometricPropertyAnalysisBin.from_row(dxf_calc.row)

        # ───── Optimisation container ─────
        optimisation_row = OptimisationResultRow.from_analysis_bin(
            analysis_bin=analysis_bin,
            run_label=run_label,
            scale_factor=self.scale_factor,
            target_Jt_mm4=self.target_Jt,
            target_Iz_mm4=self.target_Iz
        )

        self.logger.info("[%s] Optimisation result generated (Jt=%.4g, Iz=%.4g)", run_label,
                         optimisation_row.Jt_mm4, optimisation_row.Iz_mm4)

        return analysis_bin, optimisation_row