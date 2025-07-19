# process_section.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sectionproperties.analysis import Section
from section_calc_n.geometry_utils.raw_geometry import RawGeometry
from section_calc_n.geometry_utils.processed_geometry import ProcessedGeometry
from geometry_utils.mesh_dxf import MeshDXF
from section_calc_n.geometry_utils.section_properties import SectionDXF
from visualisation.mesh_visualisation import MeshVisualisation
from visualisation.convergence_visualisation import ConvergenceVisualisation
from visualisation.raw_geometry_visualisation import RawGeometryVisualisation
from visualisation.processed_geometry_visualisation import ProcessedGeometryVisualisation
import numpy as np


class ProcessSection:
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
        mode: str = "Analysis",
        scale_factor: float | None = None
    ):
        self.dxf = dxf
        self.label = label
        self.r = r
        self.r_over_R = r_over_R
        self.B_r = B_r
        self.Cx = Cx
        self.Cy = Cy
        self.hs = hs
        self.scale_factor = scale_factor

        self.mode = mode.capitalize()
        assert self.mode in {"Analysis", "Optimisation"}, "Mode must be 'Analysis' or 'Optimisation'"

        if self.mode == "Optimisation" and self.scale_factor is None:
            raise ValueError(f"[{self.label}] scale_factor must be provided in Optimisation mode.")

        self.RESULTS = results_dir / self.mode
        self.LOGS = logs_dir / self.mode
        self.section_dir = self.RESULTS / self.label
        self.section_log_dir = self.LOGS / self.label

        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.section_log_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        logging.info(
            "Station %s [%s]: DXF=%s, twist=%.1f°, centroid=(%.2f,%.2f)",
            self.label, self.mode, self.dxf, self.B_r, self.Cx, self.Cy
        )

        # ───── RAW PREVIEW ─────────────────────────────
        raw_geometry = RawGeometry(self.dxf, self.label, logs_dir=self.section_log_dir)
        RawGeometryVisualisation(raw_geometry, self.section_dir).plot()

        # ───── PROCESSED GEOMETRY ─────────────────────
        proc = ProcessedGeometry(
            filepath=self.dxf,
            label=self.label,
            logs_dir=self.section_log_dir,
            spline_delta=0.05,
            degrees_per_segment=0.5,
            exterior_nodes=400,
        )

        geom_aligned = proc.extract_and_transform(
            twist_deg=self.B_r,
            cx=self.Cx,
            cy=self.Cy
        )
        logging.info("Processed and aligned geometry extracted for %s", self.label)

        # ───── OPTIONAL VOID INSERTION ─────────────────
        if self.mode == "Optimisation":
            from section_calc_n.geometry_utils.void_builder import VoidBuilder

            optimiser = VoidBuilder(
                geometry=geom_aligned,
                label=self.label,
                log_dir=self.section_log_dir
            )

            geom_aligned = optimiser.insert_void(self.scale_factor)
            logging.info("Internal void inserted for %s at scale %.2f", self.label, self.scale_factor)

        # ───── Processed Geometry Preview ──────────────
        fig, _ = ProcessedGeometryVisualisation(geometry=geom_aligned, label=self.label).plot_te_zoom(
            te_span_pct=8,
            figsize=(7, 6),
            outline_lw=1.0,
            cp_size=10,
            legend_loc="upper right"
        )
        fig_path = self.section_dir / f"processed_{self.label}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logging.info("Processed geometry preview saved -> %s", fig_path)

        # ───── MESH + SECTION PROPERTIES ───────────────
        meshes_for_plot = []
        for i, h in enumerate(self.hs, 1):
            run_lbl = f"{self.label}_h{h:.4g}"
            logging.info("[%d/%d] meshing %s with target h = %.4g", i, len(self.hs), self.label, h)

            mesh = MeshDXF(geometry=geom_aligned, label=run_lbl, logs_dir=self.section_log_dir)
            geom_m = mesh.build(mesh_size=float(h))
            if geom_m is None:
                logging.error("[%s] mesh failed – skipping", run_lbl)
                continue

            sec_obj = Section(geometry=geom_m)
            section_csv_path = self.section_dir / "section_properties.csv"
            SectionDXF(
                run_label=run_lbl,
                mesh_h=float(h),
                section=sec_obj,
                output_path=section_csv_path,
                logs_dir=self.section_log_dir
            ).write_csv_row()
            meshes_for_plot.append((run_lbl, mesh))

        # ───── CONVERGENCE VISUALISATION ───────────────
        if meshes_for_plot:
            ConvergenceVisualisation(
                label=self.label,
                output_dir=self.section_dir
            ).plot(meshes_for_plot)