# section_calc_n/process_section_optimise.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Geometry utilities
from n_sections.geometry_utils.raw_geometry import RawGeometry
from n_sections.geometry_utils.processed_geometry import ProcessedGeometry
from n_sections.optimisation.optimisation.void_builder import VoidBuilder

# Visualisation
from n_sections.visualisation.processed_geometry_visualisation import ProcessedGeometryVisualisation

# Optimisation study
from n_sections.optimisation.optimisation.area_optimisation_study import AreaOptimisationStudy

# Result containers
from n_sections.optimisation.containers.optimisation_results_set import OptimisationResultSet
from n_sections.optimisation.containers.optimisation_metadata import OptimisationMetadata
from n_sections.optimisation.containers.optimisation_result_table import OptimisationResultTable
from n_sections.optimisation.containers.save_optimisation_results import SaveOptimisationResults

class ProcessSectionOptimise:
    def __init__(
        self,
        dxf: Path,
        label: str,
        r: float,
        r_over_R: float,
        B_r: float,
        Cx: float,
        Cy: float,
        h: float,
        results_dir: Path,
        logs_dir: Path,
        scale_factors: list[float],
        target_Jt_mm4: float,
        target_Iz_mm4: float,
    ):
        self.dxf = dxf
        self.label = label
        self.r = r
        self.r_over_R = r_over_R
        self.B_r = B_r
        self.Cx = Cx
        self.Cy = Cy
        self.h = h
        self.scale_factors = scale_factors
        self.target_Jt_mm4 = target_Jt_mm4
        self.target_Iz_mm4 = target_Iz_mm4

        self.section_dir = results_dir / "optimisation" / label
        self.section_log_dir = logs_dir / "optimisation" / label
        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.section_log_dir.mkdir(parents=True, exist_ok=True)

        self.table = OptimisationResultTable()
        self.metadata = OptimisationMetadata(
            label=label,
            r=r,
            r_over_R=r_over_R,
            B_r=B_r,
            Cx=Cx,
            Cy=Cy,
            mesh_h=h,
            dxf_path=dxf,
            scale_factors=scale_factors,
            target_Jt_mm4=target_Jt_mm4,
            target_Iz_mm4=target_Iz_mm4
        )

    def run(self) -> OptimisationResultSet:
        raw_geometry = self._process_raw_geometry()
        found_solution = False
        final_result_set = None

        for scale in self.scale_factors:
            scale_str = f"s{scale:.3f}"
            scaled_label = f"{self.label}_{scale_str}"
            logging.info("🔁 Testing scale factor: %.3f", scale)

            try:
                # Create per-step result and log directories (by scale factor)
                step_results_dir = self.section_dir / scale_str
                step_logs_dir = self.section_log_dir / scale_str
                step_results_dir.mkdir(parents=True, exist_ok=True)
                step_logs_dir.mkdir(parents=True, exist_ok=True)

                geom_with_void = self._process_and_insert_void(raw_geometry, scale, step_logs_dir)
                self._preview_processed_geometry(geom_with_void, scale, step_results_dir)

                result_set = self._optimisation_section_analysis(
                    geometry=geom_with_void,
                    scale=scale,
                    label=scaled_label,
                    output_dir=step_results_dir,
                    log_dir=step_logs_dir
                )
                final_result_set = result_set

                last_row = result_set.result_table.rows[-1]
                if last_row.Jt_mm4 >= self.target_Jt_mm4 and last_row.Iz_mm4 >= self.target_Iz_mm4:
                    logging.info("✅ Thresholds met at %.3f: Jt=%.2f, Iz=%.2f", scale, last_row.Jt_mm4, last_row.Iz_mm4)
                    found_solution = True
                else:
                    logging.info("❌ Below target at %.3f: Jt=%.2f, Iz=%.2f", scale, last_row.Jt_mm4, last_row.Iz_mm4)

            except Exception as e:
                logging.exception("🚫 Optimisation error at scale %.3f: %s", scale, str(e))
                continue

        if not found_solution:
            logging.warning("⚠️ No valid solution found within scale sweep")

        return final_result_set if final_result_set else OptimisationResultSet(metadata=self.metadata, result_table=self.table)

    def _process_raw_geometry(self) -> RawGeometry:
        return RawGeometry(self.dxf, self.label, logs_dir=self.section_log_dir)

    def _process_and_insert_void(self, raw_geometry: RawGeometry, scale: float, log_dir: Path):
        proc = ProcessedGeometry(
            filepath=self.dxf,
            label=self.label,
            logs_dir=log_dir,
            spline_delta=0.05,
            degrees_per_segment=0.5,
            exterior_nodes=400,
        )
        geom_aligned = proc.extract_and_transform(twist_deg=self.B_r, cx=self.Cx, cy=self.Cy)
        optimiser = VoidBuilder(geometry=geom_aligned, label=self.label, log_dir=log_dir)
        return optimiser.insert_void(scale)

    def _preview_processed_geometry(self, geometry, scale: float, output_dir: Path) -> None:
        fig, _ = ProcessedGeometryVisualisation(geometry=geometry, label=self.label).plot_te_zoom(
            te_span_pct=8, figsize=(7, 6), outline_lw=1.0, cp_size=10, legend_loc="upper right"
        )
        preview_path = output_dir / f"processed_geometry_{self.label}_s{scale:.3f}.png"
        fig.savefig(preview_path, dpi=300)
        plt.close(fig)
        logging.info("🖼️ Saved geometry preview → %s", preview_path)

    def _optimisation_section_analysis(self, geometry, scale: float, label: str, output_dir: Path, log_dir: Path) -> OptimisationResultSet:
        study = AreaOptimisationStudy(
            geom_aligned=geometry,
            label=label,
            mesh_h=self.h,
            scale_factor=scale,
            target_Jt=self.target_Jt_mm4,
            target_Iz=self.target_Iz_mm4,
            logs_dir=log_dir,
            results_dir=output_dir
        )
        analysis_bin, optimisation_row = study.run()

        self.table.rows.append(optimisation_row)
        self.table.compute_relative_changes()
        self._plot_metrics_snapshot(self.table, output_dir)

        result_set = OptimisationResultSet(metadata=self.metadata, result_table=self.table)
        SaveOptimisationResults(result_set=result_set, output_dir=output_dir).save()
        logging.info("📦 Saved optimisation results for step %02d", len(self.table.rows))

        return result_set

    def _plot_metrics_snapshot(self, table: OptimisationResultTable, output_dir: Path) -> None:
        rows = table.rows
        steps = list(range(1, len(rows) + 1))

        Iz_vals = [r.Iz_mm4 for r in rows]
        Jt_vals = [r.Jt_mm4 for r in rows]
        Jt_eff = [r.Jt_efficiency for r in rows]
        Iz_eff = [r.Iz_efficiency for r in rows]
        excess_Iz = [r.Excess_ratio_Iz for r in rows]
        excess_Jt = [r.Excess_ratio_Jt for r in rows]
        area_pct = [r.Area_reduction_pct for r in rows]
        area_abs = [r.area_mm2 for r in rows]

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Optimisation Metrics – {self.label}", fontsize=14)

        # Row 1: Iz and Jt
        axs[0, 0].plot(steps, Iz_vals, 'o-', color='tab:blue', label="$I_z$")
        axs[0, 0].axhline(self.target_Iz_mm4, color='tab:blue', linestyle='--', label="Target $I_z$")
        axs[0, 0].set_ylabel("$I_z$ [mm⁴]")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle="--", alpha=0.5)

        axs[0, 1].plot(steps, Jt_vals, 's-', color='tab:orange', label="$J_t$")
        axs[0, 1].axhline(self.target_Jt_mm4, color='tab:orange', linestyle='--', label="Target $J_t$")
        axs[0, 1].set_ylabel("$J_t$ [mm⁴]")
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Row 2: Efficiency (left y-axis) + Excess % (right y-axis)
        # Left: Iz Efficiency
        ax_left_Iz = axs[1, 0]
        ax_left_Iz.plot(steps, Iz_eff, 'o-', color='tab:green', label="Iz Efficiency")
        ax_left_Iz.set_ylabel("Iz Efficiency [mm²]", color='tab:green')
        ax_left_Iz.tick_params(axis='y', labelcolor='tab:green')
        ax_left_Iz.grid(True, linestyle="--", alpha=0.5)

        # Right: Iz Excess (%)
        ax_right_Iz = ax_left_Iz.twinx()
        ax_right_Iz.plot(steps, [100*x if x is not None else None for x in excess_Iz], 'x--', color='tab:purple', label="Excess $I_z$")
        ax_right_Iz.set_ylabel("Excess $I_z$ [%]", color='tab:purple')
        ax_right_Iz.tick_params(axis='y', labelcolor='tab:purple')

        # Combine legends
        lines1, labels1 = ax_left_Iz.get_legend_handles_labels()
        lines2, labels2 = ax_right_Iz.get_legend_handles_labels()
        ax_left_Iz.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Left: Jt Efficiency
        ax_left_Jt = axs[1, 1]
        ax_left_Jt.plot(steps, Jt_eff, 's-', color='tab:green', label="Jt Efficiency")
        ax_left_Jt.set_ylabel("Jt Efficiency [mm²]", color='tab:green')
        ax_left_Jt.tick_params(axis='y', labelcolor='tab:green')
        ax_left_Jt.grid(True, linestyle="--", alpha=0.5)

        # Right: Jt Excess (%)
        ax_right_Jt = ax_left_Jt.twinx()
        ax_right_Jt.plot(steps, [100*x if x is not None else None for x in excess_Jt], 'd--', color='tab:purple', label="Excess $J_t$")
        ax_right_Jt.set_ylabel("Excess $J_t$ [%]", color='tab:purple')
        ax_right_Jt.tick_params(axis='y', labelcolor='tab:purple')

        # Combine legends
        lines1, labels1 = ax_left_Jt.get_legend_handles_labels()
        lines2, labels2 = ax_right_Jt.get_legend_handles_labels()
        ax_left_Jt.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Row 3: Area metrics
        axs[2, 0].plot(steps, area_pct, '^-', color='tab:brown', label="Area Reduction [%]")
        axs[2, 0].set_ylabel("ΔArea [%]")
        axs[2, 0].set_xlabel("Optimisation Step")
        axs[2, 0].legend()
        axs[2, 0].grid(True, linestyle="--", alpha=0.5)

        axs[2, 1].plot(steps, area_abs, 'v-', color='tab:gray', label="Area [mm²]")
        axs[2, 1].set_ylabel("Area [mm²]")
        axs[2, 1].set_xlabel("Optimisation Step")
        axs[2, 1].legend()
        axs[2, 1].grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"optimisation_metrics_{self.label}_step{len(rows):02}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logging.info("🧾 Saved optimisation metrics snapshot → %s", out_path)
