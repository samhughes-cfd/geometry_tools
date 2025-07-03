# visualisation/mesh_visualisation.py
import matplotlib.pyplot as plt
from geometry_utils.mesh_dxf import MeshDXF
from section_calc_n.geometry_utils.processed_geometry import ProcessedGeometry
import logging
from pathlib import Path

class MeshVisualisation:
    def __init__(self, geometry, proc: ProcessedGeometry, label: str, output_dir: Path, logs_dir: Path):
        self.geometry = geometry
        self.proc = proc
        self.label = label
        self.output_dir = output_dir
        self.logs_dir = logs_dir

    def plot_processed_geometry(self):
        try:
            fig, (ax_full, ax_te) = plt.subplots(2, 1, figsize=(6, 8))

            # Full chord
            self.proc.plot(ax_full, outline_lw=1.0, cp_size=5, legend_loc="upper right")
            ax_full.set_title(f"Processed Geometry: {self.label} (full chord)")

            # Trailing edge zoom
            mesh_dummy = MeshDXF(self.geometry, self.label, logs_dir=self.logs_dir)
            mesh_dummy.mesh_generated = True
            mesh_dummy.geometry = self.geometry
            mesh_dummy.plot(ax_te, zoom_te_pct=5.0)
            ax_te.set_title(f"TE Zoom: {self.label}")

            fig.tight_layout()
            output_path = self.output_dir / f"processed_te_zoom_{self.label}.png"
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            logging.info("Processed geometry saved -> %s", output_path)

        except Exception as e:
            logging.exception("Failed to plot processed geometry for %s: %s", self.label, str(e))