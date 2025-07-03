# visualisation/convergence_visualisation.py
import matplotlib.pyplot as plt
from pathlib import Path
import logging

class ConvergenceVisualisation:
    def __init__(self, label: str, output_dir: Path):
        self.label = label
        self.output_dir = output_dir

    def plot(self, meshes_for_plot: list, batch_size: int = 3):
        for batch_idx, start in enumerate(range(0, len(meshes_for_plot), batch_size), 1):
            batch = meshes_for_plot[start: start + batch_size]
            n_mesh = len(batch)

            try:
                # ── Global View ──
                fig_g, axs_g = plt.subplots(n_mesh, 1, figsize=(4, 1.4 * n_mesh), sharex=True)
                if n_mesh == 1:
                    axs_g = [axs_g]
                for ax, (lbl, mesh) in zip(axs_g, batch):
                    mesh.plot(ax, zoom_te_pct=None)
                    ax.set_title(lbl, fontsize=8)

                fig_g.tight_layout()
                out_g = self.output_dir / f"mesh_conv_{self.label}_batch{batch_idx}_global.png"
                fig_g.savefig(out_g, dpi=300)
                plt.close(fig_g)
                logging.info("Global batch %d saved -> %s", batch_idx, out_g)

                # ── Local View ──
                fig_l, axs_l = plt.subplots(1, n_mesh, figsize=(4 * n_mesh, 4), sharey=True)
                if n_mesh == 1:
                    axs_l = [axs_l]
                for ax, (lbl, mesh) in zip(axs_l, batch):
                    mesh.plot(ax, zoom_te_pct=5.0)
                    ax.set_title(lbl, fontsize=8)

                fig_l.tight_layout()
                out_l = self.output_dir / f"mesh_conv_{self.label}_batch{batch_idx}_local.png"
                fig_l.savefig(out_l, dpi=300)
                plt.close(fig_l)
                logging.info("Local batch %d saved -> %s", batch_idx, out_l)

            except Exception as e:
                logging.exception("Failed to plot mesh convergence batch %d for %s: %s", batch_idx, self.label, str(e))