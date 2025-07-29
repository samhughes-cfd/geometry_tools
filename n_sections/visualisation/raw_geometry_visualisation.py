# visualisation/raw_geometry_visualisation.py

import itertools
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from n_sections.geometry_utils.raw_geometry import RawGeometry


class RawGeometryVisualisation:
    def __init__(self, raw_geometry: RawGeometry, output_dir: Path) -> None:
        self.raw_geometry = raw_geometry
        self.output_dir = output_dir

    def plot(self, annotate_every: int = 5, cmap: str = "tab20") -> Path:
        raw_pts = self.raw_geometry.extract()
        fig, ax = plt.subplots(figsize=(6, 5))

        colours = itertools.cycle(plt.get_cmap(cmap).colors)
        logging.info("Plotting %d DXF entities for preview...", len(raw_pts))

        for idx, (pts, col) in enumerate(zip(raw_pts, colours), start=1):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, lw=1.2, color=col)

            if annotate_every and idx % annotate_every == 0:
                mid = len(xs) // 2
                ax.text(xs[mid], ys[mid], str(idx), fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

        ax.set_title(f"Raw DXF Geometry: {self.raw_geometry.label}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", linewidth=0.5)

        fig.tight_layout()
        output_path = self.output_dir / f"raw_{self.raw_geometry.label}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        logging.info("Raw preview saved -> %s", output_path)
        return output_path