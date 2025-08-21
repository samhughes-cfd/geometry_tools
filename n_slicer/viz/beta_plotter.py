# n_slicer\viz\beta_plotter.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

from n_slicer.viz.base_plotter import _BaseBladePlotter
from n_slicer.containers.units import length_scale



# --- Local Style Definitions ---
@dataclass
class DistLineStyle:
    lw: float = 1.5
    ls: str = "-"
    color: str = "C1"
    label: str = "Twist (src)"


@dataclass
class DistMarkerStyle:
    show: bool = True
    marker: str = "s"
    size: float = 4.0
    face_color: str = "white"
    edge_color: str = "C1"
    edge_width: float = 0.8
    zorder: int = 3
    label: str = "Twist (dist)"


@dataclass
class DistPlotStyle:
    line: DistLineStyle = field(default_factory=DistLineStyle)
    markers: DistMarkerStyle = field(default_factory=DistMarkerStyle)
    grid: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "alpha": 0.25})
    legend: Dict[str, Any] = field(default_factory=lambda: {"fontsize": 8})


def _get_beta_style() -> DistPlotStyle:
    return DistPlotStyle()


# --- Plotter ---
@dataclass
class BetaVsZPlotter(_BaseBladePlotter):
    filename: str = "beta_vs_z.png"

    def plot(self) -> str:
        style = _get_beta_style()
        z_to_XY = length_scale(self.bin.units.L, self.bin.units.XY)

        # continuous line from SOURCE (fallback to DIST if missing)
        z_src = np.asarray(self._z_axis_for_source() * z_to_XY, float)
        b_src = np.asarray(self.bin.beta_src_deg if self.bin.beta_src_deg.size else self.bin.beta_dist_deg, float)
        if z_src.size != b_src.size:
            raise RuntimeError("Twist source distribution misaligned with its z grid.")

        m = np.isfinite(z_src) & np.isfinite(b_src)
        zc, bc = z_src[m], b_src[m]
        if zc.size < 2:
            raise RuntimeError("Insufficient finite points in twist source distribution.")
        s = np.argsort(zc)
        zc, bc = zc[s], bc[s]

        # markers from SAMPLED grid (exact n positions)
        z_dist = np.asarray(self._z_axis_for_dist() * z_to_XY, float)
        b_dist = np.asarray(self.bin.beta_dist_deg, float)
        if z_dist.size != b_dist.size:
            raise RuntimeError("Twist sampled distribution misaligned with its z grid.")

        unit = self.bin.units.XY
        angle_unit = getattr(self.bin.units, "beta_deg", "deg")

        fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)

        # line
        ax.plot(
            zc, bc,
            lw=style.line.lw,
            ls=style.line.ls,
            color=style.line.color,
            label=style.line.label,
        )

        # markers
        if style.markers.show:
            mm = np.isfinite(z_dist) & np.isfinite(b_dist)
            zm, bm = z_dist[mm], b_dist[mm]
            if zm.size:
                ax.plot(
                    zm, bm,
                    linestyle="none",
                    marker=style.markers.marker,
                    ms=style.markers.size,
                    mfc=style.markers.face_color,
                    mec=style.markers.edge_color,
                    mew=style.markers.edge_width,
                    zorder=style.markers.zorder,
                    label=style.markers.label,
                )

        ax.set_xlabel(f"z [{unit}]")
        ax.set_ylabel(f"Twist β [{angle_unit}]")
        if style.grid.get("enabled", True):
            ax.grid(True, alpha=style.grid.get("alpha", 0.25))
        ax.set_title("Twist distribution")
        ax.legend(**style.legend)

        out_path = str(Path(self.outdir) / self.filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path