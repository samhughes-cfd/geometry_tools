# n_slicer/viz/sections2d_plotter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from n_slicer.viz.base_plotter import _BaseBladePlotter
from n_slicer.containers.units import length_scale
from n_slicer.containers.section_row import SectionRow


# ─────────────────────────────────────────────────────────────
# Styles / knobs
# ─────────────────────────────────────────────────────────────

@dataclass
class Sections2DStyle:
    # geometry drawing
    line_color: str = "k"
    line_width: float = 1.0
    line_style: str = "-"

    # figure/grid layout (COMPOSITE GRID)
    cols: int = 4
    figsize_per_subplot: Tuple[float, float] = (3.2, 3.0)  # (w, h) inches per subplot in the grid
    tight_layout: bool = True
    pad_frac: float = 0.05  # symmetric pad around origin (as fraction of max(|x|,|y|))

    # axes cosmetics
    show_grid: bool = True
    grid_kwargs: Dict = field(default_factory=lambda: dict(color="0.85", linestyle="--", linewidth=0.6))
    axis_equal: bool = True
    tick_params: Dict = field(default_factory=lambda: dict(labelsize=8, pad=3))
    num_xticks: int = 4
    num_yticks: int = 4
    # labels (units appended)
    # ⬇️ Horizontal = Edgewise Y, Vertical = Flapwise X
    xlabel: Optional[str] = "Edgewise Y"
    ylabel: Optional[str] = "Flapwise X"
    xlabel_pad: int = 6
    ylabel_pad: int = 6
    show_origin_cross: bool = True
    origin_cross_kwargs: Dict = field(default_factory=lambda: dict(color="0.75", linewidth=0.8))

    # legend content / placement
    legend_loc: str = "lower left"
    legend_fontsize: int = 7
    legend_frame: bool = True

    # outputs (COMPOSITE)
    composite_filename: str = "sections_grid.png"
    dpi: int = 150

    # outputs (STANDALONE PER-SECTION)
    per_section_subdir: str = "stations"
    per_section_prefix: str = "station_"
    per_section_digits: int = 3
    per_section_include_z: bool = True   # append z in filename
    standalone_figsize: Tuple[float, float] = (6.0, 5.0)  # bigger than grid subplots
    standalone_dpi: Optional[int] = 220  # if None, fall back to `dpi` above


# ─────────────────────────────────────────────────────────────
# Plotter
# ─────────────────────────────────────────────────────────────

@dataclass
class SectionsGridPlotter(_BaseBladePlotter):
    """Renders n many 2D XY plots (one per section) and saves:
       1) A composite grid PNG with all sections (small subplots).
       2) One larger PNG per section in <outdir>/<style.per_section_subdir>/.

       Orientation for 2D sections:
       - Horizontal axis = Edgewise Y
       - Vertical axis   = Flapwise X
       Mapping assumes raw XY: [:,0] = edgewise, [:,1] = flapwise.
    """
    decimate_every: Optional[int] = None
    style: Sections2DStyle = field(default_factory=Sections2DStyle)

    # 🔑 Keep vertical axis as Flapwise X: X=edgewise, Y=flapwise → no swap
    swap_xy: bool = False

    def plot(self) -> str:
        rows = self._valid_rows()
        if not rows:
            raise RuntimeError("No rows with geometry to plot.")

        # Ensure output dirs exist
        out_dir = Path(self.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stations_dir = out_dir / self.style.per_section_subdir
        stations_dir.mkdir(parents=True, exist_ok=True)

        # Build geometry and z positions (in XY units for labeling/consistency)
        z_to_XY = length_scale(self.bin.units.L, self.bin.units.XY)
        z_vals_XY: List[float] = [self._z_for_row(r) * z_to_XY for r in rows]
        curves: List[np.ndarray] = []
        for r in rows:
            XY = self._get_XY(r, decimate_every=self.decimate_every)
            if XY is None:
                raise RuntimeError(f"Missing XY for station {getattr(r, 'station_idx', '?')}")
            curves.append(XY)

        # Precompute sampled distributions for chord & twist (align via nearest z)
        z_dist = np.asarray(self._z_axis_for_dist() * z_to_XY, float)
        c_dist = np.asarray(self.bin.c_dist, float)
        b_dist = np.asarray(self.bin.beta_dist_deg, float)
        have_c = c_dist.size == z_dist.size and z_dist.size > 0
        have_b = b_dist.size == z_dist.size and z_dist.size > 0
        c_unit = getattr(self.bin.units, "c", "")
        beta_unit = getattr(self.bin.units, "beta_deg", "deg")
        xy_unit = self.bin.units.XY

        # Composite grid figure
        n = len(curves)
        ncols = max(1, int(self.style.cols))
        nrows = int(np.ceil(n / ncols))
        fig_w = ncols * self.style.figsize_per_subplot[0]
        fig_h = nrows * self.style.figsize_per_subplot[1]
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=self.style.dpi)
        if nrows * ncols == 1:
            axes = np.array([axes])  # normalize to iterable
        axes = axes.flatten()

        # Plot each section both inside composite and as its own PNG
        for idx, (r, XY, z_here, ax) in enumerate(zip(rows, curves, z_vals_XY, axes)):
            # Mapping: X=edgewise (raw [:,0]), Y=flapwise (raw [:,1])  → vertical = Flapwise X
            X, Y = self._xy_from_raw(XY)

            # Stats
            xmin, xmax = float(np.min(X)), float(np.max(X))
            ymin, ymax = float(np.min(Y)), float(np.max(Y))

            # Nearest chord/twist (by z)
            chord_txt = "n/a"
            beta_txt = "n/a"
            if have_c or have_b:
                j = int(np.argmin(np.abs(z_dist - z_here))) if z_dist.size else 0
                if have_c:
                    chord_txt = f"{c_dist[j]:.3g} {c_unit}".strip()
                if have_b:
                    beta_txt = f"{b_dist[j]:.3g} {beta_unit}".strip()

            # DRAW (composite subplot)
            self._draw_section_2d(ax, X, Y, xmin, xmax, ymin, ymax, xy_unit)

            # Legend content
            legend_lines = []
            legend_labels = [
                f"β: {beta_txt}",
                f"c: {chord_txt}",
                f"xmin: {xmin:.3g} {xy_unit}",
                f"xmax: {xmax:.3g} {xy_unit}",
                f"ymin: {ymin:.3g} {xy_unit}",
                f"ymax: {ymax:.3g} {xy_unit}",
            ]
            for _ in legend_labels:
                h, = ax.plot([], [], alpha=0)  # invisible handle
                legend_lines.append(h)

            ax.legend(
                legend_lines, legend_labels,
                loc=self.style.legend_loc,
                fontsize=self.style.legend_fontsize,
                frameon=self.style.legend_frame,
            )

            # Save individual section (with larger standalone size/DPI)
            sec_name = self._section_filename(idx, z_here)
            sec_path = stations_dir / sec_name
            self._save_single_section_png(
                X, Y, xmin, xmax, ymin, ymax, legend_labels, sec_path, xy_unit
            )

        # Hide any unused subplots
        for k in range(n, len(axes)):
            fig.delaxes(axes[k])

        if self.style.tight_layout:
            fig.tight_layout()

        grid_path = str(out_dir / self.style.composite_filename)
        fig.savefig(grid_path, bbox_inches="tight", dpi=self.style.dpi)
        plt.close(fig)

        return grid_path  # return composite path

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _xy_from_raw(self, XY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """For vertical axis = Flapwise X:
           X (horizontal) = edgewise  = raw XY[:,0]
           Y (vertical)   = flapwise  = raw XY[:,1]
        """
        XY = np.asarray(XY, float)
        if self.swap_xy:
            # (kept for completeness; not used by default now)
            return XY[:, 1].astype(float), XY[:, 0].astype(float)
        return XY[:, 0].astype(float), XY[:, 1].astype(float)

    def _square_limits_about_origin(self, xmin, xmax, ymin, ymax) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        # symmetric square limits centered at (0,0) with padding
        half_span = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        half_span = (1.0 + self.style.pad_frac) * half_span
        return (-half_span, half_span), (-half_span, half_span)

    def _apply_axes_style(self, ax, xy_unit: str) -> None:
        if self.style.axis_equal:
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass

        ax.xaxis.set_major_locator(MaxNLocator(self.style.num_xticks))
        ax.yaxis.set_major_locator(MaxNLocator(self.style.num_yticks))

        if self.style.show_grid:
            ax.grid(True, **self.style.grid_kwargs)
        if self.style.tick_params:
            ax.tick_params(**self.style.tick_params)

        # Labels (units appended)
        if self.style.xlabel:
            ax.set_xlabel(f"{self.style.xlabel} [{xy_unit}]", labelpad=self.style.xlabel_pad)
        if self.style.ylabel:
            ax.set_ylabel(f"{self.style.ylabel} [{xy_unit}]", labelpad=self.style.ylabel_pad)

        # Origin crosshairs
        if self.style.show_origin_cross:
            ax.axhline(0.0, **self.style.origin_cross_kwargs)
            ax.axvline(0.0, **self.style.origin_cross_kwargs)

    def _draw_section_2d(self, ax, X, Y, xmin, xmax, ymin, ymax, xy_unit: str) -> None:
        # draw geometry
        ax.plot(X, Y, color=self.style.line_color, lw=self.style.line_width, ls=self.style.line_style)

        # symmetric square limits about origin
        (xlo, xhi), (ylo, yhi) = self._square_limits_about_origin(xmin, xmax, ymin, ymax)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        self._apply_axes_style(ax, xy_unit)

    def _save_single_section_png(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        legend_labels: List[str],
        path: Path,
        xy_unit: str,
    ) -> None:
        dpi = self.style.standalone_dpi if self.style.standalone_dpi is not None else self.style.dpi
        fig, ax = plt.subplots(figsize=self.style.standalone_figsize, dpi=dpi)

        self._draw_section_2d(ax, X, Y, xmin, xmax, ymin, ymax, xy_unit)

        legend_lines = []
        for _ in legend_labels:
            h, = ax.plot([], [], alpha=0)
            legend_lines.append(h)
        ax.legend(
            legend_lines, legend_labels,
            loc=self.style.legend_loc,
            fontsize=self.style.legend_fontsize,
            frameon=self.style.legend_frame,
        )

        if self.style.tight_layout:
            fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    def _section_filename(self, idx: int, z_here: float) -> str:
        stem = f"{self.style.per_section_prefix}{idx:0{self.style.per_section_digits}d}"
        if self.style.per_section_include_z:
            stem += f"_z{z_here:.4g}"
        return f"{stem}.png"