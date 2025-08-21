# n_slicer/viz/stack3d_plotter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Literal, Any
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from n_slicer.viz.base_plotter import _BaseBladePlotter
from n_slicer.containers.units import length_scale
from n_slicer.containers.section_row import SectionRow


# ─────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────

@dataclass
class Stack3DStyle:
    section_color_mode: Literal["colormap", "fixed"] = "colormap"
    section_cmap: str = "plasma"
    section_color_fixed: Optional[str] = None
    section_linestyle: str = "-"
    section_linewidth: float = 1.0
    section_color_from_z: Optional[
        Callable[[float, float, float], Tuple[float, float, float, float]]
    ] = None

    # Centroid / CoP paths
    show_centroid_axis: bool = True
    show_cop_axis: bool = True
    centroid_label: str = "C(x,y,z) Path"
    centroid_style: Dict = field(default_factory=lambda: dict(color="cyan", lw=1.0, ls="-"))
    cop_label: str = "CoP(1/3 c) Path"
    cop_style: Dict = field(default_factory=lambda: dict(color="lime", lw=1.0, ls="-"))
    legend: Dict = field(default_factory=lambda: dict(loc="upper left", frameon=False))


@dataclass
class Stack3DAxisStyle:
    # Grid + ticks
    grid: bool = True
    grid_style: Dict = field(default_factory=lambda: dict(color="0.85", linestyle="--", linewidth=0.6))
    tick_params: Dict = field(default_factory=lambda: dict(labelsize=9, pad=4))
    xticks: Optional[np.ndarray] = None
    yticks: Optional[np.ndarray] = None
    zticks: Optional[np.ndarray] = None
    num_xticks: int = 5
    num_yticks: int = 5
    num_zticks: int = 8
    xformatter: Optional[Callable] = None
    yformatter: Optional[Callable] = None
    zformatter: Optional[Callable] = None

    # Axis inversion
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False

    # Labels
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None
    xlabel_pad: int = 8
    ylabel_pad: int = 8
    zlabel_pad: int = 12

    # Camera control
    auto_view: bool = True        # Enable PCA auto-view
    elev: Optional[float] = None  # Manual override if auto_view=False
    azim: Optional[float] = None


# module-level defaults
_DEFAULT_STACK3D_STYLE = Stack3DStyle()
_DEFAULT_STACK3D_AXIS_STYLE = Stack3DAxisStyle()

def _get_stack3d_style() -> Stack3DStyle:
    return deepcopy(_DEFAULT_STACK3D_STYLE)

def _get_stack3d_axis_style() -> Stack3DAxisStyle:
    return deepcopy(_DEFAULT_STACK3D_AXIS_STYLE)

def set_stack3d_style(**kwargs) -> None:
    for k, v in kwargs.items():
        if hasattr(_DEFAULT_STACK3D_STYLE, k):
            setattr(_DEFAULT_STACK3D_STYLE, k, v)

def set_stack3d_axis_style(**kwargs) -> None:
    for k, v in kwargs.items():
        if hasattr(_DEFAULT_STACK3D_AXIS_STYLE, k):
            setattr(_DEFAULT_STACK3D_AXIS_STYLE, k, v)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _pca_view_angles(points: np.ndarray) -> Tuple[float, float]:
    """Compute PCA-based viewing angles for 3D data.
    Returns (elev, azim) in matplotlib convention."""
    pts = points - points.mean(0)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 22.0, -60.0
    try:
        _, _, Vt = np.linalg.svd(pts, full_matrices=False)
        v1 = Vt[0]  # principal direction
        azim = np.degrees(np.arctan2(v1[1], v1[0]))
        elev = np.degrees(np.arcsin(v1[2] / (np.linalg.norm(v1) + 1e-12)))
        return float(elev), float(azim)
    except Exception:
        return 22.0, -60.0


def _apply_common_axis_style(ax, axis_style: Stack3DAxisStyle, unit_xy: str) -> None:
    ax.set_xlabel(axis_style.xlabel or f"Flapwise X [{unit_xy}]", labelpad=axis_style.xlabel_pad)
    ax.set_ylabel(axis_style.ylabel or f"Edgewise Y [{unit_xy}]", labelpad=axis_style.ylabel_pad)
    ax.set_zlabel(axis_style.zlabel or f"Spanwise Z [{unit_xy}]", labelpad=axis_style.zlabel_pad)

    # tick density
    ax.xaxis.set_major_locator(MaxNLocator(axis_style.num_xticks))
    ax.yaxis.set_major_locator(MaxNLocator(axis_style.num_yticks))
    ax.zaxis.set_major_locator(MaxNLocator(axis_style.num_zticks))

    # optional fixed ticks / formatters
    if axis_style.xticks is not None: ax.set_xticks(axis_style.xticks)
    if axis_style.yticks is not None: ax.set_yticks(axis_style.yticks)
    if axis_style.zticks is not None: ax.set_zticks(axis_style.zticks)
    if axis_style.xformatter: ax.xaxis.set_major_formatter(axis_style.xformatter)
    if axis_style.yformatter: ax.yaxis.set_major_formatter(axis_style.yformatter)
    if axis_style.zformatter: ax.zaxis.set_major_formatter(axis_style.zformatter)

    # grid + tick params
    if axis_style.grid:
        ax.grid(True, **axis_style.grid_style)
    if axis_style.tick_params:
        ax.tick_params(**axis_style.tick_params)

    # inversions
    if axis_style.invert_x: ax.invert_xaxis()
    if axis_style.invert_y: ax.invert_yaxis()
    if axis_style.invert_z: ax.invert_zaxis()


def _square_xy_limits(allX: np.ndarray, allY: np.ndarray, pad_frac: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    # Square symmetric box around XY origin
    xy_max = max(float(np.max(np.abs(allX))), float(np.max(np.abs(allY))))
    xy_max = (1.0 + pad_frac) * xy_max
    return (-xy_max, xy_max), (-xy_max, xy_max)


def _z_limits(allZ: np.ndarray, pad_frac: float) -> Tuple[float, float]:
    zmin = float(np.min(allZ))
    zmax = float(np.max(allZ))
    z_range = max(1e-12, zmax - zmin)
    return (zmin - pad_frac * z_range, zmax + pad_frac * z_range)


# Canonical views adapted for swapped coordinates + optional per-view zoom
# Keys: name -> dict(elev, azim, zlim(optional [zlo, zhi]))
_CANONICAL_VIEWS: Dict[str, Dict[str, Any]] = {
    # Presentation 3D
    "isometric":          dict(elev=30,  azim=-60),
    "isometric_alt":      dict(elev=20,  azim=120),

    # Engineering orthos
    "front_elevation":    dict(elev=0,   azim=0),     # looking along +Y (edgewise)
    "side_elevation":     dict(elev=0,   azim=90),    # looking along +X (flapwise)
    "planform_top":       dict(elev=90,  azim=-90),   # looking down +Z (planform)
    "planform_bottom":    dict(elev=-90, azim=-90),   # looking up  -Z

    # Blade-centric
    "span_development":   dict(elev=15,  azim=-60),
    "root_zoom":          dict(elev=20,  azim=-45, zlim=("min", "min+0.2span")),
    "tip_zoom":           dict(elev=20,  azim=-45, zlim=("max-0.2span", "max")),

    # ── Additional oblique full-blade presets ──
    "iso_low_fore":       dict(elev=15,  azim=-35),
    "iso_mid_fore":       dict(elev=25,  azim=-50),
    "iso_high_fore":      dict(elev=35,  azim=-65),

    "iso_low_aft":        dict(elev=15,  azim=145),
    "iso_mid_aft":        dict(elev=25,  azim=130),
    "iso_high_aft":       dict(elev=35,  azim=115),

    "quarter_fore":       dict(elev=20,  azim=-30),
    "quarter_aft":        dict(elev=20,  azim=150),

    "plan_oblique":       dict(elev=70,  azim=-60),   # plan-ish with thickness cues
    "edge_oblique":       dict(elev=10,  azim=110),   # edge-ish with less overlap

    "span_oblique_fore":  dict(elev=20,  azim=-75),
    "span_oblique_aft":   dict(elev=20,  azim=105),
}


# ─────────────────────────────────────────────────────────────
# Plotter
# ─────────────────────────────────────────────────────────────

@dataclass
class Stack3DPlotter(_BaseBladePlotter):
    decimate_every: Optional[int] = None
    pad: float = 0.05
    filename: str = "blade_stack_3d.png"

    # Optional overrides
    linewidth: Optional[float] = None
    show_centroid_axis: Optional[bool] = None
    show_cop_axis: Optional[bool] = None

    def plot(self) -> str:
        style = _get_stack3d_style()
        axis_style = _get_stack3d_axis_style()

        # optional overrides
        if self.linewidth is not None:
            style.section_linewidth = float(self.linewidth)
        if self.show_centroid_axis is not None:
            style.show_centroid_axis = bool(self.show_centroid_axis)
        if self.show_cop_axis is not None:
            style.show_cop_axis = bool(self.show_cop_axis)

        rows = self._valid_rows()
        if not rows:
            raise RuntimeError("No rows with geometry to plot.")

        z_to_XY = length_scale(self.bin.units.L, self.bin.units.XY)

        # z positions and curves
        z_vals_XY: List[float] = [self._z_for_row(r) * z_to_XY for r in rows]
        curves: List[np.ndarray] = []
        for r in rows:
            XY = self._get_XY(r, decimate_every=self.decimate_every)
            if XY is None:
                raise RuntimeError(f"Missing XY for station {getattr(r, 'station_idx', '?')}")
            curves.append(XY)

        # Optional centroid / CoP axes
        cent_pts = self._axis_points(rows, use_centroid=True,  z_to_XY=z_to_XY)
        cop_pts  = self._axis_points(rows, use_centroid=False, z_to_XY=z_to_XY)

        # Figure/Axes
        fig = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Color normalization baseline
        z_arr = np.asarray(z_vals_XY, float)
        zmin, zmax = float(z_arr.min()), float(z_arr.max())
        rng = max(1e-12, zmax - zmin)
        cmap = plt.get_cmap(style.section_cmap)

        # Plot sections (with axis swap: X=flapwise, Y=edgewise)
        for XY, z_plot in zip(curves, z_vals_XY):
            Y = XY[:, 0]   # edgewise
            X = XY[:, 1]   # flapwise
            Z = np.full_like(X, z_plot, dtype=float)

            if style.section_color_mode == "fixed" and style.section_color_fixed:
                color = style.section_color_fixed
            elif style.section_color_from_z:
                color = style.section_color_from_z(z_plot, zmin, zmax)
            else:
                color = cmap((z_plot - zmin) / rng)

            ax.plot(
                X, Y, Z,
                ls=style.section_linestyle,
                lw=style.section_linewidth,
                color=color,
            )

        # Centroid / CoP paths
        if style.show_centroid_axis and cent_pts is not None:
            ax.plot(cent_pts[:, 0], cent_pts[:, 1], cent_pts[:, 2],
                    label=style.centroid_label, **style.centroid_style)
        if style.show_cop_axis and cop_pts is not None:
            ax.plot(cop_pts[:, 0], cop_pts[:, 1], cop_pts[:, 2],
                    label=style.cop_label, **style.cop_style)

        # Bounds & aspect (square XY, independent Z)
        allX = np.concatenate([c[:, 1] for c in curves])  # flapwise
        allY = np.concatenate([c[:, 0] for c in curves])  # edgewise
        allZ = np.array(z_vals_XY, float)

        (xmin, xmax), (ymin, ymax) = _square_xy_limits(allX, allY, self.pad)
        zlo, zhi = _z_limits(allZ, self.pad)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zlo, zhi)

        # Keep XY square, Z free
        ax.set_box_aspect((max(1e-12, xmax - xmin),
                           max(1e-12, ymax - ymin),
                           max(1e-12, zhi - zlo)))

        # Apply axis styling
        _apply_common_axis_style(ax, axis_style, self.bin.units.XY)

        # View: auto PCA or manual
        if axis_style.auto_view:
            pts_all = np.vstack([
                np.column_stack([c[:, 1], c[:, 0], np.full(len(c), z)])
                for c, z in zip(curves, z_vals_XY)
            ])
            elev, azim = _pca_view_angles(pts_all)
        else:
            elev = axis_style.elev if axis_style.elev is not None else 22.0
            azim = axis_style.azim if axis_style.azim is not None else -60.0

        ax.view_init(elev=elev, azim=azim)

        if (style.show_centroid_axis and cent_pts is not None) or \
           (style.show_cop_axis and cop_pts is not None):
            ax.legend(**style.legend)

        ax.set_title(self._title())
        fig.tight_layout()

        # Save main
        out_path = Path(self.outdir) / self.filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        # Save canonical views (with same styling and bounds logic)
        self._save_canonical_views(curves, z_vals_XY, cent_pts, cop_pts,
                                   axis_style=axis_style,
                                   bounds=((xmin, xmax), (ymin, ymax), (zlo, zhi)))

        return str(out_path)

    # ──────────────────────────────
    # Canonical view saver
    # ──────────────────────────────
    def _save_canonical_views(
        self,
        curves: List[np.ndarray],
        z_vals_XY: List[float],
        cent_pts: Optional[np.ndarray],
        cop_pts: Optional[np.ndarray],
        *,
        axis_style: Stack3DAxisStyle,
        bounds: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
    ) -> None:
        out_dir = Path(self.outdir) / "canonical_views"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Base color mapping
        zmin, zmax = float(min(z_vals_XY)), float(max(z_vals_XY))
        rng = max(1e-12, zmax - zmin)
        cmap = plt.get_cmap(_DEFAULT_STACK3D_STYLE.section_cmap)

        (xmin_base, xmax_base), (ymin_base, ymax_base), (zlo_base, zhi_base) = bounds
        zspan_full = zhi_base - zlo_base

        def resolve_zlim(spec, zlo, zhi):
            """Support ('min','min+0.2span','max-0.2span','max') style tokens."""
            if not spec or not isinstance(spec, tuple):
                return (zlo, zhi)

            def _tok_to_val(tok: str) -> float:
                tok = str(tok).strip().lower()
                if tok == "min":
                    return zlo_base
                if tok == "max":
                    return zhi_base
                if tok.startswith("min+"):
                    # "min+0.2span"
                    try:
                        frac = float(tok.split("+", 1)[1].replace("span", "").strip())
                        return zlo_base + frac * zspan_full
                    except Exception:
                        return zlo_base
                if tok.startswith("max-"):
                    # "max-0.2span"
                    try:
                        frac = float(tok.split("-", 1)[1].replace("span", "").strip())
                        return zhi_base - frac * zspan_full
                    except Exception:
                        return zhi_base
                # numeric fallback
                try:
                    return float(tok)
                except Exception:
                    return zlo_base

            lo_tok, hi_tok = spec
            return (_tok_to_val(lo_tok), _tok_to_val(hi_tok))

        for name, spec in _CANONICAL_VIEWS.items():
            elev = float(spec.get("elev", 22.0))
            azim = float(spec.get("azim", -60.0))
            zlim_spec = spec.get("zlim", None)
            zlo, zhi = resolve_zlim(zlim_spec, zlo_base, zhi_base)

            fig = plt.figure(figsize=(8, 6), dpi=150)
            ax = fig.add_subplot(111, projection="3d")

            # Draw blade sections
            for XY, z_plot in zip(curves, z_vals_XY):
                Y = XY[:, 0]  # edgewise
                X = XY[:, 1]  # flapwise
                Z = np.full_like(X, z_plot, dtype=float)
                color = cmap((z_plot - zmin) / rng)
                ax.plot(X, Y, Z, lw=_DEFAULT_STACK3D_STYLE.section_linewidth, color=color)

            # Extras
            if _DEFAULT_STACK3D_STYLE.show_centroid_axis and cent_pts is not None:
                ax.plot(cent_pts[:, 0], cent_pts[:, 1], cent_pts[:, 2],
                        **_DEFAULT_STACK3D_STYLE.centroid_style)
            if _DEFAULT_STACK3D_STYLE.show_cop_axis and cop_pts is not None:
                ax.plot(cop_pts[:, 0], cop_pts[:, 1], cop_pts[:, 2],
                        **_DEFAULT_STACK3D_STYLE.cop_style)

            # Apply same bounds (square XY) and per-view z zoom (if any)
            ax.set_xlim(xmin_base, xmax_base)
            ax.set_ylim(ymin_base, ymax_base)
            ax.set_zlim(zlo, zhi)
            ax.set_box_aspect((max(1e-12, xmax_base - xmin_base),
                               max(1e-12, ymax_base - ymin_base),
                               max(1e-12, zhi - zlo)))

            # Style knobs (ticks/grid/labels/formatters)
            _apply_common_axis_style(ax, axis_style, unit_xy=self.bin.units.XY)

            # Camera
            ax.view_init(elev=elev, azim=azim)

            ax.set_title(f"Canonical: {name}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
            plt.close(fig)

    # ──────────────────────────────
    # Axis helper
    # ──────────────────────────────
    def _axis_points(self, rows: List[SectionRow], *, use_centroid: bool, z_to_XY: float) -> Optional[np.ndarray]:
        pts: List[Tuple[float, float, float]] = []
        for r in rows:
            z_plot = self._z_for_row(r) * z_to_XY
            gp = (r.scaled.geom if r.scaled and r.scaled.geom else None)
            fp = (r.scaled.from_norm if r.scaled and r.scaled.from_norm else None)
            if use_centroid:
                x = (gp.cy if gp else (fp.cy if fp else None))  # flapwise -> X
                y = (gp.cx if gp else (fp.cx if fp else None))  # edgewise -> Y
            else:
                x = (gp.y_cp if gp else (fp.y_cp if fp else None))  # flapwise -> X
                y = (gp.x_cp if gp else (fp.x_cp if fp else None))  # edgewise -> Y
            if (x is None) or (y is None):
                continue
            pts.append((float(x), float(y), float(z_plot)))
        return None if not pts else np.asarray(pts, float)
