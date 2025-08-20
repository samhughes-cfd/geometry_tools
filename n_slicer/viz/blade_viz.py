# n_slicer/viz/blade_viz.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Literal
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

try:
    import ezdxf
    _HAS_EZDXF = True
except Exception:
    _HAS_EZDXF = False

from n_slicer.containers.section_bin import SectionBin
from n_slicer.containers.section_row import SectionRow
from n_slicer.containers.units import length_scale


# ─────────────────────────────────────────────────────────────────────────────
# Shared base with common helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _BaseBladePlotter:
    bin: SectionBin
    outdir: str

    # ---- helpers shared by plotters ----
    def _valid_rows(self) -> List[SectionRow]:
        out = []
        for r in self.bin.rows:
            if r.XY is not None and getattr(r.XY, "size", 0) > 0:
                out.append(r)
            elif (r.dxf_path is not None) and _HAS_EZDXF:
                out.append(r)
        out.sort(key=lambda rr: getattr(rr, "rR", 0.0))  # safer
        return out

    def _get_XY(self, r: SectionRow, decimate_every: Optional[int]) -> Optional[np.ndarray]:
        if r.XY is not None and getattr(r.XY, "size", 0) > 0:
            XY = np.asarray(r.XY, float)
        elif r.dxf_path and _HAS_EZDXF:
            XY = self._read_dxf_polyline(r.dxf_path)
            if XY is None:
                return None
        else:
            return None
        if decimate_every and decimate_every > 1 and XY.shape[0] > decimate_every:
            XY = XY[::decimate_every, :]
            if not np.allclose(XY[0], XY[-1], atol=1e-9):
                XY = np.vstack([XY, XY[0]])
        return XY

    def _read_dxf_polyline(self, path: str) -> Optional[np.ndarray]:
        try:
            doc = ezdxf.readfile(path)
            msp = doc.modelspace()
            ents = list(msp.query("LWPOLYLINE POLYLINE"))
            if not ents:
                return None
            e = ents[0]
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
                closed = bool(e.closed)
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                closed = bool(e.is_closed)
            XY = np.asarray(pts, float)
            if closed and not np.allclose(XY[0], XY[-1], atol=1e-9):
                XY = np.vstack([XY, XY[0]])
            return XY
        except Exception:
            return None

    # Physical z (in meters of L-units). Uses z = r - R_hub = rR*R - (R - L)
    def _z_for_row(self, r: SectionRow) -> float:
        L = float(self.bin.L or 0.0)
        R = float(getattr(r, "R", 0.0))
        if L <= 0 or R <= 0:
            return 0.0
        if getattr(r, "zL", None) is not None:
            return float(r.zL) * L
        rR = float(r.rR)
        return rR * R - (R - L)

    # explicit z-axes for sampled/used vs source/original
    def _z_axis_for_dist(self) -> np.ndarray:
        """
        Physical z array (meters of L-units) for the *sampled/used* distributions
        (aligned with rR_dist / c_dist / beta_dist_deg).
        Preference order:
          1) zL_dist (if present) * L
          2) map r/R -> z with first-row R and global L: z = rR*R - (R - L)
        """
        rR = np.asarray(self.bin.rR_dist, float)
        L = float(self.bin.L or 0.0)

        # Prefer explicit zL_dist if present and sized
        if getattr(self.bin, "zL_dist", np.empty(0)).size == rR.size and L > 0:
            return np.asarray(self.bin.zL_dist, float) * L

        # Legacy support: xL_dist
        if getattr(self.bin, "xL_dist", np.empty(0)).size == rR.size and L > 0:
            return np.asarray(self.bin.xL_dist, float) * L

        # Fallback mapping using R and L
        if not self.bin.rows:
            return np.zeros_like(rR)
        R = float(getattr(self.bin.rows[0], "R", 0.0))
        if L <= 0 or R <= 0:
            return np.zeros_like(rR)
        return rR * R - (R - L)

    def _z_axis_for_source(self) -> np.ndarray:
        """
        Physical z array (meters of L-units) for the *source/original* distributions
        (aligned with rR_src / c_src / beta_src_deg). Falls back to the sampled grid
        if source arrays are not populated.
        Preference order:
          1) zL_src (if present) * L
          2) map r/R (from rR_src) -> z with first-row R and global L
          3) fallback to _z_axis_for_dist()
        """
        rR_src = np.asarray(self.bin.rR_src, float)
        L = float(self.bin.L or 0.0)

        if rR_src.size == 0:
            return self._z_axis_for_dist()

        # Prefer explicit zL_src if present and sized
        if getattr(self.bin, "zL_src", np.empty(0)).size == rR_src.size and L > 0:
            return np.asarray(self.bin.zL_src, float) * L

        # Fallback mapping using R and L
        if not self.bin.rows:
            return np.zeros_like(rR_src)
        R = float(getattr(self.bin.rows[0], "R", 0.0))
        if L <= 0 or R <= 0:
            return np.zeros_like(rR_src)
        return rR_src * R - (R - L)

    # Back-compat shim (deprecated)
    def _z_axis_for_distribution(self) -> np.ndarray:
        return self._z_axis_for_dist()

    def _title(self) -> str:
        name = self.bin.blade_name or "Blade"
        if self.bin.L:
            Ltxt = f"L={self.bin.L:g}{self.bin.units.L}"
        else:
            Ltxt = "L=unknown"
        return f"{name} — sections={len(self.bin.rows)}, {Ltxt}, physical z stacking"


# ─────────────────────────────────────────────────────────────────────────────
# Internal styles & module-level defaults (kept internal to this module)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Stack3DStyle:
    section_color_mode: Literal["colormap", "fixed"] = "colormap"
    section_cmap: str = "plasma"
    section_color_fixed: Optional[str] = None
    section_linestyle: str = "-"
    section_linewidth: float = 1
    # Optional custom color mapping: f(z_value, zmin, zmax) -> RGBA
    section_color_from_z: Optional[Callable[[float, float, float], Tuple[float, float, float, float]]] = None

    # Extra axes (centroid / CoP) visibility + styles + legend labels
    show_centroid_axis: bool = True
    show_cop_axis: bool = True
    centroid_label: str = "C(x,y,z) Path"
    centroid_style: Dict = field(default_factory=lambda: dict(color="cyan", lw=1.0, ls="-"))
    cop_label: str = "CoP(1/3 c) Path"
    cop_style: Dict = field(default_factory=lambda: dict(color="lime",   lw=1.0, ls="-"))
    legend: Dict = field(default_factory=lambda: dict(loc="upper left", frameon=False))


@dataclass
class Stack3DAxisStyle:
    grid: bool = False
    tick_params: Dict = field(default_factory=lambda: dict(labelsize=9))
    xticks: Optional[np.ndarray] = None
    yticks: Optional[np.ndarray] = None
    zticks: Optional[np.ndarray] = None
    xformatter: Optional[Callable] = None
    yformatter: Optional[Callable] = None
    zformatter: Optional[Callable] = None
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    zlabel: Optional[str] = None


@dataclass
class DistMarkerStyle:
    show: bool = True
    marker: str = "o"
    size: float = 5.0
    edge_width: float = 1.2
    edge_color: str = "black"
    face_color: str = "none"   # hollow
    zorder: int = 3
    label: str = "sampled sections"


@dataclass
class DistLineStyle:
    lw: float = 2.0
    ls: str = "-"
    color: Optional[str] = None
    label: str = "source"


@dataclass
class DistPlotStyle:
    line: DistLineStyle = field(default_factory=DistLineStyle)
    markers: DistMarkerStyle = field(default_factory=DistMarkerStyle)
    legend: Dict = field(default_factory=lambda: dict(loc="best"))
    grid: Dict = field(default_factory=lambda: dict(enabled=True, alpha=0.25))


# ---- module-level default instances (single source of truth) ----
_DEFAULT_STACK3D_STYLE       = Stack3DStyle()
_DEFAULT_STACK3D_AXIS_STYLE  = Stack3DAxisStyle()
_DEFAULT_CHORD_STYLE         = DistPlotStyle(
    line=DistLineStyle(lw=2.0, ls="-", color=None, label="c(z) source"),
    markers=DistMarkerStyle(show=True, marker="o", size=5.0, edge_width=1.2,
                            edge_color="black", face_color="none", zorder=3, label="sampled sections"),
    legend=dict(loc="best"),
    grid=dict(enabled=True, alpha=0.25),
)
_DEFAULT_BETA_STYLE          = DistPlotStyle(
    line=DistLineStyle(lw=2.0, ls="-", color=None, label="β(z) source"),
    markers=DistMarkerStyle(show=True, marker="o", size=5.0, edge_width=1.2,
                            edge_color="black", face_color="none", zorder=3, label="sampled sections"),
    legend=dict(loc="best"),
    grid=dict(enabled=True, alpha=0.25),
)

# ---- internal getters return deep copies so plotting can't mutate defaults ----
def _get_stack3d_style() -> Stack3DStyle:
    return deepcopy(_DEFAULT_STACK3D_STYLE)

def _get_stack3d_axis_style() -> Stack3DAxisStyle:
    return deepcopy(_DEFAULT_STACK3D_AXIS_STYLE)

def _get_chord_style() -> DistPlotStyle:
    return deepcopy(_DEFAULT_CHORD_STYLE)

def _get_beta_style() -> DistPlotStyle:
    return deepcopy(_DEFAULT_BETA_STYLE)


# ---- optional public tuning hooks (change defaults globally without per-call args) ----
def _update_nested_style(obj, updates: Dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and hasattr(obj, k):
            sub = getattr(obj, k)
            if hasattr(sub, "__dataclass_fields__"):
                _update_nested_style(sub, v)
            elif isinstance(sub, dict):
                sub.update(v)
            else:
                raise AttributeError(f"Cannot merge dict into {type(sub).__name__}")
        elif hasattr(obj, k):
            setattr(obj, k, v)
        else:
            raise AttributeError(f"Unknown field '{k}' in {type(obj).__name__}")

def set_stack3d_style(**kwargs) -> None:
    _update_nested_style(_DEFAULT_STACK3D_STYLE, kwargs)

def set_stack3d_axis_style(**kwargs) -> None:
    _update_nested_style(_DEFAULT_STACK3D_AXIS_STYLE, kwargs)

def set_chord_style(**kwargs) -> None:
    _update_nested_style(_DEFAULT_CHORD_STYLE, kwargs)

def set_beta_style(**kwargs) -> None:
    _update_nested_style(_DEFAULT_BETA_STYLE, kwargs)

@contextmanager
def override_viz_defaults(*, stack3d=None, axis=None, chord=None, beta=None):
    snap = (
        deepcopy(_DEFAULT_STACK3D_STYLE),
        deepcopy(_DEFAULT_STACK3D_AXIS_STYLE),
        deepcopy(_DEFAULT_CHORD_STYLE),
        deepcopy(_DEFAULT_BETA_STYLE),
    )
    try:
        if stack3d: set_stack3d_style(**stack3d)
        if axis:    set_stack3d_axis_style(**axis)
        if chord:   set_chord_style(**chord)
        if beta:    set_beta_style(**beta)
        yield
    finally:
        _DEFAULT_STACK3D_STYLE, _DEFAULT_STACK3D_AXIS_STYLE, _DEFAULT_CHORD_STYLE, _DEFAULT_BETA_STYLE
        (_DEFAULT_STACK3D_STYLE,
         _DEFAULT_STACK3D_AXIS_STYLE,
         _DEFAULT_CHORD_STYLE,
         _DEFAULT_BETA_STYLE) = snap


# ─────────────────────────────────────────────────────────────────────────────
# 1) 3D Stack Plotter (reads internal defaults; no per-call style args needed)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Stack3DPlotter(_BaseBladePlotter):
    elev: float = 22.0
    azim: float = -60.0
    decimate_every: Optional[int] = None
    pad: float = 0.05
    filename: str = "blade_stack_3d.png"

    # Back-compat optional overrides (if you *do* pass them)
    linewidth: Optional[float] = None
    show_centroid_axis: Optional[bool] = None
    show_cop_axis: Optional[bool] = None

    def plot(self) -> str:
        style = _get_stack3d_style()
        axis_style = _get_stack3d_axis_style()

        # apply optional overrides if given (back-compat)
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

        # Build stacked curves + z positions (XY units)
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

        fig = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # Color normalization baseline
        z_arr = np.asarray(z_vals_XY, float)
        zmin, zmax = float(z_arr.min()), float(z_arr.max())
        rng = max(1e-12, zmax - zmin)

        # Sections with centralized style
        cmap = plt.get_cmap(style.section_cmap)
        for XY, z_plot in zip(curves, z_vals_XY):
            X, Y = XY[:, 0], XY[:, 1]
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

        extras: List[np.ndarray] = []
        if style.show_centroid_axis and cent_pts is not None:
            ax.plot(cent_pts[:, 0], cent_pts[:, 1], cent_pts[:, 2],
                    label=style.centroid_label, **style.centroid_style)
            extras.append(cent_pts)
        if style.show_cop_axis and cop_pts is not None:
            ax.plot(cop_pts[:, 0], cop_pts[:, 1], cop_pts[:, 2],
                    label=style.cop_label, **style.cop_style)
            extras.append(cop_pts)

        # Axis labels (override-capable)
        unit = self.bin.units.XY
        ax.set_xlabel(axis_style.xlabel or f"X [{unit}]")
        ax.set_ylabel(axis_style.ylabel or f"Y [{unit}]")
        ax.set_zlabel(axis_style.zlabel or f"z [{unit}]")

        # Bounds, aspect, projection
        bounds = self._compute_bounds(curves, z_vals_XY, extra_pts=extras, pad=self.pad)
        self._apply_bounds(ax, bounds)

        # Axis grid/ticks/formatters/inversions
        if axis_style.grid:
            ax.grid(True)
        if axis_style.tick_params:
            ax.tick_params(**axis_style.tick_params)
        if axis_style.xticks is not None:
            ax.set_xticks(axis_style.xticks)
        if axis_style.yticks is not None:
            ax.set_yticks(axis_style.yticks)
        if axis_style.zticks is not None:
            ax.set_zticks(axis_style.zticks)
        if axis_style.xformatter is not None:
            ax.xaxis.set_major_formatter(axis_style.xformatter)
        if axis_style.yformatter is not None:
            ax.yaxis.set_major_formatter(axis_style.yformatter)
        if axis_style.zformatter is not None:
            ax.zaxis.set_major_formatter(axis_style.zformatter)
        if axis_style.invert_x:
            ax.invert_xaxis()
        if axis_style.invert_y:
            ax.invert_yaxis()
        if axis_style.invert_z:
            ax.invert_zaxis()

        # View & legend
        ax.view_init(elev=self.elev, azim=self.azim)
        if (style.show_centroid_axis and cent_pts is not None) or \
           (style.show_cop_axis and cop_pts is not None):
            ax.legend(**style.legend)

        ax.set_title(self._title())

        out_path = str(Path(self.outdir) / self.filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Local to 3D stacker (uses base helpers where possible)
    def _axis_points(
        self,
        rows: List[SectionRow],
        *,
        use_centroid: bool,
        z_to_XY: float
    ) -> Optional[np.ndarray]:
        pts: List[Tuple[float, float, float]] = []
        for r in rows:
            z_plot = self._z_for_row(r) * z_to_XY
            gp = (r.scaled.geom if r.scaled and r.scaled.geom else None)
            fp = (r.scaled.from_norm if r.scaled and r.scaled.from_norm else None)
            if use_centroid:
                x = (gp.cx if gp else (fp.cx if fp else None))
                y = (gp.cy if gp else (fp.cy if fp else None))
            else:
                x = (gp.x_cp if gp else (fp.x_cp if fp else None))
                y = (gp.y_cp if gp else (fp.y_cp if fp else None))
            if (x is None) or (y is None):
                continue
            pts.append((float(x), float(y), float(z_plot)))
        return None if not pts else np.asarray(pts, float)

    def _compute_bounds(
        self,
        curves: List[np.ndarray],
        z_vals_XY: List[float],
        *,
        extra_pts: Optional[List[np.ndarray]] = None,
        pad: float = 0.05
    ) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
        xs_min = [float(XY[:, 0].min()) for XY in curves]
        xs_max = [float(XY[:, 0].max()) for XY in curves]
        ys_min = [float(XY[:, 1].min()) for XY in curves]
        ys_max = [float(XY[:, 1].max()) for XY in curves]
        xmin, xmax = min(xs_min), max(xs_max)
        ymin, ymax = min(ys_min), max(ys_max)

        z_arr = np.asarray(z_vals_XY, float)
        zmin, zmax = float(z_arr.min()), float(z_arr.max())

        if extra_pts:
            for P in extra_pts:
                if P is None or P.size == 0:
                    continue
                xmin = min(xmin, float(P[:, 0].min())); xmax = max(xmax, float(P[:, 0].max()))
                ymin = min(ymin, float(P[:, 1].min())); ymax = max(ymax, float(P[:, 1].max()))
                zmin = min(zmin, float(P[:, 2].min())); zmax = max(zmax, float(P[:, 2].max()))

        def _padded(lo, hi):
            size = max(1e-12, hi - lo)
            margin = pad * size
            return lo - margin, hi + margin

        return _padded(xmin, xmax), _padded(ymin, ymax), _padded(zmin, zmax)

    def _apply_bounds(
        self,
        ax,
        bounds: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
        *,
        use_ortho: bool = True
    ) -> None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        xspan = max(1e-12, xmax - xmin)
        yspan = max(1e-12, ymax - ymin)
        zspan = max(1e-12, zmax - zmin)
        try:
            ax.set_box_aspect((xspan, yspan, zspan))
        except Exception:
            ax.set_aspect("auto")
        if use_ortho:
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 2) Chord vs z Plotter (reads internal defaults)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChordVsZPlotter(_BaseBladePlotter):
    filename: str = "chord_vs_z.png"

    def plot(self) -> str:
        style = _get_chord_style()
        z_to_XY = length_scale(self.bin.units.L, self.bin.units.XY)

        # continuous line from SOURCE (fallback to DIST if missing)
        z_src = np.asarray(self._z_axis_for_source() * z_to_XY, float)
        c_src = np.asarray(self.bin.c_src if self.bin.c_src.size else self.bin.c_dist, float)
        if z_src.size != c_src.size:
            raise RuntimeError("Chord source distribution misaligned with its z grid.")

        m = np.isfinite(z_src) & np.isfinite(c_src)
        zc, cc = z_src[m], c_src[m]
        if zc.size < 2:
            raise RuntimeError("Insufficient finite points in chord source distribution.")
        s = np.argsort(zc)
        zc, cc = zc[s], cc[s]

        # markers from SAMPLED grid (exact n positions)
        z_dist = np.asarray(self._z_axis_for_dist() * z_to_XY, float)
        c_dist = np.asarray(self.bin.c_dist, float)
        if z_dist.size != c_dist.size:
            raise RuntimeError("Chord sampled distribution misaligned with its z grid.")

        unit = self.bin.units.XY
        fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)

        # line
        ax.plot(
            zc, cc,
            lw=style.line.lw,
            ls=style.line.ls,
            color=style.line.color,
            label=style.line.label,
        )

        # markers
        if style.markers.show:
            mm = np.isfinite(z_dist) & np.isfinite(c_dist)
            zm, cm = z_dist[mm], c_dist[mm]
            if zm.size:
                ax.plot(
                    zm, cm,
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
        ax.set_ylabel(f"Chord c [{self.bin.units.c}]")
        if style.grid.get("enabled", True):
            ax.grid(True, alpha=style.grid.get("alpha", 0.25))
        ax.set_title("Chord distribution")
        ax.legend(**style.legend)

        out_path = str(Path(self.outdir) / self.filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3) Beta vs z Plotter (reads internal defaults)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Optional: thin container to orchestrate all plots (API unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BladeVisualizer:
    bin: SectionBin
    outdir: str

    def plot_stack_3d(self, **kwargs) -> str:
        """Backwards-compatible wrapper."""
        return Stack3DPlotter(self.bin, self.outdir, **kwargs).plot()

    def plot_chord_vs_z(self, **kwargs) -> str:
        # Back-compat convenience: allow markers=True to toggle default markers.show
        if "markers" in kwargs:
            show = bool(kwargs.pop("markers"))
            set_chord_style(markers={"show": show})
        return ChordVsZPlotter(self.bin, self.outdir, **kwargs).plot()

    def plot_beta_vs_z(self, **kwargs) -> str:
        if "markers" in kwargs:
            show = bool(kwargs.pop("markers"))
            set_beta_style(markers={"show": show})
        return BetaVsZPlotter(self.bin, self.outdir, **kwargs).plot()