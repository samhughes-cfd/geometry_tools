# n_sections/visualisation/processed_geometry_visualisation.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Iterable, Tuple, List, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


class ProcessedGeometryVisualisation:
    def __init__(
        self,
        geometry,
        label: str,
        *,
        cop_fraction: float = 0.25,   # fraction of chord from LE→TE to mark as CoP
        show_cop: bool = True,
        show_centroid: bool = True,
        show_chord: bool = True,
        show_zoom_box: bool = True,
    ):
        """
        Parameters
        ----------
        geometry : sectionproperties Geometry or CompoundGeometry
            Must expose .plot_geometry(ax=...), and either .geom (Polygon/MultiPolygon)
            or .geoms (iterable of Geometry).
        label : str
            Title/annotation label.
        cop_fraction : float
            CoP proxy at LE + frac * (TE - LE). Only visual; the actual origin handling is upstream.
        show_* : bool
            Toggle overlays.
        """
        self.geometry = geometry
        self.label = label
        self.cop_fraction = float(cop_fraction)
        self.show_cop = bool(show_cop)
        self.show_centroid = bool(show_centroid)
        self.show_chord = bool(show_chord)
        self.show_zoom_box = bool(show_zoom_box)

        # precompute geometric features
        self._combined = self._combined_geom()
        self._LE, self._TE, self._chord_len = self._compute_chord()
        self._cop_pt = self._compute_cop_point(self.cop_fraction) if self._chord_len > 0 else None
        c = self._combined.centroid
        self._centroid = (float(c.x), float(c.y))

    # ------------------------ helpers ------------------------

    def _iter_polys(self) -> Iterable[Polygon]:
        """Yield shapely Polygons for both Geometry and CompoundGeometry inputs."""
        g = self.geometry
        if hasattr(g, "geoms") and isinstance(getattr(g, "geoms"), (list, tuple)):
            # sectionproperties.CompoundGeometry
            for sub in g.geoms:
                s = sub.geom
                if isinstance(s, Polygon):
                    yield s
                elif isinstance(s, MultiPolygon):
                    for p in s.geoms:
                        yield p
        else:
            # sectionproperties.Geometry
            s = g.geom
            if isinstance(s, Polygon):
                yield s
            elif isinstance(s, MultiPolygon):
                for p in s.geoms:
                    yield p

    def _combined_geom(self) -> Polygon | MultiPolygon:
        polys = list(self._iter_polys())
        if not polys:
            raise ValueError("ProcessedGeometryVisualisation: no polygonal geometry found.")
        if len(polys) == 1:
            return polys[0]
        # union to compute centroid over all parts
        return unary_union(polys)

    def _largest_outer_polygon(self) -> Polygon:
        polys = list(self._iter_polys())
        return max(polys, key=lambda p: p.area)

    def _compute_chord(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """Return (LE(x,y), TE(x,y), chord_length) from the largest outer polygon."""
        outer = self._largest_outer_polygon()
        coords = np.asarray(outer.exterior.coords, dtype=float)
        i_le = int(np.argmin(coords[:, 0]))
        i_te = int(np.argmax(coords[:, 0]))
        LE = tuple(coords[i_le])
        TE = tuple(coords[i_te])
        chord_len = float(np.hypot(TE[0] - LE[0], TE[1] - LE[1]))
        return LE, TE, chord_len

    def _compute_cop_point(self, frac: float) -> Tuple[float, float]:
        """CoP proxy = LE + frac * (TE − LE)."""
        (xL, yL), (xT, yT), _ = self._LE, self._TE, self._chord_len
        return (xL + frac * (xT - xL), yL + frac * (yT - yL))

    def _count_nodes(self) -> tuple[int, int]:
        """Count exterior/interior nodes from sectionproperties geometry."""
        ext = inn = 0
        for g in (self.geometry.geoms if hasattr(self.geometry, 'geoms') else [self.geometry]):
            polys = g.geom.geoms if g.geom.geom_type == "MultiPolygon" else [g.geom]
            for poly in polys:
                ext += max(0, len(poly.exterior.coords) - 1)
                inn += sum(max(0, len(r.coords) - 1) for r in poly.interiors)
        return ext, inn

    def _annotate_points_full(self, ax, ms=40) -> List:
        """Annotate chord + LE/TE + CoP + centroid on the full view."""
        handles: List = []

        # chord + endpoints
        if self.show_chord and self._chord_len > 0:
            (xL, yL), (xT, yT) = self._LE, self._TE
            h_line = ax.plot([xL, xT], [yL, yT], linestyle="--", linewidth=1.0, label="Chord")[0]
            h_le = ax.scatter([xL], [yL], marker="^", s=ms, label="LE")
            h_te = ax.scatter([xT], [yT], marker="v", s=ms, label="TE")
            handles += [h_line, h_le, h_te]

        # CoP
        if self.show_cop and self._cop_pt is not None:
            xC, yC = self._cop_pt
            h_cop = ax.scatter([xC], [yC], marker="o", s=ms*1.1, label=f"CoP ({self.cop_fraction*100:.0f}% c)")
            handles.append(h_cop)

        # centroid
        if self.show_centroid and self._centroid is not None:
            cx, cy = self._centroid
            h_cent = ax.scatter([cx], [cy], marker="x", s=ms*1.2, label="C (centroid)")
            handles.append(h_cent)

        return handles

    # ------------------------ plotting ------------------------

    def plot_full_chord(self, ax, outline_lw: float = 1.0, cp_size: float = 10, legend_loc: str = "upper right"):
        ax.set_title(f"Processed Geometry: {self.label}")

        # draw geometry outlines
        self.geometry.plot_geometry(
            ax=ax,
            legend=False,
            labels=("control_points",),
            point_size=0,
            cp_size=cp_size,
            edgecolor="black",
            facecolor="none",
            linewidth=outline_lw,
        )

        # node counts
        ext_nodes, int_nodes = self._count_nodes()
        h1 = ax.plot([], [], color="black", label=f"Exterior nodes = {ext_nodes}")[0]
        h2 = ax.plot([], [], color="none", label=f"Interior nodes = {int_nodes}")[0]

        # overlays
        anno_handles = self._annotate_points_full(ax)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(handles=[h1, h2, *anno_handles], loc=legend_loc, frameon=False, fontsize=8)

    def plot_te_zoom(
        self,
        te_span_pct: float = 8,
        figsize: Tuple[float, float] = (7, 6),
        outline_lw: float = 1.0,
        cp_size: float = 10,
        legend_loc: str = "upper right",
    ):
        """Two-panel figure: full view + zoom near TE.

        Zoom panel only shows markers that lie within the zoom window [x_zoom_min, x_TE].
        """
        # chord extents
        (xL, yL), (xT, yT), chord_len = self._LE, self._TE, self._chord_len
        x_min, x_max = xL, xT
        x_zoom_min = x_max - (te_span_pct / 100.0) * max(1e-12, (x_max - x_min))

        fig, (ax_full, ax_zoom) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[2, 1], constrained_layout=True
        )
        fig.suptitle("Cross-Section Geometry", fontsize=14)

        # --- Full panel
        self.plot_full_chord(ax_full, outline_lw=outline_lw, cp_size=cp_size, legend_loc=legend_loc)

        # optional zoom box overlay on full view
        if self.show_zoom_box:
            y0, y1 = ax_full.get_ylim()
            rect = mpatches.Rectangle(
                (x_zoom_min, y0),
                (x_max - x_zoom_min),
                (y1 - y0),
                fill=False, linestyle=":", linewidth=0.8, alpha=0.6
            )
            ax_full.add_patch(rect)

        # --- Zoom panel: draw geometry fresh
        self.geometry.plot_geometry(
            ax=ax_zoom,
            legend=False,
            labels=("control_points",),
            point_size=0,
            cp_size=cp_size,
            edgecolor="black",
            facecolor="none",
            linewidth=outline_lw,
        )

        ms = 40
        handles: List = []

        # TE (always in range)
        h_te = ax_zoom.scatter([xT], [yT], marker="v", s=ms, label="TE")
        handles.append(h_te)

        # chord segment inside zoom window
        if self.show_chord and chord_len > 0:
            if x_zoom_min <= xT:
                if (xT - xL) != 0 and x_zoom_min > xL:
                    t = (x_zoom_min - xL) / (xT - xL)
                    y_zoom_min = yL + t * (yT - yL)
                    xs = [x_zoom_min, xT]
                    ys = [y_zoom_min, yT]
                else:
                    xs = [xL, xT]
                    ys = [yL, yT]
                h_line = ax_zoom.plot(xs, ys, linestyle="--", linewidth=1.0, label="Chord")[0]
                handles.append(h_line)

        # CoP if visible
        if self.show_cop and self._cop_pt is not None:
            xC, yC = self._cop_pt
            if xC >= x_zoom_min:
                h_cop = ax_zoom.scatter([xC], [yC], marker="o", s=ms*1.1,
                                        label=f"CoP ({self.cop_fraction*100:.0f}% c)")
                handles.append(h_cop)

        # centroid if visible
        if self.show_centroid and self._centroid is not None:
            cx, cy = self._centroid
            if cx >= x_zoom_min:
                h_cent = ax_zoom.scatter([cx], [cy], marker="x", s=ms*1.2, label="C (centroid)")
                handles.append(h_cent)

        # LE rarely visible
        if xL >= x_zoom_min:
            h_le = ax_zoom.scatter([xL], [yL], marker="^", s=ms, label="LE")
            handles.append(h_le)

        # limits from actual points in the zoom region (fallback to chord endpoints)
        if hasattr(self.geometry, "points") and self.geometry.points is not None:
            pts = np.asarray(self.geometry.points, dtype=float)
            mask = pts[:, 0] >= x_zoom_min
            if np.any(mask):
                y_min = float(pts[mask][:, 1].min())
                y_max = float(pts[mask][:, 1].max())
            else:
                y_min = min(yT, yL) - 0.05 * chord_len
                y_max = max(yT, yL) + 0.05 * chord_len
        else:
            y_min = min(yT, yL) - 0.05 * chord_len
            y_max = max(yT, yL) + 0.05 * chord_len

        ax_zoom.set_xlim(x_zoom_min, x_max)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_aspect("equal", "box")
        ax_zoom.set_xlabel("x [m]")
        ax_zoom.set_ylabel("y [m]")

        # legend only for items actually plotted in the window
        if handles:
            ax_zoom.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8)

        return fig, (ax_full, ax_zoom)