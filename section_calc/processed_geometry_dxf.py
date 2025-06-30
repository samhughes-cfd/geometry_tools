# ─── section_calc/processed_geometry_dxf.py ──────────────────────────
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sectionproperties.pre.geometry import Geometry, CompoundGeometry


# ───────────────────────── tunables ──────────────────────────────────
EXTERIOR_NODES = 400    # total nodes along the air-foil exterior
SPLINE_DELTA   = 0.05   # mm – Shapely arc sampling (Geometry.from_dxf)
DEG_PER_SEG    = 0.5    # °  – Shapely arc sampling (Geometry.from_dxf)
# you can still override these via __init__ kwargs
# ─────────────────────────────────────────────────────────────────────


def _cosine_resample_exterior(poly: Polygon, n_total: int) -> Polygon:
    """Return *poly* with its exterior ring resampled by a half-cosine law."""
    if n_total < 3:
        raise ValueError("Need at least 3 exterior points")

    xy = np.asarray(poly.exterior.coords[:-1])           # closed → open list
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    length  = cum_len[-1]

    # Cosine-clustered arc-length positions
    t = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_total)))
    s_targets = t * length

    new_ext = []
    j = 0
    for s in s_targets:
        while s > cum_len[j + 1]:
            j += 1
        frac = (s - cum_len[j]) / seg_len[j]
        pt = (1 - frac) * xy[j] + frac * xy[j + 1]
        new_ext.append(tuple(pt))

    return Polygon(new_ext, [ring.coords[:] for ring in poly.interiors])


class ProcessedGeometryDXF:
    """Build a *sectionproperties* Geometry/CompoundGeometry from a DXF,
    applying cosine-law vertex clustering only on the exterior air-foil loop."""

    def __init__(
        self,
        filepath: str | Path,
        label: str,
        *,
        spline_delta: float = SPLINE_DELTA,
        degrees_per_segment: float = DEG_PER_SEG,
        exterior_nodes: int = EXTERIOR_NODES,
    ):
        self.filepath = Path(filepath)
        self.label = label
        self.spline_delta = spline_delta
        self.degrees_per_segment = degrees_per_segment
        self.exterior_nodes = exterior_nodes
        self.geometry: Geometry | CompoundGeometry | None = None

    # ---------------------------------------------------------------- extract
    def extract(self) -> Geometry | CompoundGeometry:
        """Import DXF → resample exterior via cosine law, keep holes uniform."""
        logging.info("%s – importing DXF: %s", self.label, self.filepath)

        geom_raw = Geometry.from_dxf(
            dxf_filepath=self.filepath,
            spline_delta=self.spline_delta,
            degrees_per_segment=self.degrees_per_segment,
        )

        # Unwrap every Shapely Polygon, flattening MultiPolygons
        raw_polys: list[Polygon] = []
        if isinstance(geom_raw, Geometry):           # single region
            shp = geom_raw.geom
            raw_polys.extend(
                shp.geoms if shp.geom_type == "MultiPolygon" else [shp]
            )
        else:                                        # CompoundGeometry
            for g in geom_raw.geoms:
                shp = g.geom
                raw_polys.extend(
                    shp.geoms if shp.geom_type == "MultiPolygon" else [shp]
                )

        # Cosine-cluster the exterior ring only
        dense_polys = [
            _cosine_resample_exterior(p, self.exterior_nodes) for p in raw_polys
        ]

        # Re-wrap into sectionproperties objects
        geom_list = [Geometry(geom=p) for p in dense_polys]
        self.geometry = (
            geom_list[0] if len(geom_list) == 1 else CompoundGeometry(geom_list)
        )
        return self.geometry

    # ---------------------------------------------------------------- plot
    def plot(
        self,
        ax,
        *,
        outline_lw: float = 1.0,
        cp_size: int = 10,
        legend_loc: str = "upper right",
    ) -> None:
        """Plot clean outline + CPs and list ext/int node counts in legend."""
        ax.set_title(f"Processed Geometry: {self.label}")

        if self.geometry is None:
            ax.text(0.5, 0.5, "No geometry extracted",
                    ha="center", va="center")
            ax.axis("off")
            return

        # ── 1. draw outline (no vertex dots) + control-points ────────────
        self.geometry.plot_geometry(
            ax=ax,
            legend=False,
            labels=("control_points",),
            point_size=0,          # hide vertex markers
            cp_size=cp_size,
            edgecolor="black",
            facecolor="none",
            linewidth=outline_lw,
        )

        # ── 2. compute exterior & interior node counts ───────────────────
        def _counts_from_geom(g: Geometry):
            p = g.geom
            if p.geom_type == "MultiPolygon":
                # sum all outer + inner rings in the collection
                ext = sum(len(poly.exterior.coords) - 1 for poly in p.geoms)
                inn = sum(
                    len(r.coords) - 1
                    for poly in p.geoms
                    for r in poly.interiors
                )
            else:  # single Polygon
                ext = len(p.exterior.coords) - 1
                inn = sum(len(r.coords) - 1 for r in p.interiors)
            return ext, inn

        ext_nodes = 0
        int_nodes = 0
        if isinstance(self.geometry, Geometry):
            ext_nodes, int_nodes = _counts_from_geom(self.geometry)
        else:  # CompoundGeometry
            for sub in self.geometry.geoms:
                e, i = _counts_from_geom(sub)
                ext_nodes += e
                int_nodes += i

        # ── 3. build legend handles (invisible) ──────────────────────────
        h1 = ax.plot([], [], color="black",
                     label=f"Exterior nodes = {ext_nodes}")[0]
        h2 = ax.plot([], [], color="none",
                     label=f"Interior nodes = {int_nodes}")[0]

        ax.legend(handles=[h1, h2], loc=legend_loc,
                  frameon=False, fontsize=8)

        ax.set_aspect("equal", "box")

        # ---------- helper to count exterior / interior nodes -----------------
    def _ext_int_node_counts(self) -> tuple[int, int]:
        ext = inn = 0
        for g in (self.geometry.geoms
                  if isinstance(self.geometry, CompoundGeometry)
                  else [self.geometry]):
            p = g.geom
            polys = p.geoms if p.geom_type == "MultiPolygon" else [p]
            for poly in polys:
                ext += len(poly.exterior.coords) - 1
                inn += sum(len(r.coords) - 1 for r in poly.interiors)
        return ext, inn

    # ---------- two-row subplot: full view + TE zoom ----------------------
    def plot_te_zoom(
        self,
        *,
        te_span_pct: float = 8,        # width of zoom window [% chord]
        figsize: tuple[int, int] = (7, 6),
        outline_lw: float = 1.0,
        cp_size: int = 10,
        legend_loc: str = "upper right",
    ):
        """
        Create a (fig, (ax_full, ax_zoom)) pair:
        • ax_full – full air-foil outline
        • ax_zoom – trailing-edge window covering the last *te_span_pct* %
        """
        if self.geometry is None:
            raise RuntimeError("Call .extract() before plotting")

        # -------- compute chord & zoom window ----------------------------
        pts = np.asarray(self.geometry.points)
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        chord = x_max - x_min
        x_zoom_min = x_max - (te_span_pct / 100.0) * chord

        # -------- figure & axes -----------------------------------------
        fig, (ax_full, ax_zoom) = plt.subplots(
            2, 1, figsize=figsize, sharex=False, sharey=False,
            height_ratios=[2, 1], constrained_layout=True
        )
        fig.suptitle("Cross-Section Geometry", fontsize=14)

        # -------- common draw routine -----------------------------------
        def _draw(ax):
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
            ax.set_aspect("equal", "box")

        # full outline
        _draw(ax_full)

        # trailing-edge zoom
        _draw(ax_zoom)
        mask = pts[:, 0] >= x_zoom_min
        ax_zoom.set_xlim(x_zoom_min, x_max)
        ax_zoom.set_ylim(pts[mask][:, 1].min() - 1, pts[mask][:, 1].max() + 1)
        ax_zoom.set_xlabel("x [model units]")

        # -------- legend with node counts (on full view) ----------------
        ext_n, int_n = self._ext_int_node_counts()
        h1 = ax_full.plot([], [], color="black",
                          label=f"Exterior nodes = {ext_n}")[0]
        h2 = ax_full.plot([], [], color="none",
                          label=f"Interior nodes = {int_n}")[0]
        ax_full.legend(handles=[h1, h2], loc=legend_loc,
                       frameon=False, fontsize=8)

        return fig, (ax_full, ax_zoom)
