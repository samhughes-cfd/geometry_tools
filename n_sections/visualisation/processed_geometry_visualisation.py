import numpy as np
import matplotlib.pyplot as plt

class ProcessedGeometryVisualisation:
    def __init__(self, geometry, label: str):
        self.geometry = geometry
        self.label = label

    def _count_nodes(self) -> tuple[int, int]:
        ext = inn = 0
        for g in (self.geometry.geoms if hasattr(self.geometry, 'geoms') else [self.geometry]):
            polys = g.geom.geoms if g.geom.geom_type == "MultiPolygon" else [g.geom]
            for poly in polys:
                ext += len(poly.exterior.coords) - 1
                inn += sum(len(r.coords) - 1 for r in poly.interiors)
        return ext, inn

    def plot_full_chord(self, ax, outline_lw=1.0, cp_size=10, legend_loc="upper right"):
        ax.set_title(f"Processed Geometry: {self.label}")
        self.geometry.plot_geometry(
            ax=ax,
            legend=False,
            labels=("control_points",),
            point_size=0,
            cp_size=cp_size,
            edgecolor="black",
            facecolor="none",
            linewidth=outline_lw
        )
        ext_nodes, int_nodes = self._count_nodes()
        h1 = ax.plot([], [], color="black", label=f"Exterior nodes = {ext_nodes}")[0]
        h2 = ax.plot([], [], color="none", label=f"Interior nodes = {int_nodes}")[0]
        ax.legend(handles=[h1, h2], loc=legend_loc, frameon=False, fontsize=8)
        ax.set_aspect("equal", "box")

    def plot_te_zoom(self, te_span_pct=8, figsize=(7, 6), outline_lw=1.0, cp_size=10, legend_loc="upper right"):
        pts = np.asarray(self.geometry.points)
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        chord = x_max - x_min
        x_zoom_min = x_max - (te_span_pct / 100.0) * chord

        fig, (ax_full, ax_zoom) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[2, 1], constrained_layout=True
        )
        fig.suptitle("Cross-Section Geometry", fontsize=14)

        self.plot_full_chord(ax_full, outline_lw=outline_lw, cp_size=cp_size, legend_loc=legend_loc)
        self.plot_full_chord(ax_zoom, outline_lw=outline_lw, cp_size=cp_size, legend_loc=legend_loc)

        mask = pts[:, 0] >= x_zoom_min
        ax_zoom.set_xlim(x_zoom_min, x_max)
        ax_zoom.set_ylim(pts[mask][:, 1].min() - 1, pts[mask][:, 1].max() + 1)
        ax_zoom.set_xlabel("x [model units]")

        return fig, (ax_full, ax_zoom)
