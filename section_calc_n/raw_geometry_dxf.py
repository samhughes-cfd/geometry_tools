# section_calc/raw_geometry_dxf.py

import logging
import ezdxf
import matplotlib.pyplot as plt
import itertools 
import numpy as np
from pathlib import Path
from typing import List, Tuple


class RawDXFPreview:
    """
    Dense, faithful preview of DXF linework.
    Handles LINE, (LW)POLYLINE, SPLINE and ARC entities, interpolating points so
    curves look continuous.  Stores sampled vertices for re-use.
    """

    def __init__(self, filepath: str | Path, label: str,
                 *, n_line: int = 50, n_poly: int = 10) -> None:
        self.filepath = Path(filepath)
        self.label = label
        self.n_line = max(n_line, 2)
        self.n_poly = max(n_poly, 2)
        self._raw_pts: List[List[Tuple[float, float]]] = []  # populated on first plot

    # ───────────────────────── helpers ──────────────────────────────
    @staticmethod
    def _densify(p0, p1, n) -> List[Tuple[float, float]]:
        xs = np.linspace(p0[0], p1[0], n)
        ys = np.linspace(p0[1], p1[1], n)
        return list(zip(xs, ys))

    # ───────────────────────── extraction ───────────────────────────
    def _collect_points(self) -> None:
        if self._raw_pts:      # already done
            return

        logging.info("DXF preview: reading %s", self.filepath)
        doc = ezdxf.readfile(self.filepath)

        for ent in doc.modelspace():
            typ = ent.dxftype()
            if typ not in {"LINE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ARC"}:
                continue

            try:
                if typ == "LINE":
                    p0 = (ent.dxf.start.x, ent.dxf.start.y)
                    p1 = (ent.dxf.end.x,   ent.dxf.end.y)
                    pts = self._densify(p0, p1, self.n_line)

                elif typ in {"LWPOLYLINE", "POLYLINE"}:
                    verts = [(x, y) for x, y, *_ in ent.get_points()]
                    pts: List[Tuple[float, float]] = []
                    for a, b in zip(verts, verts[1:]):
                        pts.extend(self._densify(a, b, self.n_poly)[:-1])
                    pts.append(verts[-1])

                elif typ == "SPLINE":
                    # 200 pts is plenty for visual fidelity
                    pts = [(x, y) for x, y, *_ in ent.approximate(200)]

                elif typ == "ARC":
                    th = np.linspace(np.radians(ent.dxf.start_angle),
                                     np.radians(ent.dxf.end_angle), 100)
                    cx, cy, r = ent.dxf.center.x, ent.dxf.center.y, ent.dxf.radius
                    pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in th]

                self._raw_pts.append(pts)

            except Exception as exc:         # skip malformed entities
                logging.debug("DXF entity %s skipped: %s", typ, exc)

        if not self._raw_pts:
            raise ValueError(f"No drawable entities in {self.filepath}")

    # ───────────────────────── plotting ─────────────────────────────

    def plot(
        self,
        ax,
        *,
        cmap: str = "tab20",
        annotate_every: int = 5,        # ← new kwarg
    ) -> None:
        """Plot the sampled DXF entities on *ax*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        cmap : str
            Name of any Matplotlib colormap. Each entity gets the next colour.
        annotate_every : int
            Put a tiny index label on every *n-th* entity (1-based).  Set to 0
            or ``None`` to disable labelling altogether.
        """
        self._collect_points()

        colours = itertools.cycle(plt.get_cmap(cmap).colors)

        for idx, (pts, col) in enumerate(zip(self._raw_pts, colours), start=1):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, lw=1.2, color=col)

            # --- entity index label -------------------------------------
            if annotate_every and idx % annotate_every == 0:
                # choose midpoint for readability
                mid = len(xs) // 2
                ax.text(
                    xs[mid],
                    ys[mid],
                    str(idx),
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
                )

        ax.set_title(f"Raw DXF Geometry: {self.label}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", linewidth=0.5)
