# section_calc/raw_geometry_dxf.py
from __future__ import annotations
from pathlib import Path
import logging
import ezdxf
import matplotlib.pyplot as plt
from sectionproperties.pre.geometry import Geometry, CompoundGeometry


class RawDXFPreview:
    """Quick-and-dirty DXF linework preview (lines & polylines)."""

    def __init__(self, filepath: str | Path, label: str):
        self.filepath = Path(filepath)
        self.label = label

    def plot(self, ax) -> None:
        ax.set_title(f"Raw DXF Geometry: {self.label}")
        try:
            doc = ezdxf.readfile(self.filepath)
            msp = doc.modelspace()
            for e in msp:
                if e.dxftype() == "LINE":
                    xs = [e.dxf.start.x, e.dxf.end.x]
                    ys = [e.dxf.start.y, e.dxf.end.y]
                    ax.plot(xs, ys, lw=0.4, color="black")
                elif e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                    pts = [(v[0], v[1]) for v in e.get_points()]
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, lw=0.4, color="black")
            ax.set_aspect("equal", "box")
            ax.axis("off")
        except Exception as err:
            ax.text(0.5, 0.5, f"DXF preview failed:\n{err}",
                    ha="center", va="center")
            ax.axis("off")


class ProcessedGeometryDXF:
    """Wraps sectionproperties Geometry/CompoundGeometry built from a DXF."""

    def __init__(
        self,
        filepath: str | Path,
        label: str,
        *,
        spline_delta: float = 0.05,
        degrees_per_segment: float = 0.5,
    ):
        self.filepath = Path(filepath)
        self.label = label
        self.spline_delta = spline_delta
        self.degrees_per_segment = degrees_per_segment
        self.geometry: Geometry | CompoundGeometry | None = None

    def extract(self) -> Geometry | CompoundGeometry:
        from sectionproperties.pre.geometry import Geometry  # local import

        logging.info("%s – importing DXF: %s", self.label, self.filepath)
        self.geometry = Geometry.from_dxf(
            dxf_filepath=self.filepath,
            spline_delta=self.spline_delta,
            degrees_per_segment=self.degrees_per_segment,
        )
        return self.geometry

    def plot(self, ax, *, legend: bool = False) -> None:
        ax.set_title(f"Processed Geometry: {self.label}")
        if self.geometry is None:                     # safety net
            ax.text(0.5, 0.5, "No geometry extracted",
                    ha="center", va="center")
            ax.axis("off")
        else:
            self.geometry.plot_geometry(ax=ax, legend=legend, labels=())
            ax.set_aspect("equal", "box")