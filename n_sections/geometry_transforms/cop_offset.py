# section_calc_n/geometry_transforms/cop_offset.py

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from sectionproperties.pre.geometry import Geometry

class CopOffset:
    """
    Translate the section so the origin is at a prescribed fraction along the chord
    from the leading edge (LE) to the trailing edge (TE).

    fraction: 0.0 -> LE, 0.25 -> quarter-chord, 1.0 -> TE
    """

    def __init__(
        self,
        geometry: Geometry,
        fraction: float = 0.25,
        *,
        label: str = "Unnamed",
        logs_dir: Path | None = None,
    ):
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"'fraction' must be in [0, 1]; got {fraction}")
        self.geometry = geometry
        self.fraction = fraction
        self.label = label
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self._init_logging()

    def _init_logging(self) -> None:
        self.logger = logging.getLogger(f"CopOffset.{self.label}")
        self.logger.setLevel(logging.DEBUG)

        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / "CopOffset.log"
            if not any(
                isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
                for h in self.logger.handlers
            ):
                fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                )
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

    def _largest_outer_polygon(self) -> Polygon:
        g = self.geometry.geom
        if isinstance(g, Polygon):
            return g
        if isinstance(g, MultiPolygon):
            return max(g.geoms, key=lambda p: p.area)
        if hasattr(self.geometry, "geoms"):  # CompoundGeometry
            polys = []
            for sub in self.geometry.geoms:
                s = sub.geom
                if isinstance(s, Polygon):
                    polys.append(s)
                elif isinstance(s, MultiPolygon):
                    polys.extend(list(s.geoms))
            if not polys:
                raise TypeError(f"[{self.label}] No polygonal geometry available.")
            return max(polys, key=lambda p: p.area)
        raise TypeError(f"[{self.label}] Unsupported geometry type: {type(g)}")

    def _chord_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        outer = self._largest_outer_polygon()
        coords = np.asarray(outer.exterior.coords)
        i_le = np.argmin(coords[:, 0])
        i_te = np.argmax(coords[:, 0])
        if i_le == i_te:
            raise ValueError(f"[{self.label}] Degenerate chord: LE and TE coincide.")
        LE = coords[i_le]
        TE = coords[i_te]
        return LE, TE

    def apply(self) -> Geometry:
        try:
            LE, TE = self._chord_endpoints()
            chord_vec = TE - LE
            P = LE + self.fraction * chord_vec
            self.logger.info(
                f"[{self.label}] Origin -> {self.fraction*100:.1f}% chord at "
                f"({P[0]:.6f}, {P[1]:.6f}); LE=({LE[0]:.6f},{LE[1]:.6f}), "
                f"TE=({TE[0]:.6f},{TE[1]:.6f})"
            )
            moved = self.geometry.shift_section(
                x_offset=-float(P[0]), y_offset=-float(P[1])
            )
            self.logger.info(
                f"[{self.label}] Translation applied: dx={-float(P[0]):.6f}, dy={-float(P[1]):.6f}"
            )
            return moved
        except Exception:
            self.logger.exception(f"[{self.label}] CopOffset failed.")
            raise