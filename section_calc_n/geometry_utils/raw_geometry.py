# section_calc_n/utils/raw_geometry.py

import logging
import ezdxf
import numpy as np
from pathlib import Path
from typing import List, Tuple


class RawGeometry:
    def __init__(
        self,
        filepath: str | Path,
        label: str,
        *,
        n_line: int = 50,
        n_poly: int = 10,
        logs_dir: Path = None
    ) -> None:
        self.filepath = Path(filepath)
        self.label = label
        self.n_line = max(n_line, 2)
        self.n_poly = max(n_poly, 2)
        self._raw_pts: List[List[Tuple[float, float]]] = []

        # ───── Set up logger ─────
        self.logger = logging.getLogger(f"RawGeometry.{label}")
        self.logger.setLevel(logging.INFO)

        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"raw_geometry_{label}.log"
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(file_handler)

    def _densify(self, p0, p1, n) -> List[Tuple[float, float]]:
        xs = np.linspace(p0[0], p1[0], n)
        ys = np.linspace(p0[1], p1[1], n)
        return list(zip(xs, ys))

    def extract(self) -> List[List[Tuple[float, float]]]:
        if self._raw_pts:
            return self._raw_pts

        self.logger.info("Reading DXF file: %s", self.filepath)
        try:
            doc = ezdxf.readfile(self.filepath)
        except Exception as e:
            self.logger.error("Failed to read DXF file %s: %s", self.filepath, e)
            raise

        entity_count = 0
        for ent in doc.modelspace():
            typ = ent.dxftype()
            if typ not in {"LINE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ARC"}:
                continue

            try:
                if typ == "LINE":
                    p0 = (ent.dxf.start.x, ent.dxf.start.y)
                    p1 = (ent.dxf.end.x, ent.dxf.end.y)
                    pts = self._densify(p0, p1, self.n_line)

                elif typ in {"LWPOLYLINE", "POLYLINE"}:
                    verts = [(x, y) for x, y, *_ in ent.get_points()]
                    pts = []
                    for a, b in zip(verts, verts[1:]):
                        seg_pts = self._densify(a, b, self.n_poly)[:-1]
                        pts.extend(seg_pts)
                    pts.append(verts[-1])

                elif typ == "SPLINE":
                    pts = [(x, y) for x, y, *_ in ent.approximate(200)]

                elif typ == "ARC":
                    th = np.linspace(np.radians(ent.dxf.start_angle), np.radians(ent.dxf.end_angle), 100)
                    cx, cy, r = ent.dxf.center.x, ent.dxf.center.y, ent.dxf.radius
                    pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in th]

                self._raw_pts.append(pts)
                entity_count += 1

            except Exception as exc:
                self.logger.debug("Skipped malformed entity %s: %s", typ, exc)

        if not self._raw_pts:
            raise ValueError(f"No drawable entities in {self.filepath}")

        self.logger.info("Raw geometry extraction complete. %d drawable entities parsed.", entity_count)
        return self._raw_pts