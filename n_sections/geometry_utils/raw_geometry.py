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

        # ───── Robust Logging Setup ─────
        self.logger = logging.getLogger(f"RawGeometry.{label}")
        self.logger.setLevel(logging.DEBUG)

        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"RawGeometry.log"

            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in self.logger.handlers):
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # Optional: Add console handler for real-time debugging
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

        self.logger.info("Logger initialized for RawGeometry %s", self.label)

    def _densify(self, p0: Tuple[float, float], p1: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
        """Generate `n` evenly spaced points between p0 and p1, inclusive."""
        if p0 == p1 or n <= 2:
            return [p0, p1]
        return [
            (
                p0[0] + i * (p1[0] - p0[0]) / (n - 1),
                p0[1] + i * (p1[1] - p0[1]) / (n - 1)
            )
            for i in range(n)
        ]

    def extract(self) -> List[List[Tuple[float, float]]]:
        if self._raw_pts:
            self.logger.debug("Raw geometry already extracted, skipping.")
            return self._raw_pts

        self.logger.info("Reading DXF file: %s", self.filepath)
        try:
            doc = ezdxf.readfile(self.filepath)
        except Exception as e:
            self.logger.error("Failed to read DXF file %s: %s", self.filepath, e, exc_info=True)
            raise

        entity_count = 0
        point_total = 0
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
                point_total += len(pts)
                self.logger.debug("Parsed %s entity with %d points", typ, len(pts))

            except Exception as exc:
                self.logger.warning("Skipped malformed %s entity: %s", typ, exc, exc_info=True)

        if not self._raw_pts:
            self.logger.error("No drawable entities found in DXF.")
            raise ValueError(f"No drawable entities in {self.filepath}")

        self.logger.info("Extraction complete: %d entities, %d total points.", entity_count, point_total)
        return self._raw_pts