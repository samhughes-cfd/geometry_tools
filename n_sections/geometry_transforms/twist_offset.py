# section_calc_n/geometry_transforms/twist_offset.py

import numpy as np
import logging
from pathlib import Path
from typing import Tuple
from shapely.geometry import Polygon, MultiPolygon
from sectionproperties.pre.geometry import Geometry, CompoundGeometry  # <-- add CompoundGeometry

class TwistOffset:
    """Apply twist alignment to a section based on blade twist and a defined twist axis."""

    def __init__(
        self,
        geometry: Geometry | CompoundGeometry,
        desired_twist_deg: float,
        label: str = "Unnamed",
        logs_dir: Path | None = None,
        twist_axis_ratio: float = 1 / 3  # 0.0 = leading edge, 1.0 = trailing edge
    ):
        self.geometry = geometry
        self.desired_twist_deg = desired_twist_deg
        self.label = label
        self.logs_dir = logs_dir
        self.twist_axis_ratio = twist_axis_ratio
        self._init_logging()

    def _init_logging(self):
        self.logger = logging.getLogger(f"TwistOffset.{self.label}")
        self.logger.setLevel(logging.DEBUG)
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / "TwistOffset.log"
            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
                       for h in self.logger.handlers):
                fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

    # --- NEW: robust outer polygon getter (handles CompoundGeometry, MultiPolygon, Polygon)
    def _largest_outer_polygon(self) -> Polygon:
        g = getattr(self.geometry, "geom", None)
        if isinstance(g, Polygon):
            return g
        if isinstance(g, MultiPolygon):
            return max(g.geoms, key=lambda p: p.area)
        # CompoundGeometry path: collect outer polygons from sub-geometries
        if hasattr(self.geometry, "geoms"):  # sectionproperties.CompoundGeometry
            polys = []
            for sub in self.geometry.geoms:
                s = sub.geom
                if isinstance(s, Polygon):
                    polys.append(s)
                elif isinstance(s, MultiPolygon):
                    polys.extend(list(s.geoms))
            if not polys:
                raise TypeError(f"[{self.label}] No polygonal geometry available for twist computation.")
            return max(polys, key=lambda p: p.area)
        raise TypeError(f"[{self.label}] Unsupported geometry type for twist: {type(self.geometry)}")

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

    def compute_blade_twist(self) -> float:
        """
        Compute chord angle (deg) between LE→TE and +x axis.
        Positive if TE is above LE (right-hand system).
        """
        try:
            LE, TE = self._chord_endpoints()
            dx = float(TE[0] - LE[0])
            dy = float(TE[1] - LE[1])
            chord_len = np.hypot(dx, dy)
            if chord_len < 1e-9:
                raise ValueError(f"[{self.label}] Chord length ~ 0; cannot compute twist.")
            chord_angle = np.degrees(np.arctan2(dy, dx))
            self.logger.debug(f"[{self.label}] chord_len={chord_len:.6e}, chord_angle={chord_angle:.3f}°")
            return chord_angle
        except Exception:
            self.logger.exception(f"[{self.label}] Failed to compute blade twist.")
            raise

    def compute_twist_center(self) -> tuple[float, float]:
        """Compute the twist center at `twist_axis_ratio` of chord from LE (e.g., 0.333 ≈ 33% chord)."""
        try:
            LE, TE = self._chord_endpoints()
            chord_vec = TE - LE
            center = LE + self.twist_axis_ratio * chord_vec
            return (float(center[0]), float(center[1]))
        except Exception:
            self.logger.exception(f"[{self.label}] Failed to compute twist center.")
            raise

    def apply(self) -> Geometry | CompoundGeometry:
        """
        Rotate the geometry to align with the desired chord angle
        about the twist center (ratio along LE→TE).
        """
        try:
            current_twist = self.compute_blade_twist()
            twist_center = self.compute_twist_center()
            delta_twist = self.desired_twist_deg - current_twist

            self.logger.info(
                f"[{self.label}] Current twist: {current_twist:.2f}°, "
                f"Desired: {self.desired_twist_deg:.2f}°, "
                f"Delta: {delta_twist:.2f}° @ axis_ratio={self.twist_axis_ratio:.3f}"
            )
            self.logger.info(
                f"[{self.label}] Twist center: ({twist_center[0]:.6f}, {twist_center[1]:.6f})"
            )

            if abs(delta_twist) < 0.1:
                self.logger.info(f"[{self.label}] Twist offset negligible — no rotation applied.")
                return self.geometry

            rotated = self.geometry.rotate_section(
                angle=delta_twist,
                rot_point=twist_center
            )
            self.logger.info(
                f"[{self.label}] Section rotated by {delta_twist:.2f}° about twist center."
            )
            return rotated

        except Exception:
            self.logger.exception(f"[{self.label}] Twist offset application failed.")
            raise