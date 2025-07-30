# section_calc_n\geometry_transforms\twist_offset.py

import numpy as np
import logging
from sectionproperties.pre.geometry import Geometry
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon

class TwistOffset:
    """Apply twist alignment to a section based on blade twist and a defined twist axis."""

    def __init__(
        self,
        geometry: Geometry,
        desired_twist_deg: float,
        label: str = "Unnamed",
        logs_dir: Path | None = None,
        twist_axis_ratio: float = 1 / 3 # 0.0 = leading edge, 1.0 = trailing edge
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
            log_path = self.logs_dir / f"TwistOffset.log"

            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in self.logger.handlers):
                fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        # Optional: also add console logging
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
            sh.setFormatter(formatter)
            self.logger

    def _get_exterior_coords(self) -> np.ndarray:
        """Extract the exterior coordinates of the geometry (largest polygon if MultiPolygon)."""
        geom = self.geometry.geom

        if isinstance(geom, MultiPolygon):
            largest = max(geom.geoms, key=lambda g: g.area)
            return np.asarray(largest.exterior.coords)
        elif isinstance(geom, Polygon):
            return np.asarray(geom.exterior.coords)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")

    def compute_blade_twist(self) -> float:
        """
        Compute the blade twist angle in degrees:
        Angle between the rotor plane (horizontal axis) and the chord line.
        Positive if the trailing edge is above the leading edge.
        """
        try:
            coords = self._get_exterior_coords()
            x_min_idx = np.argmin(coords[:, 0])
            x_max_idx = np.argmax(coords[:, 0])

            if x_min_idx == x_max_idx:
                raise ValueError(f"[{self.label}] Chord points degenerate — cannot compute twist.")

            dx = coords[x_max_idx, 0] - coords[x_min_idx, 0]
            dy = coords[x_max_idx, 1] - coords[x_min_idx, 1]
            chord_angle = np.degrees(np.arctan2(dy, dx))

            # Blade twist is negative of chord angle from LE to TE
            blade_twist = -chord_angle
            return blade_twist

        except Exception as e:
            self.logger.exception(f"[{self.label}] Failed to compute blade twist.")
            raise

    def compute_chord_length(self) -> float:
        """Compute length of the chord from leading to trailing edge."""
        try:
            coords = self._get_exterior_coords()
            x_le = np.min(coords[:, 0])
            x_te = np.max(coords[:, 0])
            return x_te - x_le
        except Exception as e:
            self.logger.exception(f"[{self.label}] Failed to compute chord length.")
            raise

    def apply(self) -> Geometry:
        """Rotate the geometry to align with the desired blade twist about the origin."""
        try:
            current_twist = self.compute_blade_twist()
            delta_twist = self.desired_twist_deg - current_twist

            self.logger.info(f"[{self.label}] Current blade twist: {current_twist:.2f}°, "
                             f"Desired: {self.desired_twist_deg:.2f}°, "
                             f"Delta: {delta_twist:.2f}°")

            if abs(delta_twist) < 0.1:
                self.logger.info(f"[{self.label}] Twist offset negligible — no rotation applied.")
                return self.geometry

            rotated = self.geometry.rotate_section(angle=delta_twist)

            self.logger.info(f"[{self.label}] Section rotated by {delta_twist:.2f}° "
                             f"(about origin, twist axis ratio was {self.twist_axis_ratio:.2%}).")

            return rotated

        except Exception as e:
            self.logger.exception(f"[{self.label}] Twist offset application failed.")
            raise