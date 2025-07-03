# section_calc_n\geometry_transforms\twist_offset.py

import numpy as np
import logging
from sectionproperties.pre.geometry import Geometry
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon

class TwistOffset:
    """Apply twist alignment to a section based on chord orientation."""

    def __init__(self, geometry: Geometry, desired_twist_deg: float, label: str = "Unnamed", logs_dir: Path | None = None):
        self.geometry = geometry
        self.desired_twist_deg = desired_twist_deg
        self.label = label
        self.logs_dir = logs_dir

        if self.logs_dir:
            self._setup_logger()

    def _setup_logger(self):
        log_file = Path(self.logs_dir) / f"twist_offset.log"
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

    def compute_chord_angle(self) -> float:
        try:
            geom = self.geometry.geom

            # Handle MultiPolygon by selecting the largest
            if isinstance(geom, MultiPolygon):
                largest = max(geom.geoms, key=lambda g: g.area)
                coords = np.asarray(largest.exterior.coords)
            elif isinstance(geom, Polygon):
                coords = np.asarray(geom.exterior.coords)
            else:
                raise TypeError(f"Unsupported geometry type: {type(geom)}")

            # Find leading and trailing edge (min/max x)
            x_min_idx = np.argmin(coords[:, 0])
            x_max_idx = np.argmax(coords[:, 0])

            if x_min_idx == x_max_idx:
                raise ValueError(f"[{self.label}] Chord points degenerate — cannot compute twist.")

            dx = coords[x_max_idx, 0] - coords[x_min_idx, 0]
            dy = coords[x_max_idx, 1] - coords[x_min_idx, 1]
            angle = np.degrees(np.arctan2(dy, dx))

            return angle

        except Exception as e:
            logging.exception(f"[{self.label}] Failed to compute chord angle.")
            raise

    def apply(self) -> Geometry:
        """Rotate the geometry to align with the desired twist angle."""
        try:
            current_twist = self.compute_chord_angle()
            delta_twist = self.desired_twist_deg - current_twist

            logging.info(f"[{self.label}] Current twist: {current_twist:.2f}°, "
                         f"Desired: {self.desired_twist_deg:.2f}°, "
                         f"Delta: {delta_twist:.2f}°")

            if abs(delta_twist) < 0.1:
                logging.info(f"[{self.label}] Twist offset negligible — no rotation applied.")
                return self.geometry

            rotated = self.geometry.rotate_section(angle=delta_twist)
            logging.info(f"[{self.label}] Section rotated by {delta_twist:.2f}°.")
            return rotated

        except Exception as e:
            logging.exception(f"[{self.label}] Twist offset application failed.")
            raise