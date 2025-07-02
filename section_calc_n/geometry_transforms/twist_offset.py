# geometry_transforms/twist_offset.py

import numpy as np
from sectionproperties.pre.geometry import Geometry

class TwistOffset:
    def __init__(self, geometry: Geometry, desired_twist_deg: float):
        self.geometry = geometry
        self.desired_twist_deg = desired_twist_deg

    def compute_chord_angle(self) -> float:
        points = np.array(self.geometry.geom.exterior.coords)
        idx_le = np.argmin(points[:, 0])
        idx_te = np.argmax(points[:, 0])
        x_le, y_le = points[idx_le]
        x_te, y_te = points[idx_te]

        return np.degrees(np.arctan2(y_te - y_le, x_te - x_le))

    def apply(self) -> Geometry:
        chord_angle = self.compute_chord_angle()
        delta_theta = self.desired_twist_deg - chord_angle

        if abs(delta_theta) < 0.1:
            return self.geometry  # No rotation needed
        return self.geometry.rotate_section(angle=delta_theta)
