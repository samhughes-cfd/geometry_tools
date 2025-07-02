# geometry_transforms/centroid_offset.py

from sectionproperties.pre.geometry import Geometry

class CentroidOffset:
    def __init__(self, geometry: Geometry, cx_target: float, cy_target: float):
        self.geometry = geometry
        self.cx_target = cx_target
        self.cy_target = cy_target

    def apply(self) -> Geometry:
        # Compute current centroid from geometry
        current_cx, current_cy = self.geometry.calculate_geometric_properties().get_c()
        dx = self.cx_target - current_cx
        dy = self.cy_target - current_cy

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return self.geometry  # No shift needed
        return self.geometry.shift_section(dx, dy)
