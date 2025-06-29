from svgpathtools import svg2paths2
from sectionproperties.pre import Geometry
import numpy as np


class FixedGeometry:
    def __init__(self, svg_path, label):
        self.svg_path = svg_path
        self.label = label
        self.geometry = None

    def path_to_points(self, path_obj, num_points=2000):
        """Convert an SVG path to (x, y) points."""
        points = []
        for segment in path_obj:
            for t in np.linspace(0, 1, num_points):
                pt = segment.point(t)
                x, y = pt.real, pt.imag
                if not (np.isclose(x, 0.0) and np.isclose(y, 0.0)):
                    points.append((x, y))
        return points

    def deduplicate_points(self, points, tol=1e-8):
        """Remove near-duplicate consecutive points."""
        if not points:
            return []
        deduped = [points[0]]
        for i in range(1, len(points)):
            if np.linalg.norm(np.subtract(points[i], deduped[-1])) > tol:
                deduped.append(points[i])
        return deduped

    def points_to_facets(self, points):
        """Return facet index pairs forming a closed loop."""
        return [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]

    def midpoint(self, points):
        """Return a rough centroid."""
        x_vals, y_vals = zip(*points)
        return [(sum(x_vals) / len(points), sum(y_vals) / len(points))]

    def build(self):
        """Construct and union all valid geometries in the SVG."""
        paths, _, _ = svg2paths2(self.svg_path)
        geometries = []

        for path in paths:
            pts = self.path_to_points(path)
            pts = self.deduplicate_points(pts)
            if len(pts) < 3:
                continue
            facets = self.points_to_facets(pts)
            control = self.midpoint(pts)
            try:
                geom = Geometry.from_points(pts, facets, control)
                geom.geom = geom.geom.buffer(0)
                if not geom.geom.is_empty:
                    geometries.append(geom)
            except Exception as e:
                print(f"⚠️ Skipping shape in {self.svg_path}: {e}")

        if not geometries:
            raise ValueError(f"No valid geometries from {self.svg_path}")

        base = geometries[0]
        for g in geometries[1:]:
            try:
                base = base - g
            except Exception as e:
                print(f"⚠️ Subtraction failed in {self.label}: {e}")
        self.geometry = base
        return base

    def plot(self, ax):
        """Plot the geometry."""
        if self.geometry is None:
            raise RuntimeError("Geometry not yet built. Call .build() first.")
        ax.set_title(f"Fixed Geometry: {self.label}")
        self.geometry.plot_geometry(ax=ax)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")