from svgpathtools import svg2paths2
from sectionproperties.pre import Geometry
import numpy as np
import matplotlib.pyplot as plt


class RawGeometry:
    def __init__(self, svg_path, label, blob_threshold=100):
        self.svg_path = svg_path
        self.label = label
        self.blob_threshold = blob_threshold
        self.raw_points = []
        self.geometry = None

    def path_to_points(self, path_obj, num_points=2000):
        points = []
        for segment in path_obj:
            for t in np.linspace(0, 1, num_points):
                pt = segment.point(t)
                x, y = pt.real, pt.imag
                if not (np.isclose(x, 0.0) and np.isclose(y, 0.0)):
                    points.append((x, y))
        return points

    def deduplicate_points(self, points, tol=1e-8):
        if not points:
            return []
        deduped = [points[0]]
        for i in range(1, len(points)):
            if np.linalg.norm(np.subtract(points[i], deduped[-1])) > tol:
                deduped.append(points[i])
        return deduped

    def points_to_facets(self, points):
        return [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]

    def midpoint(self, points):
        x_vals, y_vals = zip(*points)
        return [(sum(x_vals) / len(points), sum(y_vals) / len(points))]

    def extract(self):
        paths, _, _ = svg2paths2(self.svg_path)
        geometries = []
        self.raw_points.clear()

        for path in paths:
            points = self.path_to_points(path)
            if not points:
                continue

            xs, ys = zip(*points)
            if max(xs) < self.blob_threshold and max(ys) < self.blob_threshold:
                continue  # skip origin blobs

            points = self.deduplicate_points(points)
            if len(points) < 3:
                continue

            facets = self.points_to_facets(points)
            control = self.midpoint(points)
            try:
                geom = Geometry.from_points(points, facets, control)
                geom.geom = geom.geom.buffer(0)
                geometries.append(geom)
                self.raw_points.append(points)
            except Exception as e:
                print(f"⚠️ Skipping invalid path in {self.label}: {e}")

        if not geometries:
            raise ValueError(f"No valid geometry found in: {self.svg_path}")

        base = geometries[0]
        for g in geometries[1:]:
            try:
                g.geom = g.geom.buffer(0)
                base.geom = base.geom.buffer(0)
                base = base - g
            except Exception as e:
                print(f"⚠️ Boolean subtraction skipped in {self.label}: {e}")

        self.geometry = base
        return base

    def plot(self, ax):
        if not self.raw_points:
            self.extract()

        ax.set_aspect("equal")
        ax.set_title(f"Raw Geometry: {self.label}")
        legend_labels = []

        for i, pts in enumerate(self.raw_points, start=1):
            px, py = zip(*pts)
            label = f"{self.label}-{i}"
            ax.plot(px, py, label=label)
            ax.text(px[0], py[0], str(i), fontsize=8, color="black")
            legend_labels.append(label)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Legend below the plot
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -1),
            ncol=10,  # Increase this number if labels wrap or overflow
            fontsize=6,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.8
        )
