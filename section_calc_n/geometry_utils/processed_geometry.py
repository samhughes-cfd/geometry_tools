# section_calc_n/utils/processed_geometry.py

from pathlib import Path
import logging
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from geometry_transforms.twist_offset import TwistOffset
from geometry_transforms.centroid_offset import CentroidOffset


def _cosine_resample_exterior(poly: Polygon, n_total: int) -> Polygon:
    if n_total < 3:
        raise ValueError("Need at least 3 exterior points")

    xy = np.asarray(poly.exterior.coords[:-1])
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    length = cum_len[-1]
    t = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_total)))
    s_targets = t * length

    new_ext = []
    j = 0
    for s in s_targets:
        while s > cum_len[j + 1]:
            j += 1
        frac = (s - cum_len[j]) / seg_len[j]
        pt = (1 - frac) * xy[j] + frac * xy[j + 1]
        new_ext.append(tuple(pt))

    return Polygon(new_ext, [ring.coords[:] for ring in poly.interiors])


class ProcessedGeometry:
    def __init__(
        self,
        filepath: Path,
        label: str,
        logs_dir: Path,
        *,
        spline_delta: float = 0.05,
        degrees_per_segment: float = 0.5,
        exterior_nodes: int = 400,
    ):
        self.filepath = Path(filepath)
        self.label = label
        self.logs_dir = logs_dir
        self.spline_delta = spline_delta
        self.degrees_per_segment = degrees_per_segment
        self.exterior_nodes = exterior_nodes
        self.geometry: Geometry | CompoundGeometry | None = None

    def extract_and_transform(self, twist_deg: float, cx: float, cy: float) -> Geometry | CompoundGeometry:
        logging.info("[%s] Importing DXF: '%s'", self.label, self.filepath)

        geom_raw = Geometry.from_dxf(
            dxf_filepath=self.filepath,
            spline_delta=self.spline_delta,
            degrees_per_segment=self.degrees_per_segment
        )

        if isinstance(geom_raw, Geometry): 
            scaled_geom = scale(geom_raw.geom, xfact=0.001, yfact=0.001, origin=(0, 0)) # .dxf in mm scale to m
            geom_raw = Geometry(geom=scaled_geom)
        elif isinstance(geom_raw, CompoundGeometry): 
            scaled_geoms = []
            for g in geom_raw.geoms:
                scaled = scale(g.geom, xfact=0.001, yfact=0.001, origin=(0, 0)) # .dxf in mm scale to m
                scaled_geoms.append(Geometry(geom=scaled))
            geom_raw = CompoundGeometry(scaled_geoms)
        else:
            raise TypeError("Unexpected geometry type when scaling.")

        raw_polys = []
        if isinstance(geom_raw, Geometry):
            polys = geom_raw.geom.geoms if geom_raw.geom.geom_type == "MultiPolygon" else [geom_raw.geom]
            raw_polys.extend(polys)
        else:
            for g in geom_raw.geoms:
                polys = g.geom.geoms if g.geom.geom_type == "MultiPolygon" else [g.geom]
                raw_polys.extend(polys)

        dense_polys = []
        for i, p in enumerate(raw_polys):
            try:
                dense_polys.append(_cosine_resample_exterior(p, self.exterior_nodes))
            except Exception as e:
                logging.warning("[%s] Resampling polygon %d failed: %s", self.label, i, e)

        geom_list = [Geometry(geom=p) for p in dense_polys]
        geom = geom_list[0] if len(geom_list) == 1 else CompoundGeometry(geom_list)

        # ───── Rotate ─────────────────────────────────────────────
        geom_rotated = TwistOffset(
            geometry=geom,
            desired_twist_deg=twist_deg,
            label=self.label,
            logs_dir=self.logs_dir
        ).apply()

        # ───── Translate ──────────────────────────────────────────
        geom_translated = CentroidOffset(
            geometry=geom_rotated,
            cx_target=cx,
            cy_target=cy,
            label=self.label,
            logs_dir=self.logs_dir
        ).apply()

        self.geometry = geom_translated
        return self.geometry