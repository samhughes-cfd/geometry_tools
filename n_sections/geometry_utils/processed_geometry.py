# section_calc_n/utils/processed_geometry.py

from pathlib import Path
import logging
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.pre.pre import Material
from material_utils.assign_material import AssignMaterial
from typing import Union
from geometry_transforms.twist_offset import TwistOffset
from geometry_transforms.centroid_offset import CentroidOffset

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

        self._init_logging()

    def _init_logging(self) -> None:
        log_path = self.logs_dir / "ProcessedGeometry.log"  # ✅ Corrected line

        # Named logger per section
        self.logger = logging.getLogger(f"ProcessedGeometry.{self.label}")
        self.logger.propagate = False  # ⛔ Prevent double logging

        # Only add handler if it doesn't already exist
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
            for h in self.logger.handlers
        ):
            handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)
        self.logger.info("Logging initialized for ProcessedGeometry")


    def _cosine_resample_exterior(self, poly: Polygon, n_total: int) -> Polygon:
        if n_total < 3:
            self.logger.error("Need at least 3 exterior points for resampling.")
            raise ValueError("Need at least 3 exterior points for resampling.")

        xy = np.asarray(poly.exterior.coords[:-1])
        n_points = len(xy)

        if n_points < 2:
            self.logger.error("Polygon exterior has fewer than 2 points; cannot compute segment lengths.")
            raise ValueError("Polygon exterior has fewer than 2 points; cannot compute segment lengths.")

        seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        length = cum_len[-1]

        self.logger.debug("Resampling polygon: %d points, target=%d points, perimeter=%.6f", n_points, n_total, length)

        if length == 0.0:
            self.logger.error("Polygon has zero perimeter length; cannot resample.")
            raise ValueError("Polygon has zero perimeter length; cannot resample.")

        try:
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

            self.logger.debug("Resampled exterior generated with %d points", len(new_ext))
            return Polygon(new_ext, [ring.coords[:] for ring in poly.interiors])
        except Exception as e:
            self.logger.exception("Cosine resampling failed: %s", str(e))
            raise

    def extract_and_transform(self, twist_deg: float, cx: float, cy: float, material: Union[Material, dict[int, Material], None] = None) -> Geometry | CompoundGeometry:
        self.logger.info("Importing DXF geometry from %s", self.filepath)

        try:
            geom_raw = Geometry.from_dxf(
                dxf_filepath=self.filepath,
                spline_delta=self.spline_delta,
                degrees_per_segment=self.degrees_per_segment
            )
            self.logger.info("DXF import complete")
        except Exception as e:
            self.logger.error("DXF import failed: %s", e, exc_info=True)
            raise

        try:
            if isinstance(geom_raw, Geometry): 
                scaled_geom = scale(geom_raw.geom, xfact=0.001, yfact=0.001, origin=(0, 0))
                geom_raw = Geometry(geom=scaled_geom)
                self.logger.debug("Scaled single geometry to metres")
            elif isinstance(geom_raw, CompoundGeometry): 
                scaled_geoms = []
                for g in geom_raw.geoms:
                    scaled = scale(g.geom, xfact=0.001, yfact=0.001, origin=(0, 0))
                    scaled_geoms.append(Geometry(geom=scaled))
                geom_raw = CompoundGeometry(scaled_geoms)
                self.logger.debug("Scaled compound geometry to metres")
            else:
                raise TypeError("Unexpected geometry type when scaling.")
        except Exception as e:
            self.logger.error("Scaling geometry failed: %s", e, exc_info=True)
            raise

        # ───── Resample Exterior ─────
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
                dense = self._cosine_resample_exterior(p, self.exterior_nodes)
                dense_polys.append(dense)
                self.logger.debug("Resampled polygon %d with %d exterior nodes", i, self.exterior_nodes)
            except Exception as e:
                self.logger.warning("Resampling polygon %d failed: %s", i, e, exc_info=True)

        try:
            geom_list = [Geometry(geom=p) for p in dense_polys]
            geom = geom_list[0] if len(geom_list) == 1 else CompoundGeometry(geom_list)
            self.logger.info("Resampling and geometry reconstruction complete")
        except Exception as e:
            self.logger.error("Failed to reconstruct geometry from resampled polygons: %s", e, exc_info=True)
            raise

        # ───── Rotate ─────
        try:
            geom_rotated = TwistOffset(
                geometry=geom,
                desired_twist_deg=twist_deg,
                label=self.label,
                logs_dir=self.logs_dir,
                twist_axis_ratio=0.333
            ).apply()
            self.logger.info("Applied twist rotation: %.2f°", twist_deg)
        except Exception as e:
            self.logger.error("Twist transformation failed: %s", e, exc_info=True)
            raise

        # ───── Translate ─────
        try:
            geom_translated = CentroidOffset(
                geometry=geom_rotated,
                cx_target=cx,
                cy_target=cy,
                label=self.label,
                logs_dir=self.logs_dir
            ).apply()
            self.logger.info("Applied centroid translation to (%.3f, %.3f)", cx, cy)
        except Exception as e:
            self.logger.error("Centroid offset failed: %s", e, exc_info=True)
            raise
        self.geometry = geom_translated

        # ───── Assign Material(s) ─────
        try:
            AssignMaterial(
                geometry=geom_translated,
                material=material,
                logs_dir=self.logs_dir,
                label=self.label
            ).apply()

        except Exception as e:
            self.logger.error("Material assignment failed: %s", e, exc_info=True)
            raise

        
        return self.geometry