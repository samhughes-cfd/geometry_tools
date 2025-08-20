# section_calc_n/utils/processed_geometry.py

from pathlib import Path
import logging
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.pre.pre import Material
from material_utils.assign_material import AssignMaterial
from typing import Union, Literal
from collections.abc import Mapping

from geometry_transforms.twist_offset import TwistOffset
from geometry_transforms.centroid_offset import CentroidOffset
from geometry_transforms.cop_offset import CopOffset

# ─────────────────────────────────────────────────────────────────────────────
# CAD files are always in millimetres → convert to metres
MM_to_M: float = 1e-3
# ─────────────────────────────────────────────────────────────────────────────


class ProcessedGeometry:
    """
    Import a DXF airfoil section, scale (mm→m), optionally resample the exterior,
    optionally apply twist and/or origin placement, then (optionally) assign material(s).

    Toggle behaviour per-call via:
      - apply_twist: bool = False
      - apply_origin: bool = False     (ignored if False)
      - origin_mode: 'cop' | 'centroid'
    """

    def __init__(
        self,
        filepath: Path,
        label: str,
        logs_dir: Path,
        *,
        spline_delta: float = 0.05,
        degrees_per_segment: float = 0.5,
        exterior_nodes: int = 400,
        twist_axis_ratio: float = 0.333,
    ):
        self.filepath = Path(filepath)
        self.label = label
        self.logs_dir = logs_dir
        self.spline_delta = spline_delta
        self.degrees_per_segment = degrees_per_segment
        self.exterior_nodes = max(3, int(exterior_nodes))
        self.twist_axis_ratio = twist_axis_ratio
        self.geometry: Geometry | CompoundGeometry | None = None
        self._init_logging()

    # ─────────────────────────────────────────────────────────────────────────

    def _init_logging(self) -> None:
        log_path = self.logs_dir / "ProcessedGeometry.log"
        self.logger = logging.getLogger(f"ProcessedGeometry.{self.label}")
        self.logger.propagate = False
        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
            for h in self.logger.handlers
        ):
            handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Logging initialized for ProcessedGeometry")
        self.logger.info("Units scale set to %.6f (mm → m)", MM_to_M)

    # ─────────────────────────────────────────────────────────────────────────

    def _cosine_resample_exterior(self, poly: Polygon, n_total: int) -> Polygon:
        if n_total < 3:
            self.logger.error("Need at least 3 exterior points for resampling.")
            raise ValueError("Need at least 3 exterior points for resampling.")

        xy = np.asarray(poly.exterior.coords[:-1], dtype=float)
        n_points = len(xy)
        if n_points < 2:
            self.logger.error("Polygon exterior has fewer than 2 points; cannot compute segment lengths.")
            raise ValueError("Polygon exterior has fewer than 2 points; cannot compute segment lengths.")

        # De-duplicate consecutive points
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-12]
        xy = xy[keep]
        if len(xy) < 2:
            raise ValueError("Insufficient unique exterior points after deduplication.")

        seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        length = cum_len[-1]

        self.logger.debug("Resampling polygon: %d→%d points, perimeter=%.6f", n_points, n_total, length)

        if length <= 0.0 or not np.isfinite(length):
            self.logger.error("Polygon has non-positive perimeter length; cannot resample.")
            raise ValueError("Polygon has non-positive perimeter length; cannot resample.")

        try:
            # Cosine-spaced arclength targets
            t = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_total)))
            s_targets = t * length

            new_ext = []
            j = 0
            last = len(seg_len) - 1
            for s in s_targets:
                while j < last and s > cum_len[j + 1]:
                    j += 1
                if seg_len[j] < 1e-12:
                    pt = xy[j + 1]
                else:
                    frac = (s - cum_len[j]) / seg_len[j]
                    frac = min(max(frac, 0.0), 1.0)
                    pt = (1 - frac) * xy[j] + frac * xy[j + 1]
                new_ext.append((float(pt[0]), float(pt[1])))

            from shapely.validation import explain_validity
            candidate = Polygon(new_ext, [list(r.coords) for r in poly.interiors])
            if not candidate.is_valid:
                self.logger.warning("Resampled polygon invalid: %s; applying buffer(0).", explain_validity(candidate))
                candidate = candidate.buffer(0)
            return candidate
        except Exception as e:
            self.logger.exception("Cosine resampling failed: %s", str(e))
            raise

    # --- material helpers -------------------------------------------------

    def _to_material(self, mdict: dict) -> Material:
        """Build a sectionproperties Material from a CSV-style dict."""
        name = mdict.get("name", mdict.get("material_name", "material"))
        E = float(mdict.get("E", mdict.get("elastic_modulus")))
        nu = float(mdict.get("nu", mdict.get("poissons_ratio")))
        fy = float(mdict.get("fy", mdict.get("yield_strength")))
        rho = float(mdict.get("rho", mdict.get("density")))
        color = mdict.get("color", "lightgrey")
        return Material(
            name=name,
            elastic_modulus=E,
            poissons_ratio=nu,
            yield_strength=fy,
            density=rho,
            color=color,
        )

    def _normalise_material_arg(self, geom, material):
        """
        Return one of:
          - Material (for Geometry),
          - dict[int, Material] (for CompoundGeometry),
          - None
        """
        if material is None:
            return None
        if isinstance(material, Material):
            return material
        if isinstance(material, Mapping):
            # Mapping[int, Material] for per-part assignment?
            if material and all(isinstance(k, int) for k in material.keys()) and \
               all(isinstance(v, Material) for v in material.values()):
                return dict(material)
            # Otherwise assume CSV-style dict -> build a single Material
            return self._to_material(dict(material))
        raise TypeError(f"Unsupported material argument type: {type(material)}")

    # ─────────────────────────────────────────────────────────────────────────

    def extract_and_transform(
        self,
        twist_deg: float,
        *,
        # New toggles (default OFF)
        apply_twist: bool = False,
        apply_origin: bool = False,
        # Origin options (used only if apply_origin=True)
        origin_mode: Literal["cop", "centroid"] = "cop",
        cop_fraction: float = 0.25,
        cx: float = 0.0,
        cy: float = 0.0,
        # Material (optional)
        material: Union[Material, dict[int, Material], None] = None,
        # Resampling toggle (kept on by default)
        resample_exterior: bool = True,
    ) -> Geometry | CompoundGeometry:
        """
        Import + scale + (optional) resample + (optional) twist + (optional) origin placement; assign material(s).
        Returns a Geometry or CompoundGeometry in metres.
        """
        self.logger.info("Importing DXF geometry from %s", self.filepath)

        if apply_origin and origin_mode == "cop" and not (0.0 <= cop_fraction <= 1.0):
            raise ValueError(f"cop_fraction must be in [0,1]; got {cop_fraction}")

        # ── Import DXF
        try:
            geom_raw = Geometry.from_dxf(
                dxf_filepath=self.filepath,
                spline_delta=self.spline_delta,
                degrees_per_segment=self.degrees_per_segment,
            )
            self.logger.info("DXF import complete")
        except Exception as e:
            self.logger.error("DXF import failed: %s", e, exc_info=True)
            raise

        # ── Scale to metres (mm → m via global MM_to_M)
        try:
            if isinstance(geom_raw, Geometry):
                scaled_geom = scale(geom_raw.geom, xfact=MM_to_M, yfact=MM_to_M, origin=(0, 0))
                geom_raw = Geometry(geom=scaled_geom)
                self.logger.debug("Scaled single geometry by %.6f", MM_to_M)
            elif isinstance(geom_raw, CompoundGeometry):
                scaled_geoms = []
                for g in geom_raw.geoms:
                    scaled = scale(g.geom, xfact=MM_to_M, yfact=MM_to_M, origin=(0, 0))
                    scaled_geoms.append(Geometry(geom=scaled))
                geom_raw = CompoundGeometry(scaled_geoms)
                self.logger.debug("Scaled compound geometry by %.6f", MM_to_M)
            else:
                raise TypeError("Unexpected geometry type when scaling.")
        except Exception as e:
            self.logger.error("Scaling geometry failed: %s", e, exc_info=True)
            raise

        # ── Optional exterior resample
        if resample_exterior:
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
                if not geom_list:
                    self.logger.warning("All resamples failed; falling back to scaled input geometry.")
                    geom = geom_raw
                else:
                    geom = geom_list[0] if len(geom_list) == 1 else CompoundGeometry(geom_list)
                self.logger.info("Resampling and geometry reconstruction complete")
            except Exception as e:
                self.logger.error("Failed to reconstruct geometry from resampled polygons: %s", e, exc_info=True)
                raise
        else:
            geom = geom_raw
            self.logger.info("Exterior resampling disabled; using scaled input geometry.")

        # ── Twist alignment (toggle)
        if apply_twist:
            try:
                geom_rotated = TwistOffset(
                    geometry=geom,
                    desired_twist_deg=twist_deg,
                    label=self.label,
                    logs_dir=self.logs_dir,
                    twist_axis_ratio=self.twist_axis_ratio,
                ).apply()
                self.logger.info("Applied twist rotation: %.2f°", twist_deg)
            except Exception as e:
                self.logger.error("Twist transformation failed: %s", e, exc_info=True)
                raise
        else:
            geom_rotated = geom
            self.logger.info("Twist step disabled; passing geometry through unchanged.")

        # ── Origin placement (toggle)
        if apply_origin:
            try:
                if origin_mode == "cop":
                    geom_translated = CopOffset(
                        geometry=geom_rotated,
                        fraction=cop_fraction,
                        label=self.label,
                        logs_dir=self.logs_dir,
                    ).apply()
                    self.logger.info(
                        "Origin set to %.1f%% chord (centre-of-pressure proxy).",
                        100.0 * cop_fraction,
                    )
                elif origin_mode == "centroid":
                    geom_translated = CentroidOffset(
                        geometry=geom_rotated,
                        cx_target=cx,
                        cy_target=cy,
                        label=self.label,
                        logs_dir=self.logs_dir,
                    ).apply()
                    self.logger.info("Applied centroid translation to (%.3f, %.3f)", cx, cy)
                else:
                    raise ValueError(f"Unsupported origin_mode='{origin_mode}'")
            except Exception as e:
                self.logger.error("Origin placement failed: %s", e, exc_info=True)
                raise
        else:
            geom_translated = geom_rotated
            self.logger.info("Origin step disabled; passing geometry through unchanged.")

        # ── Material assignment (optional)
        try:
            material_norm = self._normalise_material_arg(geom_translated, material)
            AssignMaterial(
                geometry=geom_translated,
                material=material_norm,
                logs_dir=self.logs_dir,
                label=self.label,
            ).apply()

            if isinstance(material_norm, Material):
                self.logger.info("Material assigned: %s", material_norm.name)
            elif isinstance(material_norm, dict):
                self.logger.info("Per-part materials assigned to %d parts.", len(material_norm))
            else:
                self.logger.info("No material assigned.")
        except Exception as e:
            self.logger.error("Material assignment failed: %s", e, exc_info=True)
            raise

        # ensure attribute is set and return the built geometry
        self.geometry = geom_translated
        return geom_translated
