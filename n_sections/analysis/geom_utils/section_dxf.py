# geometry_utils/section_dxf.py

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np  # noqa: F401 (kept for downstream extensions)
from sectionproperties.analysis.section import Section

# --- Unit Conversion Constants ---
M2_TO_MM2 = 1e6
M3_TO_MM3 = 1e9
M4_TO_MM4 = 1e12
M6_TO_MM6 = 1e18
M_TO_MM = 1e3


@dataclass
class Field:
    """Declarative field spec: name, getter(self) -> value in SI, and a scale to output units."""
    name: str
    getter: Callable[["SectionDXF"], Optional[float]]
    scale: float = 1.0


class SectionDXF:
    """
    Harvest geometric, warping, and shear results from a Section in a highly extendible way.

    Design:
      - Each stage returns (pre, fields) where `pre()` runs required analysis once,
        and `fields` is a list of Field(name, getter, scale).
      - Header and row are built automatically from field specs.
      - To add properties later, just add Field(...) entries or a new stage method.
    """

    # Static leading columns
    leading_names = ["RunLabel", "Mesh_h_mm"]

    def __init__(self, run_label: str, mesh_h: float, section: Section, logs_dir: Path):
        self.run_label = run_label
        self.mesh_h = mesh_h
        self.sec = section
        self.logs_dir = logs_dir

        self.header: List[str] = []
        self.row: List[Optional[float | str]] = []
        self.start_time = time.time()

        # Create logger instance
        self.logger = logging.getLogger(f"SectionDXF.{self.run_label}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._init_logger(logs_dir)

        try:
            self.logger.info(f"Starting analysis for section '{run_label}'")
            self._validate_inputs()
            self._build_header_and_row()
            self._validate_unit_ranges()
            duration = time.time() - self.start_time
            self.logger.info(f"Analysis completed in {duration:.2f}s")
        except Exception as e:
            self.logger.critical(f"Fatal error during analysis: {str(e)}", exc_info=True)
            raise

    # -----------------------
    # Boilerplate & utilities
    # -----------------------
    def _init_logger(self, logs_dir: Path):
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"

        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def _validate_inputs(self):
        if not isinstance(self.sec, Section):
            raise TypeError(f"Expected `section` to be of type `Section`, got {type(self.sec)}")
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise ValueError("`run_label` must be a non-empty string.")
        if not isinstance(self.logs_dir, Path):
            raise TypeError("`logs_dir` must be a Path object.")

    @staticmethod
    def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        try:
            if numerator is None or denominator in (None, 0.0):
                return None
            out = numerator / denominator
            return out if not math.isnan(out) else None
        except ZeroDivisionError:
            return None

    @staticmethod
    def _safe_scale(val: Optional[float], scale: float) -> Optional[float]:
        return None if val is None else val * scale

    def _validate_unit_ranges(self):
        """Warn if any physical quantity is suspiciously large or small (simple heuristics)."""
        props = getattr(self.sec, "section_props", None)
        if not props:
            return
        area = getattr(props, "area", None)
        ip = None
        if hasattr(props, "ixx_c") and hasattr(props, "iyy_c"):
            ip = (props.ixx_c or 0.0) + (props.iyy_c or 0.0)

        area_mm2 = self._safe_scale(area, M2_TO_MM2)
        ip_mm4 = self._safe_scale(ip, M4_TO_MM4)
        if area_mm2 and area_mm2 > 1e9:
            self.logger.warning(f"[{self.run_label}] Area is unusually large ({area_mm2:.3e} mm²) — check units.")
        if ip_mm4 and ip_mm4 > 1e12:
            self.logger.warning(f"[{self.run_label}] Ip is unusually large ({ip_mm4:.3e} mm⁴) — check geometry scale.")

    # -----------------------
    # Declarative field sets
    # -----------------------
    def _bounds_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """
        Bounding-box touch points (x,y) where the axis-aligned bbox meets the section,
        plus bbox dimensions. Uses shapely if available; falls back to mesh nodes, then corners.

        Exposed names (mm):
          - xmin_touch_x_mm, xmin_touch_y_mm
          - xmax_touch_x_mm, xmax_touch_y_mm
          - ymin_touch_x_mm, ymin_touch_y_mm
          - ymax_touch_x_mm, ymax_touch_y_mm
          - bbox_width_mm, bbox_height_mm
        """
        def pre():
            if not hasattr(self.sec, "geometry"):
                raise AttributeError("Section missing geometry attribute")

            # cache extents
            self._x_min, self._x_max, self._y_min, self._y_max = self.sec.geometry.calculate_extents()

            # try cached centroid (if geometric stage ran), otherwise mid-extents fallback
            try:
                p = getattr(self.sec, "section_props", None)
                self._cx = getattr(p, "cx", None)
                self._cy = getattr(p, "cy", None)
            except Exception:
                self._cx = None
                self._cy = None
            if self._cx is None or self._cy is None:
                self._cx = 0.5 * (self._x_min + self._x_max)
                self._cy = 0.5 * (self._y_min + self._y_max)

            # compute touch points once
            (self._xmin_touch,
             self._xmax_touch,
             self._ymin_touch,
             self._ymax_touch) = self._compute_bbox_touch_points()

        fields = [
            # x = xmin touch
            Field("xmin_touch_x_mm", lambda s: s._xmin_touch[0], M_TO_MM),
            Field("xmin_touch_y_mm", lambda s: s._xmin_touch[1], M_TO_MM),
            # x = xmax touch
            Field("xmax_touch_x_mm", lambda s: s._xmax_touch[0], M_TO_MM),
            Field("xmax_touch_y_mm", lambda s: s._xmax_touch[1], M_TO_MM),
            # y = ymin touch
            Field("ymin_touch_x_mm", lambda s: s._ymin_touch[0], M_TO_MM),
            Field("ymin_touch_y_mm", lambda s: s._ymin_touch[1], M_TO_MM),
            # y = ymax touch
            Field("ymax_touch_x_mm", lambda s: s._ymax_touch[0], M_TO_MM),
            Field("ymax_touch_y_mm", lambda s: s._ymax_touch[1], M_TO_MM),
            # bbox size
            Field("bbox_width_mm",  lambda s: s._x_max - s._x_min, M_TO_MM),
            Field("bbox_height_mm", lambda s: s._y_max - s._y_min, M_TO_MM),
        ]
        return pre, fields

    def _geometric_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Geometric properties (global, centroidal, principal)."""
        def pre():
            self.sec.calculate_geometric_properties()

        p = lambda: self.sec.section_props  # shorthand

        fields = [
            # Global axis
            Field("Area_mm2",      lambda s: getattr(p(), "area", None),      M2_TO_MM2),
            Field("Perimeter_mm",  lambda s: getattr(p(), "perimeter", None), M_TO_MM),
            Field("Cx_mm",         lambda s: getattr(p(), "cx", None),        M_TO_MM),
            Field("Cy_mm",         lambda s: getattr(p(), "cy", None),        M_TO_MM),

            # Centroidal axis
            Field("Ixx_c_mm4",     lambda s: getattr(p(), "ixx_c", None),     M4_TO_MM4),
            Field("Iyy_c_mm4",     lambda s: getattr(p(), "iyy_c", None),     M4_TO_MM4),
            Field("Ixy_c_mm4",     lambda s: getattr(p(), "ixy_c", None),     M4_TO_MM4),
            Field("Ip_c_mm4",      lambda s: (getattr(p(), "ixx_c", 0.0) or 0.0) +
                                             (getattr(p(), "iyy_c", 0.0) or 0.0), M4_TO_MM4),
            Field("rx_mm",         lambda s: getattr(p(), "rx_c", None),      M_TO_MM),
            Field("ry_mm",         lambda s: getattr(p(), "ry_c", None),      M_TO_MM),
            Field("Sx_mm3",        lambda s: getattr(p(), "sxx", None),       M3_TO_MM3),
            Field("Sy_mm3",        lambda s: getattr(p(), "syy", None),       M3_TO_MM3),

            # Principal axis
            Field("Principal_angle_deg", lambda s: getattr(p(), "phi", None),    1.0),
            Field("I1_mm4",              lambda s: getattr(p(), "i11_c", None),  M4_TO_MM4),
            Field("I2_mm4",              lambda s: getattr(p(), "i22_c", None),  M4_TO_MM4),
        ]
        return pre, fields

    def _warping_shear_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Warping & shear properties. Elastic + Trefftz shear centres exposed."""
        def pre():
            # Must follow geometric analysis
            self.sec.calculate_warping_properties()

        p = lambda: self.sec.section_props

        fields = [
            # Torsion
            Field("J_mm4", lambda s: getattr(p(), "j", None), M4_TO_MM4),

            # Shear areas (global)
            Field("Asx_mm2", lambda s: getattr(p(), "a_sx", None), M2_TO_MM2),
            Field("Asy_mm2", lambda s: getattr(p(), "a_sy", None), M2_TO_MM2),

            # Shear centres (elastic method, global axes)
            Field("SCx_elastic_mm", lambda s: getattr(p(), "x_se", None), M_TO_MM),
            Field("SCy_elastic_mm", lambda s: getattr(p(), "y_se", None), M_TO_MM),

            # Shear centres (Trefftz method, global axes)
            Field("SCx_trefftz_mm", lambda s: getattr(p(), "x_st", None), M_TO_MM),
            Field("SCy_trefftz_mm", lambda s: getattr(p(), "y_st", None), M_TO_MM),

            # Shear centre components along principal axes
            Field("SC1_mm", lambda s: getattr(p(), "x11_se", None), M_TO_MM),
            Field("SC2_mm", lambda s: getattr(p(), "y22_se", None), M_TO_MM),

            # Monosymmetry constants
            Field("Beta_x_plus",  lambda s: getattr(p(), "beta_x_plus", None), 1.0),
            Field("Beta_x_minus", lambda s: getattr(p(), "beta_x_minus", None), 1.0),
            Field("Beta_y_plus",  lambda s: getattr(p(), "beta_y_plus", None), 1.0),
            Field("Beta_y_minus", lambda s: getattr(p(), "beta_y_minus", None), 1.0),

            # >>> Future easy additions:
            # Field("As11_mm2", lambda s: getattr(p(), "a_s11", None), M2_TO_MM2),
            # Field("As22_mm2", lambda s: getattr(p(), "a_s22", None), M2_TO_MM2),
            # Field("Gamma_mm6", lambda s: getattr(p(), "gamma", None), M6_TO_MM6),
            # Field("Delta_s",   lambda s: getattr(p(), "delta_s", None), 1.0),
        ]
        return pre, fields

    def _derived_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Derived parameters built from previously computed props (no extra pre)."""
        def pre():
            # nothing to do; depends on geometric + warping having run
            pass

        def props():
            return self.sec.section_props

        # Helpers reading directly from props to avoid order coupling.
        def polar_radius_m(_: "SectionDXF") -> Optional[float]:
            p = props()
            area = getattr(p, "area", None)
            ip = None
            if hasattr(p, "ixx_c") and hasattr(p, "iyy_c"):
                ip = (getattr(p, "ixx_c", 0.0) or 0.0) + (getattr(p, "iyy_c", 0.0) or 0.0)
            return None if area in (None, 0.0) else math.sqrt(ip / area) if ip is not None else None

        def shape_factor_x(_: "SectionDXF") -> Optional[float]:
            p = props()
            i1, sx = getattr(p, "i11_c", None), getattr(p, "sxx", None)
            return self._safe_div(sx, i1)

        def shape_factor_y(_: "SectionDXF") -> Optional[float]:
            p = props()
            i2, sy = getattr(p, "i22_c", None), getattr(p, "syy", None)
            return self._safe_div(sy, i2)

        def j_over_ip(_: "SectionDXF") -> Optional[float]:
            p = props()
            j = getattr(p, "j", None)
            ip = None
            if hasattr(p, "ixx_c") and hasattr(p, "iyy_c"):
                ip = (getattr(p, "ixx_c", 0.0) or 0.0) + (getattr(p, "iyy_c", 0.0) or 0.0)
            return self._safe_div(j, ip)

        def as_over_a_x(_: "SectionDXF") -> Optional[float]:
            p = props()
            return self._safe_div(getattr(p, "a_sx", None), getattr(p, "area", None))

        def as_over_a_y(_: "SectionDXF") -> Optional[float]:
            p = props()
            return self._safe_div(getattr(p, "a_sy", None), getattr(p, "area", None))

        def compactness(_: "SectionDXF") -> Optional[float]:
            p = props()
            perim = getattr(p, "perimeter", None)
            perim2 = None if perim is None else (perim ** 2)
            return self._safe_div(perim2, getattr(p, "area", None))

        def shear_offset_ratio_x(_: "SectionDXF") -> Optional[float]:
            p = props()
            cx, scx = getattr(p, "cx", None), getattr(p, "x_se", None)  # elastic centre
            return self._safe_div(abs(cx - scx) if (cx is not None and scx is not None) else None, cx)

        def shear_offset_ratio_y(_: "SectionDXF") -> Optional[float]:
            p = props()
            cy, scy = getattr(p, "cy", None), getattr(p, "y_se", None)
            return self._safe_div(abs(cy - scy) if (cy is not None and scy is not None) else None, cy)

        fields = [
            Field("ShapeFactor_x", shape_factor_x, 1.0),
            Field("ShapeFactor_y", shape_factor_y, 1.0),
            Field("PolarR_mm",     polar_radius_m, M_TO_MM),
            Field("J_over_Ip",     j_over_ip, 1.0),
            Field("Asx_over_A",    as_over_a_x, 1.0),
            Field("Asy_over_A",    as_over_a_y, 1.0),
            Field("Compactness",   compactness, 1.0),
            Field("ShearOffsetRatio_x", shear_offset_ratio_x, 1.0),
            Field("ShearOffsetRatio_y", shear_offset_ratio_y, 1.0),
            # convenience duplicates:
            Field("vx", as_over_a_x, 1.0),
            Field("vy", as_over_a_y, 1.0),
        ]
        return pre, fields

    # -----------------------
    # Orchestration
    # -----------------------
    def _build_header_and_row(self):
        """Run stages, auto-build header and row (stable order)."""
        # Stages in execution order
        stages: List[Tuple[str, Callable[[], None], List[Field]]] = []
        for name, maker in [
            ("bounds", self._bounds_fields),
            ("geometric", self._geometric_fields),
            ("warping_shear", self._warping_shear_fields),
            ("derived", self._derived_fields),
        ]:
            pre, fields = maker()
            stages.append((name, pre, fields))

        # Build header
        names: List[str] = list(self.leading_names)

        # Start row with run label + mesh size
        values: List[Optional[float | str]] = [self.run_label, self.mesh_h]

        # Execute stages
        for name, pre, fields in stages:
            try:
                self.logger.debug(f"Starting {name} stage")
                pre()
                for f in fields:
                    names.append(f.name)
                    raw = f.getter(self)
                    values.append(self._safe_scale(raw, f.scale))
            except Exception as e:
                self.logger.error(f"{name} stage failed: {e}", exc_info=True)
                for f in fields:
                    names.append(f.name)
                    values.append(None)

        self.header = names
        self.row = values
        success_rate = sum(v is not None for v in self.row[2:]) / max(1, len(self.row) - 2)
        self.logger.info(f"Property row assembled with {success_rate:.1%} completion")

    # -----------------------
    # Helpers
    # -----------------------
    def _compute_bbox_touch_points(self):
        """
        Returns four points as tuples (x, y) in metres:
          - xmin_touch: point on boundary where x == x_min (y closest to centroid)
          - xmax_touch: point on boundary where x == x_max
          - ymin_touch: point on boundary where y == y_min
          - ymax_touch: point on boundary where y == y_max
        Strategy: shapely -> mesh nodes -> centroid-aligned fallback.
        """
        x_min, x_max, y_min, y_max = self._x_min, self._x_max, self._y_min, self._y_max
        cx, cy = self._cx, self._cy

        # --- 1) Shapely route if available ---
        geom = getattr(self.sec.geometry, "geom", None)
        try:
            from shapely.geometry import LineString  # type: ignore
            if geom is not None and hasattr(geom, "boundary"):
                # Vertical lines at x_min, x_max; horizontal lines at y_min, y_max
                pad = max((x_max - x_min), (y_max - y_min)) * 2.0 or 1.0
                vline_min = LineString([(x_min, y_min - pad), (x_min, y_max + pad)])
                vline_max = LineString([(x_max, y_min - pad), (x_max, y_max + pad)])
                hline_min = LineString([(x_min - pad, y_min), (x_max + pad, y_min)])
                hline_max = LineString([(x_min - pad, y_max), (x_max + pad, y_max)])

                def pick_point(intersection, prefer_y=None, prefer_x=None):
                    # Normalize to list of points
                    try:
                        geoms = list(getattr(intersection, "geoms", [intersection]))
                    except Exception:
                        geoms = [intersection]
                    pts = []
                    for g in geoms:
                        if hasattr(g, "x") and hasattr(g, "y"):
                            pts.append((g.x, g.y))
                        elif hasattr(g, "coords"):
                            pts.extend(list(g.coords))
                    if not pts:
                        return None
                    if prefer_y is not None:
                        return min(pts, key=lambda p: abs(p[1] - prefer_y))
                    if prefer_x is not None:
                        return min(pts, key=lambda p: abs(p[0] - prefer_x))
                    return pts[0]

                xmin_touch = pick_point(geom.boundary.intersection(vline_min), prefer_y=cy)
                xmax_touch = pick_point(geom.boundary.intersection(vline_max), prefer_y=cy)
                ymin_touch = pick_point(geom.boundary.intersection(hline_min), prefer_x=cx)
                ymax_touch = pick_point(geom.boundary.intersection(hline_max), prefer_x=cx)

                if xmin_touch and xmax_touch and ymin_touch and ymax_touch:
                    return xmin_touch, xmax_touch, ymin_touch, ymax_touch
        except Exception as e:
            self.logger.debug(f"Shapely intersection fallback: {e}")

        # --- 2) Mesh nodes route ---
        try:
            # collect node coords from elements
            nodes = []
            for el in getattr(self.sec, "elements", []):
                # el.coords shape (2, n_nodes), in metres
                for i in range(el.coords.shape[1]):
                    nodes.append((float(el.coords[0, i]), float(el.coords[1, i])))
            if nodes:
                xs = [p[0] for p in nodes]
                ys = [p[1] for p in nodes]
                tol_x = max(1e-9, 1e-6 * (max(xs) - min(xs) or 1.0))
                tol_y = max(1e-9, 1e-6 * (max(ys) - min(ys) or 1.0))

                xmin_candidates = [p for p in nodes if abs(p[0] - x_min) <= tol_x]
                xmax_candidates = [p for p in nodes if abs(p[0] - x_max) <= tol_x]
                ymin_candidates = [p for p in nodes if abs(p[1] - y_min) <= tol_y]
                ymax_candidates = [p for p in nodes if abs(p[1] - y_max) <= tol_y]

                def pick_closest(cands, target_y=None, target_x=None):
                    if not cands:
                        return None
                    if target_y is not None:
                        return min(cands, key=lambda p: abs(p[1] - target_y))
                    if target_x is not None:
                        return min(cands, key=lambda p: abs(p[0] - target_x))
                    return cands[0]

                xmin_touch = pick_closest(xmin_candidates, target_y=cy)
                xmax_touch = pick_closest(xmax_candidates, target_y=cy)
                ymin_touch = pick_closest(ymin_candidates, target_x=cx)
                ymax_touch = pick_closest(ymax_candidates, target_x=cx)

                if xmin_touch and xmax_touch and ymin_touch and ymax_touch:
                    return xmin_touch, xmax_touch, ymin_touch, ymax_touch
        except Exception as e:
            self.logger.debug(f"Mesh-node fallback failed: {e}")

        # --- 3) Centroid-aligned fallback (deterministic) ---
        self.logger.warning(
            "Falling back to bbox centroid-aligned touch points (no shapely geom or mesh boundary available)."
        )
        xmin_touch = (x_min, cy)
        xmax_touch = (x_max, cy)
        ymin_touch = (cx, y_min)
        ymax_touch = (cx, y_max)
        return xmin_touch, xmax_touch, ymin_touch, ymax_touch