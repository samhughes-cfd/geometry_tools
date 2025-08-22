# geometry_utils/section_dxf.py

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np  # noqa: F401 (kept for downstream extensions)
from sectionproperties.analysis.section import Section

# --- Unit Conversion Constants --- DXF for this run is in mm already, so no conversion needed.
M2_TO_MM2 = 1  # 1e6
M3_TO_MM3 = 1  # 1e9
M4_TO_MM4 = 1  # 1e12
M6_TO_MM6 = 1  # 1e18
M_TO_MM = 1    # 1e3


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
    def _extents_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """
        Axis-aligned geometry extents from sectionproperties' native API.
        Caches values for downstream stages and exposes them as fields.
        """
        def pre():
            if not hasattr(self.sec, "geometry"):
                raise AttributeError("Section missing geometry attribute")
            # Native call:
            self._x_min, self._x_max, self._y_min, self._y_max = self.sec.geometry.calculate_extents()

            # Cache bbox size
            self._bbox_width = self._x_max - self._x_min
            self._bbox_height = self._y_max - self._y_min

            # Global extreme-fibre distances (used by global section moduli, normalisations, etc.)
            self._c_x_g = max(abs(self._y_min), abs(self._y_max))  # for Sx about x@origin
            self._c_y_g = max(abs(self._x_min), abs(self._x_max))  # for Sy about y@origin

        fields = [
            Field("x_min_mm",       lambda s: s._x_min,       M_TO_MM),
            Field("x_max_mm",       lambda s: s._x_max,       M_TO_MM),
            Field("y_min_mm",       lambda s: s._y_min,       M_TO_MM),
            Field("y_max_mm",       lambda s: s._y_max,       M_TO_MM),
            Field("bbox_width_mm",  lambda s: s._bbox_width,  M_TO_MM),
            Field("bbox_height_mm", lambda s: s._bbox_height, M_TO_MM),
            Field("c_x_g_mm",       lambda s: s._c_x_g,       M_TO_MM),
            Field("c_y_g_mm",       lambda s: s._c_y_g,       M_TO_MM),
        ]
        return pre, fields

    def _bounds_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """
        Bounding-box touch points (x,y) where the axis-aligned bbox meets the section.
        Uses shapely if available; falls back to mesh nodes, then centroid-aligned points.
        """
        def pre():
            if not hasattr(self.sec, "geometry"):
                raise AttributeError("Section missing geometry attribute")

            # Extents MUST already be cached by _extents_fields()
            for attr in ("_x_min", "_x_max", "_y_min", "_y_max"):
                if not hasattr(self, attr):
                    raise RuntimeError("Extents not initialised. Ensure _extents_fields runs before _bounds_fields.")

            # Use centroid from geometric stage if available, otherwise mid-extents fallback
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
            Field("xmin_touch_x_mm", lambda s: s._xmin_touch[0], M_TO_MM),
            Field("xmin_touch_y_mm", lambda s: s._xmin_touch[1], M_TO_MM),
            Field("xmax_touch_x_mm", lambda s: s._xmax_touch[0], M_TO_MM),
            Field("xmax_touch_y_mm", lambda s: s._xmax_touch[1], M_TO_MM),
            Field("ymin_touch_x_mm", lambda s: s._ymin_touch[0], M_TO_MM),
            Field("ymin_touch_y_mm", lambda s: s._ymin_touch[1], M_TO_MM),
            Field("ymax_touch_x_mm", lambda s: s._ymax_touch[0], M_TO_MM),
            Field("ymax_touch_y_mm", lambda s: s._ymax_touch[1], M_TO_MM),
        ]
        return pre, fields

    def _geometric_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Geometric properties reported systematically as centroidal (_c) and global (_g)."""
        def pre():
            # Requires extents stage to have run already
            self.sec.calculate_geometric_properties()
            p = self.sec.section_props

            # Cache frequently used centroidal values
            self._A   = getattr(p, "area", None)
            self._per = getattr(p, "perimeter", None)
            self._cx  = getattr(p, "cx", None)
            self._cy  = getattr(p, "cy", None)
            self._Ixx_c = getattr(p, "ixx_c", None)
            self._Iyy_c = getattr(p, "iyy_c", None)
            self._Ixy_c = getattr(p, "ixy_c", None)
            self._Ip_c  = None if None in (self._Ixx_c, self._Iyy_c) else (self._Ixx_c + self._Iyy_c)
            self._rx_c  = getattr(p, "rx_c", None)
            self._ry_c  = getattr(p, "ry_c", None)

            # Parallel-axis transform to global (DXF origin)
            self._Ixx_g, self._Iyy_g, self._Ixy_g = self._parallel_axis_to_origin(
                self._Ixx_c, self._Iyy_c, self._Ixy_c, self._A, self._cx, self._cy
            )
            self._Ip_g = None if None in (self._Ixx_g, self._Iyy_g) else (self._Ixx_g + self._Iyy_g)
            self._rx_g = None if None in (self._Ixx_g, self._A) or self._A == 0 else math.sqrt(self._Ixx_g / self._A)
            self._ry_g = None if None in (self._Iyy_g, self._A) or self._A == 0 else math.sqrt(self._Iyy_g / self._A)

            # Section moduli: centroidal (from lib) and global (we compute)
            self._Sx_c = getattr(p, "sxx", None)
            self._Sy_c = getattr(p, "syy", None)

            c_x_c, c_y_c = (None, None)
            if None not in (self._cx, self._cy):
                c_x_c, c_y_c = self._extreme_fibres_centroid(self._cx, self._cy)

            # If lib didn't provide, compute centroidal section modulus as fallback
            if self._Sx_c is None and None not in (self._Ixx_c, c_x_c) and c_x_c not in (None, 0.0):
                self._Sx_c = self._Ixx_c / c_x_c
            if self._Sy_c is None and None not in (self._Iyy_c, c_y_c) and c_y_c not in (None, 0.0):
                self._Sy_c = self._Iyy_c / c_y_c

            # Global section moduli: use global inertias + extreme fibres from origin
            c_x_g, c_y_g = self._extreme_fibres_global()
            self._Sx_g = self._Ixx_g / c_x_g if None not in (self._Ixx_g, c_x_g) and c_x_g not in (None, 0.0) else None
            self._Sy_g = self._Iyy_g / c_y_g if None not in (self._Iyy_g, c_y_g) and c_y_g not in (None, 0.0) else None

            # Principal (centroidal): as provided by sectionproperties
            self._phi   = getattr(p, "phi", None)
            self._I11_c = getattr(p, "i11_c", None)
            self._I22_c = getattr(p, "i22_c", None)

            # Principal ABOUT ORIGIN (global): diagonalise global inertia tensor
            self._I11_g, self._I22_g, self._phi_g = self._principal_about_origin()

        fields = [
            # --- Global (DXF origin) ---
            Field("Ixx_g_mm4", lambda s: s._Ixx_g, M4_TO_MM4),
            Field("Iyy_g_mm4", lambda s: s._Iyy_g, M4_TO_MM4),
            Field("Ixy_g_mm4", lambda s: s._Ixy_g, M4_TO_MM4),
            Field("Ip_g_mm4",  lambda s: s._Ip_g,  M4_TO_MM4),
            Field("rx_g_mm",   lambda s: s._rx_g,  M_TO_MM),
            Field("ry_g_mm",   lambda s: s._ry_g,  M_TO_MM),
            Field("Sx_g_mm3",  lambda s: s._Sx_g,  M3_TO_MM3),
            Field("Sy_g_mm3",  lambda s: s._Sy_g,  M3_TO_MM3),
            # Principal (about origin)
            Field("Principal_angle_g_deg", lambda s: s._phi_g, 1.0),
            Field("I11_g_mm4",             lambda s: s._I11_g, M4_TO_MM4),
            Field("I22_g_mm4",             lambda s: s._I22_g, M4_TO_MM4),

            # --- Centroidal (as in library) ---
            Field("Area_mm2",      lambda s: s._A,   M2_TO_MM2),
            Field("Perimeter_mm",  lambda s: s._per, M_TO_MM),
            Field("Cx_mm",         lambda s: s._cx,  M_TO_MM),
            Field("Cy_mm",         lambda s: s._cy,  M_TO_MM),

            Field("Ixx_c_mm4",     lambda s: s._Ixx_c, M4_TO_MM4),
            Field("Iyy_c_mm4",     lambda s: s._Iyy_c, M4_TO_MM4),
            Field("Ixy_c_mm4",     lambda s: s._Ixy_c, M4_TO_MM4),
            Field("Ip_c_mm4",      lambda s: s._Ip_c,  M4_TO_MM4),
            Field("rx_c_mm",       lambda s: s._rx_c,  M_TO_MM),   # (alias of previous rx_mm)
            Field("ry_c_mm",       lambda s: s._ry_c,  M_TO_MM),   # (alias of previous ry_mm)
            Field("Sx_c_mm3",      lambda s: s._Sx_c,  M3_TO_MM3), # (alias of previous Sx_mm3)
            Field("Sy_c_mm3",      lambda s: s._Sy_c,  M3_TO_MM3), # (alias of previous Sy_mm3)

            # --- Principal (about centroid) ---
            Field("Principal_angle_deg", lambda s: s._phi,   1.0),
            Field("I11_c_mm4",           lambda s: s._I11_c, M4_TO_MM4),
            Field("I22_c_mm4",           lambda s: s._I22_c, M4_TO_MM4),

            # --- Back-compat aliases (optional; keep if downstream expects old names) ---
            Field("rx_mm",    lambda s: s._rx_c,  M_TO_MM),
            Field("ry_mm",    lambda s: s._ry_c,  M_TO_MM),
            Field("Sx_mm3",   lambda s: s._Sx_c,  M3_TO_MM3),
            Field("Sy_mm3",   lambda s: s._Sy_c,  M3_TO_MM3),
            Field("I1_mm4",   lambda s: s._I11_c, M4_TO_MM4),
            Field("I2_mm4",   lambda s: s._I22_c, M4_TO_MM4),
        ]
        return pre, fields

    def _warping_shear_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Warping & shear: J and As* are translation-invariant; shear centre positions + offsets reported."""
        def pre():
            self.sec.calculate_warping_properties()
            p = self.sec.section_props

            self._J   = getattr(p, "j", None)
            self._Asx = getattr(p, "a_sx", None)
            self._Asy = getattr(p, "a_sy", None)

            # Shear centre absolute (global/DXF)
            self._SCx = getattr(p, "x_se", None)
            self._SCy = getattr(p, "y_se", None)

            # Offsets
            self._SCx_from_centroid = None if None in (self._SCx, self._cx) else (self._SCx - self._cx)
            self._SCy_from_centroid = None if None in (self._SCy, self._cy) else (self._SCy - self._cy)

            # Normalised offsets (by global extreme fibres)
            c_x_g, c_y_g = self._extreme_fibres_global()
            self._SCx_norm_origin    = self._safe_div(abs(self._SCx), c_y_g) if self._SCx is not None else None
            self._SCy_norm_origin    = self._safe_div(abs(self._SCy), c_x_g) if self._SCy is not None else None
            self._SCx_norm_centroid  = self._safe_div(abs(self._SCx_from_centroid), c_y_g) if self._SCx_from_centroid is not None else None
            self._SCy_norm_centroid  = self._safe_div(abs(self._SCy_from_centroid), c_x_g) if self._SCy_from_centroid is not None else None

            # Principal-axis shear centre components (about centroid axes)
            self._SC1 = getattr(p, "x11_se", None)
            self._SC2 = getattr(p, "y22_se", None)

            # Monosymmetry
            self._beta_x_plus  = getattr(p, "beta_x_plus", None)
            self._beta_x_minus = getattr(p, "beta_x_minus", None)
            self._beta_y_plus  = getattr(p, "beta_y_plus", None)
            self._beta_y_minus = getattr(p, "beta_y_minus", None)

        fields = [
            # Invariant / direct
            Field("J_mm4",    lambda s: s._J,   M4_TO_MM4),
            Field("Asx_mm2",  lambda s: s._Asx, M2_TO_MM2),
            Field("Asy_mm2",  lambda s: s._Asy, M2_TO_MM2),

            # Shear centre absolute (global coords)
            Field("SCx_global_mm", lambda s: s._SCx, M_TO_MM),
            Field("SCy_global_mm", lambda s: s._SCy, M_TO_MM),

            # Shear centre offsets
            Field("SCx_from_centroid_mm", lambda s: s._SCx_from_centroid, M_TO_MM),
            Field("SCy_from_centroid_mm", lambda s: s._SCy_from_centroid, M_TO_MM),
            Field("SCx_norm_origin",      lambda s: s._SCx_norm_origin,  1.0),
            Field("SCy_norm_origin",      lambda s: s._SCy_norm_origin,  1.0),
            Field("SCx_norm_centroid",    lambda s: s._SCx_norm_centroid,1.0),
            Field("SCy_norm_centroid",    lambda s: s._SCy_norm_centroid,1.0),

            # Principal-axis components (about centroid)
            Field("SC1_mm", lambda s: s._SC1, M_TO_MM),
            Field("SC2_mm", lambda s: s._SC2, M_TO_MM),

            # Monosymmetry constants
            Field("Beta_x_plus",  lambda s: s._beta_x_plus,  1.0),
            Field("Beta_x_minus", lambda s: s._beta_x_minus, 1.0),
            Field("Beta_y_plus",  lambda s: s._beta_y_plus,  1.0),
            Field("Beta_y_minus", lambda s: s._beta_y_minus, 1.0),
        ]
        return pre, fields

    def _derived_fields(self) -> Tuple[Callable[[], None], List[Field]]:
        """Derived parameters; provide both centroidal (_c) and global (_g) forms where applicable."""
        def pre():
            # nothing extra; relies on geometric + warping having run
            pass

        # Centroidal
        def polar_radius_c(_: "SectionDXF") -> Optional[float]:
            return None if self._A in (None, 0.0) or self._Ip_c is None else math.sqrt(self._Ip_c / self._A)

        def shape_factor_x_c(_: "SectionDXF") -> Optional[float]:
            # Historical definition using centroidal principal inertia
            return self._safe_div(self._Sx_c, self._I11_c)

        def shape_factor_y_c(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._Sy_c, self._I22_c)

        def j_over_ip_c(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._J, self._Ip_c)

        # Global
        def polar_radius_g(_: "SectionDXF") -> Optional[float]:
            return None if self._A in (None, 0.0) or self._Ip_g is None else math.sqrt(self._Ip_g / self._A)

        def shape_factor_x_g(_: "SectionDXF") -> Optional[float]:
            # Define wrt global x-axis: Sx_g / Ixx_g
            return self._safe_div(self._Sx_g, self._Ixx_g)

        def shape_factor_y_g(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._Sy_g, self._Iyy_g)

        def j_over_ip_g(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._J, self._Ip_g)

        # Invariants (no translation effect)
        def as_over_a_x(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._Asx, self._A)

        def as_over_a_y(_: "SectionDXF") -> Optional[float]:
            return self._safe_div(self._Asy, self._A)

        def compactness(_: "SectionDXF") -> Optional[float]:
            perim2 = None if self._per is None else (self._per ** 2)
            return self._safe_div(perim2, self._A)

        fields = [
            # Shape factors
            Field("ShapeFactor_x_c", shape_factor_x_c, 1.0),
            Field("ShapeFactor_y_c", shape_factor_y_c, 1.0),
            Field("ShapeFactor_x_g", shape_factor_x_g, 1.0),
            Field("ShapeFactor_y_g", shape_factor_y_g, 1.0),

            # Polar radii
            Field("PolarR_c_mm", polar_radius_c, M_TO_MM),
            Field("PolarR_g_mm", polar_radius_g, M_TO_MM),

            # J/Ip
            Field("J_over_Ip_c", j_over_ip_c, 1.0),
            Field("J_over_Ip_g", j_over_ip_g, 1.0),

            # Invariants
            Field("Asx_over_A", as_over_a_x, 1.0),
            Field("Asy_over_A", as_over_a_y, 1.0),
            Field("Compactness", compactness, 1.0),

            # Convenience duplicates (unchanged)
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
            ("extents", self._extents_fields),       # NEW: extents first (native geometry.calculate_extents)
            ("geometric", self._geometric_fields),   # depends on extents for fibre distances
            ("bounds", self._bounds_fields),         # uses centroid if available; else mid-extents fallback
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

    # -----------------------
    # Local helpers
    # -----------------------
    def _parallel_axis_to_origin(
        self,
        Ixx_c: Optional[float],
        Iyy_c: Optional[float],
        Ixy_c: Optional[float],
        A: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (Ixx_g, Iyy_g, Ixy_g) about the DXF origin via parallel-axis theorem."""
        if None in (Ixx_c, Iyy_c, Ixy_c, A, cx, cy):
            return None, None, None
        return (
            Ixx_c + A * (cy ** 2),
            Iyy_c + A * (cx ** 2),
            Ixy_c + A * (cx * cy),
        )

    def _extreme_fibres_centroid(self, cx: float, cy: float) -> Tuple[Optional[float], Optional[float]]:
        """Return (c_x_c, c_y_c): extreme fibre distances relative to centroidal axes."""
        if None in (self._y_min, self._y_max, self._x_min, self._x_max, cx, cy):
            return None, None
        c_x_c = max(cy - self._y_min, self._y_max - cy)  # distance to top/bottom from centroid (for Sx about x@centroid)
        c_y_c = max(cx - self._x_min, self._x_max - cx)  # distance to left/right from centroid (for Sy about y@centroid)
        return c_x_c, c_y_c

    def _extreme_fibres_global(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (c_x_g, c_y_g) from cached extents (origin-referenced extreme fibre distances)."""
        if hasattr(self, "_c_x_g") and hasattr(self, "_c_y_g"):
            return self._c_x_g, self._c_y_g
        # Fallback if extents weren't cached for some reason
        if None in (self._y_min, self._y_max, self._x_min, self._x_max):
            return None, None
        c_x_g = max(abs(self._y_min), abs(self._y_max))
        c_y_g = max(abs(self._x_min), abs(self._x_max))
        return c_x_g, c_y_g

    def _principal_about_origin(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (I11_g, I22_g, phi_g_deg) from the global inertia components."""
        Ixx, Iyy, Ixy = getattr(self, "_Ixx_g", None), getattr(self, "_Iyy_g", None), getattr(self, "_Ixy_g", None)
        if None in (Ixx, Iyy, Ixy):
            return None, None, None
        d = Ixx - Iyy
        R = math.hypot(d, 2.0 * Ixy)  # sqrt(d^2 + (2 Ixy)^2)
        I11 = 0.5 * (Ixx + Iyy) + 0.5 * R
        I22 = 0.5 * (Ixx + Iyy) - 0.5 * R
        phi = 0.5 * math.atan2(2.0 * Ixy, d)  # radians
        phi_deg = math.degrees(phi)
        # Wrap to [-90, 90] for reporting clarity
        if phi_deg > 90.0:
            phi_deg -= 180.0
        elif phi_deg < -90.0:
            phi_deg += 180.0
        return I11, I22, phi_deg