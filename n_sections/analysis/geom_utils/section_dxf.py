# geometry_utils/section_dxf.py

import logging
import math
import numpy as np
from pathlib import Path
from sectionproperties.analysis.section import Section

# --- Unit Conversion Constants ---
M2_TO_MM2 = 1e6
M3_TO_MM3 = 1e9
M4_TO_MM4 = 1e12
M6_TO_MM6 = 1e18
M_TO_MM = 1e3


class SectionDXF:
    """Harvest every geometric, plastic, and warping result from a Section upon instantiation."""

    header = [
        "RunLabel", "Mesh_h_mm",
        # Section bounds
        "x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm", "max_width_mm", "max_height_mm"
        # Geometric
        "Area_mm2", "Perimeter_mm",
        "Cx_mm", "Cy_mm",
        "Ixx_c_mm4", "Iyy_c_mm4", "Ixy_c_mm4", "Ip_c_mm4",
        "Principal_angle_deg", "I1_mm4", "I2_mm4",
        "rx_mm", "ry_mm",
        "Sx_mm3", "Sy_mm3",
        # Warping
        "J_mm4", "Asx_mm2", "Asy_mm2", "SCx_mm", "SCy_mm",
        "Beta_x_plus", "Beta_x_minus", "Beta_y_plus", "Beta_y_minus",
        # Derived Metrics
        "ShapeFactor_x", "ShapeFactor_y",
        "PolarR_mm", "J_over_Ip",
        "Asx_over_A", "Asy_over_A",
        "Compactness", "ShearOffsetRatio_x", "ShearOffsetRatio_y",
        "vx", "vy",
    ]

    def __init__(self, run_label: str, mesh_h: float, section: Section, logs_dir: Path):
        self.label = run_label
        self.mesh_h = mesh_h
        self.sec = section
        self.logs_dir = logs_dir
        self.row = None

        # Create logger instance and initialize file handler
        self.logger = logging.getLogger(f"SectionDXF.{self.label}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent double logging if root logger is configured
        self._init_logger(self.logs_dir)
        self._validate_inputs()
        self._extract_property_row()
        self._validate_unit_ranges()


    def _init_logger(self, logs_dir: Path):
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def _validate_inputs(self):
        if not isinstance(self.sec, Section):
            raise TypeError(f"Expected `section` to be of type `Section`, got {type(self.sec)}")
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise ValueError("`run_label` must be a non-empty string.")
        if self.h <= 0.0:
            raise ValueError("Mesh size `mesh_h` must be positive.")
        if not isinstance(self.logs_dir, Path):
            raise TypeError("`logs_dir` must be a Path object.")

    def _validate_unit_ranges(self):
        """Warn if any physical quantity is suspiciously large or small."""
        area_mm2 = self._safe_mul(self.area, M2_TO_MM2)
        ip_mm4 = self._safe_mul(self.ip, M4_TO_MM4)
        if area_mm2 and area_mm2 > 1e9:
            self.logger.warning(f"[{self.run_label}] Area is unusually large ({area_mm2:.3e} mm²) — check units.")
        if ip_mm4 and ip_mm4 > 1e12:
            self.logger.warning(f"[{self.run_label}] Ip is unusually large ({ip_mm4:.3e} mm⁴) — check geometry scale.")

    def _extract_property_row(self):
        row = [self.run_label, self.h]

        # --- Run core analysis routines ---
        bounds = self._compute_section_bounds()
        geo = self._geometric_analysis()
        warp = self._warping_analysis()
        derived = self._derived_parameters()

        # --- Define expected field counts ---
        NUM_BOUNDS_FIELDS = 6
        NUM_GEOMETRIC_FIELDS = 17
        NUM_WARPING_FIELDS = 8
        NUM_DERIVED_FIELDS = 11

        # --- Fill in missing values with None if any stage fails ---
        if bounds is None or any(b is None for b in bounds):
            self.logger.error(f"[{self.run_label}] Section bounds computation failed — skipping values.")
            bounds = [None] * NUM_BOUNDS_FIELDS

        if geo is None:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed — skipping values.")
            geo = [None] * NUM_GEOMETRIC_FIELDS

        if warp is None:
            self.logger.error(f"[{self.run_label}] Warping analysis failed — skipping values.")
            warp = [None] * NUM_WARPING_FIELDS

        if derived is None:
            self.logger.error(f"[{self.run_label}] Derived parameter calculation failed — skipping values.")
            derived = [None] * NUM_DERIVED_FIELDS

        # --- Assemble full row ---
        self.row = row + bounds + geo + warp + derived

        # --- Final confirmation log ---
        self.logger.info(f"[{self.run_label}] SectionDXF row assembled with partial success.")

    @staticmethod
    def _safe_mul(val, factor):
        return val * factor if val is not None else None

    @staticmethod
    def _safe_div(numerator, denominator):
        try:
            if numerator is None or denominator in (None, 0.0):
                return None
            result = numerator / denominator
            return result if not isinstance(result, float) or not math.isnan(result) else None
        except ZeroDivisionError:
            return None
        

    def _compute_section_bounds(self):
        """
        Compute the min and max coordinates in both x and y directions (in mm)
        using the geometry object's `calculate_extents()` method.

        Returns:
            (xmin_mm, xmax_mm, ymin_mm, ymax_mm, width_mm, height_mm)
        """
        try:
            if not hasattr(self.sec, "geometry") or not hasattr(self.sec.geometry, "calculate_extents"):
                self.logger.error(f"[{self.run_label}] Section geometry does not support extent calculation.")
                return None, None, None, None, None, None

            x_min, x_max, y_min, y_max = self.sec.geometry.calculate_extents()

            # Convert to mm
            xmin_mm = x_min * M_TO_MM
            xmax_mm = x_max * M_TO_MM
            ymin_mm = y_min * M_TO_MM
            ymax_mm = y_max * M_TO_MM

            width_mm = xmax_mm - xmin_mm
            height_mm = ymax_mm - ymin_mm

            self.logger.info(
                f"[{self.run_label}] Section bounds:\n"
                f"  x: min = {xmin_mm:.3f} mm, max = {xmax_mm:.3f} mm, width = {width_mm:.3f} mm\n"
                f"  y: min = {ymin_mm:.3f} mm, max = {ymax_mm:.3f} mm, height = {height_mm:.3f} mm"
            )

            return xmin_mm, xmax_mm, ymin_mm, ymax_mm, width_mm, height_mm

        except Exception as e:
            self.logger.error(f"[{self.run_label}] Failed to compute section bounds: {e}", exc_info=True)
            return None, None, None, None, None, None


    def _geometric_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting geometric analysis...")
            self.sec.calculate_geometric_properties()
            props = self.sec.section_props

            self.area = self.sec.get_area()
            self.perimeter = self.sec.get_perimeter()
            self.cx, self.cy = self.sec.get_c()
            self.ixx, self.iyy, self.ixy = self.sec.get_ic()
            self.ip = self.ixx + self.iyy if self.ixx is not None and self.iyy is not None else None
            self.phi_deg = props.phi
            self.i1 = props.i11_c
            self.i2 = props.i22_c
            self.rx, self.ry = props.rx_c, props.ry_c
            self.sx, self.sy = props.sxx, props.syy

            # Fallbacks and Warnings

            if self.area is None:
                self.logger.warning(f"[{self.run_label}] Area is None. Defaulting to 0.0.")
                self.area = 0.0

            if self.perimeter is None:
                self.logger.warning(f"[{self.run_label}] Perimeter is None. Defaulting to 0.0.")
                self.perimeter = 0.0

            if self.cx is None or self.cy is None:
                self.logger.warning(f"[{self.run_label}] Centroid cx/cy is None. Defaulting to 0.0.")
                self.cx = self.cx or 0.0
                self.cy = self.cy or 0.0

            if self.ip is None:
                self.logger.warning(f"[{self.run_label}] Ip is None. Approximating as Ixx + Iyy.")
                self.ip = self.ixx + self.iyy if self.ixx is not None and self.iyy is not None else 0.0

            if self.phi_deg is None:
                self.logger.warning(f"[{self.run_label}] Principal angle phi is None. Defaulting to 0.0°.")
                self.phi_deg = 0.0

            if self.i1 is None or self.i2 is None:
                self.logger.warning(f"[{self.run_label}] Principal moments I1/I2 are None. Defaulting to 0.0.")
                self.i1 = props.i11_c() or 0.0
                self.i2 = props.i22_c() or 0.0

            if self.rx is None or self.ry is None:
                self.logger.warning(f"[{self.run_label}] Radii of gyration rx/ry are None. Defaulting to 0.0.")
                self.rx = props.rx_c or 0.0
                self.ry = props.ry_c or 0.0

            if self.sx is None or self.sy is None:
                self.logger.warning(f"[{self.run_label}] Section modulus Sx/Sy is None. Defaulting to 0.0.")
                self.sx = props.sxx or 0.0
                self.sy = props.syy or 0.0

            # Final info log
            self.logger.info(f"[{self.run_label}] Geometric analysis complete. Key values:")
            self.logger.info(f"  Area = {self._safe_mul(self.area, M2_TO_MM2):.6g} mm²")
            self.logger.info(f"  Perimeter = {self._safe_mul(self.perimeter, M_TO_MM):.6g} mm")
            self.logger.info(f"  Centroid = ({self._safe_mul(self.cx, M_TO_MM):.6g}, {self._safe_mul(self.cy, M_TO_MM):.6g}) mm")
            self.logger.info(f"  Moments of inertia: Ixx = {self._safe_mul(self.ixx, M4_TO_MM4):.6g} mm⁴, "
                 f"Iyy = {self._safe_mul(self.iyy, M4_TO_MM4):.6g} mm⁴, "
                 f"Ixy = {self._safe_mul(self.ixy, M4_TO_MM4):.6g} mm⁴")
            self.logger.info(f"  Polar moment: Ip = {self._safe_mul(self.ip, M4_TO_MM4):.6g} mm⁴")
            self.logger.info(f"  Principal I1 = {self._safe_mul(self.i1, M4_TO_MM4):.6g} mm⁴, "
                 f"I2 = {self._safe_mul(self.i2, M4_TO_MM4):.6g} mm⁴ at φ = {self.phi_deg:.3f}°")
            self.logger.info(f"  Radii of gyration rx = {self._safe_mul(self.rx, M_TO_MM):.6g} mm, "
                 f"ry = {self._safe_mul(self.ry, M_TO_MM):.6g} mm")
            self.logger.info(f"  Section moduli Sx = {self._safe_mul(self.sx, M3_TO_MM3):.6g} mm³, "
                 f"Sy = {self._safe_mul(self.sy, M3_TO_MM3):.6g} mm³")

            return [
                self._safe_mul(self.area, M2_TO_MM2),
                self._safe_mul(self.perimeter, M_TO_MM),
                self._safe_mul(self.cx, M_TO_MM), self._safe_mul(self.cy, M_TO_MM),
                self._safe_mul(self.ixx, M4_TO_MM4), self._safe_mul(self.iyy, M4_TO_MM4),
                self._safe_mul(self.ixy, M4_TO_MM4), self._safe_mul(self.ip, M4_TO_MM4),
                self.phi_deg,
                self._safe_mul(self.i1, M4_TO_MM4), self._safe_mul(self.i2, M4_TO_MM4),
                self._safe_mul(self.rx, M_TO_MM), self._safe_mul(self.ry, M_TO_MM),
                self._safe_mul(self.sx, M3_TO_MM3), self._safe_mul(self.sy, M3_TO_MM3),
            ]

        except Exception as e:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed: {e}", exc_info=True)
            return None


    def _warping_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting warping analysis...")
            self.sec.calculate_warping_properties()
            props = self.sec.section_props

            self.j = self.sec.get_j()
            self.asx, self.asy = self.sec.get_as()
            self.scx, self.scy = self.sec.get_sc()
            self.beta_xp = props.beta_x_plus
            self.beta_xm = props.beta_x_minus
            self.beta_yp = props.beta_y_plus
            self.beta_ym = props.beta_y_minus

            # Fallbacks and Warnings
            if self.j is None:
                self.logger.warning(f"[{self.run_label}] J is None. Defaulting to 0.0.")
                self.j = 0.0

            if self.asx is None or self.asy is None:
                self.logger.warning(f"[{self.run_label}] Asx/Asy is None. Defaulting to 0.0.")
                self.asx = self.asx or 0.0
                self.asy = self.asy or 0.0

            if self.scx is None or self.scy is None:
                self.logger.warning(f"[{self.run_label}] SCx/SCy is None. Defaulting to 0.0.")
                self.scx = self.scx or 0.0
                self.scy = self.scy or 0.0

            if any(val is None for val in [self.beta_xp, self.beta_xm, self.beta_yp, self.beta_ym]):
                self.logger.warning(f"[{self.run_label}] One or more beta parameters are None. Defaulting to 0.0.")
                self.beta_xp = self.beta_xp or 0.0
                self.beta_xm = self.beta_xm or 0.0
                self.beta_yp = self.beta_yp or 0.0
                self.beta_ym = self.beta_ym or 0.0

            # Final info log
            self.logger.info(f"[{self.run_label}] Warping analysis complete. Key values:")
            self.logger.info(f"  J = {self._safe_mul(self.j, M4_TO_MM4):.6g} mm⁴")
            self.logger.info(f"  Asx = {self._safe_mul(self.asx, M2_TO_MM2):.6g} mm², "
                 f"Asy = {self._safe_mul(self.asy, M2_TO_MM2):.6g} mm²")
            self.logger.info(f"  Shear centre = ({self._safe_mul(self.scx, M_TO_MM):.6g}, "
                 f"{self._safe_mul(self.scy, M_TO_MM):.6g}) mm")
            self.logger.info(f"  Beta coefficients:")
            self.logger.info(f"    Beta_x+ = {self.beta_xp:.6g}, Beta_x- = {self.beta_xm:.6g}")
            self.logger.info(f"    Beta_y+ = {self.beta_yp:.6g}, Beta_y- = {self.beta_ym:.6g}")

            return [
                self._safe_mul(self.j, M4_TO_MM4),
                self._safe_mul(self.asx, M2_TO_MM2), self._safe_mul(self.asy, M2_TO_MM2),
                self._safe_mul(self.scx, M_TO_MM), self._safe_mul(self.scy, M_TO_MM),
                self.beta_xp, self.beta_xm, self.beta_yp, self.beta_ym
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Warping analysis failed: {e}", exc_info=True)
            return None


    def _derived_parameters(self):
        try:
            self.logger.info(f"[{self.run_label}] Calculating derived section parameters...")

            shape_factor_x = self._safe_div(self.sx, self.i1) if self.sx and self.i1 else None
            shape_factor_y = self._safe_div(self.sy, self.i2) if self.sy and self.i2 else None
            polar_r = self._safe_div(self.ip, self.area) ** 0.5 if self.ip and self.area else None
            j_over_ip = self._safe_div(self.j, self.ip) if self.j and self.ip else None
            asx_ratio = self._safe_div(self.asx, self.area) if self.asx and self.area else None
            asy_ratio = self._safe_div(self.asy, self.area) if self.asy and self.area else None
            compactness = self._safe_div(self.perimeter**2, self.area) if self.perimeter and self.area else None
            shear_offset_ratio_x = self._safe_div(abs(self.cx - self.scx), self.cx) if self.cx and self.scx else None
            shear_offset_ratio_y = self._safe_div(abs(self.cy - self.scy), self.cy) if self.cy and self.scy else None
            vx = self._safe_div(self.asx, self.area) if self.area and self.asx else None
            vy = self._safe_div(self.asy, self.area) if self.area and self.asy else None

            polar_r_mm = self._safe_mul(polar_r, M_TO_MM)

            # Logging results with unit annotations and symbolic expressions where applicable
            if shape_factor_x is not None:
                self.logger.info(f"[{self.run_label}] Shape Factor X = Sx / I1 = {self.sx:.6g} / {self.i1:.6g} = {shape_factor_x:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Shape Factor X is None.")

            if shape_factor_y is not None:
                self.logger.info(f"[{self.run_label}] Shape Factor Y = Sy / I2 = {self.sy:.6g} / {self.i2:.6g} = {shape_factor_y:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Shape Factor Y is None.")

            if polar_r_mm is not None:
                self.logger.info(f"[{self.run_label}] Polar Radius = sqrt(Ip / A) = sqrt({self.ip:.6g} / {self.area:.6g}) = {polar_r_mm:.6g} mm")
            else:
                self.logger.warning(f"[{self.run_label}] Polar Radius is None.")

            if j_over_ip is not None:
                self.logger.info(f"[{self.run_label}] J / Ip = {self.j:.6g} / {self.ip:.6g} = {j_over_ip:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] J / Ip is None.")

            if asx_ratio is not None:
                self.logger.info(f"[{self.run_label}] Asx / A = {self.asx:.6g} / {self.area:.6g} = {asx_ratio:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Asx / A is None.")

            if asy_ratio is not None:
                self.logger.info(f"[{self.run_label}] Asy / A = {self.asy:.6g} / {self.area:.6g} = {asy_ratio:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Asy / A is None.")

            if compactness is not None:
                self.logger.info(f"[{self.run_label}] Compactness = Perimeter^2 / A = {self.perimeter:.6g}^2 / {self.area:.6g} = {compactness:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Compactness is None.")

            if shear_offset_ratio_x is not None:
                self.logger.info(f"[{self.run_label}] Shear Offset Ratio X = |Cx - SCx| / Cx = |{self.cx:.6g} - {self.scx:.6g}| / {self.cx:.6g} = {shear_offset_ratio_x:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Shear Offset Ratio X is None.")

            if shear_offset_ratio_y is not None:
                self.logger.info(f"[{self.run_label}] Shear Offset Ratio Y = |Cy - SCy| / Cy = |{self.cy:.6g} - {self.scy:.6g}| / {self.cy:.6g} = {shear_offset_ratio_y:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Shear Offset Ratio Y is None.")

            if vx is not None:
                self.logger.info(f"[{self.run_label}] Timoshenko Coefficient vx = Asx / A = {self.asx:.6g} / {self.area:.6g} = {vx:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Timoshenko Coefficient vx is None.")

            if vy is not None:
                self.logger.info(f"[{self.run_label}] Timoshenko Coefficient vy = Asy / A = {self.asy:.6g} / {self.area:.6g} = {vy:.6g}")
            else:
                self.logger.warning(f"[{self.run_label}] Timoshenko Coefficient vy is None.")

            return [
                shape_factor_x, shape_factor_y,
                polar_r_mm, j_over_ip,
                asx_ratio, asy_ratio,
                compactness, shear_offset_ratio_x, shear_offset_ratio_y,
                vx, vy
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Derived parameter calculation failed: {e}", exc_info=True)
            return None