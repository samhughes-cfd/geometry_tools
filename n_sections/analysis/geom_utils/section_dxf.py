# geometry_utils/section_dxf.py

import logging
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
        "vx", "vy"
    ]

    def __init__(self, run_label: str, mesh_h: float, section: Section, logs_dir: Path):
        self.run_label = run_label
        self.h = mesh_h
        self.sec = section
        self.logs_dir = logs_dir
        self.row = None

        self.logger = logging.getLogger(f"SectionDXF.{run_label}")
        self.logger.setLevel(logging.INFO)
        self._init_logger(logs_dir)

        self._extract_property_row()

    def _init_logger(self, logs_dir: Path):
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def _extract_property_row(self):
        row = [self.run_label, self.h]

        geo = self._geometric_analysis()
        warp = self._warping_analysis()
        derived = self._derived_parameters()

        if geo is None:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed — skipping values.")
            geo = [None] * 17
        if warp is None:
            self.logger.error(f"[{self.run_label}] Warping analysis failed — skipping values.")
            warp = [None] * 8
        if derived is None:
            self.logger.error(f"[{self.run_label}] Derived parameter calculation failed — skipping values.")
            derived = [None] * 10

        self.row = row + geo + warp + derived
        self.logger.info(f"[{self.run_label}] SectionDXF row assembled with partial success.")

    @staticmethod
    def _safe_mul(val, factor):
        return val * factor if val is not None else None

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
                self.logger.warning(f"[{self.run_label}] Principal angle φ is None. Defaulting to 0.0°.")
                self.phi_deg = 0.0

            if self.i1 is None or self.i2 is None:
                self.logger.warning(f"[{self.run_label}] Principal moments I1/I2 are None. Defaulting to 0.0.")
                self.i1 = self.sec.get_i1() or 0.0
                self.i2 = self.sec.get_i2() or 0.0

            if self.rx is None or self.ry is None:
                self.logger.warning(f"[{self.run_label}] Radii of gyration rx/ry are None. Defaulting to 0.0.")
                self.rx = self.sec.get_rx() or 0.0
                self.ry = self.sec.get_ry() or 0.0

            if self.sx is None or self.sy is None:
                self.logger.warning(f"[{self.run_label}] Section modulus Sx/Sy is None. Defaulting to 0.0.")
                self.sx = self.sec.get_sx() or 0.0
                self.sy = self.sec.get_sy() or 0.0

            # Final info log
            self.logger.info(f"[{self.run_label}] Geometric analysis complete. Key values:")
            self.logger.info(f"  Area = {self.area:.6g} m²")
            self.logger.info(f"  Perimeter = {self.perimeter:.6g} m")
            self.logger.info(f"  Centroid = ({self.cx:.6g}, {self.cy:.6g}) m")
            self.logger.info(f"  Moments of inertia: Ixx = {self.ixx:.6g} m⁴, Iyy = {self.iyy:.6g} m⁴, Ixy = {self.ixy:.6g} m⁴")
            self.logger.info(f"  Polar moment: Ip = {self.ip:.6g} m⁴")
            self.logger.info(f"  Principal I1 = {self.i1:.6g} m⁴, I2 = {self.i2:.6g} m⁴ at φ = {self.phi_deg:.3f}°")
            self.logger.info(f"  Radii of gyration rx = {self.rx:.6g} m, ry = {self.ry:.6g} m")
            self.logger.info(f"  Section moduli Sx = {self.sx:.6g} m³, Sy = {self.sy:.6g} m³")

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
            self.logger.info(f"  J = {self.j:.6g} m⁴")
            self.logger.info(f"  Asx = {self.asx:.6g} m², Asy = {self.asy:.6g} m²")
            self.logger.info(f"  Shear centre = ({self.scx:.6g}, {self.scy:.6g}) m")
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

            shape_factor_x = self.sx / self.i1 if self.sx and self.i1 else None
            shape_factor_y = self.sy / self.i2 if self.sy and self.i2 else None
            polar_r = (self.ip / self.area) ** 0.5 if self.ip and self.area else None
            j_over_ip = self.j / self.ip if self.j and self.ip else None
            asx_ratio = self.asx / self.area if self.asx and self.area else None
            asy_ratio = self.asy / self.area if self.asy and self.area else None
            compactness = self.perimeter ** 2 / self.area if self.perimeter and self.area else None
            shear_offset_ratio_x = abs((self.cx - self.scx) / self.cx) if self.cx and self.scx else None
            shear_offset_ratio_y = abs((self.cy - self.scy) / self.cy) if self.cy and self.scy else None
            vx = self.area / self.asx if self.area and self.asx else None
            vy = self.area / self.asy if self.area and self.asy else None

            # Fallback logging
            for name, val in [
                ("Shape Factor X", shape_factor_x),
                ("Shape Factor Y", shape_factor_y),
                ("Polar Radius", polar_r),
                ("J / Ip", j_over_ip),
                ("Asx / Area", asx_ratio),
                ("Asy / Area", asy_ratio),
                ("Compactness", compactness),
                ("Shear Offset Ratio X", shear_offset_ratio_x),
                ("Shear Offset Ratio Y", shear_offset_ratio_y),
                ("Timoshenko Coefficient vx", vx),
                ("Timoshenko Coefficient vy", vy)
            ]:
                if val is None:
                    self.logger.warning(f"[{self.run_label}] {name} is None.")
                else:
                    self.logger.info(f"[{self.run_label}] {name} = {val:.6g}")

            return [
                shape_factor_x, shape_factor_y,
                self._safe_mul(polar_r, M_TO_MM), j_over_ip,
                asx_ratio, asy_ratio,
                compactness, shear_offset_ratio_x, shear_offset_ratio_y,
                vx, vy
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Derived parameter calculation failed: {e}", exc_info=True)
            return None
