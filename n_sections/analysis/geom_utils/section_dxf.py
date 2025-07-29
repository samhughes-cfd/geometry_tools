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
        "Beta_x_plus", "Beta_x_minus", "Beta_y_plus", "Beta_y_minus"
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

        if geo is None:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed — skipping values.")
            geo = [None] * 17
        if warp is None:
            self.logger.error(f"[{self.run_label}] Warping analysis failed — skipping values.")
            warp = [None] * 0

        self.row = row + geo + warp
        self.logger.info(f"[{self.run_label}] SectionDXF row assembled with partial success.")

    def _geometric_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting geometric analysis...")
            self.sec.calculate_geometric_properties()
            props = self.sec.section_props

            area = self.sec.get_area()
            perimeter = self.sec.get_perimeter()
            cx, cy = self.sec.get_c()
            ixx, iyy, ixy = self.sec.get_ic()
            ip = ixx + iyy
            phi_deg = props.phi
            i1 = props.i11_c
            i2 = props.i22_c
            rx, ry = props.rx_c, props.ry_c
            sx, sy = props.sxx, props.syy

            return [
                area * M2_TO_MM2,
                perimeter * M_TO_MM,
                cx * M_TO_MM, cy * M_TO_MM,
                ixx * M4_TO_MM4, iyy * M4_TO_MM4, ixy * M4_TO_MM4, ip * M4_TO_MM4,
                phi_deg,
                i1 * M4_TO_MM4, i2 * M4_TO_MM4,
                rx * M_TO_MM, ry * M_TO_MM,
                sx * M3_TO_MM3, sy * M3_TO_MM3,
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed: {e}", exc_info=True)
            return None

    def _warping_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting warping analysis...")
            self.sec.calculate_warping_properties()
            props = self.sec.section_props
            j = self.sec.get_j()
            asx, asy = self.sec.get_as()
            scx, scy = self.sec.get_sc()
            beta_xp = props.beta_x_plus
            beta_xm = props.beta_x_minus
            beta_yp = props.beta_y_plus
            beta_ym = props.beta_y_minus


            return [
                j * M4_TO_MM4,
                asx * M2_TO_MM2, asy * M2_TO_MM2,
                scx * M_TO_MM, scy * M_TO_MM,
                beta_xp, beta_xm, beta_yp, beta_ym
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Warping analysis failed: {e}", exc_info=True)
            return None
