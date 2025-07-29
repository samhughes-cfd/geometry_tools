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
        "Area_mm2", "Perimeter_mm", "Mass_kg", "Axial_rigidity_N",
        "Cx_mm", "Cy_mm",
        "Ixx_c_mm4", "Iyy_c_mm4", "Ixy_c_mm4", "Ip_c_mm4",
        "Principal_angle_deg", "I1_mm4", "I2_mm4",
        "rx_mm", "ry_mm",
        "Sx_mm3", "Sy_mm3",

        # Plastic
        "Zx_mm3", "Zy_mm3", "Z1_mm3", "Z2_mm3",
        "Shape_factor_x", "Shape_factor_y", "Shape_factor_1", "Shape_factor_2",

        # Warping
        "J_mm4", "Asx_mm2", "Asy_mm2", "SCx_mm", "SCy_mm", "Cw_mm6", "Beta_monosym"
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
        pl = self._plastic_analysis()
        warp = self._warping_analysis()

        if geo is None:
            self.logger.error(f"[{self.run_label}] Geometric analysis failed — skipping values.")
            geo = [None] * 17
        if pl is None:
            self.logger.error(f"[{self.run_label}] Plastic analysis failed — skipping values.")
            pl = [None] * 8
        if warp is None:
            self.logger.error(f"[{self.run_label}] Warping analysis failed — skipping values.")
            warp = [None] * 7

        self.row = row + geo + pl + warp
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
            rx, ry = self.sec.get_rc()
            sx = self.sec.get_sx()
            sy = self.sec.get_sy()

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

    def _plastic_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting plastic analysis...")
            self.sec.calculate_plastic_properties()
            props = self.sec.section_props

            zx = self.sec.get_zx()
            zy = self.sec.get_zy()
            z1 = props.z11
            z2 = props.z22
            sx = self.sec.get_sx()
            sy = self.sec.get_sy()
            s1 = props.s11
            s2 = props.s22

            shape_factor_x = zx / sx if sx else float('nan')
            shape_factor_y = zy / sy if sy else float('nan')
            shape_factor_1 = z1 / s1 if s1 else float('nan')
            shape_factor_2 = z2 / s2 if s2 else float('nan')

            return [
                zx * M3_TO_MM3, zy * M3_TO_MM3, z1 * M3_TO_MM3, z2 * M3_TO_MM3,
                shape_factor_x, shape_factor_y,
                shape_factor_1, shape_factor_2
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Plastic analysis failed: {e}", exc_info=True)
            return None

    def _warping_analysis(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting warping analysis...")
            self.sec.calculate_warping_properties()

            j = self.sec.get_j()
            asx, asy = self.sec.get_as()
            scx, scy = self.sec.get_sc()
            cw = self.sec.get_cw()
        beta_vals = self.sec.get_beta()  # returns 4 values


            return [
                j * M4_TO_MM4,
                asx * M2_TO_MM2, asy * M2_TO_MM2,
                scx * M_TO_MM, scy * M_TO_MM,
                cw * M6_TO_MM6,
                beta
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Warping analysis failed: {e}", exc_info=True)
            return None
