# section_calc/section_dxf.py

import os
import csv
import logging
from sectionproperties.analysis.section import Section

# ───────── Logging ─────────
logger = logging.getLogger("SectionDXF")
logger.setLevel(logging.DEBUG)
os.makedirs("section_calc/logs", exist_ok=True)
fh = logging.FileHandler("section_calc/logs/section_dxf.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# ───────── Results directory ─────────
RES_CSV = "section_calc/results/section_results.csv"
os.makedirs("section_calc/results", exist_ok=True)

class SectionDXF:
    """Harvest every geometric result from a Section and append to CSV."""

    header = [
        "RunLabel", "Mesh_h_mm",
        "Area_mm2", "Perimeter_mm",
        "Cx_mm", "Cy_mm",
        "Ixx_c_mm4", "Iyy_c_mm4", "Ixy_c_mm4", "Ip_c_mm4",
        "Principal_angle_deg", "I1_mm4", "I2_mm4",
        "J_mm4", "rx_mm", "ry_mm"
    ]

    def __init__(self, run_label: str, mesh_h: float, section: Section):
        self.run_label = run_label
        self.h         = mesh_h
        self.sec       = section

    def write_csv_row(self):
        """Append a single convergence row (one mesh) to CSV."""
        try:
            # 1) geometric analysis
            self.sec.calculate_geometric_properties()

            # 2) warping analysis (required for torsion constant)
            self.sec.calculate_warping_properties()

            # 3) area & perimeter
            area      = self.sec.get_area()

            # 4) centroid
            cx, cy = self.sec.get_c()

            # 5) centroidal moments of inertia
            ixx_c, iyy_c, ixy_c = self.sec.get_ic()
            ip_c = ixx_c + iyy_c

            # 6) principal axes from SectionProperties
            #    phi (°) and principal second moments i11_c, i22_c
            phi_deg = self.sec.section_props.phi
            i1_mm4  = self.sec.section_props.i11_c
            i2_mm4  = self.sec.section_props.i22_c

            # 7) torsion constant
            j_mm4 = self.sec.get_j()

            # 8) radii of gyration about centroidal axes
            rx_mm, ry_mm = self.sec.get_rc()

            # 9) build the CSV row
            row = [
                self.run_label, self.h,
                area,
                cx, cy,
                ixx_c, iyy_c, ixy_c, ip_c,
                phi_deg, i1_mm4, i2_mm4,
                j_mm4, rx_mm, ry_mm
            ]

            # 10) write header if needed and append row
            write_header = not os.path.isfile(RES_CSV)
            with open(RES_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.header)
                writer.writerow(row)

            logger.info(f"[{self.run_label}] row appended (h={self.h})")

        except Exception as e:
            logger.error(f"[{self.run_label}] failed to compute/write properties: {e}")