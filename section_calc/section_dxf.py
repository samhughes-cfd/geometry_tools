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
            self.sec.calculate_geometric_properties()

            cx, cy = self.sec.get_c()
            ic     = self.sec.get_ic()
            ixx, ixy, iyy = ic[0][0], ic[0][1], ic[1][1]
            ip     = ixx + iyy
            phi, i1, i2 = self.sec.get_principal_properties()
            j_torsion   = self.sec.get_j()
            rx, ry      = self.sec.get_r_g()

            row = [
                self.run_label, self.h,
                self.sec.get_area(),    self.sec.perimeter,
                cx, cy,
                ixx, iyy, ixy, ip,
                phi, i1, i2,
                j_torsion, rx, ry
            ]

            write_header = not os.path.isfile(RES_CSV)
            with open(RES_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.header)
                writer.writerow(row)

            logger.info(f"[{self.run_label}] row appended (h={self.h})")

        except Exception as e:
            logger.error(f"[{self.run_label}] failed to compute/write properties: {e}")