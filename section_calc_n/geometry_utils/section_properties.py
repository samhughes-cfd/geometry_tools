# section_calc_n\utils\section_properties.py

import csv
import logging
from pathlib import Path
from sectionproperties.analysis.section import Section


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

    def __init__(
        self,
        run_label: str,
        mesh_h: float,
        section: Section,
        output_path: Path,
        logs_dir: Path
    ):
        self.run_label = run_label
        self.h = mesh_h
        self.sec = section
        self.output_path = output_path

        # Logger setup for per-section
        self.logger = logging.getLogger(f"SectionDXF.{run_label}")
        self.logger.setLevel(logging.INFO)

        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"

        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def write_csv_row(self):
        try:
            # Geometric properties
            self.sec.calculate_geometric_properties()
            self.sec.calculate_warping_properties()

            # Raw quantities (in meters and m^2, m^4 etc)
            area_m2 = self.sec.get_area()
            perimeter_m = self.sec.get_perimeter()
            cx_m, cy_m = self.sec.get_c()
            ixx_c_m4, iyy_c_m4, ixy_c_m4 = self.sec.get_ic()
            ip_c_m4 = ixx_c_m4 + iyy_c_m4
            phi_deg = self.sec.section_props.phi
            i1_m4 = self.sec.section_props.i11_c
            i2_m4 = self.sec.section_props.i22_c
            j_m4 = self.sec.get_j()
            rx_m, ry_m = self.sec.get_rc()

            # Convert to mm and mm-based units
            area_mm2 = area_m2 * 1e6
            perimeter_mm = perimeter_m * 1e3
            cx_mm, cy_mm = cx_m * 1e3, cy_m * 1e3
            ixx_c_mm4 = ixx_c_m4 * 1e12
            iyy_c_mm4 = iyy_c_m4 * 1e12
            ixy_c_mm4 = ixy_c_m4 * 1e12
            ip_c_mm4 = ip_c_m4 * 1e12
            i1_mm4 = i1_m4 * 1e12
            i2_mm4 = i2_m4 * 1e12
            j_mm4 = j_m4 * 1e12
            rx_mm = rx_m * 1e3
            ry_mm = ry_m * 1e3

            row = [
                self.run_label, self.h,
                area_mm2, perimeter_mm,
                cx_mm, cy_mm,
                ixx_c_mm4, iyy_c_mm4, ixy_c_mm4, ip_c_mm4,
                phi_deg, i1_mm4, i2_mm4,
                j_mm4, rx_mm, ry_mm
            ]

            write_header = not self.output_path.exists()
            with open(self.output_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.header)
                writer.writerow(row)

            self.logger.info(f"[{self.run_label}] Row appended (h={self.h})")

        except Exception as e:
            self.logger.error(f"[{self.run_label}] Failed to compute/write properties: {e}", exc_info=True)
