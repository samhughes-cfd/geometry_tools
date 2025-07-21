# geometry_utils/section_dxf.py

import logging
from pathlib import Path
from sectionproperties.analysis.section import Section


class SectionDXF:
    """Harvest every geometric result from a Section upon instantiation"""

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
        logs_dir: Path
    ):
        self.run_label = run_label
        self.h = mesh_h
        self.sec = section
        self.logs_dir = logs_dir
        self.row = None  # Will only be filled on full success

        # Logger setup
        self.logger = logging.getLogger(f"SectionDXF.{run_label}")
        self.logger.setLevel(logging.INFO)
        self._init_logger(logs_dir)

        # Compute properties on instantiation
        self._extract_property_row()

    def _init_logger(self, logs_dir: Path):
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"

        # Avoid duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def _extract_property_row(self):
        try:
            self.logger.info(f"[{self.run_label}] Starting geometric analysis...")
            self.sec.calculate_geometric_properties()

            self.logger.info(f"[{self.run_label}] Starting warping analysis (required for J)...")
            self.sec.calculate_warping_properties()

            # Extract properties (SI units)
            area = self.sec.get_area()
            perimeter = self.sec.get_perimeter()
            cx, cy = self.sec.get_c()
            ixx, iyy, ixy = self.sec.get_ic()
            ip = ixx + iyy
            phi_deg = self.sec.section_props.phi
            i1 = self.sec.section_props.i11_c
            i2 = self.sec.section_props.i22_c
            j = self.sec.get_j()
            rx, ry = self.sec.get_rc()

            # Convert to mm-based units
            self.row = [
                self.run_label, self.h,
                area * 1e6,          # mm²
                perimeter * 1e3,     # mm
                cx * 1e3, cy * 1e3,  # mm
                ixx * 1e12, iyy * 1e12, ixy * 1e12, ip * 1e12,
                phi_deg,
                i1 * 1e12, i2 * 1e12,
                j * 1e12,
                rx * 1e3, ry * 1e3
            ]
            self.logger.info(f"[{self.run_label}] Success at h = {self.h:.4f}")

        except Exception as e:
            self.row = None
            self.logger.error(f"[{self.run_label}] Failed to compute full properties: {e}", exc_info=True)
            raise