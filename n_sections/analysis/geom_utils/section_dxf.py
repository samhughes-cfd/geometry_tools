# geometry_utils/section_dxf.py

import logging
import math
import time
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
        "x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm", "max_width_mm", "max_height_mm",
        # Geometric (Global Axis)
        "Area_mm2", "Perimeter_mm", "Cx_mm", "Cy_mm",
        # Geometric (Centroidal Axis) 
        "Ixx_c_mm4", "Iyy_c_mm4", "Ixy_c_mm4", "Ip_c_mm4",
        "rx_mm", "ry_mm", "Sx_mm3", "Sy_mm3",
        # Geometric (Principal Axis)
        "Principal_angle_deg", "I1_mm4", "I2_mm4",
        # Warping and Shear
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
        self.run_label = run_label
        self.mesh_h = mesh_h
        self.sec = section
        self.logs_dir = logs_dir
        self.row = None
        self.start_time = time.time()

        # Create logger instance
        self.logger = logging.getLogger(f"SectionDXF.{self.run_label}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._init_logger(logs_dir)
        
        try:
            self.logger.info(f"Starting analysis for section '{run_label}'")
            self._validate_inputs()
            self._extract_property_row()
            self._validate_unit_ranges()
            duration = time.time() - self.start_time
            self.logger.info(f"Analysis completed in {duration:.2f}s")
        except Exception as e:
            self.logger.critical(f"Fatal error during analysis: {str(e)}", exc_info=True)
            raise

    def _init_logger(self, logs_dir: Path):
        """Initialize logger with file handler"""
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "section_dxf.log"
        
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                   for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def _validate_inputs(self):
        """Validate constructor inputs with detailed errors"""
        if not isinstance(self.sec, Section):
            raise TypeError(f"Expected `section` to be of type `Section`, got {type(self.sec)}")
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise ValueError("`run_label` must be a non-empty string.")
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
        """Orchestrate property extraction with error handling"""
        row = [self.run_label, self.mesh_h]

        # Define expected field counts for each analysis stage
        analysis_stages = [
            ("bounds", self._bounds_analysis, 6),
            ("geometric", self._geometric_analysis, 17),  # 4 global + 8 centroidal + 5 principal
            ("warping_shear", self._warping_shear_analysis, 9),  # 1 torsion + 8 shear
            ("derived", self._derived_parameters, 11)
        ]
    
        results = {}
        for name, fn, field_count in analysis_stages:
            try:
                self.logger.debug(f"Starting {name} analysis")
                result = fn()
                results[name] = result if result is not None else [None] * field_count
            except Exception as e:
                self.logger.error(f"{name} analysis failed: {str(e)}", exc_info=True)
                results[name] = [None] * field_count

        # Assemble full row - now guaranteed to have lists
        self.row = row + results["bounds"] + results["geometric"] + results["warping_shear"] + results["derived"]
    
        success_rate = sum(x is not None for x in self.row)/len(self.row)
        self.logger.info(f"Property row assembled with {success_rate:.1%} completion")

    @staticmethod
    def _safe_mul(val, factor):
        return val * factor if val is not None else None

    @staticmethod
    def _safe_div(numerator, denominator):
        try:
            if numerator is None or denominator in (None, 0.0):
                return None
            result = numerator / denominator
            return result if not math.isnan(result) else None
        except ZeroDivisionError:
            return None

    def _bounds_analysis(self):
        """Compute section boundaries with detailed diagnostics"""
        try:
            self.logger.info("Computing section bounds")
            if not hasattr(self.sec, "geometry"):
                raise AttributeError("Section missing geometry attribute")
            
            x_min, x_max, y_min, y_max = self.sec.geometry.calculate_extents()
            return [
                x_min * M_TO_MM,
                x_max * M_TO_MM,
                y_min * M_TO_MM,
                y_max * M_TO_MM,
                (x_max - x_min) * M_TO_MM,
                (y_max - y_min) * M_TO_MM
            ]
        except Exception as e:
            self.logger.error(f"Boundary calculation failed: {str(e)}", exc_info=True)
            return [None] * 6  # Return 6 Nones for bounds fields

    def _geometric_analysis(self):
        """Orchestrate full geometric property analysis"""
        try:
            self.logger.info("Starting geometric analysis")
            self.sec.calculate_geometric_properties()
            props = self.sec.section_props
        
            global_results = self._global_axis_properties(props) or [None]*4
            centroidal_results = self._centroidal_axis_properties(props) or [None]*8
            principal_results = self._principal_axis_properties(props) or [None]*3
        
            return global_results + centroidal_results + principal_results
        except Exception as e:
            self.logger.error(f"Geometric analysis failed: {str(e)}", exc_info=True)
            return [None] * 17  # 4 global + 8 centroidal + 5 principal
        
    def _global_axis_properties(self, props):
        """Calculate global axis geometric properties"""
        try:
            self.area = props.area or 0.0
            self.perimeter = props.perimeter or 0.0
            self.cx, self.cy = props.cx or 0.0, props.cy or 0.0
            
            return [
                self.area * M2_TO_MM2,
                self.perimeter * M_TO_MM,
                self.cx * M_TO_MM,
                self.cy * M_TO_MM
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Global axis properties failed: {e}", exc_info=True)
            return [None] * 4

    def _centroidal_axis_properties(self, props):
        """Calculate centroidal axis properties including section moduli Sx, Sy"""
        try:
            self.ixx = props.ixx_c or 0.0
            self.iyy = props.iyy_c or 0.0
            self.ixy = props.ixy_c or 0.0
            self.ip = self.ixx + self.iyy
            self.rx = props.rx_c or 0.0
            self.ry = props.ry_c or 0.0
            self.sx = props.sxx or 0.0  # Elastic section modulus x
            self.sy = props.syy or 0.0  # Elastic section modulus y
            
            return [
                self.ixx * M4_TO_MM4,
                self.iyy * M4_TO_MM4,
                self.ixy * M4_TO_MM4,
                self.ip * M4_TO_MM4,
                self.rx * M_TO_MM,
                self.ry * M_TO_MM,
                self.sx * M3_TO_MM3,
                self.sy * M3_TO_MM3
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Centroidal axis properties failed: {e}", exc_info=True)
            return [None] * 8

    def _principal_axis_properties(self, props):
        """Calculate principal axis properties"""
        try:
            self.phi_deg = props.phi or 0.0
            self.i1 = props.i11_c or 0.0
            self.i2 = props.i22_c or 0.0
            
            return [
                self.phi_deg,
                self.i1 * M4_TO_MM4,
                self.i2 * M4_TO_MM4
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Principal axis properties failed: {e}", exc_info=True)
            return [None] * 3

    def _warping_shear_analysis(self):
        """Orchestrate warping and shear property analysis"""
        try:
            self.logger.info("Starting warping and shear analysis")
            self.sec.calculate_warping_properties()
            props = self.sec.section_props
        
            torsion_results = self._torsion_properties(props) or [None]
            shear_results = self._shear_properties(props) or [None]*8
        
            return torsion_results + shear_results
        except Exception as e:
            self.logger.error(f"Warping/shear analysis failed: {str(e)}", exc_info=True)
            return [None] * 9  # 1 torsion + 8 shear

    def _torsion_properties(self, props):
        """Calculate torsion-specific properties"""
        try:
            self.j = props.j or 0.0
            return [self.j * M4_TO_MM4]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Torsion properties failed: {e}", exc_info=True)
            return [None]

    def _shear_properties(self, props):
        """Calculate shear-specific properties"""
        try:
            self.asx = props.asx or 0.0
            self.asy = props.asy or 0.0
            self.scx = props.scx or 0.0
            self.scy = props.scy or 0.0
            self.beta_xp = props.beta_x_plus or 0.0
            self.beta_xm = props.beta_x_minus or 0.0
            self.beta_yp = props.beta_y_plus or 0.0
            self.beta_ym = props.beta_y_minus or 0.0
            
            return [
                self.asx * M2_TO_MM2,
                self.asy * M2_TO_MM2,
                self.scx * M_TO_MM,
                self.scy * M_TO_MM,
                self.beta_xp,
                self.beta_xm,
                self.beta_yp,
                self.beta_ym
            ]
        except Exception as e:
            self.logger.error(f"[{self.run_label}] Shear properties failed: {e}", exc_info=True)
            return [None] * 8

    def _derived_parameters(self):
        """Calculate all derived parameters"""
        try:
            self.logger.info("Calculating derived parameters")
        
            return [
                self._safe_div(self.sx, self.i1),
                self._safe_div(self.sy, self.i2),
                self._safe_div(self.ip, self.area) ** 0.5 * M_TO_MM,
                self._safe_div(self.j, self.ip),
                self._safe_div(self.asx, self.area),
                self._safe_div(self.asy, self.area),
                self._safe_div(self.perimeter**2, self.area),
                self._safe_div(abs(self.cx - self.scx), self.cx),
                self._safe_div(abs(self.cy - self.scy), self.cy),
                self._safe_div(self.asx, self.area),
                self._safe_div(self.asy, self.area)
            ]
        except Exception as e:
            self.logger.error(f"Derived parameters failed: {str(e)}", exc_info=True)
            return [None] * 11