from dataclasses import dataclass
from typing import Optional

@dataclass
class GeometricPropertyAnalysisBin:
    run_label: str
    mesh_h: float

    # Section Bounds
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    max_width_mm: float
    max_height_mm: float

    # Global Axis Properties
    area_mm2: float
    perimeter_mm: float
    cx_mm: float
    cy_mm: float

    # Centroidal Axis Properties
    ixx_c_mm4: float
    iyy_c_mm4: float
    ixy_c_mm4: float
    ip_c_mm4: float
    rx_mm: float
    ry_mm: float
    sx_mm3: float  # Elastic section modulus x
    sy_mm3: float  # Elastic section modulus y

    # Principal Axis Properties
    phi_deg: float  # Principal axis angle
    i1_mm4: float   # Major principal moment
    i2_mm4: float   # Minor principal moment

    # Torsion Properties
    j_mm4: float    # Torsion constant

    # Shear Properties
    asx_mm2: float  # Shear area x
    asy_mm2: float  # Shear area y
    scx_mm: float   # Shear center x
    scy_mm: float   # Shear center y
    beta_x_plus: float
    beta_x_minus: float
    beta_y_plus: float
    beta_y_minus: float

    # Derived Metrics
    shape_factor_x: float
    shape_factor_y: float
    polar_r_mm: float
    j_over_ip: float
    asx_over_a: float
    asy_over_a: float
    compactness: float
    shear_offset_ratio_x: float
    shear_offset_ratio_y: float
    vx: float  # Timoshenko coefficient x
    vy: float  # Timoshenko coefficient y

    @classmethod
    def from_row(cls, row: list) -> "GeometricPropertyAnalysisBin":
        """Create instance from analysis results row with explicit grouping"""
        return cls(
            # Identification
            run_label=row[0],
            mesh_h=row[1],
            
            # Section Bounds (6 fields)
            x_min_mm=row[2],
            x_max_mm=row[3],
            y_min_mm=row[4],
            y_max_mm=row[5],
            max_width_mm=row[6],
            max_height_mm=row[7],

            # Global Axis Properties (4 fields)
            area_mm2=row[8],
            perimeter_mm=row[9],
            cx_mm=row[10],
            cy_mm=row[11],

            # Centroidal Axis Properties (8 fields)
            ixx_c_mm4=row[12],
            iyy_c_mm4=row[13],
            ixy_c_mm4=row[14],
            ip_c_mm4=row[15],
            rx_mm=row[19],  # Note: Adjusted indices to match calculation order
            ry_mm=row[20],
            sx_mm3=row[21],
            sy_mm3=row[22],

            # Principal Axis Properties (3 fields)
            phi_deg=row[16],
            i1_mm4=row[17],
            i2_mm4=row[18],

            # Torsion Property (1 field)
            j_mm4=row[23],

            # Shear Properties (8 fields)
            asx_mm2=row[24],
            asy_mm2=row[25],
            scx_mm=row[26],
            scy_mm=row[27],
            beta_x_plus=row[28],
            beta_x_minus=row[29],
            beta_y_plus=row[30],
            beta_y_minus=row[31],

            # Derived Metrics (11 fields)
            shape_factor_x=row[32],
            shape_factor_y=row[33],
            polar_r_mm=row[34],
            j_over_ip=row[35],
            asx_over_a=row[36],
            asy_over_a=row[37],
            compactness=row[38],
            shear_offset_ratio_x=row[39],
            shear_offset_ratio_y=row[40],
            vx=row[41],
            vy=row[42],
        )

    def get_property_groups(self) -> dict:
        """Return properties organized by analysis category"""
        return {
            "section_bounds": {
                "x_min_mm": self.x_min_mm,
                "x_max_mm": self.x_max_mm,
                "y_min_mm": self.y_min_mm,
                "y_max_mm": self.y_max_mm,
                "max_width_mm": self.max_width_mm,
                "max_height_mm": self.max_height_mm,
            },
            "global_axis": {
                "area_mm2": self.area_mm2,
                "perimeter_mm": self.perimeter_mm,
                "cx_mm": self.cx_mm,
                "cy_mm": self.cy_mm,
            },
            "centroidal_axis": {
                "ixx_c_mm4": self.ixx_c_mm4,
                "iyy_c_mm4": self.iyy_c_mm4,
                "ixy_c_mm4": self.ixy_c_mm4,
                "ip_c_mm4": self.ip_c_mm4,
                "rx_mm": self.rx_mm,
                "ry_mm": self.ry_mm,
                "sx_mm3": self.sx_mm3,
                "sy_mm3": self.sy_mm3,
            },
            "principal_axis": {
                "phi_deg": self.phi_deg,
                "i1_mm4": self.i1_mm4,
                "i2_mm4": self.i2_mm4,
            },
            "torsion": {
                "j_mm4": self.j_mm4,
            },
            "shear": {
                "asx_mm2": self.asx_mm2,
                "asy_mm2": self.asy_mm2,
                "scx_mm": self.scx_mm,
                "scy_mm": self.scy_mm,
                "beta_x_plus": self.beta_x_plus,
                "beta_x_minus": self.beta_x_minus,
                "beta_y_plus": self.beta_y_plus,
                "beta_y_minus": self.beta_y_minus,
            },
            "derived": {
                "shape_factor_x": self.shape_factor_x,
                "shape_factor_y": self.shape_factor_y,
                "polar_r_mm": self.polar_r_mm,
                "j_over_ip": self.j_over_ip,
                "asx_over_a": self.asx_over_a,
                "asy_over_a": self.asy_over_a,
                "compactness": self.compactness,
                "shear_offset_ratio_x": self.shear_offset_ratio_x,
                "shear_offset_ratio_y": self.shear_offset_ratio_y,
                "vx": self.vx,
                "vy": self.vy,
            }
        }