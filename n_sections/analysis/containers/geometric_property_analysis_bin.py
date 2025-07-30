# section_calc_n\containers\analysis\geometric_property_analysis_bin.py

from dataclasses import dataclass


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

    # Geometric
    area_mm2: float
    perimeter_mm: float
    cx_mm: float
    cy_mm: float
    ixx_c_mm4: float
    iyy_c_mm4: float
    ixy_c_mm4: float
    ip_c_mm4: float
    phi_deg: float
    i1_mm4: float
    i2_mm4: float
    rx_mm: float
    ry_mm: float
    sx_mm3: float
    sy_mm3: float

    # Warping
    j_mm4: float
    asx_mm2: float
    asy_mm2: float
    scx_mm: float
    scy_mm: float
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
    vx: float
    vy: float

    @classmethod
    def from_row(cls, row: list) -> "GeometricPropertyAnalysisBin":
        return cls(
            run_label=row[0],
            mesh_h=row[1],

            # Section Bounds
            x_min_mm=row[2],
            x_max_mm=row[3],
            y_min_mm=row[4],
            y_max_mm=row[5],
            max_width_mm=row[6],
            max_height_mm=row[7],

            # Geometric
            area_mm2=row[8],
            perimeter_mm=row[9],
            cx_mm=row[10],
            cy_mm=row[11],
            ixx_c_mm4=row[12],
            iyy_c_mm4=row[13],
            ixy_c_mm4=row[14],
            ip_c_mm4=row[15],
            phi_deg=row[16],
            i1_mm4=row[17],
            i2_mm4=row[18],
            rx_mm=row[19],
            ry_mm=row[20],
            sx_mm3=row[21],
            sy_mm3=row[22],

            # Warping
            j_mm4=row[23],
            asx_mm2=row[24],
            asy_mm2=row[25],
            scx_mm=row[26],
            scy_mm=row[27],
            beta_x_plus=row[28],
            beta_x_minus=row[29],
            beta_y_plus=row[30],
            beta_y_minus=row[31],

            # Derived
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
