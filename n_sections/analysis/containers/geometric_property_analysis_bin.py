# section_calc_n\containers\analysis\geometric_property_analysis_bin.py

from dataclasses import dataclass


@dataclass
class GeometricPropertyAnalysisBin:
    run_label: str
    mesh_h: float
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
    j_mm4: float
    rx_mm: float
    ry_mm: float
    sx_mm3: float
    sy_mm3: float


    @classmethod
    def from_row(cls, row: list) -> "GeometricPropertyAnalysisBin":
        return cls(
            run_label=row[0],
            mesh_h=row[1],
            area_mm2=row[2],
            perimeter_mm=row[3],
            cx_mm=row[4],
            cy_mm=row[5],
            ixx_c_mm4=row[6],
            iyy_c_mm4=row[7],
            ixy_c_mm4=row[8],
            ip_c_mm4=row[9],
            phi_deg=row[10],
            i1_mm4=row[11],
            i2_mm4=row[12],
            j_mm4=row[13],
            rx_mm=row[14],
            ry_mm=row[15],
            sx_mm3=row[16],
            sy_mm3=row[17],
        )
