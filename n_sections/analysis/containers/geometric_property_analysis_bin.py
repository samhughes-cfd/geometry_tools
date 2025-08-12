from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class GeometricPropertyAnalysisBin:
    # Identification
    run_label: str
    mesh_h: float

    # ---- Section Bounds (touch points + bbox size; all in mm) ----
    xmin_touch_x_mm: Optional[float]
    xmin_touch_y_mm: Optional[float]
    xmax_touch_x_mm: Optional[float]
    xmax_touch_y_mm: Optional[float]
    ymin_touch_x_mm: Optional[float]
    ymin_touch_y_mm: Optional[float]
    ymax_touch_x_mm: Optional[float]
    ymax_touch_y_mm: Optional[float]
    bbox_width_mm: Optional[float]
    bbox_height_mm: Optional[float]

    # ---- Global Axis Properties ----
    area_mm2: Optional[float]
    perimeter_mm: Optional[float]
    cx_mm: Optional[float]
    cy_mm: Optional[float]

    # ---- Centroidal Axis Properties ----
    ixx_c_mm4: Optional[float]
    iyy_c_mm4: Optional[float]
    ixy_c_mm4: Optional[float]
    ip_c_mm4: Optional[float]
    rx_mm: Optional[float]
    ry_mm: Optional[float]
    sx_mm3: Optional[float]  # Elastic section modulus x
    sy_mm3: Optional[float]  # Elastic section modulus y

    # ---- Principal Axis Properties ----
    phi_deg: Optional[float]   # Principal axis angle
    i1_mm4: Optional[float]    # Major principal moment
    i2_mm4: Optional[float]    # Minor principal moment

    # ---- Torsion ----
    j_mm4: Optional[float]

    # ---- Shear Areas (global) ----
    asx_mm2: Optional[float]
    asy_mm2: Optional[float]

    # ---- Shear Centres ----
    # Elastic approach (global axes)
    scx_elastic_mm: Optional[float]  # x_se
    scy_elastic_mm: Optional[float]  # y_se
    # Trefftz approach (global axes)
    scx_trefftz_mm: Optional[float]  # x_st
    scy_trefftz_mm: Optional[float]  # y_st
    # Principal-axis components
    sc1_mm: Optional[float]          # x11_se
    sc2_mm: Optional[float]          # y22_se

    # ---- Monosymmetry ----
    beta_x_plus: Optional[float]
    beta_x_minus: Optional[float]
    beta_y_plus: Optional[float]
    beta_y_minus: Optional[float]

    # ---- Derived Metrics ----
    shape_factor_x: Optional[float]
    shape_factor_y: Optional[float]
    polar_r_mm: Optional[float]
    j_over_ip: Optional[float]
    asx_over_a: Optional[float]
    asy_over_a: Optional[float]
    compactness: Optional[float]
    shear_offset_ratio_x: Optional[float]
    shear_offset_ratio_y: Optional[float]
    vx: Optional[float]  # convenience duplicate of asx_over_a
    vy: Optional[float]  # convenience duplicate of asy_over_a

    # -------------------------
    # Factory / helpers
    # -------------------------
    @classmethod
    def from_row(cls, row: List[Any]) -> "GeometricPropertyAnalysisBin":
        """
        Create instance from a SectionDXF row.
        Index mapping matches the refactored SectionDXF header order:
          [0] run_label, [1] mesh_h,
          [2..11] bounds (touch points + bbox),
          [12..26] geometric,
          [27..39] warping/shear,
          [40..50] derived
        """
        return cls(
            # Identification
            run_label=row[0],
            mesh_h=row[1],

            # Section Bounds (10)
            xmin_touch_x_mm=row[2],
            xmin_touch_y_mm=row[3],
            xmax_touch_x_mm=row[4],
            xmax_touch_y_mm=row[5],
            ymin_touch_x_mm=row[6],
            ymin_touch_y_mm=row[7],
            ymax_touch_x_mm=row[8],
            ymax_touch_y_mm=row[9],
            bbox_width_mm=row[10],
            bbox_height_mm=row[11],

            # Global Axis (4)
            area_mm2=row[12],
            perimeter_mm=row[13],
            cx_mm=row[14],
            cy_mm=row[15],

            # Centroidal Axis (8)
            ixx_c_mm4=row[16],
            iyy_c_mm4=row[17],
            ixy_c_mm4=row[18],
            ip_c_mm4=row[19],
            rx_mm=row[20],
            ry_mm=row[21],
            sx_mm3=row[22],
            sy_mm3=row[23],

            # Principal Axis (3)
            phi_deg=row[24],
            i1_mm4=row[25],
            i2_mm4=row[26],

            # Torsion (1)
            j_mm4=row[27],

            # Shear Areas (2)
            asx_mm2=row[28],
            asy_mm2=row[29],

            # Shear Centres (6)
            scx_elastic_mm=row[30],
            scy_elastic_mm=row[31],
            scx_trefftz_mm=row[32],
            scy_trefftz_mm=row[33],
            sc1_mm=row[34],
            sc2_mm=row[35],

            # Monosymmetry (4)
            beta_x_plus=row[36],
            beta_x_minus=row[37],
            beta_y_plus=row[38],
            beta_y_minus=row[39],

            # Derived (11)
            shape_factor_x=row[40],
            shape_factor_y=row[41],
            polar_r_mm=row[42],
            j_over_ip=row[43],
            asx_over_a=row[44],
            asy_over_a=row[45],
            compactness=row[46],
            shear_offset_ratio_x=row[47],
            shear_offset_ratio_y=row[48],
            vx=row[49],
            vy=row[50],
        )

    def get_property_groups(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Return properties organized by analysis category (readable structure)."""
        return {
            "section_bounds": {
                "xmin_touch_x_mm": self.xmin_touch_x_mm,
                "xmin_touch_y_mm": self.xmin_touch_y_mm,
                "xmax_touch_x_mm": self.xmax_touch_x_mm,
                "xmax_touch_y_mm": self.xmax_touch_y_mm,
                "ymin_touch_x_mm": self.ymin_touch_x_mm,
                "ymin_touch_y_mm": self.ymin_touch_y_mm,
                "ymax_touch_x_mm": self.ymax_touch_x_mm,
                "ymax_touch_y_mm": self.ymax_touch_y_mm,
                "bbox_width_mm": self.bbox_width_mm,
                "bbox_height_mm": self.bbox_height_mm,
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
            "shear_areas": {
                "asx_mm2": self.asx_mm2,
                "asy_mm2": self.asy_mm2,
            },
            "shear_centres": {
                "scx_elastic_mm": self.scx_elastic_mm,
                "scy_elastic_mm": self.scy_elastic_mm,
                "scx_trefftz_mm": self.scx_trefftz_mm,
                "scy_trefftz_mm": self.scy_trefftz_mm,
                "sc1_mm": self.sc1_mm,
                "sc2_mm": self.sc2_mm,
            },
            "monosymmetry": {
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
            },
        }

    # Optional convenience:
    def as_dict(self) -> Dict[str, Any]:
        """Flatten to a single dict (useful for DataFrame construction)."""
        out: Dict[str, Any] = {
            "run_label": self.run_label,
            "mesh_h": self.mesh_h,
        }
        for group, vals in self.get_property_groups().items():
            for k, v in vals.items():
                out[k] = v
        return out
