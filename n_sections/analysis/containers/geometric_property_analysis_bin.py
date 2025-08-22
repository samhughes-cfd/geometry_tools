from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class GeometricPropertyAnalysisBin:
    # Identification
    run_label: str
    mesh_h: float

    # ---- Extents (mm) ----
    x_min_mm: Optional[float]
    x_max_mm: Optional[float]
    y_min_mm: Optional[float]
    y_max_mm: Optional[float]
    bbox_width_mm: Optional[float]
    bbox_height_mm: Optional[float]
    c_x_g_mm: Optional[float]  # extreme fibre distance wrt global x (for Sx about origin)
    c_y_g_mm: Optional[float]  # extreme fibre distance wrt global y (for Sy about origin)

    # ---- Global axis (about origin) ----
    ixx_g_mm4: Optional[float]
    iyy_g_mm4: Optional[float]
    ixy_g_mm4: Optional[float]
    ip_g_mm4: Optional[float]
    rx_g_mm: Optional[float]
    ry_g_mm: Optional[float]
    sx_g_mm3: Optional[float]
    sy_g_mm3: Optional[float]

    # ---- Principal about origin ----
    principal_angle_g_deg: Optional[float]
    i11_g_mm4: Optional[float]
    i22_g_mm4: Optional[float]

    # ---- Centroidal/library values & centroid location (mm / mm^n) ----
    area_mm2: Optional[float]
    perimeter_mm: Optional[float]
    cx_mm: Optional[float]
    cy_mm: Optional[float]
    ixx_c_mm4: Optional[float]
    iyy_c_mm4: Optional[float]
    ixy_c_mm4: Optional[float]
    ip_c_mm4: Optional[float]
    rx_c_mm: Optional[float]
    ry_c_mm: Optional[float]
    sx_c_mm3: Optional[float]
    sy_c_mm3: Optional[float]

    # ---- Principal about centroid ----
    principal_angle_deg: Optional[float]
    i11_c_mm4: Optional[float]
    i22_c_mm4: Optional[float]

    # ---- Back-compat aliases (as exposed by SectionDXF) ----
    rx_mm: Optional[float]
    ry_mm: Optional[float]
    sx_mm3: Optional[float]
    sy_mm3: Optional[float]
    i1_mm4: Optional[float]
    i2_mm4: Optional[float]

    # ---- Bounding-box touch points (mm) ----
    xmin_touch_x_mm: Optional[float]
    xmin_touch_y_mm: Optional[float]
    xmax_touch_x_mm: Optional[float]
    xmax_touch_y_mm: Optional[float]
    ymin_touch_x_mm: Optional[float]
    ymin_touch_y_mm: Optional[float]
    ymax_touch_x_mm: Optional[float]
    ymax_touch_y_mm: Optional[float]

    # ---- Warping / Shear ----
    j_mm4: Optional[float]
    asx_mm2: Optional[float]
    asy_mm2: Optional[float]
    scx_global_mm: Optional[float]
    scy_global_mm: Optional[float]
    scx_from_centroid_mm: Optional[float]
    scy_from_centroid_mm: Optional[float]
    scx_norm_origin: Optional[float]
    scy_norm_origin: Optional[float]
    scx_norm_centroid: Optional[float]
    scy_norm_centroid: Optional[float]
    sc1_mm: Optional[float]
    sc2_mm: Optional[float]
    beta_x_plus: Optional[float]
    beta_x_minus: Optional[float]
    beta_y_plus: Optional[float]
    beta_y_minus: Optional[float]

    # ---- Derived ----
    shape_factor_x_c: Optional[float]
    shape_factor_y_c: Optional[float]
    shape_factor_x_g: Optional[float]
    shape_factor_y_g: Optional[float]
    polar_r_c_mm: Optional[float]
    polar_r_g_mm: Optional[float]
    j_over_ip_c: Optional[float]
    j_over_ip_g: Optional[float]
    asx_over_a: Optional[float]
    asy_over_a: Optional[float]
    compactness: Optional[float]
    vx: Optional[float]
    vy: Optional[float]

    # -------------------------
    # Factory / helpers
    # -------------------------
    @classmethod
    def from_row(cls, row: List[Any]) -> "GeometricPropertyAnalysisBin":
        """
        Create instance from a SectionDXF row.

        Index mapping matches the refactored SectionDXF header order:
          0  run_label
          1  mesh_h
          -- extents (2..9)
          2..9: x_min, x_max, y_min, y_max, bbox_w, bbox_h, c_x_g, c_y_g
          -- geometric (10..41)
          10..17:   Ixx_g, Iyy_g, Ixy_g, Ip_g, rx_g, ry_g, Sx_g, Sy_g
          18..20:   Principal_angle_g, I11_g, I22_g
          21..24:   Area, Perimeter, Cx, Cy
          25..32:   Ixx_c, Iyy_c, Ixy_c, Ip_c, rx_c, ry_c, Sx_c, Sy_c
          33..35:   Principal_angle (centroid), I11_c, I22_c
          36..41:   rx_mm, ry_mm, Sx_mm3, Sy_mm3, I1_mm4, I2_mm4
          -- bounds (42..49)
          42..49:   xmin_touch(x,y), xmax_touch(x,y), ymin_touch(x,y), ymax_touch(x,y)
          -- warping/shear (50..66)
          50..66:   J, Asx, Asy, SCx_g, SCy_g, SCx_from_c, SCy_from_c,
                    SCx_norm_origin, SCy_norm_origin, SCx_norm_centroid, SCy_norm_centroid,
                    SC1, SC2, beta_x_plus, beta_x_minus, beta_y_plus, beta_y_minus
          -- derived (67..79)
          67..79:   ShapeFactor_x_c, ShapeFactor_y_c, ShapeFactor_x_g, ShapeFactor_y_g,
                    PolarR_c, PolarR_g, J/Ip_c, J/Ip_g, Asx/A, Asy/A, Compactness, vx, vy
        """
        return cls(
            # Identification
            run_label=row[0],
            mesh_h=row[1],

            # Extents
            x_min_mm=row[2],
            x_max_mm=row[3],
            y_min_mm=row[4],
            y_max_mm=row[5],
            bbox_width_mm=row[6],
            bbox_height_mm=row[7],
            c_x_g_mm=row[8],
            c_y_g_mm=row[9],

            # Global (origin)
            ixx_g_mm4=row[10],
            iyy_g_mm4=row[11],
            ixy_g_mm4=row[12],
            ip_g_mm4=row[13],
            rx_g_mm=row[14],
            ry_g_mm=row[15],
            sx_g_mm3=row[16],
            sy_g_mm3=row[17],

            # Principal about origin
            principal_angle_g_deg=row[18],
            i11_g_mm4=row[19],
            i22_g_mm4=row[20],

            # Centroidal/library
            area_mm2=row[21],
            perimeter_mm=row[22],
            cx_mm=row[23],
            cy_mm=row[24],
            ixx_c_mm4=row[25],
            iyy_c_mm4=row[26],
            ixy_c_mm4=row[27],
            ip_c_mm4=row[28],
            rx_c_mm=row[29],
            ry_c_mm=row[30],
            sx_c_mm3=row[31],
            sy_c_mm3=row[32],

            # Principal about centroid
            principal_angle_deg=row[33],
            i11_c_mm4=row[34],
            i22_c_mm4=row[35],

            # Back-compat aliases
            rx_mm=row[36],
            ry_mm=row[37],
            sx_mm3=row[38],
            sy_mm3=row[39],
            i1_mm4=row[40],
            i2_mm4=row[41],

            # Bounds touch points
            xmin_touch_x_mm=row[42],
            xmin_touch_y_mm=row[43],
            xmax_touch_x_mm=row[44],
            xmax_touch_y_mm=row[45],
            ymin_touch_x_mm=row[46],
            ymin_touch_y_mm=row[47],
            ymax_touch_x_mm=row[48],
            ymax_touch_y_mm=row[49],

            # Warping / Shear
            j_mm4=row[50],
            asx_mm2=row[51],
            asy_mm2=row[52],
            scx_global_mm=row[53],
            scy_global_mm=row[54],
            scx_from_centroid_mm=row[55],
            scy_from_centroid_mm=row[56],
            scx_norm_origin=row[57],
            scy_norm_origin=row[58],
            scx_norm_centroid=row[59],
            scy_norm_centroid=row[60],
            sc1_mm=row[61],
            sc2_mm=row[62],
            beta_x_plus=row[63],
            beta_x_minus=row[64],
            beta_y_plus=row[65],
            beta_y_minus=row[66],

            # Derived
            shape_factor_x_c=row[67],
            shape_factor_y_c=row[68],
            shape_factor_x_g=row[69],
            shape_factor_y_g=row[70],
            polar_r_c_mm=row[71],
            polar_r_g_mm=row[72],
            j_over_ip_c=row[73],
            j_over_ip_g=row[74],
            asx_over_a=row[75],
            asy_over_a=row[76],
            compactness=row[77],
            vx=row[78],
            vy=row[79],
        )

    def get_property_groups(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Return properties organized by analysis category (readable structure)."""
        return {
            "extents": {
                "x_min_mm": self.x_min_mm,
                "x_max_mm": self.x_max_mm,
                "y_min_mm": self.y_min_mm,
                "y_max_mm": self.y_max_mm,
                "bbox_width_mm": self.bbox_width_mm,
                "bbox_height_mm": self.bbox_height_mm,
                "c_x_g_mm": self.c_x_g_mm,
                "c_y_g_mm": self.c_y_g_mm,
            },
            "global_origin": {
                "ixx_g_mm4": self.ixx_g_mm4,
                "iyy_g_mm4": self.iyy_g_mm4,
                "ixy_g_mm4": self.ixy_g_mm4,
                "ip_g_mm4": self.ip_g_mm4,
                "rx_g_mm": self.rx_g_mm,
                "ry_g_mm": self.ry_g_mm,
                "sx_g_mm3": self.sx_g_mm3,
                "sy_g_mm3": self.sy_g_mm3,
            },
            "principal_about_origin": {
                "principal_angle_g_deg": self.principal_angle_g_deg,
                "i11_g_mm4": self.i11_g_mm4,
                "i22_g_mm4": self.i22_g_mm4,
            },
            "centroidal": {
                "area_mm2": self.area_mm2,
                "perimeter_mm": self.perimeter_mm,
                "cx_mm": self.cx_mm,
                "cy_mm": self.cy_mm,
                "ixx_c_mm4": self.ixx_c_mm4,
                "iyy_c_mm4": self.iyy_c_mm4,
                "ixy_c_mm4": self.ixy_c_mm4,
                "ip_c_mm4": self.ip_c_mm4,
                "rx_c_mm": self.rx_c_mm,
                "ry_c_mm": self.ry_c_mm,
                "sx_c_mm3": self.sx_c_mm3,
                "sy_c_mm3": self.sy_c_mm3,
            },
            "principal_about_centroid": {
                "principal_angle_deg": self.principal_angle_deg,
                "i11_c_mm4": self.i11_c_mm4,
                "i22_c_mm4": self.i22_c_mm4,
            },
            "compat_aliases": {
                "rx_mm": self.rx_mm,
                "ry_mm": self.ry_mm,
                "sx_mm3": self.sx_mm3,
                "sy_mm3": self.sy_mm3,
                "i1_mm4": self.i1_mm4,
                "i2_mm4": self.i2_mm4,
            },
            "bounds_touch_points": {
                "xmin_touch_x_mm": self.xmin_touch_x_mm,
                "xmin_touch_y_mm": self.xmin_touch_y_mm,
                "xmax_touch_x_mm": self.xmax_touch_x_mm,
                "xmax_touch_y_mm": self.xmax_touch_y_mm,
                "ymin_touch_x_mm": self.ymin_touch_x_mm,
                "ymin_touch_y_mm": self.ymin_touch_y_mm,
                "ymax_touch_x_mm": self.ymax_touch_x_mm,
                "ymax_touch_y_mm": self.ymax_touch_y_mm,
            },
            "warping_shear": {
                "j_mm4": self.j_mm4,
                "asx_mm2": self.asx_mm2,
                "asy_mm2": self.asy_mm2,
                "scx_global_mm": self.scx_global_mm,
                "scy_global_mm": self.scy_global_mm,
                "scx_from_centroid_mm": self.scx_from_centroid_mm,
                "scy_from_centroid_mm": self.scy_from_centroid_mm,
                "scx_norm_origin": self.scx_norm_origin,
                "scy_norm_origin": self.scy_norm_origin,
                "scx_norm_centroid": self.scx_norm_centroid,
                "scy_norm_centroid": self.scy_norm_centroid,
                "sc1_mm": self.sc1_mm,
                "sc2_mm": self.sc2_mm,
                "beta_x_plus": self.beta_x_plus,
                "beta_x_minus": self.beta_x_minus,
                "beta_y_plus": self.beta_y_plus,
                "beta_y_minus": self.beta_y_minus,
            },
            "derived": {
                "shape_factor_x_c": self.shape_factor_x_c,
                "shape_factor_y_c": self.shape_factor_y_c,
                "shape_factor_x_g": self.shape_factor_x_g,
                "shape_factor_y_g": self.shape_factor_y_g,
                "polar_r_c_mm": self.polar_r_c_mm,
                "polar_r_g_mm": self.polar_r_g_mm,
                "j_over_ip_c": self.j_over_ip_c,
                "j_over_ip_g": self.j_over_ip_g,
                "asx_over_a": self.asx_over_a,
                "asy_over_a": self.asy_over_a,
                "compactness": self.compactness,
                "vx": self.vx,
                "vy": self.vy,
            },
        }

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