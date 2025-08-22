import ezdxf
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity

# section-properties: match your working imports
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section


# ─────────────────────────────────────────────────────────────────────────────
# DXF → coordinate extraction
# ─────────────────────────────────────────────────────────────────────────────
def read_dxf_section(dxf_path: str) -> np.ndarray:
    """Read a DXF and return boundary coordinates as numpy array (N, 2) in mm."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    pts = []

    # Prefer LWPOLYLINE (often closed)
    for e in msp.query("LWPOLYLINE"):
        xy = [(p[0], p[1]) for p in e.get_points()]
        if e.closed and xy and xy[0] != xy[-1]:
            xy.append(xy[0])
        pts.extend(xy)

    # Legacy POLYLINE
    for e in msp.query("POLYLINE"):
        xy = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        if getattr(e, "is_closed", False) and xy and xy[0] != xy[-1]:
            xy.append(xy[0])
        pts.extend(xy)

    # If drawn as discrete LINEs, we’ll just collect end points (ordering may be arbitrary)
    for e in msp.query("LINE"):
        s = (e.dxf.start.x, e.dxf.start.y)
        t = (e.dxf.end.x,   e.dxf.end.y)
        pts.extend([s, t])

    coords = np.asarray(pts, dtype=float)
    if coords.size == 0:
        raise ValueError("No usable outline entities found in DXF.")

    # De-duplicate consecutive duplicates
    if len(coords) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-12]
        coords = coords[keep]

    # Ensure closed loop if it looks like a single boundary
    if coords.shape[0] >= 3 and not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    return coords


# ─────────────────────────────────────────────────────────────────────────────
# Polygon integrals (centroidal Area, Ixx, Iyy)
# ─────────────────────────────────────────────────────────────────────────────
def polygon_section_properties(coords: np.ndarray):
    """Compute area, centroid, and Ixx/Iyy about centroid (mm, mm², mm⁴)."""
    x, y = coords[:, 0], coords[:, 1]
    if not np.allclose(coords[0], coords[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * np.sum(a)

    Cx = (1.0 / (6.0 * A)) * np.sum((x[:-1] + x[1:]) * a)
    Cy = (1.0 / (6.0 * A)) * np.sum((y[:-1] + y[1:]) * a)

    Ixx_o = (1.0 / 12.0) * np.sum((y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2) * a)
    Iyy_o = (1.0 / 12.0) * np.sum((x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2) * a)

    # Shift to centroidal axes
    Ixx_c = Ixx_o - A * Cy**2
    Iyy_c = Iyy_o - A * Cx**2

    return abs(A), (Cx, Cy), Ixx_c, Iyy_c


# ─────────────────────────────────────────────────────────────────────────────
# Warping-based torsional constant using section-properties
# ─────────────────────────────────────────────────────────────────────────────
def torsional_constant(coords: np.ndarray, mesh_size: float = 2.0) -> float:
    """
    Compute torsional constant Jt [mm^4] using FE warping analysis.
    Requires geometric properties to be calculated first.
    """
    # Ensure closed ring
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import explain_validity
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid:
            raise ValueError(f"Invalid polygon for warping analysis: {explain_validity(poly)}")

    from sectionproperties.pre.geometry import Geometry
    from sectionproperties.analysis.section import Section

    geom = Geometry(geom=poly)
    geom.create_mesh(mesh_sizes=[mesh_size])   # single-part geometry → scalar/list is fine

    sec = Section(geom)

    # REQUIRED: compute geometric props first (centroid, etc.)
    sec.calculate_geometric_properties()

    # Now run the warping (Saint-Venant) analysis for torsion
    # (you can pass solver_type="direct" or "cgs" if you prefer)
    sec.calculate_warping_properties()

    # Read Jt (API varies slightly across versions)
    if hasattr(sec, "section_props") and hasattr(sec.section_props, "j"):
        return float(sec.section_props.j)
    if hasattr(sec, "get_j"):
        return float(sec.get_j())

    raise AttributeError("Could not retrieve torsional constant J from Section.")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dxf_path = r"C:\Users\samea\Desktop\geometry_tools\n_sections\blade\sec_000_r0.125.dxf"
    coords = read_dxf_section(dxf_path)

    A, (Cx, Cy), Ixx_c, Iyy_c = polygon_section_properties(coords)
    Jt = torsional_constant(coords, mesh_size=2.0)

    print(f"Area [mm²]: {A:.3f}")
    print(f"Centroid [mm]: ({Cx:.3f}, {Cy:.3f})")

    # Remember: in your convention x is vertical (flapwise), y is horizontal (edgewise).
    print("\nCentroidal inertias (about axes at the centroid):")
    print(f" Ixx (about x-axis = vertical flapwise axis, resists edgewise bending): {Ixx_c:.3e} mm⁴")
    print(f" Iyy (about y-axis = horizontal edgewise axis, resists flapwise bending): {Iyy_c:.3e} mm⁴")

    print(f"\nTorsional constant Jt: {Jt:.3e} mm⁴")