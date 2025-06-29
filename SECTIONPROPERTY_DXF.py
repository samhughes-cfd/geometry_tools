import os
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from shapely.geometry import LineString, Polygon
from sectionproperties.pre import Material, Geometry
from sectionproperties.analysis import Section
from tabulate import tabulate
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ezdxf")

# 1. Custom arc processing function
def safe_process_arc(arc_entity, degrees_per_segment=1.0):
    """Process arc entities with validation to prevent degenerate cases"""
    try:
        center = (arc_entity.dxf.center.x, arc_entity.dxf.center.y)
        radius = arc_entity.dxf.radius
        start_angle = arc_entity.dxf.start_angle
        end_angle = arc_entity.dxf.end_angle
        
        # Validate arc parameters
        if radius <= 1e-6:  # Skip zero-radius arcs
            return None
            
        # Handle angle wrapping
        if end_angle < start_angle:
            end_angle += 360
        
        # Skip degenerate arcs (less than 1 degree)
        angle_span = abs(end_angle - start_angle)
        if angle_span < 1.0:
            return None
        
        # Calculate number of segments
        num_segments = max(2, int(angle_span * degrees_per_segment))
        
        # Generate arc points
        pts = []
        angles = np.linspace(start_angle, end_angle, num_segments + 1)
        
        for angle in angles:
            rad = np.radians(angle)
            x = center[0] + radius * np.cos(rad)
            y = center[1] + radius * np.sin(rad)
            pts.append((x, y))
        
        # Skip if we don't have enough points
        if len(pts) < 2:
            return None
            
        return LineString(pts)
    except Exception as e:
        print(f"  ⚠️ Error processing arc: {str(e)}")
        return None

# 2. Robust DXF loader
def load_dxf_with_retry(dxf_path, max_attempts=3):
    """Attempt to load DXF with multiple strategies"""
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempt {attempt}/{max_attempts}: Loading DXF directly...")
            return Geometry.from_dxf(dxf_path)
        except Exception as e:
            print(f"  ❌ Direct load failed: {str(e)}")
            
            if attempt < max_attempts:
                print("  ⚙️ Trying alternative approach...")
                try:
                    # Create geometry from facets
                    doc = ezdxf.readfile(dxf_path)
                    msp = doc.modelspace()
                    
                    geometry = Geometry()
                    entity_count = 0
                    
                    for e in msp:
                        try:
                            if e.dxftype() == 'LINE':
                                line = LineString([(e.dxf.start.x, e.dxf.start.y), 
                                                  (e.dxf.end.x, e.dxf.end.y)])
                                coords = np.array(line.coords)
                                for i in range(len(coords) - 1):
                                    geometry.add_facet([tuple(coords[i]), tuple(coords[i+1])])
                                entity_count += 1
                                
                            elif e.dxftype() == 'ARC':
                                arc_line = safe_process_arc(e)
                                if arc_line:
                                    coords = np.array(arc_line.coords)
                                    for i in range(len(coords) - 1):
                                        geometry.add_facet([tuple(coords[i]), tuple(coords[i+1])])
                                    entity_count += 1
                                    
                            elif e.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                                points = [v.dxf.location[:2] for v in e.vertices]
                                if e.closed:
                                    points.append(points[0])
                                if len(points) > 1:
                                    for i in range(len(points) - 1):
                                        geometry.add_facet([tuple(points[i]), tuple(points[i+1])])
                                    entity_count += 1
                        except Exception as entity_error:
                            print(f"  ⚠️ Skipping entity {e.dxftype()}: {str(entity_error)}")
                    
                    if entity_count == 0:
                        raise ValueError("No valid entities found in DXF file")
                    
                    geometry = geometry.close()
                    print(f"  ✅ Created geometry from {entity_count} entities")
                    return geometry
                    
                except Exception as alt_error:
                    print(f"  ❌ Alternative approach failed: {str(alt_error)}")
    
    raise RuntimeError(f"Failed to load DXF after {max_attempts} attempts")

# 3. Material definition
al_7075_t6 = Material(
    name="Al_7075_T6",
    elastic_modulus=71700,    # MPa
    poissons_ratio=0.33,
    yield_strength=503,       # MPa
    density=2.81e-6,          # kg/mm³
    color="silver"
)

# 4. Main analysis
try:
    dxf_path = "Station8.dxf"
    print(f"Processing DXF: {os.path.basename(dxf_path)}")
    print(f"File size: {os.path.getsize(dxf_path):,} bytes")
    
    # Load geometry with robust retry mechanism
    geometry = load_dxf_with_retry(dxf_path)
    geometry.material = al_7075_t6
    
    # Mesh generation
    print("Generating mesh...")
    geometry.create_mesh(mesh_sizes=[0.5])
    
    # Section analysis
    section = Section(geometry)
    print("Calculating geometric properties...")
    section.calculate_geometric_properties()
    print("Calculating plastic properties...")
    section.calculate_plastic_properties()
    print("Calculating warping properties...")
    section.calculate_warping_properties()
    
    # Calculate shear modulus
    shear_modulus = al_7075_t6.elastic_modulus / (2 * (1 + al_7075_t6.poissons_ratio))
    
    # Extract properties
    props = section.section_properties
    
    # Prepare property dictionaries
    geometric = {
        "Area (A)": [props.area, "mm²", "Mass calculation"],
        "Centroid (Cx)": [props.cx, "mm", "Load application point"],
        "Centroid (Cy)": [props.cy, "mm", "Load application point"],
        "Ixx": [props.ixx_c, "mm⁴", "Bending stiffness (x-axis)"],
        "Iyy": [props.iyy_c, "mm⁴", "Bending stiffness (y-axis)"],
        "Ixy": [props.ixy_c, "mm⁴", "Product of inertia"],
        "Polar Moment (Iz)": [props.i_zz_c, "mm⁴", "Axial torsion (circular only)"],
        "rx": [props.rx_c, "mm", "Buckling resistance (x-axis)"],
        "ry": [props.ry_c, "mm", "Buckling resistance (y-axis)"],
        "rz": [props.rz_c, "mm", "Torsional buckling"]
    }

    torsional = {
        "Torsional Constant (J)": [props.j, "mm⁴", "Twist calculation: θ = TL/GJ"],
        "Shear Center (Sx)": [props.x_s, "mm", "Avoid unintended torsion"],
        "Shear Center (Sy)": [props.y_s, "mm", "Avoid unintended torsion"],
        "Warping Constant (Γ)": [props.gamma, "mm⁶", "Constrained warping analysis"],
        "Max Shear Stress (τ_max)": [props.tau_max, "MPa", "Fatigue life prediction"]
    }

    strength = {
        "Zxx+": [props.zxx_plus, "mm³", "Elastic modulus (+x bending)"],
        "Zxx-": [props.zxx_minus, "mm³", "Elastic modulus (-x bending)"],
        "Zyy+": [props.zyy_plus, "mm³", "Elastic modulus (+y bending)"],
        "Zyy-": [props.zyy_minus, "mm³", "Elastic modulus (-y bending)"],
        "Sxx": [props.sxx, "mm³", "Plastic modulus (x-axis)"],
        "Syy": [props.syy, "mm³", "Plastic modulus (y-axis)"],
        "Shape Factor (ϕ_x)": [props.shape_factor_xx, "-", "Ductility assessment (x)"],
        "Shape Factor (ϕ_y)": [props.shape_factor_yy, "-", "Ductility assessment (y)"],
        "κx": [props.kappa_x, "-", "Shear deformation (x-direction)"],
        "κy": [props.kappa_y, "-", "Shear deformation (y-direction)"]
    }

    # Create report tables
    def create_table(data_dict, title):
        table = []
        for prop, (value, unit, desc) in data_dict.items():
            if abs(value) < 0.01:  # Format small numbers
                value_str = f"{value:.4e}"
            else:
                value_str = f"{value:,.2f}"
            table.append([prop, value_str, unit, desc])
        return tabulate(table, headers=["Property", "Value", "Units", "Application"], tablefmt="grid")

    # Print report
    print("\n" + "="*80)
    print(f"ALUMINIUM 7075-T6 CROSS-SECTION PROPERTIES")
    print(f"Material: E = {al_7075_t6.elastic_modulus:,.0f} MPa, G = {shear_modulus:,.0f} MPa")
    print(f"Mesh Elements: {len(section.elements):,} | Area: {props.area:,.2f} mm²")
    print("="*80)

    print("\n[GEOMETRIC PROPERTIES]")
    print(create_table(geometric, "Geometric Properties"))

    print("\n[TORSIONAL PROPERTIES]")
    print(create_table(torsional, "Torsional Properties"))

    print("\n[STRENGTH PROPERTIES]")
    print(create_table(strength, "Strength Properties"))

    # Visualizations
    print("\nGenerating visualizations...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mesh Plot
    section.plot_mesh(ax=axs[0, 0], title="Finite Element Mesh", 
                     materials=True, node_indexes=False)
    
    # Centroids
    section.plot_centroids(ax=axs[0, 1], title="Centers of Action")
    
    # Principal Axes
    section.plot_principal_axes(ax=axs[1, 0], title="Principal Axes & Moments")
    
    # Torsion Shear Stress
    section.plot_stress_psi(ax=axs[1, 1], title="Torsion Shear Stress Distribution")
    
    plt.tight_layout()
    plt.savefig("section_properties_visualization.png", dpi=300)
    print("Visualizations saved to section_properties_visualization.png")
    
    # Additional outputs
    print("\nCRITICAL PROPERTIES SUMMARY:")
    print(f"- Torsional Constant (J): {props.j:,.2f} mm⁴")
    print(f"- Polar Moment (Iz): {props.i_zz_c:,.2f} mm⁴")
    print(f"- Max Shear Stress: {props.tau_max:.1f} MPa")
    print(f"- Shear Center: ({props.x_s:.2f}, {props.y_s:.2f}) mm")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nTROUBLESHOOTING RECOMMENDATIONS:")
    print("1. Check DXF file integrity in a viewer (e.g., LibreCAD)")
    print("2. Re-export DXF with these settings:")
    print("   - Format: AutoCAD 2010 DXF")
    print("   - Explode all blocks/hatches")
    print("   - Include only 2D geometry")
    print("3. Try converting to SVG using online tools")
    print("4. Test with a simple DXF file first")

finally:
    print("\nAnalysis completed")