import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section

# ------------------------------------------------
#  Arc Generation (unchanged)
# ------------------------------------------------
def generate_arc_points(xc, yc, radius, theta_start_deg, theta_end_deg, resolution=100):
    theta_start = np.radians(theta_start_deg)
    theta_end   = np.radians(theta_end_deg)
    if theta_end < theta_start:
        theta_end += 2 * np.pi
    theta = np.linspace(theta_start, theta_end, resolution)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return np.column_stack((x, y))

# ------------------------------------------------
#  Generate Enhanced Geometry (unchanged)
# ------------------------------------------------
def generate_enhanced_geometry(translated_points, connectivity, arc_resolution=20):
    arc_definitions = {
        (2, 3):   {'center': (-40.75, -12.5), 'radius': 7.5, 'start_deg': 180, 'end_deg': 270},
        (10, 11): {'center': (-40.75, -12.5), 'radius': 2.5, 'start_deg': 180, 'end_deg': 270},
        (4, 5):   {'center': (40.75, -12.5),  'radius': 7.5, 'start_deg': 270, 'end_deg': 360},
        (8, 9):   {'center': (40.75, -12.5),  'radius': 2.5, 'start_deg': 270, 'end_deg': 360},
    }
    enhanced_points = []
    for (start, end) in connectivity:
        pt_start = translated_points[start - 1]
        pt_end   = translated_points[end   - 1]
        enhanced_points.append(pt_start)
        if (start, end) in arc_definitions:
            arc = arc_definitions[(start, end)]
            arc_pts = generate_arc_points(
                arc['center'][0], arc['center'][1],
                arc['radius'], arc['start_deg'], arc['end_deg'],
                resolution=arc_resolution
            )
            enhanced_points.extend(arc_pts)
        else:
            enhanced_points.append(pt_end)
    return np.array(enhanced_points)

# ------------------------------------------------
#  SectionProperties Analysis Core
# ------------------------------------------------
def analyze_with_sectionproperties(points, angle=0):
    # Create closed polygon
    closed_points = np.vstack([points, points[0]]) if not np.allclose(points[0], points[-1]) else points
    
    # Create sectionproperties geometry
    facets = [(i, i+1) for i in range(len(closed_points)-1)]
    control_points = [closed_points.mean(axis=0)]  # Centroid as control point
    
    geom = Geometry.from_points(
        points=closed_points,
        facets=facets,
        control_points=control_points,
        holes=None
    )
    
    # Rotate geometry
    if angle != 0:
        geom = geom.rotate_geometry(np.radians(angle))
    
    # Create mesh and section
    try:
        geom.create_mesh(
            mesh_sizes=[1.0],  # Finer mesh for better stability
            min_angle=30,      # Prevent degenerate elements
            coarse_size=10.0
        )
    except Exception as e:
        print(f"Meshing failed: {e}")
        geom.plot_geometry(title="Problematic Geometry")
        plt.show()
        raise

    # Create section with validity checks
    section = Section(geom)
    
    try:
        section.calculate_geometric_properties()
        section.calculate_warping_properties()
    except Exception as e:
        print(f"Analysis failed: {e}")
        section.plot_mesh(title="Failed Analysis Mesh")
        plt.show()
        raise

    return section

# ------------------------------------------------
#  Main script (modified)
# ------------------------------------------------
if __name__ == "__main__":
    # Load and prepare geometry
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'input.csv')
    df         = pd.read_csv(csv_path)
    points     = df[['z', 'y']].values
    
    # Generate enhanced geometry
    npts = len(points)
    connectivity = list(zip(range(1, npts+1), np.roll(range(1, npts+1), -1)))
    enhanced = generate_enhanced_geometry(points, connectivity, arc_resolution=80)
    
    # Plot setup
    resolutions = [5, 10, 20, 40, 80]
    angles = [0, 90, 180, 270]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Analysis loop
    for idx, angle in enumerate(angles):
        ax = axs[idx // 2, idx % 2]
        izz_vals, iyy_vals, izy_vals = [], [], []
        
        for res in resolutions:
            # Generate geometry with current resolution
            enhanced_geo = generate_enhanced_geometry(points, connectivity, res)
            
            # Analyze with sectionproperties
            section = analyze_with_sectionproperties(enhanced_geo, angle)
            
            # Store results
            izz, iyy, izy = section.get_ic()[:3]
            izz_vals.append(izz)
            iyy_vals.append(iyy)
            izy_vals.append(izy)
        
        # Plot results
        x_vals = [len(generate_enhanced_geometry(points, connectivity, res)) for res in resolutions]
        ax.plot(x_vals, izz_vals, 'o-', label=f"Izz = {izz_vals[-1]:.2e}")
        ax.plot(x_vals, iyy_vals, 's--', label=f"Iyy = {iyy_vals[-1]:.2e}")
        ax.plot(x_vals, izy_vals, 'd-.', label=f"Izy = {izy_vals[-1]:.2e}")
        
        # Plot configuration
        ax.set_title(f"{angle}° Orientation")
        ax.set_xlabel("Polygon Node Count")
        ax.set_ylabel("Moment of Inertia [mm⁴]")
        ax.grid(True)
        ax.legend()
        
        # Inset plot
        inset = inset_axes(ax, width=1.5, height=1.5, loc='upper left')
        section.plot_geometry(ax=inset, alpha=0.8)
        inset.set_title("")
        inset.axis('off')

    plt.tight_layout()
    plt.show()