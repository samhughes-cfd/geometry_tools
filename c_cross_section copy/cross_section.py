import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
#  Compute 2D Polygon Inertia (unchanged)
# ------------------------------------------------
def compute_polygon_inertia(coords):
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    z = coords[:, 0]
    y = coords[:, 1]
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = (z * y_next - z_next * y)
    area = 0.5 * np.sum(cross)
    Izz = (1/12) * np.sum((y**2 + y*y_next + y_next**2) * cross)
    Iyy = (1/12) * np.sum((z**2 + z*z_next + z_next**2) * cross)
    Izy = (1/24) * np.sum((z*y_next + 2*z*y + 2*z_next*y_next + z_next*y) * cross)
    return Izz, Iyy, Izy, area

# ------------------------------------------------
#  Rotate Points (unchanged)
# ------------------------------------------------
def rotate_points(coords, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return coords @ R.T

# ------------------------------------------------
#  Centroid (unchanged)
# ------------------------------------------------
def compute_polygon_centroid(coords):
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    z = coords[:, 0]
    y = coords[:, 1]
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = (z * y_next - z_next * y)
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-14:
        return (0.0, 0.0)
    Cz = (1/(6*A)) * np.sum((z + z_next) * cross)
    Cy = (1/(6*A)) * np.sum((y + y_next) * cross)
    return (Cz, Cy)

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
        enhanced_points.append(pt_start)  # always start
        if (start, end) in arc_definitions:
            arc = arc_definitions[(start, end)]
            arc_pts = generate_arc_points(
                arc['center'][0], arc['center'][1],
                arc['radius'], arc['start_deg'], arc['end_deg'],
                resolution=arc_resolution
            )
            for pt in arc_pts:
                enhanced_points.append(pt)
        else:
            enhanced_points.append(pt_end)
    return np.array(enhanced_points)

# ------------------------------------------------
#  Draw local Y-Z axes (unchanged)
# ------------------------------------------------
def draw_local_axes(ax, cx, cy, axis_len):
    clearance = axis_len * 0.15
    arrow_len = axis_len
    ax.text(cx, cy, r'$\otimes$', color='red', fontsize=14,
            ha='center', va='center')
    ax.arrow(cx + clearance, cy, arrow_len, 0, color='red',
             head_width=0.1*arrow_len, width=0.02*arrow_len, length_includes_head=True)
    ax.text(cx + clearance + arrow_len * 1.05, cy, "Z+", fontsize=12, color='red', va='center')
    ax.arrow(cx, cy + clearance, 0, arrow_len, color='red',
             head_width=0.1*arrow_len, width=0.02*arrow_len, length_includes_head=True)
    ax.text(cx, cy + clearance + arrow_len * 1.05, "Y+", fontsize=12, color='red', ha='center')

# ------------------------------------------------
#  Main script
# ------------------------------------------------
if __name__ == "__main__":

    # --- Step 1: Load CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, 'input.csv')
    df         = pd.read_csv(csv_path)
    points     = df[['z', 'y']].values

    # Translate to shift centroid near (0,0)
    centroid   = points.mean(axis=0)
    translated = points - centroid

    # Build connectivity
    npts = len(points)
    connectivity = list(zip(range(1, npts+1),
                            np.roll(range(1, npts+1), -1)))

    # Plot shapes before any rotation/inertia
    fig_geom, axs_geom = plt.subplots(1, 3, figsize=(18, 6))
    labels = ['Input Geometry', 'Translated Geometry', 'Enhanced Geometry']
    shapes = [points, translated, generate_enhanced_geometry(translated, connectivity, arc_resolution=80)]

    for i, (shape, label) in enumerate(zip(shapes, labels)):
        ax = axs_geom[i]
        closed_shape = np.vstack([shape, shape[0]]) if not np.allclose(shape[0], shape[-1]) else shape
        ax.plot(closed_shape[:, 0], closed_shape[:, 1], 'k-', lw=1.5)
        ax.set_title(label)
        ax.set_aspect('equal')
        ax.grid(True)

        if label != 'Input Geometry':
            cz, cy = compute_polygon_centroid(closed_shape)
            axis_len = 0.25 * np.max(np.ptp(closed_shape, axis=0))
            draw_local_axes(ax, cz, cy, axis_len=axis_len)

    plt.tight_layout()
    plt.show()

    # ------------------------------------
    #  Compute inertia for each resolution
    # ------------------------------------
    resolutions = [5, 10, 20, 40, 80]
    angles      = [0, 90, 180, 270]
    results     = {a: {} for a in angles}
    node_count_map = {}

    for res in resolutions:
        enhanced = generate_enhanced_geometry(translated, connectivity, arc_resolution=res)
        if not np.allclose(enhanced[0], enhanced[-1]):
            enhanced = np.vstack([enhanced, enhanced[0]])
        node_count_map[res] = len(enhanced)
        for angle in angles:
            rotated = rotate_points(enhanced, angle)
            Izz, Iyy, Izy, area = compute_polygon_inertia(rotated)
            Ixx = Izz + Iyy
            results[angle][res] = (Izz, Iyy, Izy, Ixx)

    # ------------------------------------
    #  Plot: x-axis = # of polygon nodes
    # ------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    final_shape = generate_enhanced_geometry(translated, connectivity, arc_resolution=max(resolutions))
    if not np.allclose(final_shape[0], final_shape[-1]):
        final_shape = np.vstack([final_shape, final_shape[0]])

    all_angles_pts = []
    for a in angles:
        all_angles_pts.append(rotate_points(final_shape, a))
    all_angles_pts = np.vstack(all_angles_pts)
    xmin, ymin = all_angles_pts.min(axis=0)
    xmax, ymax = all_angles_pts.max(axis=0)
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    xlim = (xmin - pad, xmax + pad)
    ylim = (ymin - pad, ymax + pad)

    for idx, angle in enumerate(angles):
        ax = axs[idx // 2, idx % 2]
        x_vals = [node_count_map[r] for r in resolutions]
        y_izz = [results[angle][r][0] for r in resolutions]
        y_iyy = [results[angle][r][1] for r in resolutions]
        y_izy = [results[angle][r][2] for r in resolutions]
        y_ixx = [results[angle][r][3] for r in resolutions]

        ax.plot(x_vals, y_izz, 'o-', label=f"Izz={y_izz[-1]:.2e}")
        ax.plot(x_vals, y_iyy, 's--', label=f"Iyy={y_iyy[-1]:.2e}")
        ax.plot(x_vals, y_izy, 'd-.', label=f"Izy={y_izy[-1]:.2e}")
        ax.plot(x_vals, y_ixx, 'v:', label=f"Ixx={y_ixx[-1]:.2e}")

        ax.set_title(f"{angle}° Orientation")
        ax.set_xlabel("Polygon Node Count [-]")
        ax.set_ylabel("Inertia [m^4]")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

        inset = inset_axes(ax, width=1.5, height=1.5, loc='center left',
                           bbox_to_anchor=(1.05, 0.5),
                           bbox_transform=ax.transAxes, borderpad=0)
        shape_rot = rotate_points(final_shape, angle)
        inset.plot(shape_rot[:,0], shape_rot[:,1], '-', color='black')

        cZ, cY = compute_polygon_centroid(shape_rot)
        axis_length = 0.25 * max(xmax - xmin, ymax - ymin)
        draw_local_axes(inset, cZ, cY, axis_length)

        inset.set_xlim(xlim)
        inset.set_ylim(ylim)
        inset.set_aspect('equal')
        inset.axis('off')

    plt.tight_layout()
    plt.show()