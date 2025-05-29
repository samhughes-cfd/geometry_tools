import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Arc Generation Function ---
def generate_arc_points(xc, yc, radius, theta_start_deg, theta_end_deg, resolution=100):
    theta_start = np.radians(theta_start_deg)
    theta_end = np.radians(theta_end_deg)
    if theta_end < theta_start:
        theta_end += 2 * np.pi
    theta = np.linspace(theta_start, theta_end, resolution)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return np.column_stack((x, y))

# --- Compute Polygon Inertia ---
def compute_polygon_inertia(coords):
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    z = coords[:, 0]  # Z is horizontal axis
    y = coords[:, 1]  # Y is vertical axis
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = (z * y_next - z_next * y)
    area = 0.5 * np.sum(cross)
    Izz = (1/12) * np.sum((y**2 + y*y_next + y_next**2) * cross)  # bending about Z
    Iyy = (1/12) * np.sum((z**2 + z*z_next + z_next**2) * cross)  # bending about Y
    Izy = (1/24) * np.sum((z*y_next + 2*z*y + 2*z_next*y_next + z_next*y) * cross)  # product of inertia
    return Izz, Iyy, Izy, area

# --- Rotate Points ---
def rotate_points(coords, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return coords @ R.T

# --- Polygon Centroid ---
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

# --- Enhanced Geometry Builder ---
def generate_enhanced_geometry(translated_points, connectivity, arc_resolution=20):
    arc_definitions = {
        (2, 3):   {'center': (-40.75, -12.5), 'radius': 7.5, 'start_deg': 180, 'end_deg': 270},
        (10, 11): {'center': (-40.75, -12.5), 'radius': 2.5, 'start_deg': 180, 'end_deg': 270},
        (4, 5):   {'center': (40.75, -12.5),  'radius': 7.5, 'start_deg': 270, 'end_deg': 360},
        (8, 9):   {'center': (40.75, -12.5),  'radius': 2.5, 'start_deg': 270, 'end_deg': 360},
    }
    enhanced_points = []
    current_node_id = 1
    previous_node_id = None
    for (start, end) in connectivity:
        pt_start = translated_points[start - 1]
        pt_end   = translated_points[end - 1]
        start_id = current_node_id
        enhanced_points.append(pt_start)
        current_node_id += 1
        if (start, end) in arc_definitions:
            arc = arc_definitions[(start, end)]
            arc_pts = generate_arc_points(
                arc['center'][0], arc['center'][1],
                arc['radius'], arc['start_deg'], arc['end_deg'],
                resolution=arc_resolution
            )
            for pt in arc_pts:
                enhanced_points.append(pt)
                current_node_id += 1
            end_id = current_node_id - 1
        else:
            enhanced_points.append(pt_end)
            current_node_id += 1
            end_id = current_node_id - 1
        previous_node_id = end_id
    return np.array(enhanced_points)

# --- Coordinate System Drawing ---
def draw_local_axes(ax, cx, cy, axis_len):
    clearance = axis_len * 0.15
    arrow_len = axis_len
    ax.text(cx, cy, r'$⊗$', color='red', fontsize=14,
            ha='center', va='center')
    ax.arrow(cx + clearance, cy, arrow_len, 0, color='red',
             head_width=0.1*arrow_len, width=0.02*arrow_len, length_includes_head=True)
    ax.text(cx + clearance + arrow_len * 1.05, cy, "Z+", fontsize=12, color='red', va='center')
    ax.arrow(cx, cy + clearance, 0, arrow_len, color='red',
             head_width=0.1*arrow_len, width=0.02*arrow_len, length_includes_head=True)
    ax.text(cx, cy + clearance + arrow_len * 1.05, "Y+", fontsize=12, color='red', ha='center')
    r = clearance + 1.05 * arrow_len
    zx = cx + r * np.cos(np.radians(225))
    zy = cy + r * np.sin(np.radians(225))
    ax.text(zx, zy, "X+", fontsize=12, color='red', ha='center', va='center')

# --- Main Script ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'input.csv')
    df = pd.read_csv(csv_path)
    points = df[['z', 'y']].values  # Note: z-y input per custom coordinate system
    centroid = points.mean(axis=0)
    translated = points - centroid
    connectivity = list(zip(range(1, len(points)+1), np.roll(range(1, len(points)+1), -1)))
    resolutions = [5, 10, 20, 40, 80]
    angles = [0, 90, 180, 270]
    thickness = 0.01  # [m] extrusion depth for torsional Ixx
    results = {a: {} for a in angles}
    for res in resolutions:
        enhanced = generate_enhanced_geometry(translated, connectivity, arc_resolution=res)
        if not np.allclose(enhanced[0], enhanced[-1]):
            enhanced = np.vstack([enhanced, enhanced[0]])
        for angle in angles:
            rotated = rotate_points(enhanced, angle)
            Izz, Iyy, Izy, area = compute_polygon_inertia(rotated)
            Ixx = area * (thickness ** 2) / 12  # torsional in custom system
            results[angle][res] = (Izz, Iyy, Izy, Ixx)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    final_shape = generate_enhanced_geometry(translated, connectivity, arc_resolution=max(resolutions))
    if not np.allclose(final_shape[0], final_shape[-1]):
        final_shape = np.vstack([final_shape, final_shape[0]])

    bbox = np.vstack([rotate_points(final_shape, a) for a in angles])
    xmin, ymin = bbox.min(axis=0)
    xmax, ymax = bbox.max(axis=0)
    padding = 0.05 * max(xmax - xmin, ymax - ymin)
    xlim = (xmin - padding, xmax + padding)
    ylim = (ymin - padding, ymax + padding)

    for idx, angle in enumerate(angles):
        ax = axs[idx//2, idx%2]
        x = resolutions
        y1 = [results[angle][r][0] for r in x]  # Izz (bending)
        y2 = [results[angle][r][1] for r in x]  # Iyy (bending)
        y3 = [results[angle][r][2] for r in x]  # Izy
        y4 = [results[angle][r][3] for r in x]  # Ixx (torsion)
        ax.plot(x, y1, 'o-', label=f"Izz = {y1[-1]:.2e}")
        ax.plot(x, y2, 's--', label=f"Iyy = {y2[-1]:.2e}")
        ax.plot(x, y3, 'd-.', label=f"Izy = {y3[-1]:.2e}")
        ax.plot(x, y4, 'v:', label=f"Ixx = {y4[-1]:.2e}")
        ax.set_title(f"{angle}° Orientation")
        ax.set_xlabel("Nodes [-]")
        ax.set_ylabel("Inertia $I$ [m$^4$]")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)
        inset = inset_axes(ax, width=1.5, height=1.5, loc='center left',
                           bbox_to_anchor=(1.05, 0.5), bbox_transform=ax.transAxes, borderpad=0)
        rotated = rotate_points(final_shape, angle)
        inset.plot(rotated[:,0], rotated[:,1], '-', color='black')
        cx, cy = compute_polygon_centroid(rotated)
        draw_local_axes(inset, cx, cy, axis_len=0.25 * max(xmax - xmin, ymax - ymin))
        inset.set_xlim(xlim)
        inset.set_ylim(ylim)
        inset.set_aspect('equal')
        inset.axis('off')

    plt.tight_layout()
    plt.show()