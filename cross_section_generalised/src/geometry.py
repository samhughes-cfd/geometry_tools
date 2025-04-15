# cross_section_generalised\src\geometry.py

import numpy as np
import os
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'geometry.log')
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - GEOMETRY - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

logger.info("Geometry module logger initialized.")

def generate_arc_points(xc, yc, radius, theta_start_deg, theta_end_deg, resolution=100):
    theta_start = np.radians(theta_start_deg)
    theta_end = np.radians(theta_end_deg)
    if theta_end < theta_start:
        theta_end += 2 * np.pi
    theta = np.linspace(theta_start, theta_end, resolution)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return np.column_stack((x, y))

def rotate_points(coords, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return coords @ R.T

def compute_polygon_centroid(coords):
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    z, y = coords[:, 0], coords[:, 1]
    z_next, y_next = np.roll(z, -1), np.roll(y, -1)
    cross = (z * y_next - z_next * y)
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-14:
        return (0.0, 0.0)
    Cz = (1/(6*A)) * np.sum((z + z_next) * cross)
    Cy = (1/(6*A)) * np.sum((y + y_next) * cross)
    return (Cz, Cy)

def generate_enhanced_geometry(translated_points, connectivity, arc_resolution=20):
    arc_definitions = {
        (2, 3):   {'center': (-40.75, -12.5), 'radius': 7.5, 'start_deg': 180, 'end_deg': 270},
        (10, 11): {'center': (-40.75, -12.5), 'radius': 2.5, 'start_deg': 180, 'end_deg': 270},
        (4, 5):   {'center': (40.75, -12.5),  'radius': 7.5, 'start_deg': 270, 'end_deg': 360},
        (8, 9):   {'center': (40.75, -12.5),  'radius': 2.5, 'start_deg': 270, 'end_deg': 360},
    }

    enhanced_points = []
    last_point = None

    for start, end in connectivity:
        pt_start = translated_points[start - 1]
        pt_end = translated_points[end - 1]

        if last_point is None or not np.allclose(pt_start, last_point, atol=1e-10):
            enhanced_points.append(pt_start)
            last_point = pt_start

        if (start, end) in arc_definitions:
            arc = arc_definitions[(start, end)]
            logger.debug(f"Inserting arc from node {start} → {end}")
            arc_pts = generate_arc_points(
                arc['center'][0], arc['center'][1],
                arc['radius'], arc['start_deg'], arc['end_deg'],
                resolution=arc_resolution
            )
            if np.allclose(arc_pts[0], last_point, atol=1e-10):
                arc_pts = arc_pts[1:]
            enhanced_points.extend(arc_pts)
            last_point = arc_pts[-1]
        else:
            if not np.allclose(pt_end, last_point, atol=1e-10):
                enhanced_points.append(pt_end)
                last_point = pt_end

    logger.info(f"Enhanced geometry finalised with {len(enhanced_points)} points.")
    return np.array(enhanced_points)

def draw_local_axes(ax, cx, cy, axis_len):
    clearance = axis_len * 0.15
    arrow_len = axis_len
    ax.text(cx, cy, r'$\otimes$', color='red', fontsize=14, ha='center', va='center')
    ax.arrow(cx + clearance, cy, arrow_len, 0, color='red',
             head_width=0.1 * arrow_len, width=0.02 * arrow_len, length_includes_head=True)
    ax.text(cx + clearance + arrow_len * 1.05, cy, "Z+", fontsize=12, color='red', va='center')
    ax.arrow(cx, cy + clearance, 0, arrow_len, color='red',
             head_width=0.1 * arrow_len, width=0.02 * arrow_len, length_includes_head=True)
    ax.text(cx, cy + clearance + arrow_len * 1.05, "Y+", fontsize=12, color='red', ha='center')

def build_connectivity_from_polyline(points):
    n = len(points)
    return list(zip(range(1, n + 1), np.roll(range(1, n + 1), -1)))