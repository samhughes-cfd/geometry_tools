# cross_section_generalised\src\mechanics.py

import os
import numpy as np
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'mechanics.log')  # absolute path!
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - MECHANICS - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def compute_polygon_inertia(coords):
    logger.info("Computing second moments of area for polygon")
    if not np.allclose(coords[0], coords[-1]):
        logger.warning("Polygon is not closed. Automatically closing it.")
        coords = np.vstack([coords, coords[0]])

    z = coords[:, 0]
    y = coords[:, 1]
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = z * y_next - z_next * y
    area = 0.5 * np.sum(cross)
    logger.debug(f"Computed polygon area: {area:.6f}")

    Izz = (1/12) * np.sum((y**2 + y*y_next + y_next**2) * cross)
    Iyy = (1/12) * np.sum((z**2 + z*z_next + z_next**2) * cross)
    Izy = (1/24) * np.sum((z*y_next + 2*z*y + 2*z_next*y_next + z_next*y) * cross)
    
    logger.info(f"Inertia results: Izz={Izz:.6f}, Iyy={Iyy:.6f}, Izy={Izy:.6f}")
    return Izz, Iyy, Izy, area

def compute_polygon_centroid(coords):
    logger.info("Computing centroid of polygon")
    if not np.allclose(coords[0], coords[-1]):
        logger.warning("Polygon is not closed. Automatically closing it.")
        coords = np.vstack([coords, coords[0]])

    z = coords[:, 0]
    y = coords[:, 1]
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = z * y_next - z_next * y
    A = 0.5 * np.sum(cross)
    logger.debug(f"Computed polygon area: {A:.6f}")

    if abs(A) < 1e-14:
        logger.warning("Polygon area is near zero — possible degenerate shape.")
        return (0.0, 0.0)

    Cz = (1/(6*A)) * np.sum((z + z_next) * cross)
    Cy = (1/(6*A)) * np.sum((y + y_next) * cross)
    logger.info(f"Computed centroid: Cz={Cz:.6f}, Cy={Cy:.6f}")
    return (Cz, Cy)

def rotate_points(coords, angle_deg):
    logger.info(f"Rotating coordinates by {angle_deg} degrees")
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    rotated = coords @ R.T
    logger.debug(f"Rotation matrix:\n{R}")
    return rotated

def analyze_geometry(enhanced_coords, angles):
    logger.info(f"Analyzing geometry for {len(angles)} angles")
    results = {}
    for angle in angles:
        logger.info(f"Analyzing angle {angle}°")
        rotated = rotate_points(enhanced_coords, angle)
        Izz, Iyy, Izy, area = compute_polygon_inertia(rotated)
        results[angle] = (Izz, Iyy, Izy, Izz + Iyy)
        logger.debug(f"Angle {angle}° → Izz={Izz:.4f}, Iyy={Iyy:.4f}, Izy={Izy:.4f}, Ixx={Izz + Iyy:.4f}")
    return results