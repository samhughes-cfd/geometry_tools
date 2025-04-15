# cross_section_generalised\src\section.py

import numpy as np
import os
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'section.log')
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - SECTION - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

logger.info("Section module logger initialized.")


def compute_polygon_inertia(coords):
    """
    Computes area and second moments of inertia for a closed polygon.

    Args:
        coords (np.ndarray): Nx2 array of (z, y) coordinates.

    Returns:
        tuple: (Izz, Iyy, Izy, area)
    """
    logger.info("Computing polygon inertia.")
    if not np.allclose(coords[0], coords[-1]):
        logger.debug("Polygon not closed — auto-closing by appending first point.")
        coords = np.vstack([coords, coords[0]])

    z = coords[:, 0]
    y = coords[:, 1]
    z_next = np.roll(z, -1)
    y_next = np.roll(y, -1)
    cross = (z * y_next - z_next * y)
    area = 0.5 * np.sum(cross)

    if abs(area) < 1e-14:
        logger.warning("Polygon area is very small or zero — possible degenerate geometry.")

    Izz = (1 / 12) * np.sum((y**2 + y * y_next + y_next**2) * cross)
    Iyy = (1 / 12) * np.sum((z**2 + z * z_next + z_next**2) * cross)
    Izy = (1 / 24) * np.sum((z * y_next + 2 * z * y + 2 * z_next * y_next + z_next * y) * cross)

    logger.debug(f"Area: {area:.6e}, Izz: {Izz:.6e}, Iyy: {Iyy:.6e}, Izy: {Izy:.6e}")
    return Izz, Iyy, Izy, area