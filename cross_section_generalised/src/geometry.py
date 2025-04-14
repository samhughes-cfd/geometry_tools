# cross_section_generalised\src\geometry.py

import numpy as np
import os
import logging

import os
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'geometry.log')  # absolute path!
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - GEOMETRY - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def generate_arc_points(xc, yc, radius, theta_start_deg, theta_end_deg, resolution=100):
    logger.info(f"Generating arc: center=({xc}, {yc}), radius={radius}, "
                f"start={theta_start_deg}°, end={theta_end_deg}°, resolution={resolution}")
    theta_start = np.radians(theta_start_deg)
    theta_end = np.radians(theta_end_deg)
    if theta_end < theta_start:
        logger.debug("Arc end angle less than start angle. Adjusting by adding 2π.")
        theta_end += 2 * np.pi
    theta = np.linspace(theta_start, theta_end, resolution)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    logger.debug(f"Generated {len(x)} arc points.")
    return np.column_stack((x, y))

def generate_enhanced_geometry(translated_points, connectivity, arc_resolution=20, arc_definitions=None):
    from config.parameters import ARC_DEFINITIONS
    enhanced_points = []
    arc_definitions = arc_definitions or ARC_DEFINITIONS
    logger.info(f"Generating enhanced geometry with {len(connectivity)} connections")

    for (start, end) in connectivity:
        pt_start = translated_points[start - 1]
        pt_end = translated_points[end - 1]
        enhanced_points.append(pt_start)
        logger.debug(f"Processing segment from point {start} to {end}")

        if (start, end) in arc_definitions:
            arc = arc_definitions[(start, end)]
            logger.info(f"Adding arc between points {start} and {end}")
            arc_pts = generate_arc_points(
                arc['center'][0], arc['center'][1],
                arc['radius'], arc['start_deg'], arc['end_deg'],
                resolution=arc_resolution
            )
            enhanced_points.extend(arc_pts)
        else:
            enhanced_points.append(pt_end)
            logger.debug(f"Added straight segment to point {end}")

    logger.info(f"Total enhanced points generated: {len(enhanced_points)}")
    return np.array(enhanced_points)

def prepare_geometry(points, centroid):
    logger.info("Translating geometry to centroid")
    translated = points - centroid
    npts = len(points)
    connectivity = list(zip(range(1, npts+1), np.roll(range(1, npts+1), -1)))
    logger.debug(f"Generated {len(connectivity)} connectivity pairs")
    return translated, connectivity