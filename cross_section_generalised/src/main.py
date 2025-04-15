# cross_section_generalised\src\main.py

import os
import pandas as pd
import numpy as np
from src.geometry import generate_enhanced_geometry, build_connectivity_from_polyline
from src.geometry_visualisation import plot_geometry_stages
from src.convergence_visualisation import plot_convergence
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'main.log')
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("Main module logger initialized.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'input.csv')
    df = pd.read_csv(csv_path)
    points = df[['z', 'y']].values

    # Translate to centroid
    centroid = points.mean(axis=0)
    translated = points - centroid

    # Original connectivity (for enhancement only)
    npts = len(points)
    connectivity = list(zip(range(1, npts+1), np.roll(range(1, npts+1), -1)))

    # Generate enhanced geometry and matching connectivity
    enhanced = generate_enhanced_geometry(translated, connectivity, arc_resolution=80)
    new_connectivity = build_connectivity_from_polyline(enhanced)

    # Plot all three stages
    plot_geometry_stages(points, translated, enhanced, new_connectivity)

    # Perform convergence analysis
    resolutions = [5, 10, 20, 40, 80]
    angles = [0, 90, 180, 270]
    plot_convergence(translated, connectivity, resolutions, angles)
