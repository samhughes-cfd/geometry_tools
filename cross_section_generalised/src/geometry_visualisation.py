# cross_section_generalised\src\geometry_visualisation.py

import matplotlib.pyplot as plt
import numpy as np
from src.geometry import compute_polygon_centroid, draw_local_axes, generate_enhanced_geometry
import os
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'geometry_visualisation.log')  # Change per module
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - GEOMETRY VISUALISATION - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info("Geometry visualisation module logger initialized.")

def plot_geometry_stages(points, translated, enhanced, connectivity):
    """
    Plots three stages of the cross-section geometry:
    - Raw input
    - Translated to origin
    - Enhanced with arcs and rebuilt connectivity

    Args:
        points (np.ndarray): Original input geometry
        translated (np.ndarray): Geometry translated to centroid
        enhanced (np.ndarray): Enhanced geometry with arcs
        connectivity (list of tuples): Rebuilt connectivity for enhanced geometry
    """
    logger.info("Plotting geometry stages.")

    labels = ['Input Geometry', 'Translated Geometry', 'Enhanced Geometry']
    shapes = [points, translated, enhanced]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, (shape, label) in enumerate(zip(shapes, labels)):
        ax = axs[i]

        # Plot the enhanced geometry using connectivity
        if label == 'Enhanced Geometry':
            for start, end in connectivity:
                p1 = shape[start - 1]
                p2 = shape[end - 1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1.5)
        else:
            # For points and translated shapes, plot as closed polygon
            closed = shape
            if not np.allclose(shape[0], shape[-1], atol=1e-10):
                closed = np.vstack([shape, shape[0]])
            ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1.5)

        ax.set_title(label)
        ax.set_aspect('equal')
        ax.grid(True)

        # Draw local axes for translated and enhanced
        if label != 'Input Geometry':
            cz, cy = compute_polygon_centroid(shape)
            axis_len = 0.25 * np.max(np.ptp(shape, axis=0))
            draw_local_axes(ax, cz, cy, axis_len=axis_len)

    plt.tight_layout()
    plt.show()
    logger.info("Finished plotting geometry stages.")