import numpy as np
import matplotlib.pyplot as plt
from src.geometry import rotate_points, generate_enhanced_geometry, compute_polygon_centroid, draw_local_axes
from src.section import compute_polygon_inertia
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import logging

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'convergence_visualisation.log')
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - CONVERGENCE VISUALISATION - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

logger.info("Convergence visualisation module logger initialized.")


def plot_convergence(translated, connectivity, resolutions, angles):
    """
    Plots convergence of section inertia values as arc resolution increases
    across multiple rotated orientations.

    Args:
        translated (np.ndarray): Translated geometry.
        connectivity (list): Connectivity list of tuples (1-based indexing).
        resolutions (list): Arc resolutions to test.
        angles (list): Angles to rotate the shape for each convergence plot.
    """
    logger.info("Starting convergence plotting.")
    results = {a: {} for a in angles}
    node_count_map = {}

    for res in resolutions:
        logger.debug(f"Generating enhanced shape at arc resolution {res}")
        enhanced = generate_enhanced_geometry(translated, connectivity, arc_resolution=res)
        if not np.allclose(enhanced[0], enhanced[-1]):
            logger.warning(f"Polygon not closed at resolution {res}; closing it manually.")
            enhanced = np.vstack([enhanced, enhanced[0]])
        node_count_map[res] = len(enhanced)

        for angle in angles:
            rotated = rotate_points(enhanced, angle)
            Izz, Iyy, Izy, area = compute_polygon_inertia(rotated)
            Ixx = Izz + Iyy
            results[angle][res] = (Izz, Iyy, Izy, Ixx)
            logger.debug(f"Angle {angle}°, Res {res}: Izz={Izz:.3e}, Iyy={Iyy:.3e}, Ixy={Izy:.3e}, Area={area:.3e}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    logger.info("Generating final high-resolution shape for insets.")
    final_shape = generate_enhanced_geometry(translated, connectivity, arc_resolution=max(resolutions))
    if not np.allclose(final_shape[0], final_shape[-1]):
        final_shape = np.vstack([final_shape, final_shape[0]])

    all_rotated = np.vstack([rotate_points(final_shape, angle) for angle in angles])
    xmin, ymin = all_rotated.min(axis=0)
    xmax, ymax = all_rotated.max(axis=0)
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

        logger.info(f"Plotting convergence curve for angle {angle}°.")

        ax.plot(x_vals, y_izz, 'o-', label=f"Izz={y_izz[-1]:.2e}")
        ax.plot(x_vals, y_iyy, 's--', label=f"Iyy={y_iyy[-1]:.2e}")
        ax.plot(x_vals, y_izy, 'd-.', label=f"Izy={y_izy[-1]:.2e}")
        ax.plot(x_vals, y_ixx, 'v:', label=f"Ixx={y_ixx[-1]:.2e}")

        shape_rot = rotate_points(final_shape, angle)
        cZ, cY = compute_polygon_centroid(shape_rot)
        axis_length = 0.25 * max(xmax - xmin, ymax - ymin)

        inset = inset_axes(ax, width=1.5, height=1.5, loc='center left',
                           bbox_to_anchor=(1.05, 0.5), bbox_transform=ax.transAxes, borderpad=0)
        inset.plot(shape_rot[:, 0], shape_rot[:, 1], '-', color='black')
        draw_local_axes(inset, cZ, cY, axis_length)
        inset.set_xlim(xlim)
        inset.set_ylim(ylim)
        inset.set_aspect('equal')
        inset.axis('off')

        ax.set_title(f"{angle}° Orientation")
        ax.set_xlabel("Polygon Node Count [-]")
        ax.set_ylabel("Inertia [m^4]")
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

    plt.tight_layout()
    plt.show()
    logger.info("Finished plotting convergence analysis.")