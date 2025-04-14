# cross_section_generalised\src\visualisation.py

import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.mechanics import rotate_points, compute_polygon_centroid

# Set logs directory based on project root: cross_section_generalised/logs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
log_file_path = os.path.join(LOG_DIR, 'visualisation.log')
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - VISUAL - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def draw_local_axes(ax, cx, cy, axis_len):
    logger.debug(f"Drawing local axes at ({cx:.4f}, {cy:.4f}) with length {axis_len:.4f}")
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

def setup_convergence_plot():
    logger.info("Setting up 2x2 convergence subplot layout")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    return fig, axs

def plot_convergence_data(ax, x_vals, inertia_data, angle):
    logger.info(f"Plotting convergence data for angle {angle}°")
    y_izz = [d[0] for d in inertia_data]
    y_iyy = [d[1] for d in inertia_data]
    y_izy = [d[2] for d in inertia_data]
    y_ixx = [d[3] for d in inertia_data]

    ax.plot(x_vals, y_izz, 'o-', label=f"Izz={y_izz[-1]:.2e}")
    ax.plot(x_vals, y_iyy, 's--', label=f"Iyy={y_iyy[-1]:.2e}")
    ax.plot(x_vals, y_izy, 'd-.', label=f"Izy={y_izy[-1]:.2e}")
    ax.plot(x_vals, y_ixx, 'v:', label=f"Ixx={y_ixx[-1]:.2e}")

    ax.set_title(f"{angle}° Orientation")
    ax.set_xlabel("Polygon Node Count [-]")
    ax.set_ylabel("Inertia [m^4]")
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

def add_shape_inset(ax, final_shape, angle, xlim, ylim, axis_len_factor=0.25):
    logger.info(f"Adding shape inset for angle {angle}°")

    # Bigger inset, same clean positioning below the legend
    inset = inset_axes(ax, width="50%", height="65%", loc='upper left',
                       bbox_to_anchor=(1.05, 0.0, 0.5, 0.65), bbox_transform=ax.transAxes,
                       borderpad=0)

    shape_rot = rotate_points(final_shape, angle)
    inset.plot(shape_rot[:, 0], shape_rot[:, 1], '-', color='black')

    cZ, cY = compute_polygon_centroid(shape_rot)

    axis_length = axis_len_factor * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    draw_local_axes(inset, cZ, cY, axis_length)

    inset.set_xlim(xlim)
    inset.set_ylim(ylim)
    inset.set_aspect('equal')
    inset.axis('off')