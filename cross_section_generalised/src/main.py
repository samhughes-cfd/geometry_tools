# cross_section_generalised\src\main.py

import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root to path for importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.parameters import *
from src.geometry import *
from src.mechanics import *
from src.convergence_visualisation import *
from src.geometry_visualisation import plot_geometry_stages

# Setup logs directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(LOG_DIR, 'main.log'))
formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def main():
    try:
        logger.info("Starting cross-section analysis pipeline")

        # Load input data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '..', 'input.csv')
        logger.info(f"Reading input points from: {csv_path}")
        df = pd.read_csv(csv_path)
        points = df[['z', 'y']].values
        logger.info(f"Loaded {len(points)} points")

        # Initial processing
        centroid = compute_polygon_centroid(points)
        logger.info(f"Centroid computed: (Z={centroid[0]:.4f}, Y={centroid[1]:.4f})")
        translated, connectivity = prepare_geometry(points, centroid)
        logger.debug(f"Connectivity established with {len(connectivity)} segments")

        # === Geometry enhancement phase ===
        logger.info("Generating enhanced geometry across resolutions")
        enhanced_dict = {}
        results = {}
        node_counts = {}
        final_shape = None

        for res in RESOLUTIONS:
            logger.info(f"Enhancing geometry at resolution {res}")
            enhanced = generate_enhanced_geometry(translated, connectivity, res)
            if not np.allclose(enhanced[0], enhanced[-1]):
                logger.warning("Enhanced geometry not closed. Closing polygon.")
                enhanced = np.vstack([enhanced, enhanced[0]])
            enhanced_dict[res] = enhanced
            node_counts[res] = len(enhanced)
            results[res] = analyze_geometry(enhanced, ANGLES)

            if res == max(RESOLUTIONS):
                final_shape = enhanced
                logger.info("Stored final (max-resolution) shape for inset plots")

        # === NEW: Geometry stage visualisation ===
        logger.info("Plotting geometry transformation stages")
        plot_geometry_stages(points, translated, enhanced_dict)

        # === Convergence plot setup ===
        logger.info("Setting up convergence plots")
        fig, axs = setup_convergence_plot()

        # Get global bounds for insets
        logger.info("Computing global plot bounds")
        all_rotated = [rotate_points(final_shape, a) for a in ANGLES]
        xmin = min([pts[:, 0].min() for pts in all_rotated])
        xmax = max([pts[:, 0].max() for pts in all_rotated])
        ymin = min([pts[:, 1].min() for pts in all_rotated])
        ymax = max([pts[:, 1].max() for pts in all_rotated])
        pad = 0.05 * max(xmax - xmin, ymax - ymin)
        xlim = (xmin - pad, xmax + pad)
        ylim = (ymin - pad, ymax + pad)
        logger.debug(f"Plot limits: xlim={xlim}, ylim={ylim}")

        # Plot each orientation
        logger.info("Plotting results for each rotation angle")
        for idx, angle in enumerate(ANGLES):
            logger.info(f"Plotting angle {angle}°")
            ax = axs[idx // 2, idx % 2]
            x_vals = [node_counts[r] for r in RESOLUTIONS]
            inertia_data = [results[r][angle] for r in RESOLUTIONS]

            plot_convergence_data(ax, x_vals, inertia_data, angle)
            add_shape_inset(ax, final_shape, angle, xlim, ylim, AXIS_LEN_FACTOR)

        logger.info("Displaying plots")
        plt.show()
        logger.info("Execution completed successfully")

    except Exception as e:
        logger.exception("Unhandled exception occurred during execution.")
        raise

if __name__ == "__main__":
    main()