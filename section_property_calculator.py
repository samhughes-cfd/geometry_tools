import logging
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
from functools import wraps
from time import time
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section

# ------------------------------------------------
# Logging Configuration
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geometry_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add timeout decorator
def timeout(seconds=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            if time() - start > seconds:
                raise TimeoutError(f"Function {func.__name__} exceeded {seconds} seconds")
            return result
        return wrapper
    return decorator

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # Bytes
    return f"{mem / (1024 * 1024):.1f} MB"

# ------------------------------------------------
# Arc Generation
# ------------------------------------------------
def generate_arc_points(xc, yc, radius, theta_start_deg, theta_end_deg, resolution=100):
    try:
        if radius <= 0:
            raise ValueError(f"Invalid radius: {radius}. Must be positive.")
        
        theta_start = np.radians(theta_start_deg)
        theta_end = np.radians(theta_end_deg)
        
        if theta_end < theta_start:
            theta_end += 2 * np.pi
            
        theta = np.linspace(theta_start, theta_end, resolution)
        x = xc + radius * np.cos(theta)
        y = yc + radius * np.sin(theta)
        
        logger.debug(f"Generated arc with {resolution} points between {theta_start_deg}°-{theta_end_deg}°")
        return np.column_stack((x, y))
    
    except Exception as e:
        logger.error(f"Arc generation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

# ------------------------------------------------
# Geometry Builder
# ------------------------------------------------
def generate_enhanced_geometry(points, connectivity, arc_resolution=20):
    try:
        arc_definitions = {
            (2, 3):   {'center': (-40.75, -12.5), 'radius': 7.5, 'start_deg': 180, 'end_deg': 270},
            (10, 11): {'center': (-40.75, -12.5), 'radius': 2.5, 'start_deg': 180, 'end_deg': 270},
            (4, 5):   {'center': (40.75, -12.5),  'radius': 7.5, 'start_deg': 270, 'end_deg': 360},
            (8, 9):   {'center': (40.75, -12.5),  'radius': 2.5, 'start_deg': 270, 'end_deg': 360},
        }

        enhanced_points = []
        max_index = len(points)
        
        for (start, end) in connectivity:
            if start < 1 or end < 1 or start > max_index or end > max_index:
                raise ValueError(f"Invalid connectivity index: ({start}, {end}) for {max_index} points")
            
            pt_start = points[start - 1]
            pt_end = points[end - 1]
            enhanced_points.append(pt_start)
            
            if (start, end) in arc_definitions:
                logger.debug(f"Processing arc {start}-{end}")
                arc = arc_definitions[(start, end)]
                arc_pts = generate_arc_points(
                    arc['center'][0], arc['center'][1],
                    arc['radius'], arc['start_deg'], arc['end_deg'],
                    resolution=arc_resolution
                )
                enhanced_points.extend(arc_pts)
            else:
                enhanced_points.append(pt_end)
        
        logger.info(f"Generated geometry with {len(enhanced_points)} points")
        return np.array(enhanced_points)
    
    except Exception as e:
        logger.error(f"Geometry generation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

# ------------------------------------------------
# Rotation Function
# ------------------------------------------------
def rotate_points(coords, angle_deg):
    try:
        if not isinstance(coords, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(coords)}")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Need Nx2 array, got {coords.shape}")
        if not isinstance(angle_deg, (int, float)):
            raise TypeError(f"Angle must be numeric, got {type(angle_deg)}")

        logger.debug(f"Rotating {len(coords)} points by {angle_deg}°")
        
        if len(coords) == 0:
            logger.warning("Empty coordinates array")
            return np.empty((0, 2))

        theta = np.radians(angle_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = coords @ R.T
        
        if np.any(np.isnan(rotated)):
            raise ValueError("Rotation produced NaN values")
            
        return rotated

    except Exception as e:
        logger.error(f"Rotation failed: {str(e)}")
        logger.debug(f"Input: angle={angle_deg}°, shape={coords.shape}")
        raise

# ------------------------------------------------
# Inertia Calculations
# ------------------------------------------------
def compute_polygon_inertia(coords):
    try:
        if len(coords) < 3:
            raise ValueError("Need at least 3 points")
        
        if not np.allclose(coords[0], coords[-1]):
            logger.warning("Closing unclosed polygon")
            coords = np.vstack([coords, coords[0]])
        
        z = coords[:, 0]
        y = coords[:, 1]
        z_next = np.roll(z, -1)
        y_next = np.roll(y, -1)
        cross = (z * y_next - z_next * y)
        area = 0.5 * np.sum(cross)
        
        if area <= 0:
            raise ValueError("Negative area - check orientation")
        
        Izz = (1/12) * np.sum((y**2 + y*y_next + y_next**2) * cross)
        Iyy = (1/12) * np.sum((z**2 + z*z_next + z_next**2) * cross)
        Izy = (1/24) * np.sum((z*y_next + 2*z*y + 2*z_next*y_next + z_next*y) * cross)
        
        logger.debug(f"Manual results: Izz={Izz:.1f}, Iyy={Iyy:.1f}")
        return Izz, Iyy, Izy, Izz+Iyy, area
    
    except Exception as e:
        logger.error(f"Inertia calculation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

# ------------------------------------------------
# SectionProperties Analysis
# ------------------------------------------------
def analyze_with_sectionproperties(coords, angle=0, web_thickness=5.0, mesh_factor=0.2):
    try:
        logger.info("== [Analysis Pipeline Start] ==")
        
        # Validate input parameters first
        if not isinstance(coords, np.ndarray) or coords.shape[1] != 2:
            raise ValueError("Coordinates must be Nx2 numpy array")
        if web_thickness <= 0:
            raise ValueError("Web thickness must be positive")
        if mesh_factor <= 0.01:
            logger.warning("Very small mesh factor may cause instability")

        # Calculate and validate mesh size
        mesh_size = mesh_factor * web_thickness
        logger.info(f">> Target mesh size: {mesh_size:.2f}mm")
        
        # Enforce minimum mesh size
        min_mesh_size = 0.1  # 0.1mm minimum
        if mesh_size < min_mesh_size:
            logger.warning(f">> Mesh size {mesh_size:.2f}mm is too small, using minimum {min_mesh_size}mm")
            mesh_size = min_mesh_size
            
        # Ensure closed polygon with tolerance check
        closure_tol = 1e-6  # 1 micron tolerance
        if not np.allclose(coords[0], coords[-1], atol=closure_tol):
            logger.warning("Auto-closing polygon with tolerance %.2e", closure_tol)
            coords = np.vstack([coords, coords[0]])

        # Geometry creation with sanity checks
        facets = [(i, i+1) for i in range(len(coords)-1)]
        centroid = np.mean(coords, axis=0)
        
        try:
            geom = Geometry.from_points(
                points=coords,
                facets=facets,
                control_points=[centroid]
            )
            logger.info(">> Geometry created successfully")
        except Exception as e:
            logger.error(">> Geometry creation failed! First 3 points: %s", coords[:3])
            raise

        # Rotation handling with bounds check
        if angle != 0:
            logger.info(f">> Rotating by {angle}°")
            if not (-360 <= angle <= 360):
                logger.warning("Unusual rotation angle: %d°", angle)
            try:
                geom = geom.rotate_geometry(np.radians(angle))
            except Exception as e:
                logger.error(">> Rotation failed: %s", str(e))
                raise

        # Meshing with stability checks
        @timeout(30)
        def safe_mesh():
            try:
                geom.create_mesh(mesh_sizes=[mesh_size])
            except Exception as e:
                if "No geometry bounds" in str(e):
                    logger.error(">> Invalid geometry for meshing")
                raise
            
        try:
            logger.info(">> Starting meshing process")
            safe_mesh()
            logger.info(">> Meshing successful")
            
            # Element quality check
            try:
                poor_quality = [el for el in geom.mesh.elements if el.area < 1e-6]
                if len(poor_quality) > 0:
                    logger.warning(">> Found %d tiny elements (<1e-6mm²)", len(poor_quality))
            except AttributeError:
                logger.warning(">> Could not access mesh elements for quality check")
                
        except TimeoutError:
            logger.error(">> Meshing timeout! Geometry too complex?")
            raise
        except Exception as e:
            logger.error(">> Meshing failed: %s", str(e))
            raise

        # Analysis with numerical stability checks
        logger.info(">> Starting analysis")
        section = Section(geom)
        try:
            section.calculate_geometric_properties()
            
            # Warping properties with fallback
            try:
                section.calculate_warping_properties()
            except Exception as e:
                logger.warning(">> Warping properties unavailable: %s", str(e))
                
        except Exception as e:
            if "singular matrix" in str(e).lower():
                logger.error(">> Numerical instability detected in analysis")
            raise

        results = section.get_ic()[:3]
        logger.info("<< Analysis completed >>")
        return results

    except Exception as e:
        logger.error("<< Pipeline failed: %s >>", str(e))
        return (np.nan, np.nan, np.nan)
    
# ------------------------------------------------
# Main Execution
# ------------------------------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting analysis")
        
        # Load input data
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'input.csv')
            df = pd.read_csv(csv_path)
            points = df[['z', 'y']].values
            npts = len(points)
            connectivity = list(zip(range(1, npts+1), np.roll(range(1, npts+1), -1)))
            logger.info(f"Loaded {npts} points from {csv_path}")
        except Exception as e:
            logger.error(f"Data load failed: {str(e)}")
            raise

        # Configuration
        web_thickness = 5.0
        mesh_factor = 0.01
        resolutions = [5, 10, 20, 40, 80]
        angles = [0, 90, 180, 270]
        results = {angle: {'manual': [], 'sectionprops': []} for angle in angles}

        # Process configurations
        for angle in angles:
            logger.info(f"\n{'='*40}\nProcessing angle {angle}°\n{'='*40}")
            
            for res in resolutions:
                try:
                    logger.info(f"Resolution {res}")
                    raw = generate_enhanced_geometry(points, connectivity, res)
                    raw_centered = raw - raw.mean(axis=0)
                    rot = rotate_points(raw_centered, angle)

                    # Manual calculation
                    try:
                        Izz, Iyy, Izy, *_ = compute_polygon_inertia(rot)
                        results[angle]['manual'].append((res, Izz, Iyy, Izy))
                    except Exception as e:
                        logger.warning(f"Manual calculation failed: {str(e)}")
                        results[angle]['manual'].append((res, np.nan, np.nan, np.nan))

                    # SectionProperties
                    try:
                        sp_results = analyze_with_sectionproperties(raw_centered, angle, web_thickness, mesh_factor)
                        results[angle]['sectionprops'].append((res, *sp_results))
                    except Exception as e:
                        logger.warning(f"Section analysis failed: {str(e)}")
                        results[angle]['sectionprops'].append((res, np.nan, np.nan, np.nan))

                except Exception as e:
                    logger.error(f"Resolution {res} failed: {str(e)}")
                    continue

        # Plotting
        try:
            logger.info("Generating plots")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            for idx, angle in enumerate(angles):
                ax = axs[idx//2, idx%2]
                
                # Prepare data
                manual = np.atleast_2d(results[angle]['manual'])
                section = np.atleast_2d(results[angle]['sectionprops'])
                nodes = [len(generate_enhanced_geometry(points, connectivity, r)) for r in resolutions]

                # Plot manual
                if manual.size > 0 and not np.all(np.isnan(manual)):
                    ax.plot(nodes, manual[:,1], 'o-', label='Manual Izz')
                    ax.plot(nodes, manual[:,2], 's--', label='Manual Iyy')
                    ax.plot(nodes, manual[:,3], 'd-.', label='Manual Izy')

                # Plot sectionprops
                if section.size > 0 and not np.all(np.isnan(section)):
                    ax.plot(nodes, section[:,1], 'o-', alpha=0.5, label='Section Izz')
                    ax.plot(nodes, section[:,2], 's--', alpha=0.5, label='Section Iyy')
                    ax.plot(nodes, section[:,3], 'd-.', alpha=0.5, label='Section Izy')

                ax.set_title(f"{angle}° Orientation")
                ax.set_xlabel("Node Count")
                ax.set_ylabel("Inertia (mm⁴)")
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.savefig("results.png")
            logger.info("Analysis completed successfully")
            plt.show()
        
        except Exception as e:
            logger.error(f"Plotting failed: {str(e)}")
            raise

    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")
        exit(1)