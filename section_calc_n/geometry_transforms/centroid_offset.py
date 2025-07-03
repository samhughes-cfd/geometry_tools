# geometry_transforms/centroid_offset.py

import logging
from pathlib import Path
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section


class CentroidOffset:
    """Translate a section so its centroid aligns with a target location."""

    def __init__(self, geometry: Geometry, cx_target: float = 0.0, cy_target: float = 0.0,
                 mesh_sizes: float = 0.5, label: str = "Unnamed", logs_dir: Path | None = None):
        self.geometry = geometry
        self.cx_target = cx_target
        self.cy_target = cy_target
        self.mesh_sizes = mesh_sizes
        self.label = label
        self.logs_dir = logs_dir

        if self.logs_dir:
            self._setup_logger()

    def _setup_logger(self):
        log_file = Path(self.logs_dir) / f"centroid_offset.log"
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

    def apply(self) -> Geometry:
        """Translate the geometry to align its centroid with the desired (cx, cy) target."""
        try:
            from shapely.validation import explain_validity

            # Validate original geometry
            if hasattr(self.geometry, "geom"):
                geom_valid = self.geometry.geom.is_valid
                if not geom_valid:
                    reason = explain_validity(self.geometry.geom)
                    logging.warning(f"[{self.label}] Geometry is invalid: {reason}")

                    # Attempt fix via .buffer(0)
                    logging.info(f"[{self.label}] Attempting geometry fix with .buffer(0).")
                    self.geometry.geom = self.geometry.geom.buffer(0)
                    logging.info(f"[{self.label}] Geometry repair attempted.")

            # Apply mesh size to all sub-geometries
            if hasattr(self.geometry, "geoms"):
                for geom in self.geometry.geoms:
                    geom.mesh_sizes = self.mesh_sizes

            # Mesh geometry
            try:
                self.geometry.create_mesh(mesh_sizes=self.mesh_sizes)
                logging.info(f"[{self.label}] Geometry meshed with mesh_sizes={self.mesh_sizes}.")
            except Exception:
                logging.exception(f"[{self.label}] Mesh creation failed.")
                raise

            # Check mesh is valid
            if not hasattr(self.geometry, "mesh") or self.geometry.mesh is None:
                raise ValueError(f"[{self.label}] Mesh creation appears to have failed — no mesh present.")

            # Create Section and calculate centroid
            section = Section(geometry=self.geometry)
            section.calculate_geometric_properties()
            cx_actual, cy_actual = section.get_c()

            dx = self.cx_target - cx_actual
            dy = self.cy_target - cy_actual

            logging.info(f"[{self.label}] Actual centroid: ({cx_actual:.4f}, {cy_actual:.4f})")
            logging.info(f"[{self.label}] Desired centroid: ({self.cx_target:.4f}, {self.cy_target:.4f})")
            logging.info(f"[{self.label}] Applying translation dx={dx:.4f}, dy={dy:.4f}")

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                logging.info(f"[{self.label}] Offset negligible — no translation applied.")
                return self.geometry

            translated = self.geometry.shift_section(x_offset=dx, y_offset=dy)
            logging.info(f"[{self.label}] Section translated.")
            return translated

        except Exception:
            logging.exception(f"[{self.label}] Centroid offset application failed.")
            raise
