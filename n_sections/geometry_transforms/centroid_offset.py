# geometry_transforms/centroid_offset.py

import logging
from pathlib import Path
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis.section import Section


class CentroidOffset:
    """Translate a section so its centroid aligns with a target location."""

    def __init__(
        self,
        geometry: Geometry,
        cx_target: float = 0.0,
        cy_target: float = 0.0,
        mesh_sizes: float = 0.5,
        label: str = "Unnamed",
        logs_dir: Path | None = None,
    ):
        self.geometry = geometry
        self.cx_target = cx_target
        self.cy_target = cy_target
        self.mesh_sizes = mesh_sizes
        self.label = label
        self.logs_dir = logs_dir

        self._setup_logger()

    def _setup_logger(self):
        logger_name = f"centroid_offset.{self.label}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = Path(self.logs_dir) / f"centroid_offset_{self.label.replace(' ', '_')}.log"

            if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                       for h in self.logger.handlers):
                fh = logging.FileHandler(log_file, mode='w')
                fh.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console.setFormatter(formatter)
            self.logger.addHandler(console)

        self.logger.info(f"Logger initialized for CentroidOffset [{self.label}]")

    def apply(self) -> Geometry:
        """Translate the geometry to align its centroid with the desired (cx, cy) target."""
        try:
            from shapely.validation import explain_validity

            # ───── Validate original geometry ─────
            if hasattr(self.geometry, "geom"):
                if not self.geometry.geom.is_valid:
                    reason = explain_validity(self.geometry.geom)
                    self.logger.warning(f"Geometry invalid: {reason}")
                    self.logger.info("Attempting geometry fix via .buffer(0)")
                    self.geometry.geom = self.geometry.geom.buffer(0)
                    self.logger.info("Geometry repair attempted")

            # ───── Apply mesh size ─────
            if hasattr(self.geometry, "geoms"):
                for geom in self.geometry.geoms:
                    geom.mesh_sizes = self.mesh_sizes

            # ───── Mesh geometry ─────
            try:
                self.geometry.create_mesh(mesh_sizes=self.mesh_sizes)
                self.logger.info(f"Geometry meshed with mesh_sizes = {self.mesh_sizes}")
            except Exception as e:
                self.logger.exception("Mesh creation failed", exc_info=True)
                raise

            if not hasattr(self.geometry, "mesh") or self.geometry.mesh is None:
                raise ValueError("Mesh creation appears to have failed — no mesh present")

            # ───── Compute centroid and offset ─────
            section = Section(geometry=self.geometry)
            section.calculate_geometric_properties()
            cx_actual, cy_actual = section.get_c()

            dx = self.cx_target - cx_actual
            dy = self.cy_target - cy_actual

            self.logger.info(f"Actual centroid: ({cx_actual:.4f}, {cy_actual:.4f})")
            self.logger.info(f"Target centroid: ({self.cx_target:.4f}, {self.cy_target:.4f})")
            self.logger.info(f"Translation to apply: dx={dx:.4f}, dy={dy:.4f}")

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                self.logger.info("Offset negligible — no translation applied.")
                return self.geometry

            translated = self.geometry.shift_section(x_offset=dx, y_offset=dy)
            self.logger.info("Section successfully translated.")
            return translated

        except Exception as e:
            self.logger.exception("Centroid offset application failed", exc_info=True)
            raise