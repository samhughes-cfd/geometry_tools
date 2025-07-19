# section_calc_n\geometry_utils\void_builder.py

from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale
from sectionproperties.pre.geometry import Geometry, CompoundGeometry
import logging


class VoidBuilder:
    def __init__(self, geometry: Geometry | CompoundGeometry, label: str, log_dir):
        self.geometry = geometry
        self.label = label
        self.log_dir = log_dir
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"void_builder.{self.label}")
        logger.setLevel(logging.DEBUG)
        log_file = self.log_dir / f"void_builder_{self.label.replace(' ', '_')}.log"
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False
        return logger

    def _get_outer_polygon(self) -> Polygon:
        """Extract the largest outer polygon from geometry."""
        if isinstance(self.geometry, Geometry):
            geom = self.geometry.geom
        elif isinstance(self.geometry, CompoundGeometry):
            # Combine sub-geometries into one multipolygon
            geom = MultiPolygon([g.geom for g in self.geometry.geoms])
        else:
            raise TypeError(f"[{self.label}] Unexpected geometry type: {type(self.geometry)}")

        if geom.geom_type == "Polygon":
            return geom
        elif geom.geom_type == "MultiPolygon":
            return max(geom.geoms, key=lambda g: g.area)
        else:
            raise ValueError(f"[{self.label}] Geometry must be Polygon or MultiPolygon")

    def insert_void(self, scale_factor: float) -> Geometry | CompoundGeometry:
        """Insert a scaled internal void based on the outer boundary."""
        outer = self._get_outer_polygon()
        cx, cy = outer.centroid.coords[0]

        self.logger.info(f"Scaling outer polygon inward by factor {scale_factor:.2f} at centroid ({cx:.4f}, {cy:.4f})")

        # Scale the outer boundary inward to make the void
        void_poly = scale(outer, xfact=scale_factor, yfact=scale_factor, origin=(cx, cy))
        if not void_poly.is_valid or void_poly.area <= 0:
            raise ValueError(f"[{self.label}] Void polygon invalid or collapsed at scale {scale_factor}")

        # Insert as interior ring in the outer polygon
        polygon_with_hole = Polygon(outer.exterior.coords, [void_poly.exterior.coords])

        # Wrap into a sectionproperties Geometry object
        new_geom = Geometry(geom=polygon_with_hole)

        self.logger.info(f"Inserted internal void. Outer area: {outer.area:.6f}, Void area: {void_poly.area:.6f}")
        return new_geom