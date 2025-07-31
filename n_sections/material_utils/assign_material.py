# n_sections/material_utils/assign_material.py

from sectionproperties.pre.geometry import Geometry, CompoundGeometry
from sectionproperties.pre.pre import Material
from typing import Union
import logging
from pathlib import Path


class AssignMaterial:
    def __init__(
        self,
        geometry: Geometry | CompoundGeometry,
        material: Union[None, Material, dict[int, Material]],
        logs_dir: Path,
        label: str,
    ):
        self.geometry = geometry
        self.material = material
        self.logger = self._init_logger(logs_dir, label)

    def _init_logger(self, logs_dir: Path, label: str) -> logging.Logger:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "AssignMaterial.log"

        logger = logging.getLogger(f"AssignMaterial.{label}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
            for h in logger.handlers
        ):
            handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.info("Logging initialized for AssignMaterial")
        return logger

    def apply(self) -> None:
        if self.material is None:
            self._no_assignment()
        elif isinstance(self.geometry, Geometry):
            self._assign_single_geometry()
        elif isinstance(self.geometry, CompoundGeometry):
            if isinstance(self.material, Material):
                self._assign_uniform_compound()
            elif isinstance(self.material, dict):
                self._assign_indexed_compound()
            else:
                raise TypeError("Material for CompoundGeometry must be a Material or dict[int, Material]")
        else:
            raise TypeError(f"Unsupported geometry type: {type(self.geometry)}")

    def _no_assignment(self):
        self.logger.info("No material assignment provided.")

    def _assign_single_geometry(self):
        if not isinstance(self.material, Material):
            raise TypeError("Expected a single Material for a Geometry instance.")
        self.geometry.material = self.material
        self.logger.info("Assigned material to single geometry: %s", self.material.name)

    def _assign_uniform_compound(self):
        for i, geom_i in enumerate(self.geometry.geoms):
            geom_i.material = self.material
            self.logger.debug("Assigned common material '%s' to region %d", self.material.name, i)
        self.logger.info("Assigned common material to all %d regions", len(self.geometry.geoms))

    def _assign_indexed_compound(self):
        assigned = 0
        for i, geom_i in enumerate(self.geometry.geoms):
            mat = self.material.get(i)
            if mat is None:
                self.logger.warning("No material specified for region %d; skipping assignment.", i)
                continue
            if not isinstance(mat, Material):
                raise TypeError(f"Material at index {i} must be of type Material.")
            geom_i.material = mat
            self.logger.debug("Assigned material '%s' to region %d", mat.name, i)
            assigned += 1
        self.logger.info("Assigned indexed materials to %d regions", assigned)