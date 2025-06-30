# section_calc/mesh_dxf.py

import logging
from sectionproperties.analysis import Section

logger = logging.getLogger("MeshDXF")

class MeshDXF:
    """
    Wrapper for meshing a sectionproperties geometry using a given mesh size.
    Removes any property calculation logic.
    """

    def __init__(self, geometry, label):
        self.geometry = geometry
        self.label = label
        self.mesh_generated = False

    def build(self, mesh_size: float):
        """
        Generate a mesh using the specified mesh size.
        Returns the meshed geometry object, or None on failure.
        """
        try:
            self.geometry.create_mesh(mesh_sizes=[mesh_size])
            self.mesh_generated = True
            logger.info(f"[{self.label}] mesh created with h = {mesh_size:.4g}")
            return self.geometry
        except Exception as e:
            logger.error(f"[{self.label}] mesh failed: {e}")
            return None

    def plot(self, ax):
        """
        Plot the mesh if it was successfully generated.
        """
        if not self.mesh_generated:
            ax.text(0.5, 0.5, "Mesh Not Generated", ha="center", va="center")
            return

        try:
            section = Section(geometry=self.geometry)
            section.plot_mesh(ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot Failed\n{e}", ha="center", va="center")
            logger.warning(f"[{self.label}] mesh plotting failed: {e}")