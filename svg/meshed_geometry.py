from sectionproperties.analysis import Section
import multiprocessing as mp
import matplotlib.pyplot as plt


class MeshedGeometry:
    def __init__(self, geometry, label):
        self.geometry = geometry
        self.label = label
        self.section = None

    def mesh(self, mesh_size=0.01):
        """Generate mesh and compute section properties."""
        self.geometry.create_mesh(mesh_sizes=[mesh_size])
        self.section = Section(geometry=self.geometry)
        self.section.calculate_geometric_properties()
        return self.section

    def plot(self, ax):
        """Plot the mesh or show failure."""
        ax.set_title(self.label)
        if self.section:
            self.section.plot_mesh(ax=ax)
            print(f"✅ {self.label} - Area: {self.section.get_area():.6f}")
        else:
            ax.text(0.5, 0.5, "Meshing Failed", ha="center", va="center")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")


def parallel_mesh(jobs, mesh_size=0.01):
    """Mesh multiple geometries in parallel."""
    def worker(job):
        label, geometry = job
        try:
            mg = MeshedGeometry(geometry, label)
            mg.mesh(mesh_size)
            return mg
        except Exception as e:
            print(f"❌ {label} failed: {e}")
            return None

    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(worker, jobs)