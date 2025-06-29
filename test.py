from sectionproperties.pre import Geometry
from sectionproperties.analysis import Section
import matplotlib.pyplot as plt

# Step 1: Load the geometry from SVG
geom = Geometry.from_svg("Station8_group_union.svg")

# Step 2: Create a mesh (adjust mesh size as needed)
geom.create_mesh(mesh_sizes=[0.01])

# Step 3: Plot the generated mesh
section = Section(geometry=geom)
section.plot_mesh(materials=False)

# Optional: Show the plot
plt.show()
