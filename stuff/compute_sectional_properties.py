from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
import csv

def load_outer_skin(step_file_path):
    """
    Loads the STEP file and returns the shape.

    Parameters:
    - step_file_path (str): Path to the input STEP file.

    Returns:
    - TopoDS_Shape: The shape object extracted from the STEP file.
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)

    if status != 1:
        raise ValueError("Error: Cannot read the STEP file.")

    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape

def slice_geometry(shape, interval):
    """
    Slices the geometry along the x-axis at regular intervals.

    Parameters:
    - shape (TopoDS_Shape): The shape to be sliced.
    - interval (float): The distance between each slice.

    Returns:
    - List[TopoDS_Shape]: A list of sliced shapes.
    """
    # Get the bounding box of the shape
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    slices = []
    x = xmin

    while x <= xmax:
        # Create a plane for slicing at position x
        section = BRepAlgoAPI_Section(shape, BRepPrimAPI_MakeBox(x, ymin, zmin, x + interval, ymax, zmax).Shape())
        section.Build()

        if not section.IsDone():
            raise RuntimeError(f"Slicing failed at x = {x}")

        slices.append(section.Shape())
        x += interval

    if not slices:
        raise ValueError("No valid slices were produced.")
    
    return slices

def compute_section_properties(sliced_shapes):
    """
    Computes the geometric properties (centroid and moment of inertia) for each sliced shape.

    Parameters:
    - sliced_shapes (List[TopoDS_Shape]): List of sliced shapes.

    Returns:
    - List[Dict]: List of dictionaries containing geometric properties for each slice.
    """
    properties = []
    
    for i, slice_shape in enumerate(sliced_shapes):
        props = GProp_GProps()
        brepgprop_SurfaceProperties(slice_shape, props)

        # Extract properties
        centroid = props.CentreOfMass()
        inertia = props.MatrixOfInertia()

        # Convert inertia matrix to a dictionary with labeled axes
        inertia_dict = {
            'I_xx': inertia(0, 0),
            'I_yy': inertia(1, 1),
            'I_zz': inertia(2, 2),
            'I_xy': inertia(0, 1),
            'I_xz': inertia(0, 2),
            'I_yx': inertia(1, 0),
            'I_yz': inertia(1, 2),
            'I_zx': inertia(2, 0),
            'I_zy': inertia(2, 1),
        }

        properties.append({
            'slice_index': i,
            'centroid_x': centroid.X(),
            'centroid_y': centroid.Y(),
            'centroid_z': centroid.Z(),
            **inertia_dict
        })
    
    return properties

def save_properties_to_csv(properties, output_csv_file):
    """
    Saves the computed properties to a CSV file.

    Parameters:
    - properties (List[Dict]): List of dictionaries containing geometric properties.
    - output_csv_file (str): Path to the output CSV file.
    """
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            'slice_index', 'centroid_x', 'centroid_y', 'centroid_z',
            'I_xx', 'I_yy', 'I_zz', 'I_xy', 'I_xz', 'I_yx', 'I_yz', 'I_zx', 'I_zy'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for prop in properties:
            writer.writerow(prop)

def main(outer_skin_step_file, output_csv_file, slice_interval):
    """
    Main function to slice the outer skin and compute geometric properties.

    Parameters:
    - outer_skin_step_file (str): Path to the STEP file containing the outer skin geometry.
    - output_csv_file (str): Path to the CSV file where the properties will be saved.
    - slice_interval (float): Interval between slices along the x-axis.
    """
    # Load the outer skin from the STEP file
    outer_skin = load_outer_skin(outer_skin_step_file)

    # Slice the geometry at regular intervals
    sliced_shapes = slice_geometry(outer_skin, slice_interval)

    # Compute geometric properties for each slice
    properties = compute_section_properties(sliced_shapes)

    # Save the computed properties to a CSV file
    save_properties_to_csv(properties, output_csv_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python compute_section_properties.py <outer_skin_step_file> <output_csv_file> <slice_interval>")
        sys.exit(1)
    
    outer_skin_step_file = sys.argv[1]
    output_csv_file = sys.argv[2]
    slice_interval = float(sys.argv[3])
    
    main(outer_skin_step_file, output_csv_file, slice_interval)
