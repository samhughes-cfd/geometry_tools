import OCC.Core.STEPControl as STEPControl
import OCC.Core.BRep as BRep
import OCC.Core.BRepTools as BRepTools
import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
import OCC.Core.BRepGProp as BRepGProp
import OCC.Core.BRepProp as BRepProp
import OCC.Core.TopExp as TopExp
import OCC.Core.TopAbs as TopAbs
import OCC.Core.TopLoc as TopLoc
import OCC.Core.gp as gp
import OCC.Core.TopoDS as TopoDS
import OCC.Core.TopoDS as TopoDS

def load_step_file(file_path):
    """
    Loads a STEP file and returns the shape object.
    
    Parameters:
    - file_path (str): Path to the input STEP file.
    
    Returns:
    - shape (TopoDS_Shape): The shape object representing the 3D model.
    """
    step_reader = STEPControl.STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    
    if status != STEPControl.STEPControl_OK:
        raise RuntimeError(f"Error reading STEP file: {status}")
    
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape

def save_outer_skin_to_step(shape, file_path):
    """
    Saves the shape to a STEP file.
    
    Parameters:
    - shape (TopoDS_Shape): The shape object to be saved.
    - file_path (str): Path to the output STEP file.
    """
    step_writer = STEPControl.STEPControl_Writer()
    step_writer.Transfer(shape, STEPControl.STEPControl_AsIs)
    status = step_writer.Write(file_path)
    
    if status != STEPControl.STEPControl_OK:
        raise RuntimeError(f"Error writing STEP file: {status}")

def is_shape_watertight(shape):
    """
    Checks if the shape is watertight (i.e., has no holes or gaps).
    
    Parameters:
    - shape (TopoDS_Shape): The shape object to be checked.
    
    Returns:
    - is_watertight (bool): True if the shape is watertight, False otherwise.
    """
    # Create a BRepTools_WireExplorer to explore edges
    wire_explorer = BRepTools.BRepTools_WireExplorer()
    wire_explorer.Initialize(shape)
    
    is_watertight = True
    
    for edge in wire_explorer.Edges():
        if not wire_explorer.IsEdgeClosed(edge):
            is_watertight = False
            break
    
    return is_watertight

def calculate_face_normal(face):
    """
    Calculates the normal vector of a face.
    
    Parameters:
    - face (TopoDS_Face): The face to be analyzed.
    
    Returns:
    - normal (gp_Vec): The normal vector of the face.
    """
    normal = gp.gp_Vec()
    face_geom = BRep.BRep_Tool.Surface(face)
    face_plane = gp.gp_Pln(face_geom)
    normal = face_plane.Axis().Direction()
    return normal

def analyze_face(face, shape_bbox):
    """
    Analyzes a face to determine if it's part of the outer skin of the shape and calculates face normal.
    
    Parameters:
    - face (TopoDS_Face): The face to be analyzed.
    - shape_bbox (gp_BBox): The bounding box of the entire shape.
    
    Returns:
    - is_outer (bool): True if the face is part of the outer skin, False otherwise.
    - face_info (dict): A dictionary containing the face normal and whether it is part of the outer skin.
    """
    face_bbox = face.BoundingBox()
    
    # Check if the face's bounding box intersects with the shape's bounding box
    is_outer = (
        face_bbox.CornerMin().X() == shape_bbox.CornerMin().X() or
        face_bbox.CornerMax().X() == shape_bbox.CornerMax().X() or
        face_bbox.CornerMin().Y() == shape_bbox.CornerMin().Y() or
        face_bbox.CornerMax().Y() == shape_bbox.CornerMax().Y() or
        face_bbox.CornerMin().Z() == shape_bbox.CornerMin().Z() or
        face_bbox.CornerMax().Z() == shape_bbox.CornerMax().Z()
    )
    
    # Calculate face normal vector for additional context
    normal = calculate_face_normal(face)
    
    face_info = {
        "is_outer": is_outer,
        "normal": normal
    }
    
    return face_info

def extract_outer_skin(shape):
    """
    Extracts the outer skin of the shape based on bounding box analysis.
    
    Parameters:
    - shape (TopoDS_Shape): The shape object to be processed.
    
    Returns:
    - outer_skin (TopoDS_Shape): The shape object representing the outer skin.
    """
    outer_skin = TopoDS.TopoDS_Compound()
    compound_builder = BRepBuilderAPI.BRepBuilderAPI_Compound()
    
    # Create a bounding box for the entire shape
    shape_bbox = shape.BoundingBox()
    
    # Iterate through faces of the shape and check if they are part of the outer skin
    explorer = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_FACE)
    
    while explorer.More():
        face = TopoDS.TopoDS_Face(explorer.Current())
        face_info = analyze_face(face, shape_bbox)
        
        if face_info["is_outer"]:
            compound_builder.Add(face)
            # Print additional information for each outer face
            print(f"Face Normal: {face_info['normal']}")
        
        explorer.Next()
    
    outer_skin = compound_builder.Compound()
    
    # Check if the extracted shape is watertight
    if not is_shape_watertight(outer_skin):
        raise RuntimeError("Extracted outer skin is not watertight.")
    
    return outer_skin

def main(input_step_file, output_step_file):
    """
    Main function to process a STEP file, extract the outer skin of the shape, and save the result to a new STEP file.
    
    Parameters:
    - input_step_file (str): The path to the input STEP file containing the 3D model.
    - output_step_file (str): The path to the output STEP file where the extracted outer skin will be saved.
    """
    # Load the shape from the STEP file specified by input_step_file
    shape = load_step_file(input_step_file)
    
    # Extract the outer skin from the loaded shape
    outer_skin = extract_outer_skin(shape)
    
    # Save the extracted outer skin to a new STEP file specified by output_step_file
    save_outer_skin_to_step(outer_skin, output_step_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_outer_skin_from_step.py <input_step_file> <output_step_file>")
        sys.exit(1)
    
    input_step_file = sys.argv[1]
    output_step_file = sys.argv[2]

    """
    ensure terminal in correct directory: cd "C:\Users\s1834431\OneDrive - University of Edinburgh\PhD\Codes"
    enter into terminal: {script file name}.py {step file name}.stp {new step file name}.stp
    
    """
    
    main(input_step_file, output_step_file)