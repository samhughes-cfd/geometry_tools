import FreeCAD
import Part
import Import

# Load the DXF
doc = FreeCAD.newDocument()
Import.importDXF("Station8.dxf")

# Get all shapes and convert to wires
for obj in doc.Objects:
    if hasattr(obj, "Shape"):
        shape = obj.Shape
        wires = shape.Wires
        for wire in wires:
            Part.show(wire)

# Export cleaned DXF
Import.export(doc.Objects, "Station8_clean.dxf")
