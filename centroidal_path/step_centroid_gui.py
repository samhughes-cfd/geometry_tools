from __future__ import annotations

from pathlib import Path
import csv
import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import sys  # Added for system exit
import numpy as np

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='centroid_seeder.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Silence noisy deprecation spam from OCCT wrappers
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenCASCADE imports (with fallbacks)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Bnd import Bnd_Box

try:
    from OCC.Core.BRepBndLib import Add as _occ_add_bnd
except ImportError:
    from OCC.Core.BRepBndLib import brepbndlib_Add as _occ_add_bnd

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.BOPAlgo import BOPAlgo_Section
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.ShapeFix import ShapeFix_Wire
from OCC.Core.ShapeExtend import ShapeExtend_OK
from OCC.Core.Geom import Geom_Plane
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

from OCC.Display.SimpleGui import init_display

# Import ttkbootstrap with proper error handling
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError as e:
    logger.error(f"ttkbootstrap import failed: {e}")
    messagebox.showerror("Import Error", "ttkbootstrap package is required. Please install with: pip install ttkbootstrap")
    sys.exit(1)


def _bounding_box(shape):
    bb = Bnd_Box()
    _occ_add_bnd(shape, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    logger.debug(f"Bounding box: X({xmin:.3f},{xmax:.3f}) Y({ymin:.3f},{ymax:.3f}) Z({zmin:.3f},{zmax:.3f})")
    return xmin, ymin, zmin, xmax, ymax, zmax


def _first_solid(shape):
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    first_solid = None
    while explorer.More():
        solid_count += 1
        if first_solid is None:
            first_solid = explorer.Current()
        explorer.Next()
    
    logger.debug(f"Found {solid_count} solids in STEP file")
    return first_solid if first_solid else shape


def section_wires(solid, axis, coord, tol=1e-6):
    logger.debug(f"Creating section at {axis}={coord:.3f} with tolerance={tol}")
    
    # Define plane based on axis
    if axis == 0:  # X-axis
        plane = gp_Pln(gp_Pnt(coord, 0, 0), gp_Dir(1, 0, 0))
    elif axis == 1:  # Y-axis
        plane = gp_Pln(gp_Pnt(0, coord, 0), gp_Dir(0, 1, 0))
    else:  # Z-axis
        plane = gp_Pln(gp_Pnt(0, 0, coord), gp_Dir(0, 0, 1))
    
    face = BRepBuilderAPI_MakeFace(plane, -1e5, 1e5, -1e5, 1e5).Shape()
    
    sect = BOPAlgo_Section()
    sect.AddArgument(solid)
    sect.AddArgument(face)
    sect.SetFuzzyValue(tol)
    sect.SetRunParallel(True)
    logger.debug("Performing section operation...")
    
    # Handle different API versions of OpenCASCADE
    if hasattr(sect, 'Build'):
        sect.Build()  # Older versions (pre-7.5)
    elif hasattr(sect, 'Perform'):
        sect.Perform()  # Newer versions (7.5+)
    else:
        logger.error("Unsupported BOPAlgo_Section API")
        return []
    
    if sect.HasErrors():
        logger.error(f"Section operation failed at {axis}={coord:.3f} with errors")
        return []
    
    edges = sect.Shape()
    fb = ShapeAnalysis_FreeBounds(edges, tol, True, True)
    wires = [w for w in fb.GetClosedWires().Iterable()]
    logger.debug(f"Found {len(wires)} wire(s) in section at {axis}={coord:.3f}")
    
    fixed = []
    for i, w in enumerate(wires):
        fw = ShapeFix_Wire(w)
        fw.Perform()
        if fw.Status(ShapeExtend_OK):
            fixed.append(fw.Wire())
            logger.debug(f"Fixed wire #{i+1} at {axis}={coord:.3f}")
        else:
            logger.warning(f"Failed to fix wire #{i+1} at {axis}={coord:.3f}")
    
    return fixed


def _centroid_at_coord(solid, axis, coord):
    logger.info(f"Computing centroid at axis={axis} coord={coord:.3f}")
    wires = section_wires(solid, axis, coord)
    
    if not wires:
        logger.warning(f"No wires found at axis={axis} coord={coord:.3f}")
        return None
        
    centroids = []
    logger.debug(f"Processing {len(wires)} wire(s) at axis={axis} coord={coord:.3f}")
    
    # Define plane for face creation based on axis
    if axis == 0:  # X-axis
        plane = gp_Pln(gp_Pnt(coord, 0, 0), gp_Dir(1, 0, 0))
    elif axis == 1:  # Y-axis
        plane = gp_Pln(gp_Pnt(0, coord, 0), gp_Dir(0, 1, 0))
    else:  # Z-axis
        plane = gp_Pln(gp_Pnt(0, 0, coord), gp_Dir(0, 0, 1))
    
    for i, wire in enumerate(wires):
        try:
            logger.debug(f"Processing wire #{i+1}")
            face = BRepBuilderAPI_MakeFace(plane, wire, False).Face()
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            c = props.CentreOfMass()
            centroid = np.array([c.X(), c.Y(), c.Z()])
            centroids.append(centroid)
            logger.debug(f"Wire #{i+1} centroid: {centroid}")
        except Exception as e:
            logger.error(f"Error processing wire #{i+1}: {str(e)}")
            continue
    
    if not centroids:
        logger.error(f"No valid centroids at axis={axis} coord={coord:.3f}")
        return None
    
    mean_centroid = np.mean(centroids, axis=0).tolist()
    logger.info(f"Mean centroid at axis={axis} coord={coord:.3f}: {mean_centroid}")
    return mean_centroid


def _determine_longest_axis(shape):
    """Determine the longest axis of the bounding box for slender structures"""
    solid = _first_solid(shape)
    xmin, ymin, zmin, xmax, ymax, zmax = _bounding_box(solid)
    
    # Calculate spans for each dimension
    x_span = xmax - xmin
    y_span = ymax - ymin
    z_span = zmax - zmin
    
    spans = [x_span, y_span, z_span]
    longest_axis = np.argmax(spans)
    
    logger.info(f"Bounding box spans: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
    logger.info(f"Selected longest axis: {'X' if longest_axis == 0 else 'Y' if longest_axis == 1 else 'Z'}")
    
    return longest_axis, xmin, xmax, ymin, ymax, zmin, zmax


def compute_centroids(shape, n_slices):
    logger.info(f"Computing centroids with {n_slices} slices")
    
    # Determine longest axis and get bounding box
    axis, xmin, xmax, ymin, ymax, zmin, zmax = _determine_longest_axis(shape)
    
    # Get min/max coordinates for the longest axis
    if axis == 0:  # X-axis
        min_coord, max_coord = xmin, xmax
    elif axis == 1:  # Y-axis
        min_coord, max_coord = ymin, ymax
    else:  # Z-axis
        min_coord, max_coord = zmin, zmax
    
    if abs(max_coord - min_coord) < 1e-6:
        logger.error("Zero span in longest dimension")
        return []
    
    xs = np.linspace(min_coord, max_coord, n_slices)
    centroids = []
    
    for i, coord in enumerate(xs):
        logger.info(f"Processing slice {i+1}/{n_slices} at coord={coord:.3f}")
        centroid = _centroid_at_coord(shape, axis, coord)
        if centroid:
            centroids.append(centroid)
            logger.debug(f"Added centroid: {centroid}")
        else:
            logger.warning(f"Skipping slice at coord={coord:.3f} - no centroid")
    
    logger.info(f"Computed {len(centroids)}/{n_slices} centroids")
    return centroids


def make_polyline(points):
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for p, q in zip(points[:-1], points[1:]):
        edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*p), gp_Pnt(*q)).Edge()
        builder.Add(comp, edge)
    return comp


class CentroidGUI(ttk.Window):  # Now inheriting from ttk.Window
    def __init__(self):
        # Initialize with superhero theme
        super().__init__(themename="superhero")
        self.title("STEP Centroid Seeder")
        self.geometry("1280x900")

        self.display, self.start_display, *extra = init_display("tk")
        if not self.display:
            logger.error("Failed to initialize 3D display")
            messagebox.showerror("Display Error", "Failed to initialize 3D display")
            sys.exit(1)

        self.step_path: Path | None = None
        self.shape = None
        self.ais_shape = None
        self.centroids: list[list[float]] = []
        self._opacity = 0.0

        self._build_layout()

    def _build_layout(self):
        bar = ttk.Frame(self, padding=5)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Import .STEP", command=self.load_step).pack(side=tk.LEFT, padx=4)

        ttk.Label(bar, text="n sections:").pack(side=tk.LEFT, padx=4)
        self.n_var = ttk.IntVar(value=10)
        ttk.Entry(bar, textvariable=self.n_var, width=6).pack(side=tk.LEFT)

        ttk.Button(bar, text="Generate Preview", command=self.generate_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        ttk.Label(bar, text="Opacity:").pack(side=tk.LEFT, padx=(20, 4))
        s = ttk.Scale(bar, from_=0, to=1, orient=tk.HORIZONTAL, command=self.set_opacity, length=150)
        s.set(0.0)
        s.pack(side=tk.LEFT)

        # Handle different display configurations
        try:
            # For embedded displays
            canvas = self.display.GetCanvas()
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except AttributeError:
            # For separate window displays
            info = ttk.Label(self, text="▶ 3-D view is opened in a separate window", bootstyle="secondary")
            info.pack(side=tk.TOP, fill=tk.X)

    def _safe_message(self, title: str, msg: str, kind: str = "error"):
        try:
            getattr(messagebox, kind)(title, msg)
        except Exception:
            print(f"{title}: {msg}")

    def load_step(self):
        file_path = filedialog.askopenfilename(filetypes=[("STEP", "*.step *.stp")])
        if not file_path:
            return
            
        self.step_path = Path(file_path)
        logger.info(f"Loading STEP file: {self.step_path}")

        reader = STEPControl_Reader()
        if reader.ReadFile(str(self.step_path)) != IFSelect_RetDone:
            error_msg = "Failed to read STEP file"
            logger.error(error_msg)
            self._safe_message("STEP Error", error_msg)
            return
            
        reader.TransferRoot()
        self.shape = reader.OneShape()
        logger.info("STEP file loaded successfully")

        self.display.EraseAll()
        self.ais_shape = self.display.DisplayShape(self.shape, transparency=self._opacity, update=True)[0]
        self.display.FitAll()

    def _sphere_radius(self) -> float:
        if self.shape is None:
            return 1.0
        
        # Calculate sphere radius based on overall bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = _bounding_box(self.shape)
        spans = [xmax - xmin, ymax - ymin, zmax - zmin]
        max_span = max(spans)
        radius = max(max_span * 0.01, 1e-3)
        logger.debug(f"Sphere radius calculated: {radius:.4f}")
        return radius

    def generate_preview(self):
        if self.shape is None:
            warning_msg = "Load a STEP file first"
            logger.warning(warning_msg)
            self._safe_message("No STEP", warning_msg, "warning")
            return
            
        n = max(2, self.n_var.get())
        logger.info(f"Generating preview with {n} sections")
        
        try:
            self.centroids = compute_centroids(self.shape, n)
        except Exception as e:
            error_msg = f"Centroid error: {str(e)}"
            logger.exception(error_msg)
            self._safe_message("Centroid error", error_msg)
            return
            
        if not self.centroids:
            warning_msg = "Could not compute centroids - check geometry orientation"
            logger.warning(warning_msg)
            self._safe_message("Centroids", warning_msg)
            return

        logger.info(f"Displaying {len(self.centroids)} centroids")
        self.display.EraseAll()
        self.ais_shape = self.display.DisplayShape(self.shape, transparency=self._opacity, update=False)[0]

        rad = self._sphere_radius()
        logger.debug(f"Using sphere radius: {rad:.4f}")
        
        for p in self.centroids:
            sphere = BRepPrimAPI_MakeSphere(gp_Pnt(*p), rad).Shape()
            self.display.DisplayShape(sphere, color="RED", update=False)
            
        polyline = make_polyline(self.centroids)
        self.display.DisplayShape(polyline, color="BLACK", line_width=2, update=True)
        self.display.FitAll()

    def set_opacity(self, val):
        self._opacity = float(val)
        logger.debug(f"Set opacity to {self._opacity}")
        if self.ais_shape is not None:
            try:
                self.display.Context.SetTransparency(self.ais_shape, self._opacity, True)
            except TypeError:
                self.display.Context.SetTransparency(self.ais_shape, self._opacity)

    def export_csv(self):
        if not self.centroids:
            warning_msg = "Generate a preview first"
            logger.warning(warning_msg)
            self._safe_message("Nothing to export", warning_msg, "warning")
            return
            
        out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_path:
            return
            
        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerows(self.centroids)
            success_msg = f"Saved {len(self.centroids)} points to {os.path.basename(out_path)}"
            logger.info(success_msg)
            self._safe_message("Exported", success_msg, kind="info")
        except OSError as e:
            error_msg = f"File error: {str(e)}"
            logger.error(error_msg)
            self._safe_message("File error", error_msg)


if __name__ == "__main__":
    try:
        app = CentroidGUI()
        app.mainloop()
    except Exception as e:
        logger.exception("Critical error in application")
        messagebox.showerror("Fatal Error", f"Application crashed: {str(e)}")