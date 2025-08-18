# n_sections/main_analysis.py

import sys
from pathlib import Path
from datetime import datetime

# ───── Add project root to sys.path BEFORE any project imports ─────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # One level above 'n_sections'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───── Now do the rest of your imports ─────
import csv
import logging
import numpy as np
from typing import Literal

from n_sections.analysis.process_section_analysis import ProcessSectionAnalysis


# ───── Main function to run the analysis pipeline ─────
def main():
    # ───── Setup directories ─────
    BASE_DIR = PROJECT_ROOT / "n_sections"
    BLADE_DIR = BASE_DIR / "blade"

    # ⏱️ Timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Timestamped subdirectories
    RESULTS = BASE_DIR / "results/analysis" / timestamp
    LOGS = BASE_DIR / "logs/analysis" / timestamp
    for d in (RESULTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    # ───── Logging setup ─────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "main.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("🌀 DXF mesh-convergence pipeline started")
    logging.info("⏱ Timestamped output directory: %s", timestamp)

    # ───── Origin placement settings (choose here) ─────
    ORIGIN_MODE: Literal["cop", "centroid"] = "cop"  # "cop" (chord fraction) or "centroid" (uses Cx,Cy)
    COP_FRACTION: float = 0.33                       # used only if ORIGIN_MODE == "cop" (0.0–1.0)
    if ORIGIN_MODE == "cop" and not (0.0 <= COP_FRACTION <= 1.0):
        raise ValueError(f"COP_FRACTION must be in [0, 1]; got {COP_FRACTION}")
    logging.info(
        "Origin mode: %s%s",
        ORIGIN_MODE,
        f" (fraction={COP_FRACTION:.2f})" if ORIGIN_MODE == "cop" else "",
    )

    # ───── Load blade station metadata ─────
    stations_csv = BLADE_DIR / "blade_stations.csv"
    stations = []
    with open(stations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_over_R = float(row["r/R [-]"])
            Cx = float(row["Cx [mm]"]) / 1000.0
            Cy = float(row["Cy [mm]"]) / 1000.0
            r = float(row["Cz [mm]"]) / 1000.0
            B_r = float(row["B [deg]"])
            filename = row["filename"]
            dxf = BLADE_DIR / filename
            label = Path(filename).stem
            stations.append((dxf, label, r, r_over_R, B_r, Cx, Cy))

    # ───── Load material definitions ─────
    materials_csv = BLADE_DIR / "materials.csv"
    material_dict = {}
    with open(materials_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].lower()
            material_dict[filename] = {
                "name": row["material_name"],
                "E": float(row["elastic_modulus"]),
                "nu": float(row["poissons_ratio"]),
                "fy": float(row["yield_strength"]),
                "rho": float(row["density"]),
                "color": row["color"],
            }

    # ───── Mesh convergence parameters ─────
    N = 3                           # Number of mesh refinement levels
    h0 = 40.0                       # Base element "size" (your convention)
    hs = h0 / (4 ** np.arange(N))   # e.g., [40, 10, 2.5]
    logging.info("Mesh size targets: %s", hs)

    # ───── Process each blade station ─────
    for dxf, label, r, r_over_R, B_r, Cx, Cy in stations:
        material = material_dict.get(dxf.name.lower())
        if material is None:
            logging.error("No material found for DXF file: %s", dxf.name)
            continue

        logging.info("— Station %s: DXF=%s, twist=%.2f°, r/R=%.3f —", label, dxf.name, B_r, r_over_R)

        section = ProcessSectionAnalysis(
            dxf=dxf,
            label=label,
            r=r,
            r_over_R=r_over_R,
            B_r=B_r,
            Cx=Cx,
            Cy=Cy,
            material=material,     
            hs=hs,
            results_dir=RESULTS,
            logs_dir=LOGS,
            # Pass origin choice through to ProcessedGeometry via ProcessSectionAnalysis
            origin_mode=ORIGIN_MODE,
            cop_fraction=COP_FRACTION,
            # Optional: tweak import/processing knobs if needed:
            # spline_delta=0.05, degrees_per_segment=0.5, exterior_nodes=400,
        )
        section.run()

    logging.info("✅ SECTION ANALYSIS pipeline complete — results saved in: %s", RESULTS)


# ───── Entry Point ─────
if __name__ == "__main__":
    main()