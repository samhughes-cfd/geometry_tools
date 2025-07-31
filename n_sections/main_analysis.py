# n_sections/main_analysis.py

import sys
from pathlib import Path
from datetime import datetime  # ⏱️ Added for timestamping

# ───── Add project root to sys.path BEFORE any project imports ─────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # This points to geometry_tools
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───── Now do the rest of your imports ─────
import csv
import logging
import numpy as np

from analysis.process_section_analysis import ProcessSectionAnalysis


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

    # ───── Load blade station metadata ─────
    stations_csv = BLADE_DIR / "blade_stations.csv"
    stations = []

    with open(stations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_over_R = float(row["r/R [-]"])
            Cx = float(row["Cx [mm]"]) / 1000
            Cy = float(row["Cy [mm]"]) / 1000
            r = float(row["Cz [mm]"]) / 1000
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
    N = 3                         # Number of mesh refinement levels
    h0 = 40.0                     # Base element area (coarsest) in mm²
    hs = h0 / (4 ** np.arange(N))  # e.g., [40, 10, 2.5, ...]
    logging.info("Mesh size targets: %s", hs)

    # ───── Process each blade station ─────
    for dxf, label, r, r_over_R, B_r, Cx, Cy in stations:
        material = material_dict.get(dxf.name.lower())
        if material is None:
            logging.error(f"No material found for DXF file: {dxf.name}")
            continue

        section = ProcessSectionAnalysis(
            dxf=dxf,
            label=label,
            r=r,
            r_over_R=r_over_R,
            B_r=B_r,
            Cx=Cx,
            Cy=Cy,
            material=material,  # <-- Now includes "name" key
            hs=hs,
            results_dir=RESULTS,
            logs_dir=LOGS,
        )
        section.run()

    logging.info("✅ SECTION ANALYSIS pipeline complete — results saved in: %s", RESULTS)


# ───── Entry Point ─────
if __name__ == "__main__":
    main()