# section_calc_n\main.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import csv
import numpy as np

from section_calc_n.process_section import ProcessSection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ───── Setup directories ─────
BASE_DIR   = Path("section_calc_n")
BLADE_DIR  = BASE_DIR / "blade"
RESULTS    = BASE_DIR / "results"
LOGS       = BASE_DIR / "logs"

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

# ───── Load blade station metadata ─────
stations_csv = BLADE_DIR / "blade_stations.csv"
stations = []

with open(stations_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        r_over_R = float(row["r/R [-]"])
        Cx       = float(row["Cx [mm]"]) / 1000
        Cy       = float(row["Cy [mm]"]) / 1000
        r        = float(row["Cz [mm]"]) / 1000
        B_r      = float(row["B [deg]"])
        filename = row["filename"]
        dxf      = BLADE_DIR / filename
        label = Path(filename).stem
        stations.append((dxf, label, r, r_over_R, B_r, Cx, Cy))

# ───── Mesh convergence parameters ─────
N  = 3
h0 = 3.0  # target element area [mm²]
hs = h0 / (4 ** np.arange(N))
logging.info("Mesh size targets: %s", hs)

# ───── Process each blade station ─────
for dxf, label, r, r_over_R, B_r, Cx, Cy in stations:
    section = ProcessSection(
        dxf=dxf,
        label=label,
        r=r,
        r_over_R=r_over_R,
        B_r=B_r,
        Cx=Cx,
        Cy=Cy,
        hs=hs,
        results_dir=RESULTS,
        logs_dir=LOGS
    )
    section.run()

logging.info("✅ Pipeline complete — results saved in: %s", RESULTS)