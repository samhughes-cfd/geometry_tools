# section_calc_n/main_optimise.py

import sys
from pathlib import Path
import logging
import csv
import numpy as np
import pandas as pd

# ───── Path setup ─────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from section_calc_n.process_section_optimise import ProcessSectionOptimise

BASE_DIR  = ROOT / "section_calc_n"
BLADE_DIR = BASE_DIR / "blade"
LIMIT_DIR = BASE_DIR / "blade_optimisation_limit"
RESULTS   = BASE_DIR / "results"
LOGS      = BASE_DIR / "logs"

# ───── Directory structure ─────
for d in (RESULTS / "optimisation", LOGS / "optimisation"):
    d.mkdir(parents=True, exist_ok=True)

# ───── Logging ─────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "Optimisation" / "main_optimisation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.info("Starting section optimisation pipeline")

# ───── Load station metadata ─────
try:
    stations_csv = BLADE_DIR / "blade_stations.csv"
    stations = []
    logging.debug("Reading station metadata from: %s", stations_csv)
    with open(stations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append(row)
    logging.debug("Loaded %d station rows", len(stations))
except Exception as e:
    logging.exception("Failed to load station metadata: %s", e)
    sys.exit(1)

# ───── Load optimisation limits ─────
try:
    limits_csv = LIMIT_DIR / "target_section_properties.csv"
    logging.debug("Reading optimisation limits from: %s", limits_csv)
    limits_df = pd.read_csv(limits_csv).set_index("filename")
    logging.debug("Loaded optimisation limits for %d sections", len(limits_df))
except Exception as e:
    logging.exception("Failed to load optimisation limits: %s", e)
    sys.exit(1)

# ───── Mesh size (coarse only) ─────
hs = np.array([3.0])
logging.info("Using single coarse mesh size: %s", hs)

# ───── Void scale factor sweep (exponential) ─────
n_steps = 10
scale_factors = np.geomspace(0.6, 0.995, num=n_steps)
logging.info("Void scale factor sweep: exponential from 0.1 to 0.99 → %d values", len(scale_factors))


# ───── Process each station ─────
for row in stations:
    try:
        filename = row["filename"]
        label    = Path(filename).stem
        dxf_path = BLADE_DIR / filename
        r_over_R = float(row["r/R [-]"])
        r        = float(row["Cz [mm]"]) / 1000
        Cx       = float(row["Cx [mm]"]) / 1000
        Cy       = float(row["Cy [mm]"]) / 1000
        B_r      = float(row["B [deg]"])

        logging.info("Processing station: %s", label)

        if filename not in limits_df.index:
            logging.warning("⏭️  Skipping %s: no optimisation limits defined", filename)
            continue

        target_Jt = limits_df.loc[filename, "Jt [mm⁴]"]
        target_Iz = limits_df.loc[filename, "Iz [mm⁴]"]
        logging.info("Target properties for %s → Jt ≥ %.2f, Iz ≥ %.2f", label, target_Jt, target_Iz)

        section = ProcessSectionOptimise(
            dxf=dxf_path,
            label=label,
            r=r,
            r_over_R=r_over_R,
            B_r=B_r,
            Cx=Cx,
            Cy=Cy,
            h=hs[0],
            results_dir=RESULTS,
            logs_dir=LOGS,
            scale_factors=scale_factors,
            target_Jt_mm4=target_Jt,
            target_Iz_mm4=target_Iz,
        )

        result_set = section.run()

    except Exception as e:
        logging.exception("Failed to process station %s: %s", label, str(e))

logging.info("Section optimisation process complete — results in: %s", RESULTS / "optimisation")