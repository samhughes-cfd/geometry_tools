# n_sections/main_optimise.py

import sys
from pathlib import Path
from datetime import datetime  # ⬅ for timestamping

# ───── Add project root to sys.path BEFORE any project imports ─────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]  # → geometry_tools
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───── Now do the rest of your imports ─────
import csv
import logging
import numpy as np
import pandas as pd

from optimisation.process_section_optimise import ProcessSectionOptimise


# ───── Main function to run the optimisation pipeline ─────
def main():
    # ───── Setup directories ─────
    BASE_DIR = PROJECT_ROOT / "n_sections"
    BLADE_DIR = BASE_DIR / "blade"
    LIMIT_DIR = BASE_DIR / "blade_optimisation_limits"  # plural for consistency
    RESULTS = BASE_DIR / "results"
    LOGS = BASE_DIR / "logs"

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = RESULTS / "optimisation" / timestamp
    LOGS_DIR = LOGS / "optimisation" / timestamp

    for d in (RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ───── Logging setup ─────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "main_optimisation.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("🚀 Starting section optimisation pipeline")
    logging.info("🗂 Results stored in: %s", RESULTS_DIR)

    # ───── Load blade station metadata ─────
    stations_csv = BLADE_DIR / "blade_stations.csv"
    try:
        stations = []
        logging.debug("Reading station metadata from: %s", stations_csv)
        with open(stations_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stations.append(row)
        logging.debug("Loaded %d station rows", len(stations))
    except Exception as e:
        logging.exception("❌ Failed to load station metadata: %s", e)
        sys.exit(1)

    # ───── Load optimisation limits ─────
    limits_csv = LIMIT_DIR / "target_section_properties.csv"
    try:
        logging.debug("Reading optimisation limits from: %s", limits_csv)
        limits_df = pd.read_csv(limits_csv).set_index("filename")
        logging.debug("Loaded optimisation limits for %d sections", len(limits_df))
    except Exception as e:
        logging.exception("❌ Failed to load optimisation limits: %s", e)
        sys.exit(1)

    # ───── Mesh size (coarse only) ─────
    hs = np.array([3.0])
    logging.info("Using single coarse mesh size: %s", hs)

    # ───── Void scale factor sweep ─────
    n_steps = 10
    scale_factors = np.geomspace(0.6, 0.995, num=n_steps)
    logging.info("Void scale factor sweep: exponential from 0.6 to 0.995 → %d values", len(scale_factors))

    # ───── Process each station ─────
    all_results = []
    for row in stations:
        try:
            filename = row["filename"]
            label = Path(filename).stem
            dxf_path = BLADE_DIR / filename
            r_over_R = float(row["r/R [-]"])
            r = float(row["Cz [mm]"]) / 1000
            Cx = float(row["Cx [mm]"]) / 1000
            Cy = float(row["Cy [mm]"]) / 1000
            B_r = float(row["B [deg]"])

            logging.info("⚙ Processing station: %s", label)

            if filename not in limits_df.index:
                logging.warning("⏭️  Skipping %s: no optimisation limits defined", filename)
                continue

            target_Jt = limits_df.loc[filename, "Jt [mm⁴]"]
            target_Iz = limits_df.loc[filename, "Iz [mm⁴]"]
            logging.info("🎯 Target for %s → Jt ≥ %.2f, Iz ≥ %.2f", label, target_Jt, target_Iz)

            section = ProcessSectionOptimise(
                dxf=dxf_path,
                label=label,
                r=r,
                r_over_R=r_over_R,
                B_r=B_r,
                Cx=Cx,
                Cy=Cy,
                h=hs[0],
                results_dir=RESULTS_DIR,  # ⬅ store results in timestamped folder
                logs_dir=LOGS_DIR,
                scale_factors=scale_factors,
                target_Jt_mm4=target_Jt,
                target_Iz_mm4=target_Iz,
            )

            result_set = section.run()
            all_results.append(result_set)

        except Exception as e:
            logging.exception("❌ Failed to process station %s: %s", label, str(e))

    # ───── Save summary across all stations ─────
    if all_results:
        try:
            df = pd.DataFrame(all_results)
            df.to_csv(RESULTS_DIR / "summary.csv", index=False)
            logging.info("📊 Saved summary of results to: %s", RESULTS_DIR / "summary.csv")
        except Exception as e:
            logging.exception("❌ Failed to save results summary: %s", e)

    logging.info("✅ Section optimisation process complete — results in: %s", RESULTS_DIR)


# ───── Entry Point ─────
if __name__ == "__main__":
    main()