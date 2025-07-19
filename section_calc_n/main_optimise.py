# section_calc_n\main_optimise.py

import sys
from pathlib import Path
import logging
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───── Path setup ─────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from section_calc_n.process_section import ProcessSection

BASE_DIR  = ROOT / "section_calc_n"
BLADE_DIR = BASE_DIR / "blade"
LIMIT_DIR = BASE_DIR / "blade_optimisation_limit"
RESULTS   = BASE_DIR / "results"
LOGS      = BASE_DIR / "logs"

for d in (RESULTS / "Optimisation", LOGS / "Optimisation"):
    d.mkdir(parents=True, exist_ok=True)

# ───── Logging ─────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "Optimisation" / "main_optimisation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.info("🛠️ Starting section optimisation pipeline")

# ───── Matplotlib Interactive Mode ─────
plt.ion()

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
    logging.exception("❌ Failed to load station metadata: %s", e)
    sys.exit(1)

# ───── Load optimisation limits ─────
try:
    limits_csv = LIMIT_DIR / "target_section_properties.csv"
    logging.debug("Reading optimisation limits from: %s", limits_csv)
    limits_df = pd.read_csv(limits_csv)
    limits_df = limits_df.set_index("filename")
    logging.debug("Loaded optimisation limits for %d sections", len(limits_df))
except Exception as e:
    logging.exception("❌ Failed to load optimisation limits: %s", e)
    sys.exit(1)

# ───── Mesh size setup (coarse only) ─────
hs = np.array([3.0])
logging.info("Using single coarse mesh size: %s", hs)

# ───── Void scaling setup ─────
scale_factors = np.arange(0.1, 0.99, 0.025)
logging.debug("Defined void scaling factors: %s", scale_factors)

# ───── Process each blade station ─────
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

        logging.debug("Processing station: %s", label)
        logging.debug("  ↳ r/R: %.3f, r: %.3f m, Cx: %.3f m, Cy: %.3f m, B: %.2f°", r_over_R, r, Cx, Cy, B_r)

        if filename not in limits_df.index:
            logging.warning("⏭️  Skipping %s: no optimisation limits defined", filename)
            continue

        target_Jt = limits_df.loc[filename, "Jt [mm⁴]"]
        target_Iz = limits_df.loc[filename, "Iz [mm⁴]"]
        logging.info("⚙️  Optimising %s → Target Jt ≥ %.2f, Iz ≥ %.2f", label, target_Jt, target_Iz)

        areas, jts, izs, scales = [], [], [], []
        found_solution = False

        # Live plot setup
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title(f"{label} – Optimisation Progress")
        fig.tight_layout()
        fig.show()

        for scale in scale_factors:
            scaled_label = f"{label}_s{scale:.3f}"
            logging.debug("  🔁 Testing scale factor: %.3f", scale)

            try:
                section = ProcessSection(
                    dxf=dxf_path,
                    label=scaled_label,
                    r=r,
                    r_over_R=r_over_R,
                    B_r=B_r,
                    Cx=Cx,
                    Cy=Cy,
                    hs=hs,
                    results_dir=RESULTS,
                    logs_dir=LOGS,
                    mode="Optimisation",
                    scale_factor=scale,
                )
                section.run()

                csv_path = RESULTS / "Optimisation" / scaled_label / "section_properties.csv"
                if not csv_path.exists():
                    logging.warning("❌ Missing section properties file: %s", csv_path)
                    continue

                df = pd.read_csv(csv_path)
                A  = df.loc[0, "Area [mm^2]"]
                Jt = df.loc[0, "Jt [mm^4]"]
                Iz = df.loc[0, "Iz [mm^4]"]

                areas.append(A)
                jts.append(Jt)
                izs.append(Iz)
                scales.append(scale)

                # Live plot update
                ax.clear()
                ax.plot(areas, jts, marker='o', linestyle='-', color="tab:blue", label="Jt [mm⁴]")
                ax.plot(areas, izs, marker='s', linestyle='--', color="tab:orange", label="Iz [mm⁴]")
                ax.axhline(target_Jt, color="tab:blue", linestyle=":", label="Target Jt")
                ax.axhline(target_Iz, color="tab:orange", linestyle=":", label="Target Iz")
                ax.set_xlabel("Area [mm²]")
                ax.set_ylabel("Property Value [mm⁴]")
                ax.set_title(f"{label} – Property vs Area")
                ax.grid(True)
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.05)

                if Jt >= target_Jt and Iz >= target_Iz:
                    logging.info("✅ %s passed thresholds: Jt=%.2f, Iz=%.2f", scaled_label, Jt, Iz)
                    found_solution = True
                    break
                else:
                    logging.info("❌ %s failed threshold: Jt=%.2f, Iz=%.2f", scaled_label, Jt, Iz)

            except Exception as e:
                logging.exception("💥 Exception during optimisation of %s (scale %.3f): %s", label, scale, str(e))

        try:
            fig_path = RESULTS / "Optimisation" / f"{label}_property_vs_area.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            logging.info("📈 Saved optimisation plot for %s → %s", label, fig_path)
        except Exception as e:
            logging.exception("❌ Failed to save plot for %s: %s", label, str(e))

        if not found_solution:
            logging.warning("⚠️  No valid solution found for %s within tested scale range", label)

    except Exception as e:
        logging.exception("💥 Exception while processing station %s: %s", label, str(e))

logging.info("🏁 Section optimisation process complete — results stored in: %s", RESULTS / "Optimisation")