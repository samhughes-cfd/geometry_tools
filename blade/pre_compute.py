# section_calc_n\blade\pre_compute.py

from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d

# ───── paths ─────
BASE_DIR = Path(__file__).resolve().parent
CENTROID_CSV = BASE_DIR / "centroid_profile.csv"
TWIST_CSV = BASE_DIR / "chord_and_twist_profile.csv"
OUTPUT_CSV = BASE_DIR / "blade_stations.csv"

# ───── load and clean centroid data ─────
centroid_df = pd.read_csv(CENTROID_CSV, skiprows=1)
centroid_df.columns = ["r/R", "Cx [mm]", "Cy [mm]", "Cz [mm]", "filename"]
centroid_df = centroid_df.dropna(subset=["filename"])
centroid_df["r/R"] = centroid_df["r/R"].astype(float)

# ───── load and interpolate twist distribution ─────
twist_df = pd.read_csv(TWIST_CSV, skiprows=1)
twist_df.columns = ["r/R", "B [deg]", "Chord [m]"]
twist_df["r/R"] = twist_df["r/R"].astype(float)
twist_df["B [deg]"] = twist_df["B [deg]"].astype(float)

twist_interp = interp1d(
    twist_df["r/R"],
    twist_df["B [deg]"],
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate"
)

# ───── apply interpolation ─────
centroid_df["B [deg]"] = twist_interp(centroid_df["r/R"])

# ───── reorder and save ─────
output_df = centroid_df[["r/R", "Cx [mm]", "Cy [mm]", "Cz [mm]", "B [deg]", "filename"]]
output_df.columns = ["r/R [-]", "Cx [mm]", "Cy [mm]", "Cz [mm]", "B [deg]", "filename"]
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"✔️ blade_stations.csv written to {OUTPUT_CSV}")