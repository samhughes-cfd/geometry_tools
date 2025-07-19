import os
import pandas as pd
import numpy as np

# --- Setup paths ---
base_path = "section_calc_n"
blade_dir = os.path.join(base_path, "blade")
limits_dir = os.path.join(base_path, "blade_optimisation_limits")

stations_csv = os.path.join(blade_dir, "blade_stations.csv")
ei_poly_csv = os.path.join(limits_dir, "EI_polynomials.csv")
gj_poly_csv = os.path.join(limits_dir, "GJ_polynomials.csv")
target_csv = os.path.join(limits_dir, "target_section_properties.csv")

# --- Step 1: Read blade stations ---
stations_df = pd.read_csv(stations_csv)

# --- Step 2: Load polynomial coefficients (b0 to b5) ---
ei_coeffs = pd.read_csv(ei_poly_csv).iloc[0, :-1].to_numpy()[::-1]  # Reverse for np.polyval
gj_coeffs = pd.read_csv(gj_poly_csv).iloc[0, :-1].to_numpy()[::-1]

# --- Step 3: Evaluate polynomials at r/R ---
r_by_R = stations_df["r/R [-]"].to_numpy()
Iz_vals = np.polyval(ei_coeffs, r_by_R)
Jt_vals = np.polyval(gj_coeffs, r_by_R)

# --- Step 4: Construct output DataFrame ---
target_df = pd.DataFrame({
    "r/R [-]": r_by_R,
    "Jt [mm⁴]": Jt_vals,
    "Iz [mm⁴]": Iz_vals,
    "filename": stations_df["filename"]
})

# --- Step 5: Save result ---
os.makedirs(limits_dir, exist_ok=True)
target_df.to_csv(target_csv, index=False)

print(f"✅ File saved: {target_csv}")