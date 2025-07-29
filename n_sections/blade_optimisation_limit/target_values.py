import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Setup paths ---
base_path = "section_calc_n"
blade_dir = os.path.join(base_path, "blade")
limits_dir = os.path.join(base_path, "blade_optimisation_limit")

stations_csv = os.path.join(blade_dir, "blade_stations.csv")
target_csv = os.path.join(limits_dir, "target_section_properties.csv")
plot_path = os.path.join(limits_dir, "EI_GJ_curves.png")

# --- Step 1: Define blade span length and polynomials over x [m] ---
L = 0.8  # Blade span in meters

# Polynomial coefficients in descending powers of x
ei_coeffs = np.array([-148.0, 3475.0, -9146.0, 9974.0, -5167.0, 1059.0])
gj_coeffs = np.array([-970.9, 2387.0, -2314.0, 1156.0, -329.5, 48.33])

# --- Step 2: Evaluate full x-domain curves ---
n_points = 500
x_vals = np.linspace(0, L, n_points)
ei_vals = np.polyval(ei_coeffs, x_vals)
gj_vals = np.polyval(gj_coeffs, x_vals)
ei_scaled = 1.5 * ei_vals
gj_scaled = 1.5 * gj_vals

# --- Step 3: Load station locations in r/R and convert to x [m] ---
stations_df = pd.read_csv(stations_csv)
r_by_R_target = stations_df["r/R [-]"].to_numpy()
x_target = r_by_R_target * L

# Material properties
E = 71700  # MPa
G = 26900  # MPa

# Interpolate EI(x) and GJ(x) at station positions
ei_interp = np.interp(x_target, x_vals, ei_vals)
gj_interp = np.interp(x_target, x_vals, gj_vals)

Iz_vals = ei_interp / E
Jt_vals = gj_interp / G

# --- Step 4: Save target section properties ---
target_df = pd.DataFrame({
    "r/R [-]": r_by_R_target,
    "Jt [mm⁴]": Jt_vals,
    "Iz [mm⁴]": Iz_vals,
    "filename": stations_df["filename"]
})

os.makedirs(limits_dir, exist_ok=True)
target_df.to_csv(target_csv, index=False)
print(f"✅ Target section properties saved to: {target_csv}")

# --- Step 5: Plot original and scaled curves ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# EI subplot
ax1.plot(x_vals, ei_vals, label="EI", color="tab:red")
ax1.plot(x_vals, ei_scaled, label="EI × 1.5", linestyle="--", color="tab:red", alpha=0.7)
ax1.set_ylabel("EI [N·mm²]")
ax1.grid(True)
ax1.legend()
ax1.set_title("Analytical EI and GJ Curves (Original and Scaled)")

# GJ subplot
ax2.plot(x_vals, gj_vals, label="GJ", color="tab:blue")
ax2.plot(x_vals, gj_scaled, label="GJ × 1.5", linestyle="--", color="tab:blue", alpha=0.7)
ax2.set_ylabel("GJ [N·mm²]")
ax2.set_xlabel("x [m]")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"🖼️ EI and GJ curves saved to: {plot_path}")