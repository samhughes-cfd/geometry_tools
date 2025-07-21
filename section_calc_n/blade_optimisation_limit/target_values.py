import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Setup paths ---
base_path = "section_calc_n"
blade_dir = os.path.join(base_path, "blade")
limits_dir = os.path.join(base_path, "blade_optimisation_limit")

ei_poly_csv = os.path.join(limits_dir, "EI_polynomials.csv")
gj_poly_csv = os.path.join(limits_dir, "GJ_polynomials.csv")
stations_csv = os.path.join(blade_dir, "blade_stations.csv")
target_csv = os.path.join(limits_dir, "target_section_properties.csv")
ei_png = os.path.join(limits_dir, "EI_envelope.png")
gj_png = os.path.join(limits_dir, "GJ_envelope.png")

# --- Step 1: Load polynomial coefficients ---
ei_df = pd.read_csv(ei_poly_csv).iloc[:, 0:6]
gj_df = pd.read_csv(gj_poly_csv).iloc[:, 0:6]

ei_coeffs = ei_df.to_numpy()  # shape: [n_polys × 6]
gj_coeffs = gj_df.to_numpy()

# --- Step 2: Evaluation points ---
n_points = 500
r_by_R_full = np.linspace(0, 1, n_points)
x_m = r_by_R_full * 0.8  # span from 0 to 0.8 m

def evaluate_rowwise_polynomials(coeff_array, x_vals):
    n_polys = coeff_array.shape[0]
    y_vals = np.zeros((len(x_vals), n_polys))
    for i in range(n_polys):
        coeffs = coeff_array[i, ::-1]  # b5 to b0
        y_vals[:, i] = np.polyval(coeffs, x_vals)
    return y_vals

# --- Step 3: Evaluate all polynomials ---
ei_curves = evaluate_rowwise_polynomials(ei_coeffs, r_by_R_full)
gj_curves = evaluate_rowwise_polynomials(gj_coeffs, r_by_R_full)

# --- Step 4: Compute worst-case envelopes ---
ei_worst = np.max(np.abs(ei_curves), axis=1)
gj_worst = np.max(np.abs(gj_curves), axis=1)

# --- Step 5: Plot and save EI ---
plt.figure(figsize=(12, 6))
for i in range(ei_curves.shape[1]):
    plt.plot(x_m, ei_curves[:, i], color='gray', alpha=0.05)
plt.plot(x_m, ei_worst, color='red', linewidth=2.5, label="Worst-case envelope")
plt.title("EI Curves and Worst-case Envelope")
plt.xlabel("x [m]")
plt.ylabel("EI [N·mm²]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ei_png, dpi=300)
plt.close()

# --- Step 6: Plot and save GJ ---
plt.figure(figsize=(12, 6))
for i in range(gj_curves.shape[1]):
    plt.plot(x_m, gj_curves[:, i], color='gray', alpha=0.05)
plt.plot(x_m, gj_worst, color='blue', linewidth=2.5, label="Worst-case envelope")
plt.title("GJ Curves and Worst-case Envelope")
plt.xlabel("x [m]")
plt.ylabel("GJ [N·mm²]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(gj_png, dpi=300)
plt.close()

# --- Step 7: Interpolate at station locations ---
stations_df = pd.read_csv(stations_csv)
r_by_R_target = stations_df["r/R [-]"].to_numpy()

# Tidal benchmark blade 7075-T6 Aluminum
E = 71700 # MPa
G = 26900 # MPa
Iz_vals = np.interp(r_by_R_target, r_by_R_full, ei_worst) / E
Jt_vals = np.interp(r_by_R_target, r_by_R_full, gj_worst) / G

# --- Step 8: Save target section properties ---
target_df = pd.DataFrame({
    "r/R [-]": r_by_R_target,
    "Jt [mm⁴]": Jt_vals,
    "Iz [mm⁴]": Iz_vals,
    "filename": stations_df["filename"]
})

os.makedirs(limits_dir, exist_ok=True)
target_df.to_csv(target_csv, index=False)
print(f"✅ Worst-case targets saved to: {target_csv}")
print(f"🖼️ EI plot saved to: {ei_png}")
print(f"🖼️ GJ plot saved to: {gj_png}")