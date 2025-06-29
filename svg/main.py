import os
import matplotlib.pyplot as plt
from raw_geometry import RawGeometry
from fixed_geometry import FixedGeometry
from meshed_geometry import parallel_mesh

# === Create output directory ===
PLOTS_DIR = "svg/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Input SVG files ===
svg_files = [
    ("svg/svgs/Station8.svg", "A"),
    ("svg/svgs/Station8_group_union.svg", "B")
]

# === RAW GEOMETRY PLOTTING ===
print("🔹 Plotting raw geometries...")
fig_raw, axs_raw = plt.subplots(2, 1, figsize=(8, 10))
fig_raw.suptitle("Raw Geometries")

raws = [RawGeometry(path, label) for path, label in svg_files]
for ax, raw in zip(axs_raw, raws):
    raw.extract()
    raw.plot(ax)

fig_raw.tight_layout()
fig_raw.savefig(os.path.join(PLOTS_DIR, "raw_geometries.png"), dpi=300)
plt.close(fig_raw)
print("✅ Saved raw_geometries.png to /plots")

# === FIXED GEOMETRY PLOTTING ===
print("🔹 Plotting fixed geometries...")
fig_fix, axs_fix = plt.subplots(2, 1, figsize=(8, 10))
fig_fix.suptitle("Fixed Geometries")

fixeds = [FixedGeometry(path, label) for path, label in svg_files]
for ax, fix in zip(axs_fix, fixeds):
    fix.build()
    fix.plot(ax)

fig_fix.tight_layout()
fig_fix.savefig(os.path.join(PLOTS_DIR, "fixed_geometries.png"), dpi=300)
plt.close(fig_fix)
print("✅ Saved fixed_geometries.png to /plots")

# === MESHED GEOMETRY (PARALLEL) ===
print("🔹 Meshing geometries in parallel...")
meshed_jobs = [(fix.label, fix.geometry) for fix in fixeds]
meshed_results = parallel_mesh(meshed_jobs)

fig_mesh, axs_mesh = plt.subplots(2, 1, figsize=(8, 10))
fig_mesh.suptitle("Meshed Geometries")

for ax, mesh in zip(axs_mesh, meshed_results):
    if mesh:
        mesh.plot(ax)

fig_mesh.tight_layout()
fig_mesh.savefig(os.path.join(PLOTS_DIR, "meshed_geometries.png"), dpi=300)
plt.close(fig_mesh)
print("✅ Saved meshed_geometries.png to /plots")