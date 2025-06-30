# section_calc/main.py
import os, logging, numpy as np, matplotlib.pyplot as plt
from raw_geometry_dxf import RawDXFPreview
from processed_geometry_dxf import ProcessedGeometryDXF
from mesh_dxf         import MeshDXF
from section_dxf      import SectionDXF


# ───────── folders ─────────
BASE   = "section_calc"
LOGS   = f"{BASE}/logs"
PLOTS  = f"{BASE}/plots"
os.makedirs(LOGS,  exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS}/main.log", "w"),
        logging.StreamHandler()
    ],
)
logging.info("DXF mesh-convergence pipeline started")

# ───────── DXF file ─────────
dxf_path = f"{BASE}/dxf/Station8_SE.dxf"
base_lbl = "Station8_SE"

# ───────── RAW PREVIEW PNG ─────────
raw_preview = RawDXFPreview(dxf_path, base_lbl)
fig, ax = plt.subplots(figsize=(6, 5))
raw_preview.plot(ax, annotate_every = None)
fig.tight_layout()
raw_png = f"{PLOTS}/raw.png"
fig.savefig(raw_png, dpi=300)
plt.close(fig)
logging.info(f"Raw DXF preview saved -> {raw_png}")

# --- PROCESSED GEOMETRY -------------------------------------------------
proc = ProcessedGeometryDXF(
    dxf_path,
    base_lbl,
    spline_delta=0.05,
    degrees_per_segment=0.5,
)
proc.extract()
logging.info("Processed geometry extracted")

# --- create and save two-row figure ------------------------------------
fig, (ax_full, ax_zoom) = proc.plot_te_zoom(te_span_pct=5.0, cp_size=10)

proc_png = f"{PLOTS}/processed_te_zoom.png"   # ← create the name
fig.savefig(proc_png, dpi=300)                # save
plt.close(fig)

logging.info(f"Processed geometry saved -> {proc_png}")


# ─────── rest of script unchanged – use proc.geometry for meshing ───────
N  = 3
h0 = 3
mesh_sizes = h0 / (2 ** np.arange(N))
logging.info(f"Mesh sizes: {mesh_sizes}")

meshes_for_plot = []
for i, h in enumerate(mesh_sizes, 1):
    run_lbl = f"{base_lbl}_h{float(h):.4g}"
    logging.info(f"[{i}/{N}] meshing with h = {h:.4g}")

    mesh = MeshDXF(proc.geometry, run_lbl)
    sec  = mesh.build(mesh_size=float(h))

    if sec is None:
        logging.error(f"[{run_lbl}] mesh failed – row skipped")
        continue

    SectionDXF(run_lbl, float(h), sec).write_csv_row()
    meshes_for_plot.append((run_lbl, mesh))

# ───────── subplot of all meshes ─────────
if meshes_for_plot:
    fig_h = 3 * len(meshes_for_plot)
    fig, axs = plt.subplots(len(meshes_for_plot), 1,
                            figsize=(8, fig_h),
                            sharex=True, sharey=True)
    axs = [axs] if len(meshes_for_plot) == 1 else axs

    for ax, (lbl, mesh_obj) in zip(axs, meshes_for_plot):
        mesh_obj.plot(ax)
        ax.set_title(lbl, fontsize=9)

    fig.suptitle("Mesh-refinement study — Station8_SE", y=1.02)
    fig.tight_layout()
    out_png = f"{PLOTS}/mesh_convergence.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logging.info(f"Mesh-convergence plot saved → {out_png}")

logging.info("Convergence study complete — "
             "results at section_calc/results/section_results.csv")
