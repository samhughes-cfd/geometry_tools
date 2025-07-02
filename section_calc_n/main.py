# main.py

"""
Updated:
• Read blade station list from CSV in BLADE_DIR (now holds both .dxf + stations.csv).
• Loop over every station, applying twist & centroid offset *after computing actual centroid*.
• Wrap each mesh in a Section before logging properties.
• Use ASCII arrows (->) and UTF-8 everywhere to avoid Windows‐encoding issues.
"""

import logging
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from raw_geometry_dxf import RawDXFPreview
from processed_geometry_dxf import ProcessedGeometryDXF
from geometry_transforms.twist_offset import TwistOffset
from geometry_transforms.centroid_offset import CentroidOffset
from mesh_dxf import MeshDXF
from section_dxf import SectionDXF
from sectionproperties.analysis import Section

# ───────── folders ─────────
BASE      = Path("section_calc")
BLADE_DIR = BASE / "blade"     # contains BOTH .dxf files and stations.csv
LOGS      = BASE / "logs"
PLOTS     = BASE / "plots"
RESULTS   = BASE / "results"

for d in (LOGS, PLOTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "main.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.info("DXF mesh-convergence pipeline started")

# ───────── read station schedule ─────────────────────────────────────
stations_csv = BLADE_DIR / "stations.csv"
stations = []
with open(stations_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        r  = float(row["r"])
        B  = float(row["B"])
        Cx = float(row["Cx"])
        Cy = float(row["Cy"])
        stations.append((r, B, Cx, Cy))

# ───────── mesh‐convergence parameters ───────────────────────────────
N   = 3
h0  = 3.0  # target element area [mm²]
hs  = h0 / (4 ** np.arange(N))
logging.info("Mesh sizes (area targets): %s", hs)

# ───────── loop over every blade station ─────────────────────────────
for r, B_r, Cx, Cy in stations:
    label = f"r{r:.2f}"
    dxf   = BLADE_DIR / f"Station_{r:.2f}.dxf"
    logging.info(
        "Station %s: DXF=%s, twist=%.1f°, centroid=(%.2f,%.2f)",
        label, dxf, B_r, Cx, Cy
    )

    # ───── RAW PREVIEW (optional) ───────────────────────────────────
    raw_fig, raw_ax = plt.subplots(figsize=(6, 5))
    RawDXFPreview(dxf, label).plot(raw_ax)  
    raw_ax.set_title(f"Raw DXF: {label}")
    raw_png = PLOTS / f"raw_{label}.png"
    raw_fig.tight_layout()
    raw_fig.savefig(raw_png, dpi=300)
    plt.close(raw_fig)
    logging.info("Raw preview saved -> %s", raw_png)

    # ───── PROCESSED GEOMETRY ──────────────────────────────────────
    proc = ProcessedGeometryDXF(
        filepath=dxf,
        label=label,
        spline_delta=0.05,
        degrees_per_segment=0.5,
        exterior_nodes=400,
    )
    geom0 = proc.extract()
    logging.info("Processed geometry extracted for %s", label)

    # ───── Step 1: rotate by twist ──────────────────────────────────
    geom_twisted = TwistOffset(geometry=geom0, desired_twist_deg=B_r).apply()

    # ───── Step 2: get actual centroid from preview mesh ────────────
    mesh_preview = MeshDXF(geom_twisted, label + "_preview")
    geom_preview = mesh_preview.build(mesh_size=3.0)
    sec_preview = Section(geometry=geom_preview)
    sec_preview.calculate_geometric_properties()
    x_c, y_c = sec_preview.get_c()
    logging.info("Actual centroid = (%.3f, %.3f); desired = (%.3f, %.3f)", x_c, y_c, Cx, Cy)

    # ───── Step 3: align centroid ────────────────────────────────────
    geom2 = CentroidOffset(geometry=geom_twisted, cx_target=Cx, cy_target=Cy).apply()

    # ───── Two-panel figure: full chord + TE zoom ───────────────────
    fig2, (ax_full, ax_te) = plt.subplots(2, 1, figsize=(6, 8))
    proc.plot(ax_full, outline_lw=1.0, cp_size=5, legend_loc="upper right")
    ax_full.set_title(f"Processed Geometry: {label} (full chord)")
    mesh_dummy = MeshDXF(geom2, label)
    mesh_dummy.mesh_generated = True
    mesh_dummy.geometry = geom2
    mesh_dummy.plot(ax_te, zoom_te_pct=5.0)
    ax_te.set_title(f"TE Zoom: {label}")
    fig2.tight_layout()
    proc_png = PLOTS / f"processed_te_zoom_{label}.png"
    fig2.savefig(proc_png, dpi=300)
    plt.close(fig2)
    logging.info("Processed geometry saved -> %s", proc_png)

    # ───── mesh‐convergence loops ──────────────────────────────────
    meshes_for_plot = []
    for i, h in enumerate(hs, 1):
        run_lbl = f"{label}_h{h:.4g}"
        logging.info("[%d/%d] meshing %s with target h = %.4g", i, N, label, h)

        mesh = MeshDXF(geometry=geom2, label=run_lbl)
        geom_m = mesh.build(mesh_size=float(h))
        if geom_m is None:
            logging.error("[%s] mesh failed – skipping", run_lbl)
            continue

        # wrap in Section before logging properties
        sec_obj = Section(geometry=geom_m)
        SectionDXF(run_lbl, float(h), sec_obj).write_csv_row()
        meshes_for_plot.append((run_lbl, mesh))

    # ───── plotting convergence: global + local ─────────────────────
    if meshes_for_plot:
        batch_size = 3
        for batch_idx, start in enumerate(range(0, len(meshes_for_plot), batch_size), 1):
            batch = meshes_for_plot[start : start + batch_size]
            n_mesh = len(batch)

            # global (full‐chord) vertical stack
            fig_g, axs_g = plt.subplots(n_mesh, 1, figsize=(4, 1.4 * n_mesh), sharex=True)
            if n_mesh == 1: axs_g = [axs_g]
            for ax, (lbl, m) in zip(axs_g, batch):
                m.plot(ax, zoom_te_pct=None)
                ax.set_title(lbl, fontsize=8)
            fig_g.tight_layout()
            out_g = PLOTS / f"mesh_conv_{label}_batch{batch_idx}_global.png"
            fig_g.savefig(out_g, dpi=300); plt.close(fig_g)
            logging.info("Global batch %d saved -> %s", batch_idx, out_g)

            # local (TE zoom) horizontal row
            fig_l, axs_l = plt.subplots(1, n_mesh, figsize=(4 * n_mesh, 4), sharey=True)
            if n_mesh == 1: axs_l = [axs_l]
            for ax, (lbl, m) in zip(axs_l, batch):
                m.plot(ax, zoom_te_pct=5.0)
                ax.set_title(lbl, fontsize=8)
            fig_l.tight_layout()
            out_l = PLOTS / f"mesh_conv_{label}_batch{batch_idx}_local.png"
            fig_l.savefig(out_l, dpi=300); plt.close(fig_l)
            logging.info("Local batch %d saved -> %s", batch_idx, out_l)

logging.info("Pipeline complete — results in %s", RESULTS / "section_results.csv")
