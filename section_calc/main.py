"""section_calc/main.py
Updated:
• Use pathlib.Path for paths (already done).
• Replace Unicode arrow (→) with ASCII arrow (->) to prevent Windows cp1252 encoding errors.
• Fix property‑calculation failure by wrapping geometry in a `Section` object before passing to
  `SectionDXF`, because `CompoundGeometry` no longer exposes
  `calculate_geometric_properties()` in sectionproperties ≥ 4.
"""

import logging
import os  # still used by other modules; safe to keep
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from raw_geometry_dxf import RawDXFPreview
from processed_geometry_dxf import ProcessedGeometryDXF
from mesh_dxf import MeshDXF
from section_dxf import SectionDXF

# ───────── folders ─────────
BASE = Path("section_calc")
LOGS = BASE / "logs"
PLOTS = BASE / "plots"

LOGS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "main.log", mode="w", encoding="utf-8"),  # ensure UTF‑8 file
        logging.StreamHandler(),
    ],
)
logging.info("DXF mesh-convergence pipeline started")

# ───────── DXF file ─────────
dxf_path = BASE / "dxf" / "Station8_SE.dxf"
base_lbl = "Station8_SE"

# ───────── RAW PREVIEW PNG ─────────
raw_preview = RawDXFPreview(dxf_path, base_lbl)
fig, ax = plt.subplots(figsize=(6, 5))
raw_preview.plot(ax, annotate_every=None)
fig.tight_layout()
raw_png = PLOTS / "raw.png"
fig.savefig(raw_png, dpi=300)
plt.close(fig)
logging.info("Raw DXF preview saved -> %s", raw_png)

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
proc_png = PLOTS / "processed_te_zoom.png"
fig.savefig(proc_png, dpi=300)
plt.close(fig)
logging.info("Processed geometry saved -> %s", proc_png)

# ─────── rest of script unchanged – use proc.geometry for meshing ───────
N = 3
h0 = 3
mesh_sizes = h0 / (4 ** np.arange(N))           # quarter-step refinement
logging.info("Mesh sizes (area targets): %s", mesh_sizes)

meshes_for_plot = []
for i, h in enumerate(mesh_sizes, 1):
    run_lbl = f"{base_lbl}_h{float(h):.4g}"
    logging.info("[%d/%d] meshing with h = %.4g", i, N, h)

    mesh = MeshDXF(proc.geometry, run_lbl)
    geom = mesh.build(mesh_size=float(h))

    if geom is None:
        logging.error("[%s] mesh failed – row skipped", run_lbl)
        continue

    # ---------- NEW: wrap geometry in a Section before CSV ----------
    from sectionproperties.analysis import Section
    sec_obj = Section(geometry=geom)                 # ← Section object
    SectionDXF(run_lbl, float(h), sec_obj).write_csv_row()
    # ----------------------------------------------------------------

    meshes_for_plot.append((run_lbl, mesh))

# ───────── mesh-convergence plots: global and local saved separately ─────────
if meshes_for_plot:
    batch_size = 3

    for batch_idx, start in enumerate(range(0, len(meshes_for_plot), batch_size), 1):
        batch = meshes_for_plot[start:start + batch_size]  # 1–3 meshes
        n_mesh = len(batch)

        # ---------------------------------------------------------------------
        # 1️⃣  GLOBAL (full-chord) views – vertical stack
        # ---------------------------------------------------------------------
        fig_g, axs_g = plt.subplots(
            n_mesh,
            1,
            figsize=(4, 1.4 * n_mesh),
            sharex=True,        # only bottom plot shows x-label
            sharey=False,
        )

        if n_mesh == 1:
            axs_g = [axs_g]

        for ax, (lbl, mesh_obj) in zip(axs_g, batch):
            mesh_obj.plot(ax, zoom_te_pct=None)
            ax.set_title(lbl, fontsize=8)

        for ax in axs_g:
            ax.label_outer()

        fig_g.suptitle(
            f"Mesh-refinement study – batch {batch_idx} (GLOBAL)", fontsize=10, y=0.98
        )
        fig_g.tight_layout(rect=[0, 0, 1, 0.95])

        out_png_global = PLOTS / f"mesh_convergence_batch-{batch_idx}_global.png"
        fig_g.savefig(out_png_global, dpi=300)
        plt.close(fig_g)
        logging.info("Global views batch %d saved -> %s", batch_idx, out_png_global)

        # ---------------------------------------------------------------------
        # 2️⃣  LOCAL (TE zoom) views – horizontal stack
        # ---------------------------------------------------------------------
        fig_l, axs_l = plt.subplots(
            1,
            n_mesh,
            figsize=(4 * n_mesh, 4),
            sharex=False,
            sharey=True,       # only left-most plot shows y-label
        )
        if n_mesh == 1:
            axs_l = [axs_l]

        for ax, (lbl, mesh_obj) in zip(axs_l, batch):
            mesh_obj.plot(ax, zoom_te_pct=5.0)
            ax.set_title(lbl, fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")

        for ax in axs_l:
            ax.label_outer()

        fig_l.suptitle(
            f"Mesh-refinement study – batch {batch_idx} (LOCAL TE)", fontsize=10, y=0.98
        )
        fig_l.tight_layout(rect=[0, 0, 1, 0.95])

        out_png_local = PLOTS / f"mesh_convergence_batch-{batch_idx}_local.png"
        fig_l.savefig(out_png_local, dpi=300)
        plt.close(fig_l)
        logging.info("Local views batch %d saved -> %s", batch_idx, out_png_local)