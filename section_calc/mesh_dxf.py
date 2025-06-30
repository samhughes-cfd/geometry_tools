# section_calc/mesh_dxf.py

import logging, numpy as np
from pathlib import Path
from sectionproperties.analysis import Section
from sectionproperties.pre.geometry import Geometry

# ───────────────────────── logger ─────────────────────────
LOG_DIR = Path("section_calc/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("MeshDXF")
logger.setLevel(logging.INFO)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_DIR / "mesh.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

# ───────────────────────── class ──────────────────────────
class MeshDXF:
    """Meshing helper with detailed logging & optional TE zoom."""

    def __init__(self, geometry, label):
        self.geometry   = geometry
        self.label      = label
        self.mesh_generated = False
        # statistics
        self.n_nodes = self.n_elems = 0
        self.min_a = self.max_a = None
        self.min_q = self.max_q = None
        self._prev_elems = None

        # ---------------------------------------------------------------- build
    def build(self, mesh_size: float):
        """Generate mesh, refine (if possible), collect stats, log."""
        try:
            logger.info("[%s] meshing with target h = %.4g …",
                        self.label, mesh_size)

            # 1) initial mesh
            self.geometry.create_mesh(mesh_sizes=[mesh_size])

            # 2) optional second-pass refinement (Geometry only, v3)
            from sectionproperties.pre.geometry import Geometry
            if isinstance(self.geometry, Geometry) and hasattr(self.geometry, "refine_mesh"):
                self.geometry.refine_mesh(cutoff_area=mesh_size, refine_num=2)

            self.mesh_generated = True
            mesh = self.geometry.mesh          # v4 Mesh object *or* v3 dict

            # ---------- extract vertices / triangles -------------------
            if hasattr(mesh, "points"):        # v4
                verts, tris = mesh.points, mesh.triangles
            elif isinstance(mesh, dict):       # v3
                verts, tris = mesh.get("vertices"), mesh.get("triangles")
            else:
                verts = tris = None

            self.n_nodes = 0 if verts is None else verts.shape[0]
            self.n_elems = 0 if tris  is None else tris.shape[0]

            tri_area = tri_quality = None
            if verts is not None and tris is not None and len(tris):
                p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
                tri_area = 0.5 * np.abs(
                    (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                  - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
                )

                # simple quality 4√3 A / Σ(edge²)
                e0 = np.linalg.norm(p1 - p0, axis=1)
                e1 = np.linalg.norm(p2 - p1, axis=1)
                e2 = np.linalg.norm(p0 - p2, axis=1)
                tri_quality = (4 * np.sqrt(3) * tri_area) / (e0**2 + e1**2 + e2**2)

                # ---------- sliver filter (area < 1e-8) -----------------
                sliver = tri_area < 1e-8
                if sliver.any():
                    removed = int(sliver.sum())
                    tri_area    = tri_area[~sliver]
                    tri_quality = tri_quality[~sliver]
                    self.n_elems -= removed
                    logger.warning("[%s] %d sliver triangles removed",
                                   self.label, removed)

                # ---------- stats --------------------------------------
                self.min_a, self.max_a = float(tri_area.min()), float(tri_area.max())
                mean_a  = float(tri_area.mean())
                med_a   = float(np.median(tri_area))

                self.min_q, self.max_q = float(tri_quality.min()), float(tri_quality.max())
                mean_q  = float(tri_quality.mean())
            else:
                mean_a = med_a = mean_q = None

            # ---------- logging summary --------------------------------
            logger.info(
                "[%s] mesh ok | nodes=%d elems=%d | "
                "A[min=%.3g max=%.3g mean=%.3g med=%.3g] | "
                "q[min=%.3f max=%.3f mean=%.3f]",
                self.label,
                self.n_nodes, self.n_elems,
                self.min_a or 0, self.max_a or 0, mean_a or 0, med_a or 0,
                self.min_q or 0, self.max_q or 0, mean_q or 0,
            )

            # unchanged-element warning
            if getattr(self, "_prev_elems", None) == self.n_elems:
                logger.warning("[%s] element count unchanged (=%d elems)",
                               self.label, self.n_elems)
            self._prev_elems = self.n_elems

            return self.geometry

        except Exception as exc:
            logger.error("[%s] mesh failed: %s", self.label, exc, exc_info=True)
            return None


    # ------------------------------ plot -------------------------------
    def plot(self, ax, *, zoom_te_pct: float | None = None):
        if not self.mesh_generated:
            ax.text(0.5, 0.5, "Mesh Not Generated", ha="center", va="center")
            return

        try:
            Section(self.geometry).plot_mesh(
                ax=ax, materials=False, colormap="viridis",
                colorbar=True, show_nodes=False,
            )

            if zoom_te_pct is not None:
                pts = np.asarray(self.geometry.points)
                x_max = pts[:, 0].max()
                x_min = pts[:, 0].min()
                chord = x_max - x_min
                ax.set_xlim(x_max - (zoom_te_pct / 100) * chord, x_max)

            stats = (
                f"n={self.n_elems}  "
                f"A[{self.min_a:.2g},{self.max_a:.2g}]  "
                f"q[{self.min_q:.2f},{self.max_q:.2f}]"
                if self.min_a is not None else f"n={self.n_elems}"
            )
            ax.set_title(f"{self.label}\n{stats}", fontsize=8)

        except Exception as exc:
            ax.text(0.5, 0.5, f"Plot Failed\n{exc}", ha="center", va="center")
            logger.warning("[%s] mesh plotting failed: %s", self.label, exc)