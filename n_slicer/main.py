# n_slicer/main.py
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import json

# ───── Add project root to sys.path BEFORE any project imports ─────
CURRENT_FILE = Path(__file__).resolve()
candidate = CURRENT_FILE.parent
PROJECT_ROOT = None
while True:
    if (candidate / "n_slicer").is_dir():
        PROJECT_ROOT = candidate
        break
    if candidate == candidate.parent:
        break
    candidate = candidate.parent
if PROJECT_ROOT is None:
    PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───── Project imports ─────
from n_slicer.containers.units import SectionUnits, length_scale
from n_slicer.containers.geometry import BladeGeometryBin
from n_slicer.containers.sampling import DiscretisationSpec, FittingSpec
from n_slicer.parsers.distribution_parsers import DistributionParser
from n_slicer.sampling.discretise import make_rR_grid
from n_slicer.sampling.fitters import sample_distribution_df
from n_slicer.assembly.sections import build_section_bin_from_dataframe
from n_slicer.viz.blade_viz import BladeVisualizer


# =============================================================================
# User inputs / configuration
# =============================================================================
AIRFOIL_CSV      = "n_slicer/blade_input/NACA_63-415.csv"
DISTRIBUTION_CSV = "n_slicer/blade_input/chord_and_twist_profile.csv"

BASE_OUTDIR      = Path("n_slicer/blade_output")   # timestamped run folders will be created here
LABEL            = "N"                             # optional free-form tag
BLADE_NAME       = "Blade01"
BLADE_LENGTH     = 0.7  # z/L ∈ [0,1] maps to [0, L]
ROTOR_RADIUS     = 0.8  # r/R ∈ [0,1] maps to [0, R]
COP_FRACTION     = 0.33

UNITS = SectionUnits(
    rR="-", zL="-",
    c="m", L="m", R="m",
    XY="mm",
    beta_deg="deg",
)

# Transform settings
PIVOT_CHORD_FRAC = 0.333
TWIST_SIGN = 1
KEEP_PIVOT = False

# Numeric scale from chord units → geometry units (e.g., m → mm = 1000)
UNITS_SCALE = length_scale(UNITS.c, UNITS.XY)

# ---- sampling/discretisation controls ---------------------------------------
N_SECTIONS = 10
SCHEME     = "power_root"        # 'uniform' | 'cosine' | 'power_root' | 'power_tip'
POWER_EXP  = 2.5
CUSTOM_MAPPING = None         # def CUSTOM_MAPPING(u: np.ndarray) -> np.ndarray: ...

# ---- fit controls ------------------------------------------------------------
CHORD_FIT = "pchip"           # 'pchip' | 'akima' | 'spline' | 'linear'
TWIST_FIT = "pchip"
SPLINE_S  = None

# ---- output controls ---------------------------------------------------------
WRITE_DXFS = True
FILENAME_TEMPLATE = lambda i, rR, name: f"{(name or f'sec_{i:03d}')}_r{rR:.3f}.dxf"

# ---- materials.csv defaults --------------------------------------------------
# All rows use the same properties; only the filename changes.
MATERIAL_DEFAULTS = {
    "material_name": "Al_7075_T6",
    "elastic_modulus": 71700000000,   # Pa
    "poissons_ratio": 0.33,
    "yield_strength": 503000000,      # Pa
    "density": 2810000000,            # kg/m^3
    "color": "lightskyblue",
}


# =============================================================================
# Helpers
# =============================================================================
def _slug(s: str) -> str:
    return (
        s.strip()
         .replace(" ", "")
         .replace("/", "-")
         .replace("\\", "-")
         .replace("(", "")
         .replace(")", "")
         .replace(",", "")
         .replace("__", "_")
    )

def _fmt_num(v: float, dp: int = 2) -> str:
    txt = f"{v:.{dp}f}".rstrip("0").rstrip(".")
    return txt.replace(".", "p")

def _abbr_scheme(scheme: str, power: float | None, mapping_name: str | None) -> str:
    s = scheme.lower()
    if s == "uniform":     return "uni"
    if s == "cosine":      return "cos"
    if s == "power_root":  return f"pr{_fmt_num(power or 2.0, 2)}"
    if s == "power_tip":   return f"pt{_fmt_num(power or 2.0, 2)}"
    if s == "custom":      return f"cust-{_slug(mapping_name or 'map')}"
    return _slug(s)

def build_run_name(
    *,
    blade_name: str,
    label: str | None,
    n: int,
    scheme: str,
    power: float | None,
    mapping_name: str | None,
    L: float,
    L_unit: str,
    cp_frac: float,
    airfoil_csv: str,
    tz: str = "GMT",
) -> tuple[str, Path, Path, str]:
    """
    Returns (run_name, RUN_ROOT, SECTIONS_DIR, timestamp):
      <Blade>_<label?>_<Airfoil>_n<N>_<schemeAbbr>_L<X><unit>_cp<Y>pc_<timestamp>
    """
    airfoil_tag = _slug(Path(airfoil_csv).stem)
    sch = _abbr_scheme(scheme, power, mapping_name)
    L_tok = f"L{_fmt_num(L, 3)}{_slug(L_unit)}"
    cp_tok = f"cp{_fmt_num(cp_frac*100, 0)}pc"
    n_tok  = f"n{int(n)}"

    label_clean = (label or "").strip()
    prefix = f"{blade_name}_"
    if label_clean.startswith(prefix):
        label_clean = label_clean[len(prefix):]

    ts = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d-%H%M%S")
    parts = [blade_name, label_clean, airfoil_tag, n_tok, sch, L_tok, cp_tok, ts]
    parts = [p for p in parts if p]

    run_name = "_".join(parts)
    RUN_ROOT = BASE_OUTDIR / run_name
    SECTIONS_DIR = RUN_ROOT / "sections"
    return run_name, RUN_ROOT, SECTIONS_DIR, ts


def _write_manifest_csv(RUN_ROOT: Path, bin_, manifest_cols: list[str]) -> Path:
    """
    Writes manifest.csv with section metadata.
    NOTE: Internal geometry code uses airfoil convention (x = chord axis, y = thickness).
          In the blade coordinate system we reinterpret:
            - X_vert = vertical (former y)
            - Y_horiz = horizontal (former x)
    """
    rows = []
    for r in bin_.rows:
        gp = (r.scaled.geom if r.scaled and r.scaled.geom else None)
        fp = (r.scaled.from_norm if r.scaled and r.scaled.from_norm else None)

        x_cp = gp.x_cp if gp and gp.x_cp is not None else (fp.x_cp if fp else None)
        y_cp = gp.y_cp if gp and gp.y_cp is not None else (fp.y_cp if fp else None)

        rows.append({
            "station_idx": r.station_idx,
            "name": r.name,
            "rR": r.rR,
            "zL": r.zL,
            "c": r.c,
            "beta_deg": r.beta_deg,
            "dxf_path": r.dxf_path,
            # swapped axes
            "X_vert_cp": y_cp,
            "Y_horiz_cp": x_cp,
            "P":   (gp.P if gp else (fp.P if fp else None)),
            "A":   (gp.A if gp else (fp.A if fp else None)),
            "X_vert_min": (gp.ymin if gp else None),
            "X_vert_max": (gp.ymax if gp else None),
            "Y_horiz_min": (gp.xmin if gp else None),
            "Y_horiz_max": (gp.xmax if gp else None),
            "n_vertices": (gp.n_vertices if gp else r.norm.n_vertices if r.norm else None),
        })
    manifest_path = RUN_ROOT / "manifest.csv"
    pd.DataFrame(rows)[manifest_cols].to_csv(manifest_path, index=False)
    return manifest_path


def _write_blade_csv(RUN_ROOT: Path, bin_, zL_grid: np.ndarray, units: SectionUnits,
                     blade_length: float) -> Path:
    """
    Writes blade_stations.csv with centroid positions.
    NOTE: Axis reinterpretation for blade coordinate system:
          - X_vert = former y
          - Y_horiz = former x
    """
    XY_to_mm = length_scale(units.XY, "mm")
    L_to_mm  = length_scale(units.L,  "mm")

    blade_rows = []
    for r, zL in zip(bin_.rows, zL_grid):
        gp = (r.scaled.geom if r.scaled and r.scaled.geom else None)
        fp = (r.scaled.from_norm if r.scaled and r.scaled.from_norm else None)

        cx = gp.cx if (gp and gp.cx is not None) else (fp.cx if fp else None)
        cy = gp.cy if (gp and gp.cy is not None) else (fp.cy if fp else None)

        X_vert_mm = float(cy) * XY_to_mm if cy is not None else None
        Y_horiz_mm = float(cx) * XY_to_mm if cx is not None else None
        Cz_mm = float(zL) * float(blade_length) * L_to_mm

        blade_rows.append({
            "r/R [-]": r.rR,
            "X_vert [mm]": X_vert_mm,
            "Y_horiz [mm]": Y_horiz_mm,
            "Cz [mm]": Cz_mm,
            "B [deg]": r.beta_deg,
            "filename": os.path.basename(r.dxf_path) if r.dxf_path else "",
        })

    blade_csv_path = RUN_ROOT / "blade_stations.csv"
    pd.DataFrame(blade_rows).to_csv(blade_csv_path, index=False)
    return blade_csv_path


def _write_materials_csv(RUN_ROOT: Path, bin_, material_defaults: dict) -> Path:
    """
    Writes materials.csv with one row per section DXF.
    Columns: filename, material_name, elastic_modulus, poissons_ratio,
             yield_strength, density, color
    """
    rows = []
    for r in bin_.rows:
        if r.dxf_path:
            fname = os.path.basename(r.dxf_path)
        else:
            fname = FILENAME_TEMPLATE(r.station_idx, r.rR, r.name)

        row = {"filename": fname}
        row.update(material_defaults)
        rows.append(row)

    cols = [
        "filename",
        "material_name",
        "elastic_modulus",
        "poissons_ratio",
        "yield_strength",
        "density",
        "color",
    ]
    materials_csv_path = RUN_ROOT / "materials.csv"
    pd.DataFrame(rows, columns=cols).to_csv(materials_csv_path, index=False)
    return materials_csv_path


def _make_plots(RUN_ROOT: Path, bin_) -> dict[str, str]:
    PLOTS_DIR = RUN_ROOT / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    viz = BladeVisualizer(bin_, str(PLOTS_DIR))
    station_stack_path = viz.plot_stack_3d()
    chord_path = viz.plot_chord_vs_z()
    beta_path  = viz.plot_beta_vs_z()
    station_path = viz.plot_sections_2d()
    return {
        "stack_3d": str(station_stack_path),
        "chord_vs_z": str(chord_path),
        "beta_vs_z": str(beta_path),
        "sections_2d": str(station_path),
    }


def _write_run_meta(
    RUN_ROOT: Path,
    run_name: str,
    ts: str,
    blade_name: str,
    label: str,
    units: SectionUnits,
    geom: BladeGeometryBin,
    sampling_spec: DiscretisationSpec,
    fitting_spec: FittingSpec,
    bin_,
    manifest_path: Path,
    blade_csv_path: Path,
    plots: dict[str, str],
    sections_dir: Path,
) -> None:
    run_meta = {
        "run_name": run_name,
        "timestamp": ts,
        "blade_name": blade_name,
        "label": label,
        "units": {
            "rR": units.rR, "zL": units.zL, "c": units.c,
            "L": units.L, "R": units.R, "XY": units.XY,
            "beta_deg": units.beta_deg,
        },
        "geometry": {
            "R": geom.R,
            "L": geom.L,
            "R_hub": geom.R_hub,
            "rR_locii": geom.rR_locii,
            "zL_locii": geom.zL_locii,
        },
        "transform": {
            "pivot_chord_frac": PIVOT_CHORD_FRAC,
            "twist_sign": TWIST_SIGN,
            "keep_pivot_in_place": KEEP_PIVOT,
            "units_scale": UNITS_SCALE,
        },
        "sampling": sampling_spec.to_dict(),
        "fitting": fitting_spec.to_dict(),
        "summary": bin_.summary(),
        "outputs": {
            "manifest_csv": str(manifest_path),
            "blade_sections_csv": str(blade_csv_path),
            "plots": plots,
            "sections_dir": str(sections_dir),
        },
    }
    with open(RUN_ROOT / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # 1) Read the original distribution
    df_src = DistributionParser(DISTRIBUTION_CSV, assume_no_header=False).parse_dataframe()
    r_min, r_max = float(df_src["r_over_R"].min()), float(df_src["r_over_R"].max())

    # 2) Make r/R grid
    if CUSTOM_MAPPING is None:
        rR_grid = make_rR_grid(N_SECTIONS, scheme=SCHEME, power=POWER_EXP,
                               rR_min=r_min, rR_max=r_max)
        scheme_used = SCHEME
        mapping_name = None
        power_used = POWER_EXP if "power" in SCHEME else None
    else:
        rR_grid = make_rR_grid(N_SECTIONS, mapping=CUSTOM_MAPPING,
                               rR_min=r_min, rR_max=r_max)
        scheme_used = "custom"
        mapping_name = CUSTOM_MAPPING.__name__
        power_used = None

    # 2a) Geometry container
    geom = BladeGeometryBin(R=float(ROTOR_RADIUS), L=float(BLADE_LENGTH))
    zL_grid = np.clip(geom.zL_from_rR(rR_grid), 0.0, 1.0)

    # 2b) Run folder
    run_name, RUN_ROOT, SECTIONS_DIR, ts = build_run_name(
        blade_name=BLADE_NAME,
        label=LABEL,
        n=N_SECTIONS,
        scheme=scheme_used,
        power=power_used,
        mapping_name=mapping_name,
        L=BLADE_LENGTH,
        L_unit=UNITS.L,
        cp_frac=COP_FRACTION,
        airfoil_csv=AIRFOIL_CSV,
    )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Fit chord(r/R) & twist(r/R)
    df_used = sample_distribution_df(
        df_src,
        n=N_SECTIONS,
        rR_grid=rR_grid,
        chord_fit=CHORD_FIT,
        twist_fit=TWIST_FIT,
        spline_smoothing=SPLINE_S,
    )

    # Carry names by NN in r/R
    if "name" in df_src.columns:
        r_src = df_src["r_over_R"].to_numpy(float)
        names = df_src["name"].astype(str).to_numpy()
        idx_nn = np.searchsorted(r_src, rR_grid).clip(1, len(r_src)-1)
        left = idx_nn - 1
        pick = np.where(np.abs(r_src[idx_nn] - rR_grid) <
                        np.abs(rR_grid - r_src[left]), idx_nn, left)
        df_used["name"] = names[pick]

    # 4) Spec objects
    sampling_spec = DiscretisationSpec(
        n=N_SECTIONS, scheme=scheme_used,
        rR_min=r_min, rR_max=r_max,
        power=power_used, mapping_name=mapping_name,
        rR_grid=rR_grid,
    )
    fitting_spec = FittingSpec(
        chord_fit=CHORD_FIT, twist_fit=TWIST_FIT,
        spline_smoothing=SPLINE_S,
    )

    # 5) Build SectionBin
    bin_ = build_section_bin_from_dataframe(
        airfoil_csv=AIRFOIL_CSV,
        df=df_used,
        label=LABEL,
        blade_name=BLADE_NAME,
        L=geom.L, R=geom.R, units=UNITS,
        pivot_chord_frac=PIVOT_CHORD_FRAC,
        twist_sign=TWIST_FIT and TWIST_SIGN,
        keep_pivot_in_place=KEEP_PIVOT,
        units_scale=UNITS_SCALE,
        cp_default_frac=COP_FRACTION,
        write_dxfs=WRITE_DXFS,
        outdir=str(SECTIONS_DIR),
        filename=FILENAME_TEMPLATE,
        df_source=df_src,
        sampling_spec=sampling_spec,
        fitting_spec=fitting_spec,
        zL_grid=zL_grid,
    )

    # ---- summary
    print("[SUMMARY]")
    for k, v in bin_.summary().items():
        print(f"  {k}: {v}")

    # ---- manifest
    manifest_cols = [
        "station_idx", "name", "rR", "zL", "c", "beta_deg", "dxf_path",
        "X_vert_cp", "Y_horiz_cp", "P", "A",
        "X_vert_min", "X_vert_max", "Y_horiz_min", "Y_horiz_max", "n_vertices",
    ]
    manifest_path = _write_manifest_csv(RUN_ROOT, bin_, manifest_cols)

    # ---- blade CSV
    blade_csv_path = _write_blade_csv(RUN_ROOT, bin_, zL_grid, UNITS, BLADE_LENGTH)
    print(f"[OK] Blade CSV:     {blade_csv_path}")

    # ---- materials CSV
    materials_csv_path = _write_materials_csv(RUN_ROOT, bin_, MATERIAL_DEFAULTS)
    print(f"[OK] Materials CSV: {materials_csv_path}")

    # ---- plots
    plots = _make_plots(RUN_ROOT, bin_)
    print(f"[OK] 3D stack:      {plots['stack_3d']}")
    print(f"[OK] Chord vs z:    {plots['chord_vs_z']}")
    print(f"[OK] Beta  vs z:    {plots['beta_vs_z']}")

    # ---- run metadata
    _write_run_meta(
        RUN_ROOT, run_name, ts, BLADE_NAME, LABEL, UNITS, geom,
        sampling_spec, fitting_spec, bin_,
        manifest_path, blade_csv_path, plots, SECTIONS_DIR,
    )

    print(f"\n[OK] Wrote DXFs to: {SECTIONS_DIR}")
    print(f"[OK] Manifest:      {manifest_path}")
    print(f"[OK] Run folder:    {RUN_ROOT}")


if __name__ == "__main__":
    main()