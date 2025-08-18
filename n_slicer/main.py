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

# Containers + assembler + sampling
from n_slicer.containers.units import SectionUnits, length_scale
from n_slicer.containers.sampling import DiscretisationSpec, FittingSpec
from n_slicer.parsers.distribution_parsers import DistributionParser
from n_slicer.sampling.discretise import make_rR_grid
from n_slicer.sampling.fitters import sample_distribution_df
from n_slicer.assembly.sections import build_section_bin_from_dataframe

# ---- user inputs -------------------------------------------------------------
AIRFOIL_CSV      = "n_slicer/blade_input/NACA_63-415.csv"
DISTRIBUTION_CSV = "n_slicer/blade_input/chord_and_twist_profile.csv"

BASE_OUTDIR      = Path("n_slicer/blade_output")   # timestamped run folders will be created here
LABEL            = "N"                              # optional free-form tag
BLADE_NAME       = "Blade01"
BLADE_LENGTH     = 0.7
COP_FRACTION     = 0.33

UNITS = SectionUnits(
    rR="-", xL="-",
    c="m", L="m",
    XY="mm",
    beta_deg="deg",
)

# Transform settings
PIVOT_CHORD_FRAC = 0.33
TWIST_SIGN = 1
KEEP_PIVOT = False

# Numeric scale from chord units → geometry units (e.g., m → mm = 1000)
UNITS_SCALE = length_scale(UNITS.c, UNITS.XY)

# ---- sampling/discretisation controls ---------------------------------------
N_SECTIONS = 40
SCHEME     = "power_root"        # 'uniform' | 'cosine' | 'power_root' | 'power_tip'
POWER_EXP  = 2.5
CUSTOM_MAPPING = None            # def CUSTOM_MAPPING(u: np.ndarray) -> np.ndarray: ...

# ---- fit controls ------------------------------------------------------------
CHORD_FIT = "pchip"              # 'pchip' | 'akima' | 'spline' | 'linear'
TWIST_FIT = "pchip"
SPLINE_S  = None


# ---------------------------- helpers for run name ----------------------------
def _slug(s: str) -> str:
    """Safe, compact token for folder names."""
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
    """Compact numeric token: 0.70 -> '0p70', 2.5 -> '2p5' (no trailing zeros)."""
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
    tz: str = "Europe/London",
) -> tuple[str, Path, Path, str]:
    """
    Returns (run_name, RUN_ROOT, SECTIONS_DIR, timestamp) using timestamp + meta tokens:
      <Blade>_<label?>_<Airfoil>_n<N>_<schemeAbbr>_L<X><unit>_cp<Y>pc_<timestamp>
    """
    airfoil_tag = _slug(Path(airfoil_csv).stem)  # e.g. 'NACA_63-415'
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
    parts = [p for p in parts if p]  # drop empties

    run_name = "_".join(parts)
    RUN_ROOT = BASE_OUTDIR / run_name
    SECTIONS_DIR = RUN_ROOT / "sections"
    return run_name, RUN_ROOT, SECTIONS_DIR, ts
# -----------------------------------------------------------------------------

def main() -> None:
    # 1) Read the original (source) distribution — robust to '#' headers
    df_src = DistributionParser(DISTRIBUTION_CSV, assume_no_header=False).parse_dataframe()
    r_min, r_max = float(df_src["r_over_R"].min()), float(df_src["r_over_R"].max())

    # 2) Make r/R grid according to discretisation controls
    if CUSTOM_MAPPING is None:
        rR_grid = make_rR_grid(N_SECTIONS, scheme=SCHEME, power=POWER_EXP, rR_min=r_min, rR_max=r_max)
        scheme_used = SCHEME
        mapping_name = None
        power_used = POWER_EXP if "power" in SCHEME else None
    else:
        rR_grid = make_rR_grid(N_SECTIONS, mapping=CUSTOM_MAPPING, rR_min=r_min, rR_max=r_max)
        scheme_used = "custom"
        mapping_name = CUSTOM_MAPPING.__name__
        power_used = None

    # 2b) Build a self-describing run folder name and create dirs
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

    # 3) Fit chord(r/R) & twist(r/R) and sample onto our grid
    df_used = sample_distribution_df(
        df_src,
        n=N_SECTIONS,
        rR_grid=rR_grid,
        chord_fit=CHORD_FIT,
        twist_fit=TWIST_FIT,
        spline_smoothing=SPLINE_S,
    )

    # (Optional) carry names by nearest neighbour in r/R
    if "name" in df_src.columns:
        r_src = df_src["r_over_R"].to_numpy(float)
        names = df_src["name"].astype(str).to_numpy()
        idx_nn = np.searchsorted(r_src, rR_grid).clip(1, len(r_src)-1)
        left = idx_nn - 1
        pick = np.where(np.abs(r_src[idx_nn] - rR_grid) < np.abs(rR_grid - r_src[left]), idx_nn, left)
        df_used["name"] = names[pick]

    # 4) Build spec objects so the bin remembers sampling + fitting
    sampling_spec = DiscretisationSpec(
        n=N_SECTIONS,
        scheme=scheme_used,
        rR_min=r_min,
        rR_max=r_max,
        power=power_used,
        mapping_name=mapping_name,
        rR_grid=rR_grid,
    )
    fitting_spec = FittingSpec(
        chord_fit=CHORD_FIT,
        twist_fit=TWIST_FIT,
        spline_smoothing=SPLINE_S,
    )

    # 5) Build the SectionBin from the *sampled* DataFrame and attach provenance
    bin_ = build_section_bin_from_dataframe(
        airfoil_csv=AIRFOIL_CSV,
        df=df_used,
        label=LABEL,
        blade_name=BLADE_NAME,
        L=BLADE_LENGTH,
        units=UNITS,
        pivot_chord_frac=PIVOT_CHORD_FRAC,
        twist_sign=TWIST_SIGN,
        keep_pivot_in_place=KEEP_PIVOT,
        units_scale=UNITS_SCALE,
        cp_default_frac=COP_FRACTION,
        write_dxfs=True,
        outdir=str(SECTIONS_DIR),  # DXFs → timestamped + meta-labelled folder
        filename=lambda i, rR, name: f"{(name or f'sec_{i:03d}')}_r{rR:.3f}.dxf",
        # provenance:
        df_source=df_src,
        sampling_spec=sampling_spec,
        fitting_spec=fitting_spec,
    )

    # ---- text summary
    print("[SUMMARY]")
    for k, v in bin_.summary().items():
        print(f"  {k}: {v}")

    # ---- persist provenance + manifest at run root ---------------------------
    # Save distributions and grid used
    df_src.to_csv(RUN_ROOT / "source_distribution.csv", index=False)
    df_used.to_csv(RUN_ROOT / "used_distribution.csv", index=False)
    pd.DataFrame({"r_over_R": rR_grid}).to_csv(RUN_ROOT / "rR_grid.csv", index=False)

    # Manifest (metadata + a few key scaled properties)
    rows = []
    for r in bin_.rows:
        gp = (r.scaled.geom if r.scaled and r.scaled.geom else None)
        fp = (r.scaled.from_norm if r.scaled and r.scaled.from_norm else None)
        rows.append({
            "station_idx": r.station_idx,
            "name": r.name,
            "rR": r.rR,
            "c": r.c,
            "beta_deg": r.beta_deg,
            "dxf_path": r.dxf_path,
            "x_cp": (gp.x_cp if gp and gp.x_cp is not None else (fp.x_cp if fp else None)),
            "y_cp": (gp.y_cp if gp and gp.y_cp is not None else (fp.y_cp if fp else None)),
            "P":   (gp.P if gp else (fp.P if fp else None)),
            "A":   (gp.A if gp else (fp.A if fp else None)),
            "xmin": (gp.xmin if gp else None),
            "xmax": (gp.xmax if gp else None),
            "ymin": (gp.ymin if gp else None),
            "ymax": (gp.ymax if gp else None),
            "n_vertices": (gp.n_vertices if gp else r.norm.n_vertices if r.norm else None),
        })
    manifest_path = RUN_ROOT / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    # Minimal run metadata (nice for bookkeeping)
    run_meta = {
        "run_name": run_name,
        "timestamp": ts,
        "blade_name": BLADE_NAME,
        "label": LABEL,
        "units": {
            "rR": UNITS.rR, "xL": UNITS.xL, "c": UNITS.c, "L": UNITS.L, "XY": UNITS.XY, "beta_deg": UNITS.beta_deg,
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
    }
    with open(RUN_ROOT / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n[OK] Wrote DXFs to: {SECTIONS_DIR}")
    print(f"[OK] Manifest:      {manifest_path}")
    print(f"[OK] Run folder:     {RUN_ROOT}")

if __name__ == "__main__":
    main()