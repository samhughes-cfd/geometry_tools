# n_slicer/assembly/sections.py
"""
sections.py (assembly)
----------------------
Takes already-prepared distributions (typically from main.py + sampling/)
and builds SectionBin/SectionRow, computing Normalised + Scaled properties.
No curve fitting or discretisation is done here.

Spanwise axis
-------------
We use the blade span axis 'z' with the nondimensional coordinate z/L = zL.

Pivot definition
----------------
The rotation pivot is placed on the straight LE→TE chord line at a specified
fraction 0..1 measured from the LE toward the TE, computed once from the
*normalised* airfoil geometry.

Provenance
----------
Discretisation (n, grids, scheme, etc.) and fitting (chord/twist fits) are
accepted as metadata objects and attached to SectionBin; they are not produced
here to keep a strict separation of concerns.
"""
from __future__ import annotations

import os
from typing import Optional, Callable

import numpy as np
import pandas as pd

# Parsers (I/O only)
from n_slicer.parsers.naca_parser import NormalisedNACAParser
from n_slicer.parsers.distribution_parsers import DistributionParser

# Containers & properties
from n_slicer.containers.units import SectionUnits
from n_slicer.containers.section_bin import SectionBin
from n_slicer.containers.section_row import SectionRow
from n_slicer.containers.properties import NormalisedProperties, ScaledProperties
from n_slicer.containers.sampling import DiscretisationSpec, FittingSpec

# Geometry
from n_slicer.geom.transform import transform_xy, chord_pivot_norm


# --------------------------------------------------------------------------- #
# Core builder: build from a *DataFrame* that already represents the sampled
# distribution to use (columns: r_over_R, twist_deg, chord, optional name).
# This function does no sampling/fitting.
# --------------------------------------------------------------------------- #
def build_section_bin_from_dataframe(
    *,
    airfoil_csv: str,
    df: pd.DataFrame,                      # used (sampled) distribution
    label: str = "sections",
    blade_name: Optional[str] = None,
    L: float,                              # blade length [same unit as UNITS.L]
    R: float,                              # rotor tip radius [same unit as UNITS.R]
    units: Optional[SectionUnits] = None,
    pivot_chord_frac: float = 0.25,
    twist_sign: int = 1,
    keep_pivot_in_place: bool = False,
    units_scale: float = 1.0,              # chord-unit -> XY-unit scale (e.g., m->mm = 1000)
    cp_default_frac: Optional[float] = None,
    write_dxfs: bool = False,
    outdir: Optional[str] = None,
    filename: Optional[Callable[[int, float, Optional[str]], str]] = None,
    # provenance (accepted, not produced here)
    df_source: Optional[pd.DataFrame] = None,           # original distribution before resampling
    sampling_spec: Optional[DiscretisationSpec] = None,
    fitting_spec: Optional[FittingSpec] = None,
    # spanwise positions (optional, aligned with df)
    zL_grid: Optional[np.ndarray] = None,               # z/L for each sampled station
) -> SectionBin:
    """
    Build a SectionBin from a *prepared* distribution DataFrame.

    Parameters
    ----------
    airfoil_csv : str
        Path to normalised airfoil CSV (x,y in chord units).
    df : DataFrame
        Must contain columns: 'r_over_R', 'twist_deg', 'chord', optional 'name'.
        Each row becomes one station.
    L, R : float
        Blade length and rotor tip radius (units per 'units').
    units : SectionUnits, optional
        Canonical units for the bin; defaults to SI-ish (m/deg).
    zL_grid : array-like, optional
        If provided, same length/order as df; stored in the bin and on each row.
    """
    units = units or SectionUnits()
    if write_dxfs:
        if outdir is None:
            raise ValueError("outdir must be provided when write_dxfs=True.")
        os.makedirs(outdir, exist_ok=True)

    # 1) Parse airfoil & compute normalised properties once
    XY_in = NormalisedNACAParser(airfoil_csv, assume_no_header=False).parse()
    norm_props = NormalisedProperties.compute(XY_in)
    piv_xc, piv_yc = chord_pivot_norm(XY_in, pivot_chord_frac)

    # 2) Bin + distributions
    bin_ = SectionBin(label=label, blade_name=blade_name, L=L, units=units)
    bin_.cp_default_frac = cp_default_frac

    # Optional: store original/source distributions for provenance
    if df_source is not None:
        bin_.set_source_distributions(
            rR=df_source["r_over_R"].to_numpy(float),
            c=df_source["chord"].to_numpy(float),
            beta_deg=df_source["twist_deg"].to_numpy(float),
            zL=None,  # no zL at source unless caller supplies it
        )

    # Store the USED distribution (what we will actually build)
    used_rR = df["r_over_R"].to_numpy(float)
    used_c  = df["chord"].to_numpy(float)
    used_b  = df["twist_deg"].to_numpy(float)
    zL_arr  = None if zL_grid is None else np.asarray(zL_grid, float)
    bin_.set_distributions(rR=used_rR, c=used_c, beta_deg=used_b, zL=zL_arr)

    # Attach provenance specs (passed in by caller)
    bin_.sampling = sampling_spec
    bin_.fitting  = fitting_spec

    # default filename template
    if filename is None:
        filename = lambda idx, rR, name: f"{(name or f'sec_{idx:03d}')}_r{rR:.3f}.dxf"

    # 3) Build rows from the USED distribution
    for i, row in df.reset_index(drop=True).iterrows():
        rR = float(row["r_over_R"])
        c  = float(row["chord"])
        b  = float(row["twist_deg"])
        name = str(row["name"]) if "name" in df.columns and isinstance(row["name"], str) else None
        zL_val = None if zL_arr is None else float(zL_arr[i])

        # Transform geometry once for this station
        XY = transform_xy(
            XY_in,
            chord=c,
            twist_deg=b,
            pivot_xc=piv_xc,
            pivot_yc=piv_yc,
            units_scale=units_scale,
            keep_pivot_in_place=keep_pivot_in_place,
            twist_sign=twist_sign,
        )

        # Optional DXF
        dxf_path = None
        if write_dxfs:
            from ezdxf import new as _dxf_new
            doc = _dxf_new(dxfversion="R2018"); msp = doc.modelspace()
            msp.add_lwpolyline(XY.tolist(), format="xy", close=True)
            dxf_path = os.path.join(outdir, filename(i, rR, name))
            doc.saveas(dxf_path)

        # Assemble row (store resolved pivot)
        sec = SectionRow(
            station_idx=int(i),
            rR=rR, c=c, beta_deg=b, R=R,
            name=name, zL=zL_val,
            L=L, units=units,
            source="airfoil_points",
            units_scale=units_scale,
            pivot_xc=piv_xc, pivot_yc=piv_yc,
            twist_sign=twist_sign, keep_pivot_in_place=keep_pivot_in_place,
            XY_in=None, XY=XY, dxf_path=dxf_path,
            norm=norm_props, scaled=None
        )

        # Analytic-from-normalised mapping (use same pivot/rotation)
        def _map_point(pt_norm):
            v = np.array(pt_norm, float) * c
            pivot = np.array([piv_xc * c, piv_yc * c], float)
            v_shift = v - pivot
            theta = np.deg2rad(twist_sign * b)
            Rm = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]], float)
            v_rot = Rm @ v_shift
            v_final = (v_rot + (pivot if keep_pivot_in_place else 0.0)) * units_scale
            return float(v_final[0]), float(v_final[1])

        scale_total = float(c * units_scale)
        sec.scaled = ScaledProperties.from_normalised(
            norm=norm_props, map_point=_map_point, scale=scale_total, cp_frac=cp_default_frac
        )
        sec.scaled.compute_geom(XY, cp_frac=cp_default_frac)
        bin_.add(sec)

    return bin_


# --------------------------------------------------------------------------- #
# Thin wrapper: read the CSV and forward to the dataframe builder without any
# sampling/fitting. Use this only when you explicitly want all CSV rows "as is".
# (Normally main.py should resample then call build_section_bin_from_dataframe.)
# --------------------------------------------------------------------------- #
def build_section_bin_from_files(
    *,
    airfoil_csv: str,
    distribution_csv: str,
    label: str = "sections",
    blade_name: Optional[str] = None,
    L: float,
    R: float,
    units: Optional[SectionUnits] = None,
    pivot_chord_frac: float = 0.25,
    twist_sign: int = 1,
    keep_pivot_in_place: bool = False,
    units_scale: float = 1.0,
    distribution_has_no_header: bool = False,
    cp_default_frac: Optional[float] = None,
    write_dxfs: bool = False,
    outdir: Optional[str] = None,
    filename: Optional[Callable[[int, float, Optional[str]], str]] = None,
) -> SectionBin:
    """
    Convenience wrapper to use the CSV rows directly (no resampling).
    For fitted/resampled workflows, prefer build_section_bin_from_dataframe().
    """
    df_src = DistributionParser(distribution_csv, assume_no_header=distribution_has_no_header).parse_dataframe()
    return build_section_bin_from_dataframe(
        airfoil_csv=airfoil_csv,
        df=df_src,
        label=label,
        blade_name=blade_name,
        L=L,
        R=R,
        units=units,
        pivot_chord_frac=pivot_chord_frac,
        twist_sign=twist_sign,
        keep_pivot_in_place=keep_pivot_in_place,
        units_scale=units_scale,
        cp_default_frac=cp_default_frac,
        write_dxfs=write_dxfs,
        outdir=outdir,
        filename=filename,
        # provenance: source = used (since we didn't resample here)
        df_source=df_src,
        sampling_spec=None,
        fitting_spec=None,
        zL_grid=None,
    )