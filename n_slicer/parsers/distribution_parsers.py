# n_slicer/parsers/distribution_parsers.py
from __future__ import annotations
import pandas as pd
from typing import Optional, Tuple
from .base import BaseCSVParser, norm_colname

def _identify_distribution_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    candidates = {norm_colname(c): c for c in df.columns}

    r_keys = [k for k in candidates if ("r" in k and ("_over_" in k or "ratio" in k)) or k in {"r","r_over_r"}]

    preferred_twist = ["twist_deg","twist","beta_deg","b_deg","beta","b","theta_deg","theta","deg"]
    twist_col = None
    for pref in preferred_twist:
        key = norm_colname(pref)
        if key in candidates:
            twist_col = candidates[key]; break
    if twist_col is None:
        twist_keys = [k for k in candidates if any(t in k for t in ["twist","beta","b_deg","b","theta","deg"])]
        if twist_keys:
            twist_col = candidates[twist_keys[0]]

    chord_keys = [k for k in candidates if "chord" in k]
    name_keys  = [k for k in candidates if any(s in k for s in ["name","station","id"])]

    r_col     = candidates[r_keys[0]] if r_keys else None
    chord_col = candidates[chord_keys[0]] if chord_keys else None
    name_col  = candidates[name_keys[0]]  if name_keys  else None

    return r_col, twist_col, chord_col, name_col


class DistributionParser(BaseCSVParser):
    """
    Parses a spanwise CSV that may contain r/R, twist(deg), chord, and optional 'name'.
    Robust to:
      - leading '#' metadata lines
      - header row like '# r/R [-], # B [deg], # Chord [m]'
      - files with NO header at all (first data row is numeric)
    """
    def parse_dataframe(self) -> pd.DataFrame:
        # First attempt: best-effort read that strips '#'-comment lines
        df = super()._read_csv_best_effort()

        # Try to identify columns by name
        r_col, twist_col, chord_col, name_col = _identify_distribution_columns(df)

        if r_col is None or twist_col is None or chord_col is None:
            # Fallback path: treat as headerless [r/R, twist_deg, chord]
            df = pd.read_csv(self.csv_path, comment="#", header=None)
            if df.shape[1] < 3:
                raise ValueError("Distribution CSV must have at least 3 columns: r/R, twist_deg, chord.")
            df = df.iloc[:, :3]
            df.columns = ["r_over_R", "twist_deg", "chord"]
        else:
            # Standardize columns we actually have
            rename = {r_col: "r_over_R"}
            if twist_col: rename[twist_col] = "twist_deg"
            if chord_col: rename[chord_col] = "chord"
            if name_col:  rename[name_col]   = "name"
            df = df.rename(columns=rename)

        # Coerce numerics & drop invalid rows
        for c in ("r_over_R","twist_deg","chord"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if {"r_over_R","twist_deg","chord"}.issubset(df.columns):
            df = df.dropna(subset=["r_over_R","twist_deg","chord"]).reset_index(drop=True)
        else:
            df = df.dropna(subset=["r_over_R"]).reset_index(drop=True)
        return df


class TwistDistributionParser(DistributionParser):
    """Returns a DataFrame with r_over_R, twist_deg, and optional name."""
    def parse_dataframe(self) -> pd.DataFrame:
        df = super().parse_dataframe()
        needed = ["r_over_R","twist_deg"]
        if not set(needed).issubset(df.columns):
            missing = [k for k in needed if k not in df.columns]
            raise ValueError(f"Twist distribution missing: {missing}")
        cols = ["r_over_R","twist_deg"] + (["name"] if "name" in df.columns else [])
        return df[cols].copy()


class ChordDistributionParser(DistributionParser):
    """Returns a DataFrame with r_over_R, chord, and optional name."""
    def parse_dataframe(self) -> pd.DataFrame:
        df = super().parse_dataframe()
        needed = ["r_over_R","chord"]
        if not set(needed).issubset(df.columns):
            missing = [k for k in needed if k not in df.columns]
            raise ValueError(f"Chord distribution missing: {missing}")
        cols = ["r_over_R","chord"] + (["name"] if "name" in df.columns else [])
        return df[cols].copy()