# n_slicer/containers/section_bin.py

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np

from .units import SectionUnits
from .section_row import SectionRow
from .sampling import DiscretisationSpec, FittingSpec

@dataclass
class SectionBin:
    label: str
    blade_name: Optional[str] = None
    L: Optional[float] = None
    units: SectionUnits = field(default_factory=SectionUnits)

    # sampled/used distributions (aligned with rows)
    rR_dist: np.ndarray = field(default_factory=lambda: np.empty(0))
    zL_dist: np.ndarray = field(default_factory=lambda: np.empty(0))   # ← renamed
    c_dist:  np.ndarray = field(default_factory=lambda: np.empty(0))
    beta_dist_deg: np.ndarray = field(default_factory=lambda: np.empty(0))

    # optional source/original distributions
    rR_src: np.ndarray = field(default_factory=lambda: np.empty(0))
    zL_src: np.ndarray = field(default_factory=lambda: np.empty(0))    # ← renamed
    c_src:  np.ndarray = field(default_factory=lambda: np.empty(0))
    beta_src_deg: np.ndarray = field(default_factory=lambda: np.empty(0))

    sampling: Optional[DiscretisationSpec] = None
    fitting:  Optional[FittingSpec]        = None

    cp_default_frac: Optional[float] = None

    rows: List[SectionRow] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- API ------------------------------------------------------------------

    def set_source_distributions(self, rR, c, beta_deg, zL: Optional[np.ndarray] = None, sort_by_rR: bool = True) -> None:
        """Store the *original* distributions read from disk (pre-fit, pre-resample)."""
        rR = np.asarray(rR, float); c = np.asarray(c, float); b = np.asarray(beta_deg, float)
        z = None if zL is None else np.asarray(zL, float)
        if sort_by_rR:
            order = np.argsort(rR)
            rR, c, b = rR[order], c[order], b[order]
            if z is not None: z = z[order]
        self.rR_src, self.c_src, self.beta_src_deg = rR, c, b
        self.zL_src = np.empty(0) if z is None else z

    def set_distributions(self, rR, c, beta_deg, zL: Optional[np.ndarray] = None, sort_by_rR: bool = True) -> None:
        """Store the *sampled/used* distributions that rows follow."""
        rR = np.asarray(rR, float); c = np.asarray(c, float); b = np.asarray(beta_deg, float)
        z = None if zL is None else np.asarray(zL, float)
        if sort_by_rR:
            order = np.argsort(rR)
            rR, c, b = rR[order], c[order], b[order]
            if z is not None: z = z[order]
        self.rR_dist, self.c_dist, self.beta_dist_deg = rR, c, b
        self.zL_dist = np.empty(0) if z is None else z

    def add(self, row: SectionRow) -> None:
        row.units = self.units
        self.rows.append(row)

    def summary(self) -> Dict[str, Any]:
        def rng(a: np.ndarray): return None if a.size == 0 else (float(a.min()), float(a.max()))
        out = {
            "label": self.label, "blade_name": self.blade_name, "L": self.L,
            "units": asdict(self.units),
            "n_sections": len(self.rows),
            # sampled (used)
            "rR_range": rng(self.rR_dist), "zL_range": rng(self.zL_dist),   # ← renamed
            "c_range": rng(self.c_dist),  "beta_range_deg": rng(self.beta_dist_deg),
            # source (if present)
            "rR_src_range": rng(self.rR_src), "c_src_range": rng(self.c_src),
            "beta_src_range_deg": rng(self.beta_src_deg),
            "cp_default_frac": None if self.cp_default_frac is None else float(self.cp_default_frac),
            **self.meta,
        }
        if self.sampling: out["sampling"] = self.sampling.to_dict()
        if self.fitting:  out["fitting"]  = self.fitting.to_dict()
        return out