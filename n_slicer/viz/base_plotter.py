from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import numpy as np

try:
    import ezdxf
    _HAS_EZDXF = True
except Exception:
    _HAS_EZDXF = False

from n_slicer.containers.section_bin import SectionBin
from n_slicer.containers.section_row import SectionRow
from n_slicer.containers.units import length_scale


@dataclass
class _BaseBladePlotter:
    bin: SectionBin
    outdir: str

    def _valid_rows(self) -> List[SectionRow]:
        out = []
        for r in self.bin.rows:
            if r.XY is not None and getattr(r.XY, "size", 0) > 0:
                out.append(r)
            elif (r.dxf_path is not None) and _HAS_EZDXF:
                out.append(r)
        out.sort(key=lambda rr: getattr(rr, "rR", 0.0))
        return out

    def _get_XY(self, r: SectionRow, decimate_every: Optional[int]) -> Optional[np.ndarray]:
        if r.XY is not None and getattr(r.XY, "size", 0) > 0:
            XY = np.asarray(r.XY, float)
        elif r.dxf_path and _HAS_EZDXF:
            XY = self._read_dxf_polyline(r.dxf_path)
            if XY is None:
                return None
        else:
            return None

        if decimate_every and decimate_every > 1 and XY.shape[0] > decimate_every:
            XY = XY[::decimate_every, :]
            if not np.allclose(XY[0], XY[-1], atol=1e-9):
                XY = np.vstack([XY, XY[0]])
        return XY

    def _read_dxf_polyline(self, path: str) -> Optional[np.ndarray]:
        try:
            doc = ezdxf.readfile(path)
            msp = doc.modelspace()
            ents = list(msp.query("LWPOLYLINE POLYLINE"))
            if not ents:
                return None
            e = ents[0]
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
                closed = bool(e.closed)
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                closed = bool(e.is_closed)
            XY = np.asarray(pts, float)
            if closed and not np.allclose(XY[0], XY[-1], atol=1e-9):
                XY = np.vstack([XY, XY[0]])
            return XY
        except Exception:
            return None

    def _z_for_row(self, r: SectionRow) -> float:
        L = float(self.bin.L or 0.0)
        R = float(getattr(r, "R", 0.0))
        if L <= 0 or R <= 0:
            return 0.0
        if getattr(r, "zL", None) is not None:
            return float(r.zL) * L
        rR = float(r.rR)
        return rR * R - (R - L)

    def _z_axis_for_dist(self) -> np.ndarray:
        rR = np.asarray(self.bin.rR_dist, float)
        L = float(self.bin.L or 0.0)
        if getattr(self.bin, "zL_dist", np.empty(0)).size == rR.size and L > 0:
            return np.asarray(self.bin.zL_dist, float) * L
        if getattr(self.bin, "xL_dist", np.empty(0)).size == rR.size and L > 0:
            return np.asarray(self.bin.xL_dist, float) * L
        if not self.bin.rows:
            return np.zeros_like(rR)
        R = float(getattr(self.bin.rows[0], "R", 0.0))
        if L <= 0 or R <= 0:
            return np.zeros_like(rR)
        return rR * R - (R - L)

    def _z_axis_for_source(self) -> np.ndarray:
        rR_src = np.asarray(self.bin.rR_src, float)
        L = float(self.bin.L or 0.0)
        if rR_src.size == 0:
            return self._z_axis_for_dist()
        if getattr(self.bin, "zL_src", np.empty(0)).size == rR_src.size and L > 0:
            return np.asarray(self.bin.zL_src, float) * L
        if not self.bin.rows:
            return np.zeros_like(rR_src)
        R = float(getattr(self.bin.rows[0], "R", 0.0))
        if L <= 0 or R <= 0:
            return np.zeros_like(rR_src)
        return rR_src * R - (R - L)

    def _title(self) -> str:
        name = self.bin.blade_name or "Blade"
        if self.bin.L:
            Ltxt = f"L={self.bin.L:g}{self.bin.units.L}"
        else:
            Ltxt = "L=unknown"
        return f"{name} — sections={len(self.bin.rows)}, {Ltxt}, physical z stacking"
