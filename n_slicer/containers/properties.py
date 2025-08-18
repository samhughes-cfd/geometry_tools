# n_slicer/containers/properties.py
"""
properties.py
-------------
'NormalisedProperties' are computed once on XY_in (normalised shape).
'ScaledProperties' holds:
  (a) values analytically mapped from the normalised invariants, and
  (b) exact values recomputed on XY (the transformed polyline).

This avoids re-doing polygon work per station when only scale/rotation change,
while still letting you compute exact values when you have XY.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

# --- small helpers ---
def _ensure_closed(XY: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    if XY.shape[0] < 2: return XY
    return XY if np.allclose(XY[0], XY[-1], atol=atol) else np.vstack([XY, XY[0]])

def _poly_perimeter(XY: np.ndarray) -> float:
    P = _ensure_closed(XY)
    d = np.diff(P, axis=0)
    return float(np.sqrt((d**2).sum(axis=1)).sum())

def _poly_area_centroid(XY: np.ndarray) -> Tuple[float, float, float]:
    P = _ensure_closed(XY)
    x, y = P[:,0], P[:,1]
    cross = x[:-1]*y[1:] - y[:-1]*x[1:]
    A2 = float(cross.sum())
    A = 0.5 * A2
    if abs(A2) < 1e-16:
        return A, float(np.mean(x)), float(np.mean(y))
    cx = float(((x[:-1] + x[1:]) * cross).sum() / (3.0 * A2))
    cy = float(((y[:-1] + y[1:]) * cross).sum() / (3.0 * A2))
    return A, cx, cy

def _edge_length_stats(XY: np.ndarray) -> Dict[str, float]:
    P = _ensure_closed(XY)
    seg = np.diff(P, axis=0)
    L = np.sqrt((seg**2).sum(axis=1))
    return {
        "edge_len_min": float(L.min()),
        "edge_len_max": float(L.max()),
        "edge_len_mean": float(L.mean()),
    }

def _le_te_by_x(XY: np.ndarray) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    i_le = int(np.argmin(XY[:,0]))
    i_te = int(np.argmax(XY[:,0]))
    return (float(XY[i_le,0]), float(XY[i_le,1])), (float(XY[i_te,0]), float(XY[i_te,1]))

def _point_on_chord(LE, TE, frac: float):
    f = float(np.clip(frac, 0.0, 1.0))
    le, te = np.array(LE,float), np.array(TE,float)
    v = te - le
    if np.linalg.norm(v) <= 1e-15: return None
    p = le + f*v
    return float(p[0]), float(p[1])

# --- models ---
@dataclass
class NormalisedProperties:
    """Computed once on XY_in (normalised). Units: chord units."""
    n_vertices: int
    closed: bool
    P: float
    A: float
    cx: float
    cy: float
    xmin: float; xmax: float
    ymin: float; ymax: float
    edge_len_min: float; edge_len_max: float; edge_len_mean: float
    x_le: float; y_le: float
    x_te: float; y_te: float

    @staticmethod
    def compute(XY_in: np.ndarray) -> "NormalisedProperties":
        if XY_in is None or XY_in.size == 0:
            raise ValueError("XY_in is empty.")
        X = XY_in
        n_vertices = int(X.shape[0])
        closed = bool(np.allclose(X[0], X[-1], atol=1e-12))
        P = _poly_perimeter(X)
        A, cx, cy = _poly_area_centroid(X)
        xmin, xmax = float(X[:,0].min()), float(X[:,0].max())
        ymin, ymax = float(X[:,1].min()), float(X[:,1].max())
        stats = _edge_length_stats(X)
        (x_le, y_le), (x_te, y_te) = _le_te_by_x(X)
        return NormalisedProperties(
            n_vertices, closed, P, A, cx, cy,
            xmin, xmax, ymin, ymax,
            stats["edge_len_min"], stats["edge_len_max"], stats["edge_len_mean"],
            x_le, y_le, x_te, y_te
        )

@dataclass
class ScaledFromNormalised:
    """Analytically mapped from NormalisedProperties (scale/rotate/translate)."""
    P: float; A: float
    cx: float; cy: float
    x_le: float; y_le: float
    x_te: float; y_te: float
    edge_len_min: float; edge_len_max: float; edge_len_mean: float
    # optional CP
    x_cp: Optional[float] = None; y_cp: Optional[float] = None

@dataclass
class ScaledGeometryProps:
    """Exact values recomputed on XY."""
    n_vertices: int; closed: bool
    P: float; A: float; cx: float; cy: float
    xmin: float; xmax: float; ymin: float; ymax: float
    edge_len_min: float; edge_len_max: float; edge_len_mean: float
    te_gap: float
    x_le: float; y_le: float; x_te: float; y_te: float
    # optional CP
    x_cp: Optional[float] = None; y_cp: Optional[float] = None

@dataclass
class ScaledProperties:
    """Bundle of scaled properties: 'from_norm' (analytic) and 'geom' (exact)."""
    from_norm: Optional[ScaledFromNormalised] = None
    geom: Optional[ScaledGeometryProps] = None

    @staticmethod
    def from_normalised(
        norm: NormalisedProperties,
        *,
        map_point,      # callable: (x_norm,y_norm)->(x,y) after full transform
        scale: float,   # total scalar scale = chord*units_scale
        cp_frac: Optional[float] = None,
    ) -> "ScaledProperties":
        # scale-only invariants
        P = norm.P * scale
        A = norm.A * (scale**2)
        e_min, e_max, e_mean = norm.edge_len_min*scale, norm.edge_len_max*scale, norm.edge_len_mean*scale
        # map points
        cx, cy = map_point((norm.cx, norm.cy))
        x_le, y_le = map_point((norm.x_le, norm.y_le))
        x_te, y_te = map_point((norm.x_te, norm.y_te))
        x_cp = y_cp = None
        if cp_frac is not None:
            # CoP along the chord in normalised space
            le_n = np.array([norm.x_le, norm.y_le], float)
            te_n = np.array([norm.x_te, norm.y_te], float)
            p_n  = le_n + float(np.clip(cp_frac, 0.0, 1.0)) * (te_n - le_n)
            x_cp, y_cp = map_point((p_n[0], p_n[1]))
        return ScaledProperties(
            from_norm=ScaledFromNormalised(
                P=P, A=A, cx=cx, cy=cy, x_le=x_le, y_le=y_le, x_te=x_te, y_te=y_te,
                edge_len_min=e_min, edge_len_max=e_max, edge_len_mean=e_mean,
                x_cp=x_cp, y_cp=y_cp
            ),
            geom=None
        )

    def compute_geom(self, XY: np.ndarray, cp_frac: Optional[float] = None) -> None:
        if XY is None or XY.size == 0:
            self.geom = None; return
        n_vertices = int(XY.shape[0])
        closed = bool(np.allclose(XY[0], XY[-1], atol=1e-12))
        P = _poly_perimeter(XY)
        A, cx, cy = _poly_area_centroid(XY)
        xmin, xmax = float(XY[:,0].min()), float(XY[:,0].max())
        ymin, ymax = float(XY[:,1].min()), float(XY[:,1].max())
        stats = _edge_length_stats(XY)
        te_gap = float(np.linalg.norm(XY[-1] - XY[0]))
        (x_le, y_le), (x_te, y_te) = _le_te_by_x(XY)
        x_cp = y_cp = None
        if cp_frac is not None:
            pt = _point_on_chord((x_le,y_le),(x_te,y_te), cp_frac)
            if pt is not None: x_cp, y_cp = pt
        self.geom = ScaledGeometryProps(
            n_vertices, closed, P, A, cx, cy,
            xmin, xmax, ymin, ymax,
            stats["edge_len_min"], stats["edge_len_max"], stats["edge_len_mean"],
            te_gap, x_le, y_le, x_te, y_te, x_cp, y_cp
        )