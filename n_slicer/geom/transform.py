# n_slicer/geom/transform.py

from __future__ import annotations
import numpy as np
import math

def rotation_matrix(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad); s = math.sin(theta_rad)
    return np.array([[c, -s],[s, c]], dtype=float)

def _le_te_by_x(XY: np.ndarray) -> tuple[tuple[float,float], tuple[float,float]]:
    """LE/TE by min/max x in the (normalised) airfoil coordinates."""
    i_le = int(np.argmin(XY[:, 0])); i_te = int(np.argmax(XY[:, 0]))
    LE = (float(XY[i_le, 0]), float(XY[i_le, 1]))
    TE = (float(XY[i_te, 0]), float(XY[i_te, 1]))
    return LE, TE

def chord_pivot_norm(XY_in: np.ndarray, frac: float) -> tuple[float, float]:
    """
    Return the pivot point on the straight LE→TE chord line in *normalised* coords.
    'frac' is the chordwise fraction from LE toward TE (0..1).
    """
    f = float(np.clip(frac, 0.0, 1.0))
    (x_le, y_le), (x_te, y_te) = _le_te_by_x(XY_in)
    x_p = x_le + f * (x_te - x_le)
    y_p = y_le + f * (y_te - y_le)
    return float(x_p), float(y_p)

def _ensure_closed(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Append first vertex at the end if needed; ignore if already closed."""
    poly = np.asarray(poly, dtype=float)
    if np.isnan(poly).any():
        poly = poly[~np.isnan(poly).any(axis=1)]
    if len(poly) < 3:
        return poly
    if not np.allclose(poly[0], poly[-1], atol=tol, rtol=0):
        poly = np.vstack([poly, poly[0]])
    return poly

def transform_xy(
    XY_in: np.ndarray,
    *,
    chord: float,
    twist_deg: float,
    pivot_xc: float,
    pivot_yc: float,
    units_scale: float = 1.0,
    keep_pivot_in_place: bool = False,
    twist_sign: int = 1,
    close_loop: bool = True,        # NEW: close by default
    close_tol: float = 1e-9,        # NEW: tolerance for closure
) -> np.ndarray:
    """
    Scale + twist about a *given* pivot in normalised coords (x_c,y_c).
    Returns a closed polyline if close_loop=True.
    """
    if XY_in is None or XY_in.ndim != 2 or XY_in.shape[1] != 2:
        raise ValueError("XY_in must be an (N,2) array.")
    # 1) scale to chord units
    P = XY_in * chord
    # 2) rotate about pivot (given in *normalised* coords)
    pivot = np.array([pivot_xc * chord, pivot_yc * chord], float)
    P_shift = P - pivot
    theta = math.radians(twist_sign * twist_deg)
    R = rotation_matrix(theta)
    P_rot = (R @ P_shift.T).T
    # 3) place shape (pivot at origin or kept in place)
    P_final = P_rot + (pivot if keep_pivot_in_place else 0.0)
    # 4) units scaling
    P_final = P_final * units_scale
    # 5) ensure closed if requested
    return _ensure_closed(P_final, tol=close_tol) if close_loop else P_final

def transform_xy_pivot_frac(
    XY_in: np.ndarray,
    *,
    chord: float,
    twist_deg: float,
    pivot_chord_frac: float,
    units_scale: float = 1.0,
    keep_pivot_in_place: bool = False,
    twist_sign: int = 1,
    close_loop: bool = True,        # NEW
    close_tol: float = 1e-9,        # NEW
) -> np.ndarray:
    """
    Scale + twist about the *chord-line* pivot at fraction 'pivot_chord_frac' from LE→TE.
    """
    x_c, y_c = chord_pivot_norm(XY_in, pivot_chord_frac)
    return transform_xy(
        XY_in,
        chord=chord,
        twist_deg=twist_deg,
        pivot_xc=x_c,
        pivot_yc=y_c,
        units_scale=units_scale,
        keep_pivot_in_place=keep_pivot_in_place,
        twist_sign=twist_sign,
        close_loop=close_loop,
        close_tol=close_tol,
    )
