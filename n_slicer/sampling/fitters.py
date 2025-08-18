# n_slicer/sampling/fitters.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Literal, Optional

# Optional SciPy: if unavailable, we fall back to linear
try:
    from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, UnivariateSpline
    _HAS_SCIPY = True
except Exception:
    PchipInterpolator = Akima1DInterpolator = UnivariateSpline = None  # type: ignore
    _HAS_SCIPY = False

FitKind = Literal["pchip", "akima", "spline", "linear"]

def _prep_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x, and deduplicate identical x by averaging y."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # deduplicate
    if np.unique(x).size != x.size:
        out_x, out_y = [], []
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and np.isclose(x[j], x[i]):
                j += 1
            out_x.append(float(np.mean(x[i:j])))
            out_y.append(float(np.mean(y[i:j])))
            i = j
        return np.asarray(out_x), np.asarray(out_y)
    return x, y

def fit_1d(
    x: np.ndarray,
    y: np.ndarray,
    kind: FitKind = "pchip",
    smoothing: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return f(new_x) that evaluates a smooth (or linear) fit of y(x).

    - 'pchip'  : monotone, shape-preserving cubic (great for chord/twist)
    - 'akima'  : smooth, handles wiggles
    - 'spline' : UnivariateSpline(s=smoothing) if SciPy present
    - 'linear' : always available fallback
    """
    x, y = _prep_xy(x, y)

    if _HAS_SCIPY:
        if kind == "pchip":
            f = PchipInterpolator(x, y, extrapolate=True)
            return lambda xx: np.asarray(f(xx), float)
        if kind == "akima":
            f = Akima1DInterpolator(x, y)
            return lambda xx: np.asarray(f(xx), float)
        if kind == "spline":
            s = 0.0 if smoothing is None else float(smoothing)
            k = 3 if x.size >= 4 else max(1, min(3, x.size - 1))
            f = UnivariateSpline(x, y, s=s, k=k)
            return lambda xx: np.asarray(f(xx), float)

    # fallback: linear interpolation with flat extrapolation
    def _lin(xx: np.ndarray) -> np.ndarray:
        xx = np.asarray(xx, float)
        return np.interp(xx, x, y, left=y[0], right=y[-1])
    return _lin

def sample_distribution_df(
    df_in: pd.DataFrame,
    n: int,
    rR_min: float | None = None,
    rR_max: float | None = None,
    *,
    rR_grid: np.ndarray | None = None,
    chord_fit: FitKind = "pchip",
    twist_fit: FitKind = "pchip",
    spline_smoothing: float | None = None,
) -> pd.DataFrame:
    """
    Fit chord(r/R) and twist(r/R) then sample N points on a provided rR grid (or uniform if None).
    Returns DataFrame with columns: r_over_R, twist_deg, chord.
    """
    if n < 2 and rR_grid is None:
        raise ValueError("n must be >= 2 unless an rR_grid is provided.")

    r = df_in["r_over_R"].to_numpy(float)
    c = df_in["chord"].to_numpy(float)
    b = df_in["twist_deg"].to_numpy(float)

    rmin = float(np.min(r)) if rR_min is None else float(rR_min)
    rmax = float(np.max(r)) if rR_max is None else float(rR_max)

    if rR_grid is None:
        rR_grid = np.linspace(rmin, rmax, n)
    else:
        rR_grid = np.clip(np.asarray(rR_grid, float), rmin, rmax)

    f_c = fit_1d(r, c, kind=chord_fit, smoothing=spline_smoothing)
    f_b = fit_1d(r, b, kind=twist_fit, smoothing=spline_smoothing)

    return pd.DataFrame({
        "r_over_R": rR_grid,
        "twist_deg": f_b(rR_grid),
        "chord": f_c(rR_grid),
    })
