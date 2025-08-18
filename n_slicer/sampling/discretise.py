# n_slicer/sampling/discretise.py
from __future__ import annotations
import numpy as np
from typing import Callable, Literal, Optional

Scheme = Literal["uniform", "cosine", "power_root", "power_tip"]

def make_rR_grid(
    n: int,
    *,
    scheme: Scheme = "uniform",
    power: float = 2.0,
    rR_min: float = 0.0,
    rR_max: float = 1.0,
    mapping: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Build an r/R grid of length n in [rR_min, rR_max].

    - 'uniform'   : equal spacing
    - 'cosine'    : Chebyshev-style clustering toward both ends
    - 'power_root': cluster toward root with exponent 'power'  => u**power
    - 'power_tip' : cluster toward tip  with exponent 'power'  => 1-(1-u)**power
    -  mapping     : custom u->r/R callable (u in [0,1]) overrides scheme
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    a, b = float(rR_min), float(rR_max)
    u = np.linspace(0.0, 1.0, n)

    if mapping is not None:
        r = np.asarray(mapping(u), float)
    elif scheme == "uniform":
        r = u
    elif scheme == "cosine":
        r = 0.5 * (1.0 - np.cos(np.pi * u))
    elif scheme == "power_root":
        r = u**float(power)
    elif scheme == "power_tip":
        r = 1.0 - (1.0 - u)**float(power)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    r = a + (b - a) * np.clip(r, 0.0, 1.0)
    return np.asarray(r, float)