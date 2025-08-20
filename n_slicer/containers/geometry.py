# n_slicer/containers/geometry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Union
import numpy as np

Number = Union[float, np.ndarray]

@dataclass
class BladeGeometryBin:
    R: float   # Rotor tip radius [m]
    L: float   # Blade length [m]

    r_locii: Dict[str, float] = field(init=False)
    z_locii: Dict[str, float] = field(init=False)
    rR_locii: Dict[str, float] = field(init=False)
    zL_locii: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if not (self.R > 0 and self.L > 0 and self.L < self.R):
            raise ValueError("Invalid geometry: require R>0, L>0, and L<R.")
        self._compute_fixed_locii()

    @property
    def R_hub(self) -> float:
        return self.R - self.L

    def _compute_fixed_locii(self) -> None:
        self.rR_locii = {"rotor_origin": 0.0, "rotor_hub": self.R_hub / self.R, "rotor_tip": 1.0}
        self.zL_locii = {"blade_root": 0.0, "blade_tip": 1.0}
        self.r_locii = {k: v * self.R for k, v in self.rR_locii.items()}
        self.z_locii = {k: v * self.L for k, v in self.zL_locii.items()}

    # z is span from blade root; r is radial from rotor axis
    def r_from_z(self, z: Number) -> Number:          return np.asarray(z) + self.R_hub
    def z_from_r(self, r: Number) -> Number:          return np.asarray(r) - self.R_hub
    def rR_from_z(self, z: Number) -> Number:         return self.r_from_z(z) / self.R
    def zL_from_r(self, r: Number) -> Number:         return self.z_from_r(r) / self.L
    def rR_from_zL(self, zL: Number) -> Number:       return self.r_from_z(np.asarray(zL) * self.L) / self.R
    def zL_from_rR(self, rR: Number) -> Number:       return self.z_from_r(np.asarray(rR) * self.R) / self.L
