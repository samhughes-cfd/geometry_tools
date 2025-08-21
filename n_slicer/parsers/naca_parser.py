from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseCSVParser, norm_colname

class NormalisedNACAParser(BaseCSVParser):
    """
    Reads an airfoil CSV with columns x,y (normalized) or a headerless two-column file.
    Robust to a comment title line starting with '#'.
    """
    def parse(self) -> np.ndarray:
        if self.assume_no_header:
            df = pd.read_csv(self.csv_path, comment="#", header=None)
            if df.shape[1] < 2:
                raise ValueError("Airfoil CSV without header must have at least two columns (x,y).")
            df = df.iloc[:, :2]
            df.columns = ["x", "y"]
        else:
            try:
                df = pd.read_csv(self.csv_path, comment="#")
            except Exception:
                df = pd.read_csv(self.csv_path)
            cols = {norm_colname(c): c for c in df.columns}
            xcol = cols.get("x")
            ycol = cols.get("y")
            if xcol is None or ycol is None:
                # Maybe truly headerless — assume first two columns
                df = pd.read_csv(self.csv_path, comment="#", header=None)
                if df.shape[1] < 2:
                    raise ValueError("Airfoil CSV must have at least two columns for x,y.")
                df.columns = ["x", "y"] + [f"extra_{i}" for i in range(df.shape[1]-2)]
                xcol, ycol = "x", "y"
            df = df[[xcol, ycol]]

        xy = df.to_numpy(dtype=float)

        # Defensive clean-up: drop any rows with NaNs and verify shape.
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Airfoil CSV must yield an Nx2 array (x,y).")
        if not np.isfinite(xy).all():
            xy = xy[np.isfinite(xy).all(axis=1)]

        return xy
