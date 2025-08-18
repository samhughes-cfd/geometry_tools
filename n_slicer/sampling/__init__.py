# n_slicer/sampling/__init__.py
from .discretise import make_rR_grid
from .fitters import fit_1d, sample_distribution_df

__all__ = ["make_rR_grid", "fit_1d", "sample_distribution_df"]