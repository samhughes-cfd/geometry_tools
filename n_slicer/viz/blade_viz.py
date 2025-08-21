# n_slicer/viz/blade_viz.py
from __future__ import annotations
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless-safe

from n_slicer.viz.stack3d_plotter import Stack3DPlotter
from n_slicer.viz.sections2d_plotter import SectionsGridPlotter
from n_slicer.viz.chord_plotter import ChordVsZPlotter
from n_slicer.viz.beta_plotter import BetaVsZPlotter
from n_slicer.containers.section_bin import SectionBin


@dataclass
class BladeVisualizer:
    bin: SectionBin
    outdir: str

    def plot_stack_3d(self, **kwargs) -> str:
        return Stack3DPlotter(self.bin, self.outdir, **kwargs).plot()

    def plot_sections_2d(self, **kwargs) -> str:
        return SectionsGridPlotter(self.bin, self.outdir, **kwargs).plot()

    def plot_chord_vs_z(self, **kwargs) -> str:
        return ChordVsZPlotter(self.bin, self.outdir, **kwargs).plot()

    def plot_beta_vs_z(self, **kwargs) -> str:
        return BetaVsZPlotter(self.bin, self.outdir, **kwargs).plot()
