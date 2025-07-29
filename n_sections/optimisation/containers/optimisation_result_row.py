# containers/optimisation/optimisation_result_row.py

from dataclasses import dataclass
from n_sections.analysis.containers.geometric_property_analysis_bin import GeometricPropertyAnalysisBin


@dataclass
class OptimisationResultRow:
    run_label: str
    mesh_h: float
    scale_factor: float

    area_mm2: float
    Jt_mm4: float
    Iz_mm4: float

    target_Jt_mm4: float
    target_Iz_mm4: float

    delta_Jt_mm4: float
    delta_Iz_mm4: float

    passed: bool

    # Derived metrics (optional; populated in container logic)
    Jt_ratio: float = None
    Iz_ratio: float = None
    Area_reduction_pct: float = None
    Jt_efficiency: float = None
    Iz_efficiency: float = None
    Slack_Jt: float = None
    Excess_ratio_Jt: float = None
    Slack_Iz: float = None
    Excess_ratio_Iz: float = None

    @classmethod
    def from_analysis_bin(
        cls,
        analysis_bin: GeometricPropertyAnalysisBin,
        run_label: str,
        scale_factor: float,
        target_Jt_mm4: float,
        target_Iz_mm4: float,
    ) -> "OptimisationResultRow":
        Jt = analysis_bin.j_mm4
        Iz = analysis_bin.i2_mm4
        area = analysis_bin.area_mm2
        mesh_h = analysis_bin.mesh_h

        delta_Jt = Jt - target_Jt_mm4
        delta_Iz = Iz - target_Iz_mm4
        passed = Jt >= target_Jt_mm4 and Iz >= target_Iz_mm4

        return cls(
            run_label=run_label,
            mesh_h=mesh_h,
            scale_factor=scale_factor,
            area_mm2=area,
            Jt_mm4=Jt,
            Iz_mm4=Iz,
            target_Jt_mm4=target_Jt_mm4,
            target_Iz_mm4=target_Iz_mm4,
            delta_Jt_mm4=delta_Jt,
            delta_Iz_mm4=delta_Iz,
            passed=passed,
        )
