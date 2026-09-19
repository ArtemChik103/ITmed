from __future__ import annotations

import numpy as np
import pytest

from models.calibration import InterObserverDistributionModel
from plugins.hip_dysplasia.geometry import Monocular3DAcetabularReconstructor
from core.age_utils import PediatricHipPrognosticEngine
from core.preprocessor import (
    AnatomicalCoPathologyOODScreener,
    UltraLowDoseRestorationBridge,
)
from plugins.hip_dysplasia.model import OrthopedicVisualGroundingCopilot


def test_inter_observer_distribution_model():
    model = InterObserverDistributionModel(baseline_noise_sigma=1.2)

    res = model.predict_distribution(base_angle_deg=24.0, image_quality_score=0.90)
    assert res["mean_angle_deg"] == 24.0
    assert res["std_deg"] > 0.5
    assert res["ci_95"][0] < 24.0 < res["ci_95"][1]
    
    # Screener reading should be more sensitive (higher angle) than surgeon
    readings = res["reader_readings"]
    assert readings["screener_reading"] > readings["orthopedist_reading"] >= readings["surgeon_reading"]
    assert res["consensus_agreement_pct"] > 80.0

    # Test GED
    ged_identical = model.compute_generalized_energy_distance([20, 22, 24], [20, 22, 24])
    assert ged_identical == 0.0

    ged_diff = model.compute_generalized_energy_distance([20, 22], [30, 32])
    assert ged_diff > 0.0


def test_monocular_3d_acetabular_reconstructor():
    reconstructor = Monocular3DAcetabularReconstructor()

    # 1. Normal anteversion on right hip: anterior wall x=140 is medial to posterior wall x=120
    res_norm = reconstructor.estimate_3d_cup_version(
        anterior_wall_x=140.0,
        posterior_wall_x=120.0,
        cup_diameter_px=60.0,
        side="right",
    )
    assert res_norm["crossover_sign_detected"] is False
    assert res_norm["is_version_normal"] is True
    assert 14.0 <= res_norm["estimated_anteversion_deg"] <= 24.0

    # 2. Cranial retroversion (crossover sign positive: anterior wall crossed lateral to posterior wall)
    res_retro = reconstructor.estimate_3d_cup_version(
        anterior_wall_x=110.0,
        posterior_wall_x=130.0,
        cup_diameter_px=60.0,
        side="right",
    )
    assert res_retro["crossover_sign_detected"] is True
    assert res_retro["version_status"] == "cranial_retroversion_crossover_positive"
    assert res_retro["is_version_normal"] is False


def test_pediatric_hip_prognostic_engine():
    engine = PediatricHipPrognosticEngine(high_risk_threshold=0.60)

    # 1. Favorable prognosis: 3-month infant, contained hip (Reimers 15%), normal angle
    res_good = engine.predict_treatment_failure_risk(
        age_months=3.0,
        reimers_index_pct=15.0,
        acetabular_angle_deg=22.0,
        tonnis_grade=0,
    )
    assert res_good["is_high_risk"] is False
    assert res_good["treatment_failure_risk"] < 0.20
    assert "spontaneous_resolution" in res_good["prognostic_trajectory"]

    # 2. Poor prognosis: 9-month infant, severe migration (Reimers 65%), steep roof 38 deg, Tonnis Grade 3
    res_poor = engine.predict_treatment_failure_risk(
        age_months=9.0,
        reimers_index_pct=65.0,
        acetabular_angle_deg=38.0,
        tonnis_grade=3,
    )
    assert res_poor["is_high_risk"] is True
    assert res_poor["treatment_failure_risk"] > 0.60
    assert "surgical_reduction" in res_poor["prognostic_trajectory"]

    # 3. Harrell's C-Index
    risks = [0.10, 0.30, 0.75, 0.90]
    events = [0, 0, 1, 1]
    c_index = engine.compute_c_index(risks, events)
    assert c_index == 1.0


def test_anatomical_co_pathology_ood_screener():
    screener = AnatomicalCoPathologyOODScreener(ood_threshold=0.50)

    # 1. Normal anatomy
    res_norm = screener.screen_co_pathology(
        femoral_neck_shaft_angle_deg=130.0,
        femoral_head_density_std=15.0,
        physis_width_px=4.0,
    )
    assert res_norm["has_co_pathology"] is False
    assert res_norm["is_pure_dysplasia_candidate"] is True

    # 2. Coxa Vara (NSA 108 deg < 120 deg)
    res_vara = screener.screen_co_pathology(femoral_neck_shaft_angle_deg=108.0)
    assert res_vara["has_co_pathology"] is True
    assert any("Coxa_Vara" in a for a in res_vara["suspected_co_pathologies"])

    # 3. Suspected Perthes avascular necrosis (high sclerosis density std > 35)
    res_perthes = screener.screen_co_pathology(femoral_head_density_std=45.0)
    assert res_perthes["has_co_pathology"] is True
    assert any("Perthes" in a for a in res_perthes["suspected_co_pathologies"])


def test_ultra_low_dose_restoration_bridge():
    bridge = UltraLowDoseRestorationBridge()
    rng = np.random.RandomState(42)
    clean = np.ones((64, 64), dtype=np.float32) * 120.0
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.float32)

    res = bridge.restore_low_dose_image(noisy, sharpening_gain=0.30)
    assert res["is_alara_restored"] is True
    assert res["psnr_db"] > 25.0
    assert res["noise_reduction_factor"] >= 1.0


def test_orthopedic_visual_grounding_copilot():
    copilot = OrthopedicVisualGroundingCopilot()

    landmarks = {
        "femur_r": (100.0, 160.0),
        "acetabulum_r": (100.0, 120.0),
        "triradiate_r": (140.0, 150.0),
        "triradiate_l": (220.0, 150.0),
    }

    # Query 1: Shenton's arc
    res_shenton = copilot.query_grounding("Show me Shenton arc on the right hip", landmarks, side="right")
    assert res_shenton["target_structure"] == "shenton_arc"
    assert len(res_shenton["grounded_bbox_ymin_xmin_ymax_xmax"]) == 4
    assert "Shenton" in res_shenton["clinical_explanation"]

    # Query 2: Perkins line
    res_perkins = copilot.query_grounding("Where is Perkins vertical line?", landmarks, side="right")
    assert res_perkins["target_structure"] == "perkins_line"

    # Query 3: Hilgenreiner baseline
    res_hilg = copilot.query_grounding("Hilgenreiner baseline position", landmarks)
    assert res_hilg["target_structure"] == "hilgenreiner_line"
