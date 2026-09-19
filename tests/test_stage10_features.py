from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.calibration import (
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    ClinicalMatrixDirichletCalibrator,
    AdaptiveRiskControllingPredictionSets,
)
from core.preprocessor import (
    ClinicalCorruptionStressTester,
    TestTimeConsistencyStabilizer,
)
from plugins.hip_dysplasia.geometry import SubmillimeterMorphometryCalibrator
from core.age_utils import AgeConditionedFeatureGate
from models.classifier import FastQuantizedInferenceEngine


def test_calibration_metrics_and_matrix_dirichlet_calibrator():
    # Synthetic uncalibrated probabilities with overconfidence
    rng = np.random.RandomState(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Moderately miscalibrated probabilities
    raw_probs = np.array([0.25, 0.35, 0.40, 0.15, 0.30, 0.65, 0.70, 0.85, 0.90, 0.80])

    ece_raw = compute_expected_calibration_error(raw_probs, y_true, n_bins=5)
    mce_raw = compute_maximum_calibration_error(raw_probs, y_true, n_bins=5)
    bs_raw = compute_brier_score(raw_probs, y_true)

    assert ece_raw >= 0.0
    assert mce_raw >= 0.0
    assert bs_raw >= 0.0

    calibrator = ClinicalMatrixDirichletCalibrator(gamma_fn_penalty=2.5)
    calibrator.fit(raw_probs, y_true)
    assert calibrator.is_fitted is True

    calibrated_probs = calibrator.calibrate_array(raw_probs)
    assert len(calibrated_probs) == len(raw_probs)
    # Calibrated probabilities should maintain discrimination ordering
    assert np.mean(calibrated_probs[y_true == 1]) > np.mean(calibrated_probs[y_true == 0])

    bs_cal = compute_brier_score(calibrated_probs, y_true)
    assert bs_cal <= bs_raw + 0.02


def test_adaptive_risk_controlling_prediction_sets():
    rcps = AdaptiveRiskControllingPredictionSets(alpha_risk=0.01)
    
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    val_probs = np.array([0.05, 0.10, 0.12, 0.20, 0.85, 0.90, 0.95, 0.98])
    rcps.fit(val_probs, y_true)
    assert rcps.is_fitted is True

    # Clear normal
    res_norm = rcps.predict(0.08)
    assert res_norm["prediction_set"] == [0]
    assert res_norm["is_ambiguous"] is False
    assert res_norm["calibrated_label"] == "normal"

    # Clear pathology
    res_path = rcps.predict(0.92)
    assert res_path["prediction_set"] == [1]
    assert res_path["is_ambiguous"] is False
    assert res_path["calibrated_label"] == "pathology"

    # Ambiguity rate on well-separated set should be 0.0
    amb_rate = rcps.compute_ambiguity_rate(val_probs)
    assert amb_rate == 0.0


def test_clinical_corruption_stress_tester():
    stress_tester = ClinicalCorruptionStressTester(rng_seed=42)
    img = np.ones((64, 64), dtype=np.float32) * 128.0

    # Test all 12 corruptions execute without error
    for c_type in stress_tester.CORRUPTIONS:
        corrupted = stress_tester.apply_corruption(img, c_type, severity=2)
        assert corrupted.shape == img.shape
        assert not np.isnan(corrupted).any()

    # Test stress test evaluation with dummy model
    dummy_predict_fn = lambda x: float(np.mean(x) / 255.0)
    clean_prob = dummy_predict_fn(img)
    results = stress_tester.evaluate_stress_test(dummy_predict_fn, img, clean_prob, severities=[1, 2])

    assert "mean_corruption_error" in results
    assert "worst_corruption" in results
    assert "invariance_score" in results
    assert results["num_corruptions_evaluated"] == 12
    assert results["mean_corruption_error"] >= 0.0


def test_test_time_consistency_stabilizer():
    stabilizer = TestTimeConsistencyStabilizer(variance_threshold=0.05)
    img = np.ones((64, 64), dtype=np.float32) * 100.0

    dummy_predict_fn = lambda x: float(np.clip(np.mean(x) / 100.0 * 0.7, 0.0, 1.0))
    res = stabilizer.stabilize(dummy_predict_fn, img)

    assert "stabilized_probability" in res
    assert "consistency_variance" in res
    assert "is_stable" in res
    assert res["is_stable"] is True
    assert res["num_views"] >= 1


def test_submillimeter_morphometry_calibrator():
    calibrator = SubmillimeterMorphometryCalibrator(normal_acetabular_cutoff_deg=26.0)

    # 1. Hilgenreiner angle: horizontal line y=100, roof line from (50, 100) to (80, 80)
    # dx = 30, dy = -20 -> tan theta = 20 / 30 -> angle ~ 33.7 deg (dysplastic roof)
    angle_dysplastic = calibrator.compute_hilgenreiner_angle_deg(
        triradiate_pt=(50.0, 100.0),
        triradiate_opp=(200.0, 100.0),
        acetabular_rim_pt=(80.0, 80.0),
    )
    assert 30.0 < angle_dysplastic < 36.0

    # Normal shallow roof: from (50, 100) to (80, 90) -> dx = 30, dy = -10 -> angle ~ 18.4 deg
    angle_normal = calibrator.compute_hilgenreiner_angle_deg(
        triradiate_pt=(50.0, 100.0),
        triradiate_opp=(200.0, 100.0),
        acetabular_rim_pt=(80.0, 90.0),
    )
    assert 16.0 < angle_normal < 21.0

    # 2. PCK evaluation
    target_landmarks = {"triradiate_l": (50.0, 100.0), "triradiate_r": (150.0, 100.0), "acetabulum_l": (80.0, 85.0)}
    pred_exact = target_landmarks.copy()
    pck_res = calibrator.evaluate_pck(pred_exact, target_landmarks, alpha_threshold=0.02)
    assert pck_res["pck"] == 1.0
    assert pck_res["mean_radial_error_px"] == 0.0

    # 3. Subpixel corner refinement
    grad_mag = np.zeros((32, 32), dtype=np.float32)
    # Peak at (15, 15)
    grad_mag[15, 15] = 10.0
    grad_mag[15, 14] = 6.0
    grad_mag[15, 16] = 7.0
    grad_mag[14, 15] = 6.0
    grad_mag[16, 15] = 7.0

    refined = calibrator.refine_subpixel_corner((14.8, 14.9), grad_mag, search_radius=3)
    assert abs(refined[0] - 15.0) < 0.5
    assert abs(refined[1] - 15.0) < 0.5


def test_age_conditioned_feature_gate():
    gate = AgeConditionedFeatureGate(tau_ossification=4.5)

    assert gate.get_cohort(1.5) == "neonatal_0_3m"
    assert gate.get_cohort(4.5) == "early_infant_3_6m"
    assert gate.get_cohort(9.0) == "late_infant_6m_plus"

    w_neonatal = gate.compute_cohort_weights(1.5)
    w_late = gate.compute_cohort_weights(9.0)

    # In neonates, cartilage weight is significantly higher than bone ossification
    assert w_neonatal["w_cartilage"] > w_neonatal["w_bone"]
    # In older infants, bone ossification weight is dominant
    assert w_late["w_bone"] > w_late["w_cartilage"]

    # Feature gating
    dummy_feats = np.ones(30, dtype=np.float32)
    res_gated = gate.gate_features(dummy_feats, age_months=2.0)
    assert len(res_gated["gated_features"]) == 30
    assert res_gated["cohort"] == "neonatal_0_3m"


def test_fast_quantized_inference_engine():
    engine = FastQuantizedInferenceEngine()

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    net = SimpleNet().eval()
    sample_x = torch.randn(1, 32)

    # Test latency benchmark
    stats = engine.benchmark_latency(net, sample_x, num_warmup=2, num_runs=10)
    assert "mean_latency_ms" in stats
    assert "throughput_fps" in stats
    assert stats["is_realtime_ready"] is True
    assert stats["mean_latency_ms"] < 50.0

    # Test dynamic quantization
    q_net = engine.quantize(net)
    assert q_net is not None
    out_q = q_net(sample_x)
    assert out_q.shape == (1, 2)
