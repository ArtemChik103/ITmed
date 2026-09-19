from __future__ import annotations

import numpy as np
import pytest
import torch

from models.classifier import (
    BilateralDifferentialInvariance,
    DifferentiableAcetabularAngleRegressor,
    AnatomicalPelvicMaskedAutoencoder,
)
from plugins.hip_dysplasia.geometry import ShentonArcContinuityFilter
from core.preprocessor import WassersteinTriradiateMatcher
from train.aggregation import CascadeSpecificityGate


def test_bilateral_differential_invariance():
    module = BilateralDifferentialInvariance(in_channels=32)
    module.eval()

    # Case 1: Perfectly symmetric features
    f_left = torch.randn(2, 32, 16, 16)
    f_right = torch.flip(f_left, dims=[-1])

    with torch.no_grad():
        delta_f, asym_energy, sym_gate = module(f_left, f_right)
        assert delta_f.shape == f_left.shape
        assert asym_energy.shape == (2,)
        assert asym_energy.max().item() < 1e-5  # Perfectly zero difference
        assert sym_gate.mean().item() > 0.40

    # Case 2: Asymmetric features (pathology)
    f_right_asym = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        _, asym_energy_hi, _ = module(f_left, f_right_asym)
        assert asym_energy_hi.mean().item() > asym_energy.mean().item()


def test_shenton_arc_continuity_filter():
    gate = ShentonArcContinuityFilter(step_threshold_mm=3.5)

    # Continuous arc (step dy = 1.0 mm)
    res_cont = gate.evaluate_continuity(
        femoral_neck_pt=(100.0, 150.0),
        obturator_arc_pt=(100.0, 151.0),
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_cont["is_continuous"] is True
    assert res_cont["step_mm"] == 1.0
    assert res_cont["smoothness_score"] > 0.70

    # Broken arc (step dy = 7.0 mm -> subluxation)
    res_broken = gate.evaluate_continuity(
        femoral_neck_pt=(100.0, 150.0),
        obturator_arc_pt=(100.0, 157.0),
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_broken["is_continuous"] is False
    assert res_broken["step_mm"] == 7.0


def test_differentiable_acetabular_angle_regressor():
    regressor = DifferentiableAcetabularAngleRegressor(in_features=64)
    regressor.eval()

    feat = torch.randn(4, 64)
    with torch.no_grad():
        angles, max_angle, is_normal_gate = regressor(feat)
        assert angles.shape == (4, 2)
        assert max_angle.shape == (4,)
        assert is_normal_gate.shape == (4,)
        assert (angles >= 10.0).all() and (angles <= 45.0).all()


def test_wasserstein_triradiate_matcher():
    matcher = WassersteinTriradiateMatcher(template_sigma=0.25)

    # Symmetric Gaussian crop vs itself
    crop = np.zeros((32, 32), dtype=np.float32)
    crop[12:20, 12:20] = 1.0

    w_dist = matcher.compute_wasserstein_distance(crop, crop)
    assert np.isclose(w_dist, 0.0)

    # Shifted crop has higher distance
    shifted = np.zeros((32, 32), dtype=np.float32)
    shifted[20:28, 20:28] = 1.0
    w_shifted = matcher.compute_wasserstein_distance(crop, shifted)
    assert w_shifted > w_dist


def test_cascade_specificity_gate():
    cascade = CascadeSpecificityGate(screening_threshold=0.45, definite_pathology_threshold=0.70)

    # Clear normal
    c_norm = cascade.evaluate_candidate(0.20)
    assert c_norm["tier"] == "tier1_screened_normal"
    assert c_norm["is_pathology"] is False

    # Clear pathology
    c_path = cascade.evaluate_candidate(0.85)
    assert c_path["tier"] == "tier1_screened_pathology"
    assert c_path["is_pathology"] is True

    # Borderline candidate (0.58) with confirmed healthy landmarks -> reclassified to normal!
    c_border = cascade.evaluate_candidate(
        base_probability=0.58,
        shenton_is_continuous=True,
        asymmetry_energy=0.03,
        acetabular_angle_max=24.0,
        infant_age_months=3.0,
    )
    assert c_border["tier"] == "tier2_verified_normal"
    assert c_border["final_risk"] < 0.50


def test_anatomical_pelvic_masked_autoencoder():
    mae = AnatomicalPelvicMaskedAutoencoder(in_channels=3, hidden_dim=32, mask_ratio=0.70)
    mae.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        recon, mask = mae(x)
        assert recon.shape == x.shape
        assert mask.shape == (2, 1, 64, 64)
