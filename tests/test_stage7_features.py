from __future__ import annotations

import numpy as np
import pytest
import torch

from plugins.hip_dysplasia.geometry import PerkinsOmbredanneQuadrantLocalizer
from core.preprocessor import LaplacianPyramidBoneEdgeHarmonizer
from core.age_utils import TemporalMaturationKinematics
from models.classifier import (
    TransversePelvicMidlineTransformer,
    CounterfactualJointDiffusionResidual,
)
from train.aggregation import DirichletManifoldEnergyFilter


def test_perkins_ombredanne_quadrant_localizer():
    localizer = PerkinsOmbredanneQuadrantLocalizer(safety_margin_mm=1.0)

    # 1. Normal infant right hip: inferior to Hilgenreiner line (Y > 150), medial to Perkins line (X > 100)
    res_normal = localizer.locate_quadrant(
        femoral_pt=(120.0, 160.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        side="right",
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_normal["is_in_safe_quadrant"] is True
    assert res_normal["quadrant"] == "inferior_medial"
    assert res_normal["clinical_status"] == "normal"
    assert res_normal["pathology_prior_boost"] == 0.0

    # 2. Pathological subluxation: lateralized right hip (X = 80 < Perkins X = 100)
    res_sublux = localizer.locate_quadrant(
        femoral_pt=(80.0, 160.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        side="right",
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_sublux["is_in_safe_quadrant"] is False
    assert res_sublux["quadrant"] == "inferior_lateral"
    assert res_sublux["clinical_status"] == "subluxation"
    assert res_sublux["pathology_prior_boost"] > 0.0
    assert res_sublux["lateral_displacement_mm"] == 20.0


def test_laplacian_pyramid_bone_edge_harmonizer():
    harmonizer = LaplacianPyramidBoneEdgeHarmonizer(num_levels=3, edge_boost=1.3)
    img = np.ones((64, 64), dtype=np.uint8) * 128
    # Add high frequency bone edges
    img[20:44, 20:44] = 200

    pyr = harmonizer.build_pyramid(img)
    assert len(pyr) == 3

    harmonized = harmonizer.harmonize_contrast(img)
    assert harmonized.shape == img.shape
    assert harmonized.dtype == img.dtype

    energies = harmonizer.compute_band_energies(img)
    assert "high_freq_bone_ratio" in energies
    assert "bone_edge_snr" in energies
    assert energies["bone_edge_snr"] > 0.0


def test_temporal_maturation_kinematics():
    kinematics = TemporalMaturationKinematics(normal_velocity_threshold=-0.02)

    # 1. Normal physiological maturation: risk and angle drop over 3 visits (1m, 3m, 6m)
    res_norm = kinematics.evaluate_kinematics(
        time_points_months=[1.0, 3.0, 6.0],
        risk_or_angle_trajectory=[0.45, 0.38, 0.28],
    )
    assert res_norm["is_maturing_normally"] is True
    assert res_norm["velocity"] < 0.0
    assert res_norm["calibrated_patient_risk"] < 0.28

    # 2. Progressive dysplasia: risk increases over time
    res_path = kinematics.evaluate_kinematics(
        time_points_months=[1.0, 3.0, 6.0],
        risk_or_angle_trajectory=[0.45, 0.55, 0.68],
    )
    assert res_path["is_maturing_normally"] is False
    assert res_path["velocity"] > 0.0
    assert res_path["calibrated_patient_risk"] == 0.68


def test_transverse_pelvic_midline_transformer():
    model = TransversePelvicMidlineTransformer(in_channels=32, embed_dim=16, num_heads=2)
    model.eval()

    # Create symmetric pelvic feature map
    half = torch.randn(2, 32, 8, 8)
    feat_sym = torch.cat([half, torch.flip(half, dims=[-1])], dim=-1)  # [2, 32, 8, 16]

    with torch.no_grad():
        logits, sym_score, asym_res = model(feat_sym)
        assert logits.shape == (2,)
        assert sym_score.shape == (2,)
        assert asym_res.shape == (2,)
        assert asym_res.max().item() < 1e-4
        assert sym_score.min().item() > 0.80

    # Asymmetric feature map
    feat_asym = torch.randn(2, 32, 8, 16)
    with torch.no_grad():
        _, sym_score_asym, asym_res_hi = model(feat_asym)
        assert asym_res_hi.mean().item() > asym_res.mean().item()
        assert sym_score_asym.mean().item() < sym_score.mean().item()


def test_counterfactual_joint_diffusion_residual():
    module = CounterfactualJointDiffusionResidual(in_channels=3, hidden_dim=16)
    module.eval()

    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        out = module(x)
        assert "reconstruction" in out
        assert "residual_map" in out
        assert "anomaly_energy" in out
        assert "lateral_rim_bias" in out
        assert out["residual_map"].shape == (2, 1, 32, 32)
        assert out["anomaly_energy"].shape == (2,)
        assert (out["lateral_rim_bias"] > 0.0).all()


def test_dirichlet_manifold_energy_filter():
    filter_dirichlet = DirichletManifoldEnergyFilter(sigma=1.0, alpha_propagation=0.4, num_iterations=10)

    # 10 synthetic patients in 2 tight clusters: 5 normal, 5 pathology
    normal_feats = np.ones((5, 4)) * 0.1
    path_feats = np.ones((5, 4)) * 0.9
    feats = np.vstack([normal_feats, path_feats])

    # Initial probs with 1 noisy normal at 0.55
    p_init = np.array([0.15, 0.20, 0.18, 0.22, 0.55, 0.85, 0.88, 0.90, 0.82, 0.89])

    smoothed = filter_dirichlet.smooth_probabilities(p_init, feats)
    assert len(smoothed) == 10
    # Noisy normal surrounded by normal cluster should be pulled down towards normal boundary
    assert smoothed[4] < p_init[4]
    # Pathologies should remain high
    assert smoothed[7] > 0.75
