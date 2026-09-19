from __future__ import annotations

import numpy as np
import pytest
import torch

from core.preprocessor import (
    DifferentiablePelvicTiltRectifier,
    GaborTrabecularCoherenceAnalyzer,
    GonadalShieldInpaintingGate,
)
from models.classifier import PelvicMAEDistillation
from models.losses import EvidentialDirichletLoss
from train.aggregation import TestTimeManifoldOrbitConsensus


def test_differentiable_pelvic_tilt_rectifier():
    rectifier = DifferentiablePelvicTiltRectifier()
    img = np.ones((100, 100), dtype=np.uint8) * 128

    # Tilted landmark baseline: Left at (30, 40), Right at (70, 50) -> tilted by ~14 deg
    rectified, angle = rectifier.rectify_tilt(
        image=img,
        left_landmark_pt=(30.0, 40.0),
        right_landmark_pt=(70.0, 50.0),
    )
    assert rectified.shape == (100, 100)
    assert np.isclose(angle, 14.04, atol=0.2)


def test_gabor_trabecular_coherence_analyzer():
    analyzer = GaborTrabecularCoherenceAnalyzer(num_orientations=8, kernel_size=15)

    # Vertical striped image representing strong vertical compression trabeculae
    crop = np.zeros((48, 48), dtype=np.float32)
    crop[:, ::4] = 1.0  # Vertical lines

    res = analyzer.analyze_trabeculae(crop)
    assert "dominant_angle_deg" in res
    assert "trabecular_anisotropy_index" in res
    assert "is_organized_trabeculae" in res
    assert res["trabecular_anisotropy_index"] > 0.40
    assert res["is_organized_trabeculae"] is True


def test_gonadal_shield_inpainting_gate():
    gate = GonadalShieldInpaintingGate(intensity_threshold=240.0, min_area_ratio=0.02)

    img = np.ones((80, 80), dtype=np.uint8) * 100
    # Simulate saturated lead shield block in the center
    img[30:55, 30:55] = 255

    res = gate.process(img)
    assert res["is_shielded"] is True
    assert res["occlusion_ratio"] > 0.05
    assert res["cleaned_image"].shape == img.shape
    # Saturated shield pixels should be inpainted to lower intensity
    assert res["cleaned_image"][40, 40] < 255


def test_pelvic_mae_distillation():
    distill = PelvicMAEDistillation(student_dim=64, teacher_dim=32)
    distill.eval()

    student_feat = torch.randn(4, 64)
    teacher_feat = torch.randn(4, 32)

    with torch.no_grad():
        proj = distill(student_feat)
        assert proj.shape == (4, 32)

    loss_dict = distill.compute_distillation_loss(student_feat, teacher_feat)
    assert "distillation_loss" in loss_dict
    assert "mse_loss" in loss_dict
    assert "cosine_alignment" in loss_dict
    assert loss_dict["distillation_loss"].item() > 0.0


def test_evidential_dirichlet_loss():
    loss_fn = EvidentialDirichletLoss(kl_weight=0.1)

    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 1, 0])

    res = loss_fn(logits, targets)
    assert "total_loss" in res
    assert "loss_mse" in res
    assert "loss_kl" in res
    assert "prob_pathology" in res
    assert "uncertainty" in res
    assert res["total_loss"].item() > 0.0
    assert (res["prob_pathology"] >= 0.0).all() and (res["prob_pathology"] <= 1.0).all()
    assert (res["uncertainty"] > 0.0).all() and (res["uncertainty"] <= 2.0).all()

    # Backpropagation check
    res["total_loss"].backward()
    assert logits.grad is not None


def test_test_time_manifold_orbit_consensus():
    orbit = TestTimeManifoldOrbitConsensus(variance_penalty=0.15, consistency_threshold=0.02)

    # 1. High-consistency orbit around clear pathology: [0.85, 0.86, 0.84, 0.85]
    res_clean = orbit.fuse_orbit_predictions([0.85, 0.86, 0.84, 0.85])
    assert res_clean["is_consistent"] is True
    assert res_clean["fused_probability"] >= 0.85

    # 2. Inconsistent noisy orbit: [0.20, 0.80, 0.40, 0.70]
    res_noisy = orbit.fuse_orbit_predictions([0.20, 0.80, 0.40, 0.70])
    assert res_noisy["is_consistent"] is False
    assert res_noisy["orbit_variance"] > 0.03
