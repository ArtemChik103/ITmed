from __future__ import annotations

import numpy as np
import pytest
import torch

from plugins.hip_dysplasia.geometry import BiomechanicalPelvicSpringGraph
from core.preprocessor import (
    DifferentiableActiveContourSnake,
    FractalSubchondralRoughnessAnalyzer,
)
from models.classifier import (
    ScannerDomainAdversarialInvariance,
    LatentDiffusionSubluxationSynthesizer,
)
from train.aggregation import BayesianBrierHierarchicalEnsemble


def test_biomechanical_pelvic_spring_graph():
    graph = BiomechanicalPelvicSpringGraph(max_strain_threshold=0.50, energy_tolerance=5.0)

    # 1. Physiologically normal pelvic landmark layout
    normal_landmarks = {
        "triradiate_l": (100.0, 150.0),
        "triradiate_r": (200.0, 150.0),  # d_base = 100 mm
        "acetabulum_l": (100.0, 112.0),  # dist = 38 mm (ratio = 0.38)
        "acetabulum_r": (200.0, 112.0),  # dist = 38 mm (ratio = 0.38)
        "femur_l": (100.0, 178.0),       # dist = 28 mm (ratio = 0.28)
        "femur_r": (200.0, 178.0),       # dist = 28 mm (ratio = 0.28)
    }

    res_norm = graph.evaluate_graph_energy(normal_landmarks, pixel_spacing_mm=(1.0, 1.0))
    assert res_norm["is_physically_consistent"] is True
    assert res_norm["total_elastic_energy"] < 0.01
    assert res_norm["max_strain"] < 0.05
    assert res_norm["num_springs_evaluated"] == 5

    # 2. Anatomically unphysical configuration (femur displaced by 300 mm)
    broken_landmarks = normal_landmarks.copy()
    broken_landmarks["femur_l"] = (100.0, 450.0)

    res_broken = graph.evaluate_graph_energy(broken_landmarks, pixel_spacing_mm=(1.0, 1.0))
    assert res_broken["is_physically_consistent"] is False
    assert res_broken["max_strain"] > 0.50
    assert res_broken["total_elastic_energy"] > 5.0


def test_differentiable_active_contour_snake():
    snake = DifferentiableActiveContourSnake(alpha_elasticity=0.2, beta_rigidity=0.1, gamma_edge=1.0)

    img = np.zeros((64, 64), dtype=np.float32)
    # Bright horizontal boundary (bone margin) at y = 32
    img[32:, :] = 200.0

    # Initial contour at y = 28 (slightly above true boundary)
    init_pts = np.array([[float(x), 28.0] for x in range(10, 55, 5)], dtype=np.float32)

    evolved = snake.evolve_contour(img, init_pts, num_iterations=10)
    assert evolved.shape == init_pts.shape
    # Snake should have moved towards the bright edge at y = 32
    mean_y_init = float(np.mean(init_pts[:, 1]))
    mean_y_evolved = float(np.mean(evolved[:, 1]))
    assert mean_y_evolved > mean_y_init


def test_fractal_subchondral_roughness_analyzer():
    analyzer = FractalSubchondralRoughnessAnalyzer(box_sizes=[2, 4, 8, 16])

    # 1. Smooth horizontal line (D_0 ~ 1.0)
    mask_smooth = np.zeros((64, 64), dtype=np.uint8)
    mask_smooth[32, :] = 255

    res_smooth = analyzer.compute_fractal_dimension(mask_smooth)
    assert "fractal_dimension_d0" in res_smooth
    assert "is_smooth_normal_cortex" in res_smooth
    assert res_smooth["fractal_dimension_d0"] < 1.20
    assert res_smooth["is_smooth_normal_cortex"] is True

    # 2. Highly fragmented noisy mask (high roughness D_0 > 1.30)
    rng = np.random.RandomState(42)
    mask_rough = (rng.rand(64, 64) > 0.80).astype(np.uint8) * 255
    res_rough = analyzer.compute_fractal_dimension(mask_rough)
    assert res_rough["fractal_dimension_d0"] > res_smooth["fractal_dimension_d0"]


def test_scanner_domain_adversarial_invariance():
    model = ScannerDomainAdversarialInvariance(feature_dim=64, num_domains=4, lambda_grl=0.5)
    model.train()

    features = torch.randn(8, 64, requires_grad=True)
    domain_logits = model(features)
    assert domain_logits.shape == (8, 4)

    # Test gradient reversal during backward pass
    loss = domain_logits.sum()
    loss.backward()
    assert features.grad is not None


def test_latent_diffusion_subluxation_synthesizer():
    synthesizer = LatentDiffusionSubluxationSynthesizer(latent_dim=32)
    synthesizer.eval()

    latent = torch.randn(4, 32)
    # Severity 0.0 -> no shift
    synth_0 = synthesizer.synthesize(latent, severity_degree=0.0)
    assert torch.allclose(synth_0, latent, atol=1e-5)

    # Severity 0.8 -> significant shift along dysplasia manifold
    synth_hi = synthesizer.synthesize(latent, severity_degree=0.8)
    assert synth_hi.shape == (4, 32)
    diff = torch.norm(synth_hi - latent, dim=-1)
    assert (diff > 0.01).all()


def test_bayesian_brier_hierarchical_ensemble():
    ensemble = BayesianBrierHierarchicalEnsemble(temperature_tau=0.05)

    # Synthetic validation predictions from 3 models
    # Model 0 is highly accurate, Model 1 is moderate, Model 2 is poor
    rng = np.random.RandomState(42)
    y_val = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    m0 = np.where(y_val == 1, 0.90, 0.10)
    m1 = np.where(y_val == 1, 0.70, 0.30)
    m2 = np.where(y_val == 1, 0.55, 0.45)

    val_preds = np.column_stack([m0, m1, m2])
    ensemble.fit(val_preds, y_val)

    assert ensemble.model_weights is not None
    assert len(ensemble.model_weights) == 3
    assert np.isclose(np.sum(ensemble.model_weights), 1.0)
    # Accurate Model 0 should receive the highest weight
    assert ensemble.model_weights[0] > ensemble.model_weights[1] > ensemble.model_weights[2]

    # Predict test
    test_preds = np.array([[0.8, 0.6, 0.5]])
    p_ens = ensemble.predict(test_preds)
    assert 0.6 <= float(p_ens[0]) <= 0.85
