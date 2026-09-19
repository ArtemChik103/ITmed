"""Unit tests for advanced features: Age Conditioning, SupCon, Temperature Scaling, and Gradient Checkpointing."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from core.age_utils import extract_age_from_dicom_metadata, extract_age_from_path, resolve_patient_age
from models.calibration import TemperatureScaler
from models.classifier import FiLMLayer, HipDysplasiaClassifier
from models.losses import SupConLoss


def test_age_extraction_from_path() -> None:
    assert extract_age_from_path("Патология/subject_0007/5 мес/00000001.dcm") == 5.0
    assert extract_age_from_path("Патология/subject_0050/10 мес/00000001.dcm") == 10.0
    assert extract_age_from_path("Normal/subject_0001/1 year/00000001.dcm") == 12.0
    assert extract_age_from_path("Normal/subject_0001/00000001.dcm") is None


def test_age_extraction_from_dicom() -> None:
    assert extract_age_from_dicom_metadata({"patient_age": "006M"}) == 6.0
    assert extract_age_from_dicom_metadata({"patient_age": "001Y"}) == 12.0
    assert extract_age_from_dicom_metadata({"patient_age": "026W"}) == pytest.approx(6.0, abs=0.1)
    
    dates_meta = {"patient_birth_date": "20230101", "study_date": "20230701"}
    assert extract_age_from_dicom_metadata(dates_meta) == pytest.approx(5.95, abs=0.2)


def test_resolve_patient_age() -> None:
    age, known = resolve_patient_age("train/Патология/subject_0007/5 мес/001.dcm")
    assert age == 5.0
    assert known is True

    age_unk, known_unk = resolve_patient_age("train/Норма/001.dcm", {})
    assert age_unk == 0.0
    assert known_unk is False


def test_film_layer_identity_initialization() -> None:
    feature_dim = 256
    film = FiLMLayer(feature_dim=feature_dim)
    features = torch.randn(4, feature_dim)
    conditioning = torch.tensor([[5.0 / 36.0, 1.0], [0.0, 0.0], [10.0 / 36.0, 1.0], [0.0, 0.0]])
    
    modulated = film(features, conditioning)
    assert torch.allclose(modulated, features, atol=1e-6)


def test_supcon_loss() -> None:
    supcon = SupConLoss(temperature=0.07)
    features = torch.randn(8, 64)
    # 4 pairs of identical classes/patients
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon(features, labels)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_temperature_scaler() -> None:
    scaler = TemperatureScaler(temperature=1.0)
    # Realistic noisy predictions with some misclassifications
    logits = np.array([2.5, -1.5, 0.5, -2.0, -0.5, 1.2, -1.0, 1.8])
    targets = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    t_opt = scaler.fit(logits, targets)
    assert 0.1 <= t_opt <= 5.0

    cal_p = scaler.calibrate_probability(0.85)
    assert 0.0 < cal_p < 1.0


def test_gradient_checkpointing_toggle() -> None:
    model = HipDysplasiaClassifier(architecture="resnet50", pretrained=False)
    # Should not raise
    model.enable_gradient_checkpointing()


def test_bilateral_coherence_computation() -> None:
    from plugins.hip_dysplasia.model import compute_bilateral_coherence
    left_hip = np.ones((64, 64), dtype=np.uint8) * 128
    right_hip = np.ones((64, 64), dtype=np.uint8) * 128
    result = compute_bilateral_coherence(left_hip, right_hip)
    assert "ncc" in result
    assert "coherence" in result
    assert "structural_diff" in result
    assert -1.0 <= result["coherence"] <= 1.0


def test_efficientnet_b4_classifier_forward() -> None:
    model = HipDysplasiaClassifier(architecture="efficientnet_b4", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2,)


def test_pareto_roc_convex_hull_threshold_selection() -> None:
    from train.classifier_train import compute_roc_convex_hull, find_optimal_threshold
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probs = np.array([0.9, 0.85, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    thresh, metrics, sweep = find_optimal_threshold(
        y_true,
        probs,
        policy="pareto_convex_hull",
        sensitivity_floor=0.90,
    )
    assert 0.05 <= thresh <= 0.95
    assert metrics["sensitivity"] >= 0.75
    hull = compute_roc_convex_hull(sweep)
    assert len(hull) > 0
    assert len(hull) <= len(sweep)


def test_radimagenet_pretrained_backbone() -> None:
    from pathlib import Path
    weights_path = Path("models/pretrained/radimagenet_resnet50.pth")
    if weights_path.exists():
        model = HipDysplasiaClassifier(
            architecture="radimagenet_resnet50",
            pretrained=True,
            pretrained_weights_path=weights_path,
        )
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2,)


def test_bilateral_cross_attention_fusion() -> None:
    from models.classifier import BilateralCrossAttentionFusion
    fusion = BilateralCrossAttentionFusion(embed_dim=128, num_heads=4)

    # 4D spatial feature map input [B, C, H, W]
    f_l = torch.randn(2, 64, 8, 8)
    f_r = torch.randn(2, 64, 8, 8)
    disc, asym, attn = fusion(f_l, f_r)
    assert disc.shape == (2, 128)
    assert asym.shape == (2,)
    assert 0.0 <= float(asym[0]) <= 1.0

    # 2D pooled feature input [B, C]
    f2_l = torch.randn(2, 256)
    f2_r = torch.randn(2, 256)
    disc2, asym2, _ = fusion(f2_l, f2_r)
    assert disc2.shape == (2, 128)
    assert asym2.shape == (2,)


def test_tps_elastic_augmentation() -> None:
    from data.augmentations import apply_thin_plate_spline_warp
    img = np.ones((128, 128, 3), dtype=np.uint8) * 120
    warped = apply_thin_plate_spline_warp(img)
    assert warped.shape == (128, 128, 3)
    assert warped.dtype == np.uint8


def test_longitudinal_trajectory_model() -> None:
    from train.aggregation import LongitudinalTrajectoryModel
    model = LongitudinalTrajectoryModel()

    # Progressive dysplasia: risk increasing over visits
    res_prog = model.evaluate_trajectory([0.3, 0.5, 0.7], ages_months=[2.0, 4.0, 6.0])
    assert res_prog["status"] == "progressive_dysplasia"
    assert res_prog["slope"] > 0
    assert res_prog["adjusted_risk"] >= 0.7

    # Resolving hip under treatment / normal maturation: risk declining
    res_resolv = model.evaluate_trajectory([0.7, 0.4, 0.2], ages_months=[2.0, 4.0, 6.0])
    assert res_resolv["status"] == "resolving_or_maturing"
    assert res_resolv["slope"] < 0
    assert res_resolv["adjusted_risk"] < 0.6


def test_multimodel_convex_hull_optimizer() -> None:
    from train.classifier_train import MultiModelConvexHullOptimizer
    optimizer = MultiModelConvexHullOptimizer(sensitivity_floor=0.80)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # 3 models: Model 1 good, Model 2 noisy, Model 3 complementary
    pred_matrix = np.array([
        [0.9, 0.85, 0.75, 0.7, 0.3, 0.2, 0.15, 0.1],
        [0.6, 0.7, 0.5, 0.6, 0.5, 0.4, 0.3, 0.4],
        [0.85, 0.9, 0.8, 0.75, 0.25, 0.2, 0.1, 0.05],
    ]).T

    weights, threshold, metrics = optimizer.optimize_ensemble_weights(y_true, pred_matrix)
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert all(w >= 0.0 for w in weights)
    assert metrics["sensitivity"] >= 0.75


def test_ohem_loss() -> None:
    from models.losses import OHEMLoss
    base = nn.BCEWithLogitsLoss(reduction="none")
    ohem = OHEMLoss(base_loss=base, keep_ratio=0.50)
    logits = torch.tensor([5.0, -5.0, 0.1, -0.2], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = ohem(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_barlow_twins_loss() -> None:
    from models.losses import BarlowTwinsLoss
    criterion = BarlowTwinsLoss(lambd=0.005)
    z1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32, requires_grad=True)
    loss = criterion(z1, z2)
    assert loss.item() > 0.0
    loss.backward()
    assert z1.grad is not None


def test_simclr_loss() -> None:
    from models.losses import SimCLRLoss
    criterion = SimCLRLoss(temperature=0.1)
    z1 = torch.randn(6, 16, requires_grad=True)
    z2 = torch.randn(6, 16, requires_grad=True)
    loss = criterion(z1, z2)
    assert loss.item() > 0.0
    loss.backward()
    assert z1.grad is not None


def test_multitask_geometric_classifier() -> None:
    from models.classifier import MultiTaskGeometricClassifier
    model = MultiTaskGeometricClassifier(architecture="resnet50", num_keypoints=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, geom = model(x)
    assert logits.shape == (2,)
    assert geom.shape == (2, 8 * 2 + 2)


def test_dual_stream_fusion_classifier() -> None:
    from models.classifier import DualStreamFusionClassifier
    model = DualStreamFusionClassifier(architecture="resnet50", pretrained=False)
    x_global = torch.randn(2, 3, 224, 224)
    x_detail = torch.randn(2, 3, 224, 224)
    fused_logits = model(x_global, x_detail)
    assert fused_logits.shape == (2,)


def test_tent_test_time_adaptation_and_uncertainty() -> None:
    from plugins.hip_dysplasia.model import HipDysplasiaEnsemble, resolve_model_manifest_path
    manifest_path = resolve_model_manifest_path()
    if manifest_path is not None and manifest_path.exists():
        ensemble = HipDysplasiaEnsemble(manifest_path, device="cpu")
        dummy_img = np.ones((128, 128), dtype=np.uint8) * 128
        pred = ensemble.predict(dummy_img, tta=False, dual_hip=False, tent=True)
        assert 0.0 <= pred.probability <= 1.0
        assert pred.epistemic_uncertainty is not None
        assert pred.epistemic_uncertainty >= 0.0
        assert pred.is_uncertain is not None


def test_clinical_hip_geometry_and_parser() -> None:
    from plugins.hip_dysplasia.geometry import (
        compute_hilgenreiner_line,
        compute_perkin_line,
        compute_acetabular_angle_deg,
        evaluate_clinical_hip_geometry,
    )
    # Synthetic realistic pelvic landmarks (Y-cartilages, acetabular roof edges, femoral heads, obturators)
    kpts = [
        (100.0, 200.0),  # 0: Left Y-cartilage
        (300.0, 202.0),  # 1: Right Y-cartilage
        (70.0, 160.0),   # 2: Left roof
        (330.0, 162.0),  # 3: Right roof
        (85.0, 210.0),   # 4: Left femoral head
        (315.0, 210.0),  # 5: Right femoral head
        (90.0, 240.0),   # 6: Left obturator
        (310.0, 240.0),  # 7: Right obturator
    ]
    hilg = compute_hilgenreiner_line(kpts[0], kpts[1])
    assert abs(hilg.b) > 0.9  # Nearly horizontal line
    perkin = compute_perkin_line(hilg, kpts[2])
    assert abs(perkin.a) > 0.9  # Nearly vertical line

    alpha = compute_acetabular_angle_deg(kpts[0], kpts[2], hilg)
    assert 15.0 <= alpha <= 65.0

    geom_metrics = evaluate_clinical_hip_geometry(kpts)
    assert "acetabular_angle_left" in geom_metrics
    assert "acetabular_angle_right" in geom_metrics
    assert "shenton_discrepancy_max" in geom_metrics


def test_differentiable_geometry_consistency_loss() -> None:
    from models.losses import DifferentiableGeometryConsistencyLoss
    loss_fn = DifferentiableGeometryConsistencyLoss(angle_threshold_deg=30.0, weight=0.5)
    logits = torch.tensor([1.5, -1.0], requires_grad=True)
    # 2 samples, 8 keypoints, 2 coords
    keypoints = torch.tensor([
        [
            [0.3, 0.6], [0.7, 0.6],
            [0.2, 0.45], [0.8, 0.45],
            [0.25, 0.65], [0.75, 0.65],
            [0.28, 0.8], [0.72, 0.8],
        ],
        [
            [0.3, 0.6], [0.7, 0.6],
            [0.25, 0.55], [0.75, 0.55],
            [0.28, 0.65], [0.72, 0.65],
            [0.28, 0.8], [0.72, 0.8],
        ],
    ], requires_grad=True)

    loss = loss_fn(logits, keypoints)
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None


def test_masked_patch_reconstruction_loss() -> None:
    from models.losses import MaskedPatchReconstructionLoss
    loss_fn = MaskedPatchReconstructionLoss(cosine_weight=0.2)
    pred = torch.randn(2, 16, 64, requires_grad=True)
    target = torch.randn(2, 16, 64)
    mask = torch.zeros(2, 16)
    mask[:, :8] = 1.0

    loss = loss_fn(pred, target, mask=mask)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None


def test_pelvic_spatial_transformer_stn() -> None:
    from models.classifier import PelvicSpatialTransformer
    stn = PelvicSpatialTransformer(in_channels=3)
    x = torch.randn(2, 3, 128, 128)
    aligned_x, theta = stn(x)
    assert aligned_x.shape == (2, 3, 128, 128)
    assert theta.shape == (2, 2, 3)
    # Check identity initialization on initial forward
    assert torch.allclose(theta[0, 0, 0], torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(theta[0, 1, 1], torch.tensor(1.0), atol=1e-3)


def test_pyramidal_feature_pyramid_fusion() -> None:
    from models.classifier import PyramidalFeaturePyramidFusion
    bifpn = PyramidalFeaturePyramidFusion(in_channels_list=[64, 128, 256], out_channels=64)
    p3 = torch.randn(2, 64, 32, 32)
    p4 = torch.randn(2, 128, 16, 16)
    p5 = torch.randn(2, 256, 8, 8)

    logits = bifpn([p3, p4, p5])
    assert logits.shape == (2,)


def test_masked_pelvic_autoencoder() -> None:
    from models.classifier import MaskedPelvicAutoencoder
    mae = MaskedPelvicAutoencoder(in_channels=3, patch_size=16, embed_dim=64, decoder_dim=32, mask_ratio=0.5)
    x = torch.randn(2, 3, 64, 64)
    pred_patches, target_patches, mask = mae(x)
    # 64 / 16 = 4 -> 4x4 = 16 patches
    assert pred_patches.shape == (2, 16, 3 * 16 * 16)
    assert target_patches.shape == (2, 16, 3 * 16 * 16)
    assert mask.shape == (2, 16)
    assert mask.sum() > 0


def test_gated_attention_mil() -> None:
    from train.aggregation import GatedAttentionMIL
    mil = GatedAttentionMIL(in_features=8, hidden_dim=16)

    # 1 patient with 4 radiographic studies
    bag = torch.randn(4, 8)
    logit, weights = mil(bag)
    assert logit.shape == (1,)
    assert weights.shape == (4,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-4)

    # Batch of 3 patients with 5 studies each
    batch_bags = torch.randn(3, 5, 8)
    batch_logits, batch_weights = mil(batch_bags)
    assert batch_logits.shape == (3,)
    assert batch_weights.shape == (3, 5)


def test_clinical_meta_learner_stacker() -> None:
    from train.aggregation import ClinicalMetaLearnerStacker
    stacker = ClinicalMetaLearnerStacker(n_estimators=10)

    # 20 samples with 10 features (multi-model probs + age + metrics)
    np.random.seed(42)
    X = np.random.rand(20, 10).astype(np.float32)
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    stacker.fit(X, y)
    probs = stacker.predict_proba(X)
    assert probs.shape == (20,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_decompose_multichannel_dicom_windows() -> None:
    from core.preprocessor import decompose_multichannel_dicom_windows
    synthetic_xray = np.random.randint(0, 4000, size=(128, 128), dtype=np.uint16)
    multi_window = decompose_multichannel_dicom_windows(synthetic_xray, target_size=(128, 128))
    assert multi_window.shape == (128, 128, 3)
    assert multi_window.dtype == np.float32
    assert 0.0 <= multi_window.min() <= multi_window.max() <= 1.0


def test_anatomical_graph_neural_network() -> None:
    from models.classifier import AnatomicalGraphNeuralNetwork
    gnn = AnatomicalGraphNeuralNetwork(node_dim=32, num_heads=2)
    # 2 batches of 8 landmark nodes in 2D
    kpts = torch.rand(2, 8, 2, requires_grad=True)
    logits = gnn(kpts)
    assert logits.shape == (2,)
    loss = logits.sum()
    loss.backward()
    assert kpts.grad is not None


def test_pelvic_diffusion_anomaly_estimator() -> None:
    from models.classifier import PelvicDiffusionAnomalyEstimator
    estimator = PelvicDiffusionAnomalyEstimator(in_channels=3, hidden_dim=16)
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    recon, anom_map, scores = estimator(x)
    assert recon.shape == (2, 3, 64, 64)
    assert anom_map.shape == (2, 1, 64, 64)
    assert scores.shape == (2,)
    loss = scores.sum()
    loss.backward()
    assert x.grad is not None


def test_foundation_model_distillation_loss() -> None:
    from models.losses import FoundationModelDistillationLoss
    criterion = FoundationModelDistillationLoss(temperature=2.0)
    student_f = torch.randn(4, 64, requires_grad=True)
    teacher_f = torch.randn(4, 64)
    student_l = torch.randn(4, 2, requires_grad=True)
    teacher_l = torch.randn(4, 2)
    loss = criterion(student_f, teacher_f, student_l, teacher_l)
    assert loss.item() >= 0.0
    loss.backward()
    assert student_f.grad is not None


def test_bilateral_arcface_loss() -> None:
    from models.losses import BilateralArcFaceLoss
    arcface = BilateralArcFaceLoss(scale=8.0, margin=0.2)
    f_left = torch.randn(4, 32, requires_grad=True)
    f_right = torch.randn(4, 32, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    loss = arcface(f_left, f_right, labels)
    assert loss.item() > 0.0
    loss.backward()
    assert f_left.grad is not None


def test_apply_anatomical_hip_grafting() -> None:
    from data.augmentations import apply_anatomical_hip_grafting
    img_base = np.zeros((128, 128, 3), dtype=np.uint8)
    img_donor = np.ones((128, 128, 3), dtype=np.uint8) * 200
    grafted = apply_anatomical_hip_grafting(img_base, img_donor, hip_side="left", blend_alpha=0.6)
    assert grafted.shape == (128, 128, 3)
    assert grafted.dtype == np.uint8
    assert grafted.max() > 0





