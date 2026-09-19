from __future__ import annotations

import numpy as np
import pytest
import torch

from core.preprocessor import PelvicPoseRectifier, estimate_pelvic_tilt_and_rectify
from models.classifier import (
    DualScaleDeformableZoomClassifier,
    MultimodalClinicalVLMQueryClassifier,
    DeformableCrossAttention2D,
)
from core.age_utils import NormativeCartilageGrowthModel
from models.calibration import ConformalRiskPredictor
from models.losses import ClinicalConceptAlignmentLoss
from data.augmentations import Native16BitWindowJitter, apply_native_16bit_contrast_jitter


def test_pelvic_pose_rectifier_with_landmarks():
    # Synthetic image
    img = np.zeros((128, 128, 3), dtype=np.float32)
    # Tilted landmark pair: left (40, 60), right (88, 68) -> dy = 8, dx = 48
    landmarks = np.array([[40.0, 60.0], [88.0, 68.0]])
    
    rectifier = PelvicPoseRectifier()
    params = rectifier.estimate_tilt_parameters(img, landmarks=landmarks)
    assert "in_plane_angle_deg" in params
    assert abs(params["in_plane_angle_deg"] - np.degrees(np.arctan2(8.0, 48.0))) < 1e-3

    rectified, M = rectifier.rectify(img, params)
    assert rectified.shape == img.shape
    assert M.shape == (2, 3)

    # Convenience wrapper
    rect_img, res_params = estimate_pelvic_tilt_and_rectify(img)
    assert rect_img.shape == img.shape
    assert "in_plane_angle_deg" in res_params


def test_dual_scale_deformable_zoom_classifier():
    model = DualScaleDeformableZoomClassifier(in_channels=3, feature_dim=64)
    model.eval()

    # Batch of 2 images: 384x384 global
    global_x = torch.randn(2, 3, 128, 128)
    left_crop = torch.randn(2, 3, 64, 64)
    right_crop = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        # With explicit crops
        logit, aux = model(global_x, left_crop, right_crop)
        assert logit.shape == (2,)
        assert "p_left" in aux
        assert "p_right" in aux
        assert "asymmetry_mag" in aux

        # With auto-cropping
        logit_auto, aux_auto = model(global_x)
        assert logit_auto.shape == (2,)


def test_normative_cartilage_growth_model():
    model = NormativeCartilageGrowthModel(tau_ossification_months=4.5)

    # Younger infant (2 months) should have low ossification probability
    p_young = model.expected_ossification_probability(2.0)
    p_older = model.expected_ossification_probability(8.0)
    assert p_young < 0.25
    assert p_older > 0.85

    # Threshold for young infant is boosted to require higher confidence
    t_young = model.compute_biological_decision_threshold(2.0, is_known=True)
    t_older = model.compute_biological_decision_threshold(8.0, is_known=True)
    assert t_young > t_older

    # Mild border risk on 2-month old is dampened against false alarm
    mod_risk = model.modulate_risk_by_cartilage_maturity(0.55, 2.0, is_known=True)
    assert mod_risk < 0.55
    # High risk is preserved
    high_risk = model.modulate_risk_by_cartilage_maturity(0.85, 2.0, is_known=True)
    assert np.isclose(high_risk, 0.85)


def test_conformal_risk_predictor():
    predictor = ConformalRiskPredictor(alpha_normal=0.10, alpha_pathology=0.02)
    
    # Synthetic validation set
    y_val = np.array([0] * 50 + [1] * 50)
    p_val = np.concatenate([
        np.random.beta(1.5, 6.0, size=50),  # normal: mostly low
        np.random.beta(6.0, 1.5, size=50),  # pathology: mostly high
    ])

    predictor.fit(p_val, y_val)
    assert predictor.is_fitted
    assert predictor.q_hat_0 > 0
    assert predictor.q_hat_1 > 0

    # Test high probability -> pathology set
    pred_high = predictor.predict_set(0.95)
    assert 1 in pred_high["prediction_set"]

    # Test low probability -> normal set
    pred_low = predictor.predict_set(0.05)
    assert 0 in pred_low["prediction_set"]


def test_multimodal_clinical_vlm_query_classifier_and_loss():
    model = MultimodalClinicalVLMQueryClassifier(in_channels=3, embed_dim=64, num_symptoms=6)
    loss_fn = ClinicalConceptAlignmentLoss(suppression_weight=0.5, coverage_weight=0.5)

    x = torch.randn(4, 3, 128, 128)
    dysplasia_logit, symptom_logits, attn = model(x)

    assert dysplasia_logit.shape == (4,)
    assert symptom_logits.shape == (4, 6)
    assert attn.shape[0] == 4

    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = loss_fn(symptom_logits, targets, dysplasia_logit)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_native_16bit_window_jitter():
    # Test float normalized image
    img_f32 = np.random.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    jittered_f32 = apply_native_16bit_contrast_jitter(img_f32)
    assert jittered_f32.shape == img_f32.shape
    assert jittered_f32.dtype == img_f32.dtype
    assert 0.0 <= jittered_f32.min() and jittered_f32.max() <= 1.0

    # Test 16-bit uint16 image
    img_u16 = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
    jittered_u16 = apply_native_16bit_contrast_jitter(img_u16)
    assert jittered_u16.shape == img_u16.shape
    assert jittered_u16.dtype == np.uint16

    # Test Albumentations-style callable
    aug = Native16BitWindowJitter(p=1.0)
    res = aug(image=img_u16)
    assert "image" in res
    assert res["image"].shape == img_u16.shape
