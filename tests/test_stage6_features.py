from __future__ import annotations

import numpy as np
import pytest
import torch

from plugins.hip_dysplasia.geometry import KohlerTeardropCalveMorphometry
from core.preprocessor import (
    WaveletSubchondralTextureAnalyzer,
    PersistentHomologyContourFilter,
    PelvicAlignmentWarping,
)
from models.calibration import ExtremeValueTailCalibrator
from train.aggregation import CrossSubjectNormalCentroidFilter


def test_kohler_teardrop_calve_morphometry():
    morph = KohlerTeardropCalveMorphometry(min_closure_index=0.70, max_calve_step_mm=4.0)

    # Teardrop test - normal elongated U-loop
    res_td = morph.evaluate_teardrop(teardrop_bbox_or_pts=(10.0, 20.0, 10.0, 25.0))
    assert res_td["aspect_ratio"] == 2.5
    assert res_td["closure_index"] == 1.0
    assert res_td["is_normal_teardrop"] is True

    # Broken/wide teardrop
    res_td_broken = morph.evaluate_teardrop(teardrop_bbox_or_pts=(10.0, 20.0, 25.0, 10.0))
    assert res_td_broken["is_normal_teardrop"] is False

    # Calve arc test - continuous vs discontinuous
    res_arc_ok = morph.evaluate_calve_arc(
        iliac_crest_lateral_pt=(120.0, 80.0),
        superior_femoral_neck_pt=(120.0, 82.0),
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_arc_ok["is_continuous"] is True
    assert res_arc_ok["step_mm"] == 2.0

    res_arc_broken = morph.evaluate_calve_arc(
        iliac_crest_lateral_pt=(120.0, 80.0),
        superior_femoral_neck_pt=(120.0, 87.0),
        pixel_spacing_mm=(1.0, 1.0),
    )
    assert res_arc_broken["is_continuous"] is False
    assert res_arc_broken["step_mm"] == 7.0


def test_wavelet_subchondral_texture_analyzer():
    analyzer = WaveletSubchondralTextureAnalyzer()

    # Synthetic smooth bone crop with high horizontal/vertical continuity
    crop = np.zeros((32, 32), dtype=np.float32)
    crop[10:22, :] = 1.0  # Sharp horizontal plate

    subbands = analyzer.decompose_2d_haar(crop)
    assert "LL" in subbands and "LH" in subbands and "HL" in subbands and "HH" in subbands
    assert subbands["LL"].shape == (16, 16)

    metrics = analyzer.compute_subchondral_metrics(crop)
    assert "high_frequency_energy_ratio" in metrics
    assert "cortical_plate_continuity" in metrics
    assert isinstance(metrics["is_continuous_cortex"], (bool, np.bool_))


def test_persistent_homology_contour_filter():
    tda = PersistentHomologyContourFilter(num_thresholds=12)

    # Image with persistent circular loop (torus/donut hole)
    img = np.zeros((40, 40), dtype=np.float32)
    # Circle with hole in center
    y, x = np.ogrid[:40, :40]
    dist = np.sqrt((x - 20) ** 2 + (y - 20) ** 2)
    ring = ((dist >= 6) & (dist <= 14)).astype(np.float32)

    res = tda.compute_persistence_lifetime(ring)
    assert "betti_1_max" in res
    assert "betti_1_persistence_lifetime" in res
    assert res["betti_1_max"] >= 1.0
    assert res["betti_1_persistence_lifetime"] > 0.0


def test_pelvic_alignment_warping():
    warper = PelvicAlignmentWarping(target_ccd_deg=128.0)
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128

    rectified, M = warper.rectify_femoral_rotation(img, estimated_ccd_deg=136.0)
    assert rectified.shape == (64, 64, 3)
    assert M.shape == (2, 3)


def test_extreme_value_tail_calibrator():
    calibrator = ExtremeValueTailCalibrator(tail_threshold_u=0.52)

    # Low values untouched
    assert calibrator.calibrate(0.35) == 0.35

    # Borderline ambiguous values in [0.52, 0.65] are compressed downwards
    calibrated_borderline = calibrator.calibrate(0.58)
    assert calibrated_borderline < 0.58
    assert calibrated_borderline < 0.52

    # High definite pathology untouched
    assert calibrator.calibrate(0.85) == 0.85

    # Array vectorization
    probs = [0.20, 0.59, 0.90]
    out = calibrator.calibrate_array(probs)
    assert len(out) == 3
    assert out[0] == 0.20
    assert out[1] < 0.59
    assert out[2] == 0.90


def test_cross_subject_normal_centroid_filter():
    centroid_filter = CrossSubjectNormalCentroidFilter(similarity_threshold=0.85, max_dampening=0.16)

    # Generate synthetic healthy normal cohort features
    rng = np.random.RandomState(42)
    normal_feats = rng.randn(20, 32)
    # Give them a strong shared direction
    normal_feats += np.array([1.0] * 32)
    centroid_filter.fit(normal_feats)

    assert centroid_filter.normal_centroid is not None
    assert np.isclose(np.linalg.norm(centroid_filter.normal_centroid), 1.0)

    # Candidate aligned with normal centroid
    candidate_normal = centroid_filter.normal_centroid + rng.randn(32) * 0.05
    cos_sim = centroid_filter.compute_similarity(candidate_normal)
    assert cos_sim > 0.85

    # Borderline false positive should be dampened
    res_normal = centroid_filter.filter_risk(base_probability=0.58, feature_vector=candidate_normal, deep_mean=0.45)
    assert res_normal["dampened"] is True
    assert res_normal["final_risk"] < 0.58

    # True pathology safeguard: deep_mean >= 0.52 MUST PRESERVE RISK
    res_pathology = centroid_filter.filter_risk(base_probability=0.58, feature_vector=candidate_normal, deep_mean=0.55)
    assert res_pathology["dampened"] is False
    assert res_pathology["final_risk"] == 0.58
