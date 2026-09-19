from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import torch.nn as nn

from core.preprocessor import SubPixelTrabecularDiffusionSR
from plugins.hip_dysplasia.geometry import (
    BiPlanar3DGaussianSplatting,
    FiniteElementJointStressMapper,
)
from models.calibration import ContinualSelfEvolvingActiveLearner
from plugins.hip_dysplasia.model import OrthopedicReasoningClinicalAgent
from models.classifier import InBrowserWasmInferenceEngine


def test_sub_pixel_trabecular_diffusion_sr():
    sr_module = SubPixelTrabecularDiffusionSR(scale_factor=2, num_diffusion_steps=3)
    
    # Synthetic bone patch
    rng = np.random.RandomState(42)
    bone_patch = np.ones((64, 64), dtype=np.float32) * 120.0
    bone_patch += rng.normal(0.0, 10.0, (64, 64))

    res = sr_module.enhance_trabeculae(bone_patch, sharpening_alpha=0.35)
    assert res["sr_image"].shape == (128, 128)
    assert res["scale_factor"] == 2
    assert res["trabecular_sharpness_gain"] >= 1.0
    assert res["micro_snr_db"] > 15.0
    assert res["is_microstructure_restored"] is True


def test_bi_planar_3d_gaussian_splatting():
    splatting = BiPlanar3DGaussianSplatting(num_splats=2000, nominal_cup_radius_mm=14.0)

    ap_landmarks = {"triradiate_r": (100.0, 150.0)}
    fl_landmarks = {"triradiate_r": (100.0, 151.5)}

    # 1. Normal infant cup (depth 8.5 mm, anteversion 16 deg)
    res_norm = splatting.reconstruct_3d_pelvis(
        ap_landmarks=ap_landmarks,
        frog_leg_landmarks=fl_landmarks,
        acetabular_depth_mm=8.5,
        acetabular_anteversion_deg=16.0,
        side="right",
    )
    assert res_norm["reconstructed_splats_count"] == 2000
    assert 1.8 <= res_norm["true_acetabular_volume_ml"] <= 5.0
    assert res_norm["labral_coverage_3d_pct"] > 60.0
    assert res_norm["is_3d_reconstruction_valid"] is True
    assert res_norm["is_acetabular_containment_normal"] is True

    # 2. Shallow dysplastic cup (depth 4.0 mm, anteversion 26 deg)
    res_dys = splatting.reconstruct_3d_pelvis(
        ap_landmarks=ap_landmarks,
        frog_leg_landmarks=fl_landmarks,
        acetabular_depth_mm=4.0,
        acetabular_anteversion_deg=26.0,
        side="right",
    )
    assert res_dys["true_acetabular_volume_ml"] < 3.2
    assert res_dys["is_acetabular_containment_normal"] is False
    assert res_dys["recommended_3d_volume_status"] == "shallow_hypoplastic_cup"


def test_finite_element_joint_stress_mapper():
    stress_mapper = FiniteElementJointStressMapper(normal_peak_stress_threshold_mpa=1.80)

    # 1. Normal infant hip (angle 22 deg, Reimers 15%)
    res_norm = stress_mapper.compute_contact_stress(
        acetabular_angle_deg=22.0,
        reimers_index_pct=15.0,
        body_weight_kg=7.5,
    )
    assert res_norm["peak_contact_stress_mpa"] < 1.80
    assert res_norm["biomechanical_status"] == "biomechanically_compensated"
    assert res_norm["twenty_year_osteoarthritis_risk"] < 0.15
    assert res_norm["ganz_periacetabular_osteotomy_recommended"] is False
    assert res_norm["is_contact_stress_safe"] is True

    # 2. Decompensated steep subluxated hip (angle 36 deg, Reimers 45%)
    res_decomp = stress_mapper.compute_contact_stress(
        acetabular_angle_deg=36.0,
        reimers_index_pct=45.0,
        body_weight_kg=7.5,
    )
    assert res_decomp["peak_contact_stress_mpa"] > 2.20
    assert res_decomp["biomechanical_status"] == "decompensated_high_arthrosis_risk"
    assert res_decomp["twenty_year_osteoarthritis_risk"] > 0.50
    assert res_decomp["ganz_periacetabular_osteotomy_recommended"] is True


def test_continual_self_evolving_active_learner():
    learner = ContinualSelfEvolvingActiveLearner(uncertainty_entropy_threshold=0.80)

    # High confidence case (p = 0.98)
    res_conf = learner.evaluate_sample(predicted_probability=0.98, epistemic_variance=0.005)
    assert res_conf["is_active_learning_candidate"] is False
    assert res_conf["triage_action"] == "automated_high_confidence_signoff"
    assert res_conf["predictive_entropy"] < 0.30

    # Borderline ambiguous case (p = 0.52)
    res_amb = learner.evaluate_sample(predicted_probability=0.52, epistemic_variance=0.05)
    assert res_amb["is_active_learning_candidate"] is True
    assert res_amb["triage_action"] == "escalate_to_senior_radiologist"
    assert res_amb["total_stream_samples_seen"] == 2
    assert res_amb["running_calibrated_ece"] < 0.030


def test_orthopedic_reasoning_clinical_agent():
    agent = OrthopedicReasoningClinicalAgent(high_risk_cutoff=0.60)

    # 1. Normal newborn
    res_norm = agent.reason_clinical_case(
        patient_id="PAT-NORM-1",
        age_months=3.0,
        measured_metrics={"tonnis_grade": 0, "reimers_index_pct": 12.0, "acetabular_angle_deg": 21.0, "peak_contact_stress_mpa": 1.4},
        clinical_history={"breech_presentation": False, "family_history_ddh": False},
    )
    assert len(res_norm["chain_of_thought_steps"]) == 5
    assert res_norm["is_high_risk"] is False
    assert "Normal hip anatomy" in res_norm["definitive_clinical_diagnosis"]
    assert res_norm["syndromic_dysplasia_suspected"] is False

    # 2. Severe teratologic/syndromic dislocation
    res_synd = agent.reason_clinical_case(
        patient_id="PAT-SYND-2",
        age_months=7.0,
        measured_metrics={"tonnis_grade": 3, "reimers_index_pct": 65.0, "acetabular_angle_deg": 38.0, "peak_contact_stress_mpa": 2.8},
        clinical_history={"generalized_joint_laxity": True, "breech_presentation": True},
    )
    assert res_synd["is_high_risk"] is True
    assert res_synd["syndromic_dysplasia_suspected"] is True
    assert "syndromic" in res_synd["definitive_clinical_diagnosis"].lower()


def test_in_browser_wasm_inference_engine():
    engine = InBrowserWasmInferenceEngine(target_runtime="onnxruntime-web-wasm")

    dummy_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    # 1. Export WASM Manifest
    manifest = engine.export_wasm_manifest(dummy_model, model_name="test_hip_model")
    assert manifest["format_version"] == "1.0-wasm"
    assert manifest["layers_count"] == 4  # 2 weights + 2 biases
    assert manifest["client_side_zero_footprint"] is True
    assert manifest["total_weights_size_kb"] > 0.0

    # 2. Simulate in-browser WASM latency
    sample_input = torch.randn(1, 8)
    bench = engine.simulate_in_browser_latency(dummy_model, sample_input, num_runs=20)
    assert bench["simulated_browser_latency_ms"] < 25.0
    assert bench["in_browser_fps"] > 40.0
    assert bench["is_interactive_realtime"] is True
    assert bench["client_privacy_guaranteed"] is True
