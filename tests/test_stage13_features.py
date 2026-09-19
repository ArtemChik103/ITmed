from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import torch.nn as nn

from plugins.hip_dysplasia.geometry import EpipolarFrogLegMultiViewFusion
from models.calibration import FederatedDomainShiftCalibrator
from models.classifier import SpatioTemporalPelvicGAT, SparseQuantizedEdgeEngine
from core.preprocessor import DiffusionAnatomicalSaliencyAuditor
from plugins.hip_dysplasia.model import FHIRStructuredClinicalExporter


def test_epipolar_frog_leg_multi_view_fusion():
    fusion = EpipolarFrogLegMultiViewFusion(epipolar_tolerance_px=20.0)

    ap_landmarks = {
        "triradiate_r": (120.0, 150.0),
        "femur_r": (110.0, 160.0),
    }
    fl_landmarks = {
        "triradiate_r": (120.0, 154.0),  # delta y = 4 px <= 20 px
        "femur_r": (120.0, 155.0),
    }

    # Case 1: Subluxated on AP (35% migration), but centers on frog-leg (15%) -> dynamically reducible!
    res_red = fusion.fuse_views(
        ap_landmarks=ap_landmarks,
        frog_leg_landmarks=fl_landmarks,
        ap_acetabular_angle_deg=28.0,
        frog_leg_acetabular_angle_deg=22.0,
        ap_reimers_pct=35.0,
        frog_leg_reimers_pct=15.0,
        side="right",
    )
    assert res_red["is_epipolar_aligned"] is True
    assert res_red["epipolar_height_delta_px"] == 4.0
    assert res_red["is_dynamically_reducible"] is True
    assert res_red["dynamic_reducibility_status"] == "reducible_in_abduction"
    assert res_red["dynamic_pelvic_containment_index"] > 0.70
    assert res_red["conservative_treatment_candidacy"] is True

    # Case 2: Fixed dislocation (65% migration on AP, 60% on frog-leg) -> irreducible!
    res_irr = fusion.fuse_views(
        ap_landmarks=ap_landmarks,
        frog_leg_landmarks=fl_landmarks,
        ap_acetabular_angle_deg=38.0,
        frog_leg_acetabular_angle_deg=36.0,
        ap_reimers_pct=65.0,
        frog_leg_reimers_pct=60.0,
        side="right",
    )
    assert res_irr["is_dynamically_reducible"] is False
    assert res_irr["dynamic_reducibility_status"] == "irreducible_or_fixed_dislocation"
    assert res_irr["conservative_treatment_candidacy"] is False


def test_federated_domain_shift_calibrator():
    rng = np.random.RandomState(42)
    source_features = rng.normal(loc=0.0, scale=1.0, size=(100, 32))
    calibrator = FederatedDomainShiftCalibrator(source_feature_dim=32, privacy_epsilon=1.5)
    calibrator.register_source_domain(source_features)
    assert calibrator.is_source_registered is True

    # 1. In-domain target site features (Siemens vs Siemens)
    target_indomain = rng.normal(loc=0.02, scale=1.01, size=(80, 32))
    disc_in = calibrator.compute_domain_discrepancy(target_indomain, apply_dp=True)
    assert disc_in["domain_shift_detected"] is False
    assert disc_in["source_target_mmd"] < 0.20
    assert disc_in["dp_privacy_epsilon"] == 1.5

    # 2. Out-of-domain target site features with vendor shift (GE vs Siemens)
    target_ood = rng.normal(loc=1.5, scale=2.0, size=(80, 32))
    disc_ood = calibrator.compute_domain_discrepancy(target_ood, apply_dp=True)
    assert disc_ood["domain_shift_detected"] is True
    assert disc_ood["source_target_mmd"] > 0.25

    # 3. Calibration adaptation under shift
    target_probs = np.array([0.10, 0.45, 0.85, 0.95])
    cal_res = calibrator.calibrate_target_domain(target_probs, target_ood)
    assert len(cal_res["adapted_probabilities"]) == 4
    assert cal_res["domain_adaptation_factor"] > 1.0
    assert cal_res["is_calibrated"] is True


def test_spatio_temporal_pelvic_gat():
    gat = SpatioTemporalPelvicGAT(in_features=4, hidden_dim=16, num_heads=2)

    # 1. Symmetric healthy infant pelvis
    sym_dict = {
        "ilium_l": [80.0, 60.0, 1.0, 5.0],
        "ilium_r": [160.0, 60.0, 1.0, 5.0],
        "ischium_l": [90.0, 120.0, 1.0, 5.0],
        "ischium_r": [150.0, 120.0, 1.0, 5.0],
        "pubis_l": [110.0, 140.0, 1.0, 5.0],
        "pubis_r": [130.0, 140.0, 1.0, 5.0],
        "triradiate_l": [95.0, 110.0, 1.0, 5.0],
        "triradiate_r": [145.0, 110.0, 1.0, 5.0],
        "femur_nucleus_l": [92.0, 125.0, 1.0, 5.0],
        "femur_nucleus_r": [148.0, 125.0, 1.0, 5.0],
    }
    res_sym = gat.predict_pelvic_graph(sym_dict, age_months=5.0)
    assert res_sym["ossification_asymmetry_index"] == 0.0
    assert res_sym["is_developmental_delay_suspected"] is False

    # 2. Severe unilateral hypoplasia (right nucleus volume 0.1 vs left 1.0)
    asym_dict = dict(sym_dict)
    asym_dict["femur_nucleus_r"] = [148.0, 125.0, 0.1, 5.0]
    res_asym = gat.predict_pelvic_graph(asym_dict, age_months=5.0)
    assert res_asym["ossification_asymmetry_index"] > 0.80
    assert res_asym["is_developmental_delay_suspected"] is True


def test_diffusion_anatomical_saliency_auditor():
    auditor = DiffusionAnatomicalSaliencyAuditor(pixel_spacing_mm=0.15)

    # 1. Normal infant hip (angle 22 deg == normative 22 deg)
    res_norm = auditor.audit_bone_deficit(
        current_roof_angle_deg=22.0,
        normative_roof_angle_deg=22.0,
        femoral_head_radius_px=20.0,
    )
    assert res_norm["center_edge_deficit_deg"] == 0.0
    assert res_norm["acetabular_roof_deficit_mm2"] == 0.0
    assert res_norm["is_surgical_deficit_significant"] is False
    assert res_norm["recommended_osteotomy_type"] == "conservative_remodeling_observation"

    # 2. Moderate dysplastic steep roof (32 deg vs 22 deg -> 10 deg deficit)
    res_mod = auditor.audit_bone_deficit(
        current_roof_angle_deg=32.0,
        normative_roof_angle_deg=22.0,
        femoral_head_radius_px=24.0,
    )
    assert res_mod["center_edge_deficit_deg"] == 10.0
    assert res_mod["acetabular_roof_deficit_mm2"] > 0.5
    assert res_mod["is_surgical_deficit_significant"] is True
    assert res_mod["recommended_osteotomy_type"] == "salter_innominate_osteotomy"

    # 3. Severe dysplastic deficient roof (44 deg vs 22 deg -> 22 deg deficit)
    res_sev = auditor.audit_bone_deficit(
        current_roof_angle_deg=44.0,
        normative_roof_angle_deg=22.0,
        femoral_head_radius_px=24.0,
    )
    assert res_sev["center_edge_deficit_deg"] == 22.0
    assert res_sev["recommended_osteotomy_type"] == "combined_pelvic_and_femoral_vdro"
    assert res_sev["counterfactual_saliency_mask"].shape == (128, 128)
    assert np.sum(res_sev["counterfactual_saliency_mask"]) > 0


def test_sparse_quantized_edge_engine():
    engine = SparseQuantizedEdgeEngine()

    # 1. Test 2:4 structured sparsity pattern
    w = torch.tensor([
        [1.0, 5.0, 2.0, 8.0],
        [9.0, 3.0, 7.0, 1.0],
    ])
    sparse_w = engine.apply_structured_2_4_sparsity(w)
    # Check that in each row (length 4), exactly 2 zeros exist
    assert (sparse_w[0] == 0.0).sum().item() == 2
    assert (sparse_w[1] == 0.0).sum().item() == 2
    # Check that kept elements are the top-2 magnitudes
    assert sparse_w[0, 1] == 5.0 and sparse_w[0, 3] == 8.0

    # 2. Test dynamic quantization and benchmark
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    quantized_model = engine.quantize_and_compress(model)
    dummy_input = torch.randn(1, 16)
    out = quantized_model(dummy_input)
    assert out.shape == (1, 1)

    bm = engine.benchmark_edge_throughput(quantized_model, dummy_input, num_runs=20)
    assert bm["throughput_fps"] > 100.0
    assert bm["compression_ratio"] == 4.0
    assert "2:4" in bm["structured_sparsity"]


def test_fhir_structured_clinical_exporter():
    exporter = FHIRStructuredClinicalExporter(facility_name="National Children Orthopedic Center")

    measurements = {
        "acetabular_angle_r": 31.5,
        "tonnis_grade": 2,
        "reimers_index_pct": 38.0,
        "treatment_failure_risk": 0.42,
    }
    conclusions = {
        "summary": "Tönnis Grade 2 subluxation with acetabular roof deficiency. Pavlik/splint failure risk 42%.",
    }

    # 1. Export FHIR Bundle
    bundle = exporter.export_fhir_bundle(
        patient_id="PAT-09281",
        study_uid="1.2.3.4.5.6.789",
        study_date="2026-09-19",
        measurements=measurements,
        conclusions=conclusions,
    )
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "document"
    assert len(bundle["entry"]) == 5  # 1 report + 4 observations

    report_entry = bundle["entry"][0]["resource"]
    assert report_entry["resourceType"] == "DiagnosticReport"
    assert "PAT-09281" in report_entry["subject"]["reference"]
    assert len(report_entry["result"]) == 4

    # 2. Export serialized FHIR JSON
    json_str = exporter.export_fhir_json(
        patient_id="PAT-09281",
        study_uid="1.2.3.4.5.6.789",
        study_date="2026-09-19",
        measurements=measurements,
        conclusions=conclusions,
    )
    parsed = json.loads(json_str)
    assert parsed["resourceType"] == "Bundle"

    # 3. Export DICOM SR dict
    dicom_sr = exporter.export_dicom_sr_dict(
        patient_id="PAT-09281",
        study_uid="1.2.3.4.5.6.789",
        measurements=measurements,
    )
    assert dicom_sr["SOPClassUID"] == "1.2.840.10008.5.1.4.1.1.88.33"
    assert dicom_sr["MeasuredValues"]["TonnisGrade"] == 2
