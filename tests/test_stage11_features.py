from __future__ import annotations

import numpy as np
import pytest

from plugins.hip_dysplasia.geometry import (
    TonnisGradingHierarchicalHead,
    ReimersExtrusionIndexEstimator,
    CounterfactualPelvicDeficitAuditor,
)
from core.preprocessor import RadiographPositioningQAGate
from core.age_utils import GrafUltrasoundPriorDistillation
from plugins.hip_dysplasia.model import StructuredClinicalReportGenerator


def test_tonnis_grading_hierarchical_head():
    head = TonnisGradingHierarchicalHead(normal_angle_cutoff_deg=26.0)

    # 1. Normal (medial to Perkins line x=100, angle=22 deg, side=right: fx > 100)
    res_norm = head.compute_tonnis_grade(
        femoral_center=(120.0, 160.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        acetabular_roof_pt=(100.0, 120.0),
        acetabular_angle_deg=22.0,
        side="right",
    )
    assert res_norm["tonnis_grade"] == 0
    assert res_norm["is_pathology"] is False
    assert res_norm["tonnis_grade_name"] == "Normal"

    # 2. Grade 1 (medial to Perkins, but dysplastic angle 32 deg)
    res_g1 = head.compute_tonnis_grade(
        femoral_center=(120.0, 160.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        acetabular_roof_pt=(100.0, 120.0),
        acetabular_angle_deg=32.0,
        side="right",
    )
    assert res_g1["tonnis_grade"] == 1
    assert res_g1["is_pathology"] is True

    # 3. Grade 2 (lateral to Perkins fx < 100 on right side, below roof y=120 -> fy=150)
    res_g2 = head.compute_tonnis_grade(
        femoral_center=(80.0, 150.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        acetabular_roof_pt=(100.0, 120.0),
        acetabular_angle_deg=34.0,
        side="right",
    )
    assert res_g2["tonnis_grade"] == 2

    # 4. Grade 3 (level with roof y=120)
    res_g3 = head.compute_tonnis_grade(
        femoral_center=(80.0, 122.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        acetabular_roof_pt=(100.0, 120.0),
        acetabular_angle_deg=36.0,
        side="right",
    )
    assert res_g3["tonnis_grade"] == 3

    # 5. Grade 4 (high dislocation above roof fy=80 < roof_y=120)
    res_g4 = head.compute_tonnis_grade(
        femoral_center=(80.0, 80.0),
        perkins_line_x=100.0,
        hilgenreiner_line_y=150.0,
        acetabular_roof_pt=(100.0, 120.0),
        acetabular_angle_deg=38.0,
        side="right",
    )
    assert res_g4["tonnis_grade"] == 4

    # 6. Quadratic Weighted Kappa
    y_t = [0, 1, 2, 3, 4]
    y_p = [0, 1, 2, 3, 4]
    assert np.isclose(head.compute_weighted_kappa(y_t, y_p), 1.0)


def test_reimers_extrusion_index_estimator():
    estimator = ReimersExtrusionIndexEstimator(normal_cutoff_pct=33.0, dislocation_cutoff_pct=100.0)

    # 1. Normal contained femoral head (diameter 20, lateral extent 2 px -> 10%)
    res_contained = estimator.compute_migration_index(
        femur_center=(108.0, 150.0),
        femur_radius=10.0,
        perkins_line_x=100.0,
        side="right",
    )
    assert res_contained["category"] == "normal"
    assert res_contained["is_contained"] is True
    assert res_contained["reimers_index_pct"] < 33.0

    # 2. Subluxation (femur centered on Perkins line -> 50% extruded)
    res_sublux = estimator.compute_migration_index(
        femur_center=(100.0, 150.0),
        femur_radius=10.0,
        perkins_line_x=100.0,
        side="right",
    )
    assert res_sublux["category"] == "subluxation"
    assert res_sublux["is_contained"] is False
    assert 45.0 <= res_sublux["reimers_index_pct"] <= 55.0

    # 3. Dislocation (> 100% extruded)
    res_disloc = estimator.compute_migration_index(
        femur_center=(80.0, 150.0),
        femur_radius=10.0,
        perkins_line_x=100.0,
        side="right",
    )
    assert res_disloc["category"] == "dislocation"
    assert res_disloc["reimers_index_pct"] >= 100.0


def test_counterfactual_pelvic_deficit_auditor():
    auditor = CounterfactualPelvicDeficitAuditor(normative_angle_deg=22.0)

    # Normal angle: 21 deg -> 0 deficit
    res_norm = auditor.audit_acetabular_deficit(measured_angle_deg=21.0, roof_length_mm=25.0)
    assert res_norm["deficit_area_mm2"] == 0.0
    assert res_norm["has_significant_deficit"] is False
    assert res_norm["bone_coverage_adequacy_pct"] == 100.0

    # Dysplastic angle: 34 deg -> positive deficit area
    res_dys = auditor.audit_acetabular_deficit(measured_angle_deg=34.0, roof_length_mm=25.0)
    assert res_dys["deficit_area_mm2"] > 0.0
    assert res_dys["has_significant_deficit"] is True
    assert res_dys["delta_angle_deg"] == 12.0
    assert res_dys["bone_coverage_adequacy_pct"] < 60.0


def test_radiograph_positioning_qa_gate():
    qa = RadiographPositioningQAGate()

    # Optimal positioning (left/right ratio = 1.02, tilt = 1 deg)
    res_opt = qa.evaluate_positioning_qa(left_obturator_width_or_area=102.0, right_obturator_width_or_area=100.0, pelvic_tilt_angle_deg=1.0)
    assert res_opt["positioning_status"] == "optimal"
    assert res_opt["is_diagnostically_usable"] is True
    assert res_opt["positioning_qa_score"] > 0.90

    # Severe rotation (ratio = 0.50, severe asymmetric tilt)
    res_sev = qa.evaluate_positioning_qa(left_obturator_width_or_area=50.0, right_obturator_width_or_area=100.0, pelvic_tilt_angle_deg=5.0)
    assert res_sev["positioning_status"] == "severe_rotation_repeat_recommended"
    assert res_sev["is_diagnostically_usable"] is False


def test_graf_ultrasound_prior_distillation():
    distiller = GrafUltrasoundPriorDistillation()

    # 1. Normal mature (X-ray 22 deg -> alpha ~ 68 deg)
    res_norm = distiller.estimate_graf_type(acetabular_angle_xray=22.0, age_months=4.0)
    assert res_norm["graf_type"] == "Type_I"
    assert res_norm["is_ultrasound_pathology"] is False
    assert res_norm["estimated_alpha_deg"] >= 60.0

    # 2. Physiological delay in newborn < 3 months (X-ray 34 deg -> alpha ~ 56 deg)
    res_neo = distiller.estimate_graf_type(acetabular_angle_xray=34.0, age_months=1.5)
    assert res_neo["graf_type"] == "Type_IIa"
    assert res_neo["is_ultrasound_pathology"] is False

    # 3. Dysplasia in older infant (X-ray 34 deg, age 6 months -> alpha ~ 56 deg)
    res_infant = distiller.estimate_graf_type(acetabular_angle_xray=34.0, age_months=6.0)
    assert res_infant["graf_type"] == "Type_IIb"
    assert res_infant["is_ultrasound_pathology"] is True


def test_structured_clinical_report_generator():
    generator = StructuredClinicalReportGenerator()

    left_hip = {
        "hilgenreiner_angle_deg": 22.0,
        "wiberg_lce_angle_deg": 26.0,
        "reimers_index_pct": 12.0,
        "tonnis_grade": 0,
        "tonnis_grade_name": "Normal",
        "graf_type": "Type_I",
    }
    right_hip = {
        "hilgenreiner_angle_deg": 33.0,
        "wiberg_lce_angle_deg": 14.0,
        "reimers_index_pct": 42.0,
        "tonnis_grade": 2,
        "tonnis_grade_name": "Tonnis_Grade_II_Subluxation",
        "graf_type": "Type_IIb",
    }
    qa = {"positioning_status": "optimal", "positioning_qa_score": 0.98, "obturator_symmetry_ratio": 1.02}
    conformal = {"prediction_set": [1], "is_ambiguous": False}

    report = generator.generate_report(
        patient_id="PATIENT-TEST-001",
        age_months=5.5,
        left_hip=left_hip,
        right_hip=right_hip,
        positioning_qa=qa,
        conformal_info=conformal,
        calibrated_probability=0.88,
    )

    assert report["has_pathology"] is True
    assert "Peak Tönnis Grade 2" in report["overall_diagnosis"]
    assert "Pavlik harness" in report["recommendation"]
    assert "PROTOCOL OF RADIOGRAPHIC HIP INVESTIGATION" in report["formatted_markdown"]
    assert "Hilgenreiner Angle" in report["formatted_markdown"]
