from __future__ import annotations

import numpy as np
import pandas as pd

from train.aggregation import aggregate_probability, build_group_prediction_table, merge_predictions_with_manifest


def test_merge_predictions_with_manifest_attaches_group_metadata():
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "group_id": "g0",
                "group_name": "subject_0",
                "label": 0,
                "class_name": "normal",
                "source": "Норма",
                "source_code": "normal_main",
                "relative_path": "normal/0.dcm",
                "path": "C:/tmp/0.dcm",
            },
            {
                "sample_id": "s1",
                "group_id": "g1",
                "group_name": "subject_1",
                "label": 1,
                "class_name": "pathology",
                "source": "Патология",
                "source_code": "pathology_main",
                "relative_path": "pathology/1.dcm",
                "path": "C:/tmp/1.dcm",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {"sample_id": "s0", "target": 0, "probability": 0.2},
            {"sample_id": "s1", "target": 1, "probability": 0.8},
        ]
    )

    merged = merge_predictions_with_manifest(predictions, manifest)

    assert merged["group_id"].tolist() == ["g0", "g1"]
    assert merged["source_code"].tolist() == ["normal_main", "pathology_main"]


def test_merge_predictions_with_manifest_preserves_existing_columns_and_adds_missing_ones():
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "group_id": "g0",
                "group_name": "subject_0",
                "label": 0,
                "class_name": "normal",
                "source": "Норма",
                "source_code": "normal_main",
                "relative_path": "normal/0.dcm",
                "path": "C:/tmp/0.dcm",
                "modality": "RF",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "target": 0,
                "probability": 0.2,
                "group_id": "g0",
            }
        ]
    )

    merged = merge_predictions_with_manifest(predictions, manifest)

    assert merged["group_id"].tolist() == ["g0"]
    assert merged["modality"].tolist() == ["RF"]
    assert "group_id_x" not in merged.columns
    assert "group_id_y" not in merged.columns


def test_build_group_prediction_table_computes_multiple_aggregation_methods():
    predictions = pd.DataFrame(
        [
            {"sample_id": "s0", "group_id": "g0", "target": 0, "probability": 0.2, "source_code": "normal_main"},
            {"sample_id": "s1", "group_id": "g0", "target": 0, "probability": 0.6, "source_code": "normal_main"},
            {"sample_id": "s2", "group_id": "g1", "target": 1, "probability": 0.7, "source_code": "pathology_main"},
            {"sample_id": "s3", "group_id": "g1", "target": 1, "probability": 0.9, "source_code": "pathology_main"},
        ]
    )

    grouped = build_group_prediction_table(predictions)

    assert grouped["group_id"].tolist() == ["g0", "g1"]
    assert grouped["sample_count"].tolist() == [2, 2]
    assert np.isclose(grouped.loc[0, "probability_max"], 0.6)
    assert np.isclose(grouped.loc[0, "probability_mean"], 0.4)
    assert np.isclose(grouped.loc[1, "probability_topk_mean"], 0.8)
    assert 0.0 < grouped.loc[0, "probability_logit_mean"] < 1.0


def test_aggregate_probability_logit_mean_stays_between_mean_and_max():
    probability = aggregate_probability([0.2, 0.8], method="logit_mean")

    assert np.isclose(probability, 0.5)


def test_longitudinal_trajectory_slope_and_aggregation():
    from train.aggregation import compute_longitudinal_slope

    # Increasing trend across visits: 0.2 -> 0.4 -> 0.6
    slope = compute_longitudinal_slope([0.2, 0.4, 0.6])
    assert slope > 0.0
    assert np.isclose(slope, 0.2)

    # Flat trend
    assert compute_longitudinal_slope([0.5, 0.5]) == 0.0

    # Single value
    assert compute_longitudinal_slope([0.5]) == 0.0

    # Aggregation with longitudinal slope adjusts top value
    agg = aggregate_probability([0.3, 0.6], method="longitudinal_slope")
    assert agg >= 0.6


def test_clinical_consensus_aggregation_and_calibrator():
    from train.aggregation import ConsensusRankingCalibrator

    # Multi-sample consensus
    val = aggregate_probability([0.4, 0.8], method="clinical_consensus")
    assert np.isclose(val, 0.60 * 0.8 + 0.40 * 0.6)

    # Single sample consensus
    assert np.isclose(aggregate_probability([0.75], method="clinical_consensus"), 0.75)

    # Calibrator blending peak sample and deep representation consensus
    calibrator = ConsensusRankingCalibrator(deep_weight=0.45, sample_weight=0.55)
    patient_risk = calibrator.calibrate_patient_risk([0.42], [0.60])
    assert 0.42 < patient_risk < 0.60
    assert np.isclose(patient_risk, 0.55 * 0.42 + 0.45 * 0.60)


