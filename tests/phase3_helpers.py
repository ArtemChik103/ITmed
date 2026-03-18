from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from models.classifier import HipDysplasiaClassifier
from models.keypoint_detector import KeypointDetector
from tests.helpers import build_test_dicom


def create_phase3_manifest(
    tmp_path: Path,
    *,
    labels: list[int],
) -> tuple[Path, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    data_root = tmp_path / "phase3_data"
    data_root.mkdir(parents=True, exist_ok=True)

    for index, label in enumerate(labels):
        class_name = "pathology" if label else "normal"
        source_code = "pathology_main" if label else "normal_main"
        source = "Патология" if label else "Норма"
        group_name = f"subject_{index:04d}"
        dicom_parent = data_root / class_name / group_name / "study"
        dicom_parent.mkdir(parents=True, exist_ok=True)
        dicom_path = build_test_dicom(
            dicom_parent / f"{index:08d}.dcm",
            pixel_spacing=[0.2, 0.2],
            patient_id=f"PAT-{index:04d}",
            study_date="20260101",
        )
        relative_path = dicom_path.relative_to(data_root).as_posix()
        records.append(
            {
                "sample_id": f"{source_code}::{relative_path}",
                "group_id": f"{source_code}::{group_name}",
                "group_name": group_name,
                "label": int(label),
                "class_name": class_name,
                "source": source,
                "source_code": source_code,
                "relative_path": relative_path,
                "path": str(dicom_path.resolve()),
            }
        )

    dataframe = pd.DataFrame(records)
    manifest_path = tmp_path / "train_manifest.csv"
    dataframe.to_csv(manifest_path, index=False)
    return manifest_path, dataframe


def write_synthetic_checkpoint(
    checkpoint_path: Path,
    *,
    architecture: str = "resnet50",
    threshold: float = 0.5,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = HipDysplasiaClassifier(architecture=architecture, pretrained=False)
    torch.save(
        {
            "fold": 0,
            "epoch": 0,
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "best_threshold": float(threshold),
            "metrics": {"sensitivity": 0.9, "specificity": 0.8, "accuracy": 0.85, "f1": 0.86},
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_synthetic_keypoint_checkpoint(
    checkpoint_path: Path,
    *,
    input_size: int = 384,
    heatmap_size: int = 96,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = KeypointDetector(pretrained=False, num_keypoints=8)
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "training_config": {
                "experiment": "synthetic_mtddh_keypoints_v1",
                "input_size": int(input_size),
                "heatmap_size": int(heatmap_size),
                "device": "cpu",
            },
            "metrics": {
                "mean_normalized_distance": 0.05,
                "pck_005": 0.8,
                "pck_010": 0.95,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_synthetic_model_manifest(
    manifest_path: Path,
    *,
    checkpoint_paths: list[Path],
    input_size: int,
    threshold: float = 0.5,
    preprocessing_profile: str = "default",
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "plugin": "hip_dysplasia",
        "experiment": "synthetic_phase3",
        "architecture": "resnet50",
        "preprocessing_profile": preprocessing_profile,
        "input_size": int(input_size),
        "ensemble_threshold": float(threshold),
        "folds": [
            {"fold": index, "checkpoint": str(path.resolve()), "threshold": float(threshold)}
            for index, path in enumerate(checkpoint_paths)
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_split_snapshot(
    path: Path,
    *,
    manifest_path: Path,
    holdout_sample_ids: list[str],
    val_sample_ids: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_path": str(manifest_path.resolve()),
        "holdout_sample_ids": holdout_sample_ids,
        "folds": [{"fold": 0, "train_sample_ids": [], "val_sample_ids": val_sample_ids or holdout_sample_ids}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
