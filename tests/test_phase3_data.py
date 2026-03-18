from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import HipDysplasiaDataset
from data.split_dataset import build_manifest, create_split
from tests.helpers import build_test_dicom

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN_ROOT = WORKSPACE_ROOT / "train"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_manifest_counts_and_excludes_jpeg_if_workspace_dataset_available():
    if not RAW_TRAIN_ROOT.exists():
        pytest.skip("External train dataset is not available in this workspace.")

    manifest = build_manifest(RAW_TRAIN_ROOT)

    assert len(manifest) == 290
    assert int(manifest["label"].sum()) == 167
    assert int((manifest["label"] == 0).sum()) == 123
    assert all(path.endswith(".dcm") for path in manifest["relative_path"])
    assert not any("патология_jpg" in path for path in manifest["relative_path"])


def test_group_split_has_no_leakage_if_workspace_dataset_available():
    if not RAW_TRAIN_ROOT.exists():
        pytest.skip("External train dataset is not available in this workspace.")

    manifest = build_manifest(RAW_TRAIN_ROOT)
    split_payload, _ = create_split(manifest, seed=42)

    holdout_groups = set(split_payload["holdout_group_ids"])
    for fold in split_payload["folds"]:
        train_groups = set(fold["train_group_ids"])
        val_groups = set(fold["val_group_ids"])
        assert holdout_groups.isdisjoint(train_groups)
        assert holdout_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(val_groups)


def test_dataset_returns_three_channel_tensor(tmp_path: Path):
    dicom_path = build_test_dicom(tmp_path / "sample.dcm", pixel_spacing=[0.2, 0.2])
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "normal_main::sample.dcm",
                "group_id": "normal_main::subject_0001",
                "group_name": "subject_0001",
                "label": 0,
                "class_name": "normal",
                "source": "Норма",
                "source_code": "normal_main",
                "relative_path": "Норма/sample.dcm",
                "path": str(dicom_path),
            }
        ]
    )

    dataset = HipDysplasiaDataset(manifest, image_size=384, train=False)
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 384, 384)
    assert sample["target"].ndim == 0


def test_eval_augmentations_fallback_without_albumentations(monkeypatch: pytest.MonkeyPatch):
    from data import augmentations

    monkeypatch.setattr(augmentations, "_HAS_ALBUMENTATIONS", False)
    transform = augmentations.get_eval_augmentations(image_size=64)
    result = transform(image=np.ones((16, 16, 3), dtype=np.float32))

    assert tuple(result["image"].shape) == (3, 64, 64)
    assert result["image"].dtype == torch.float32


def test_split_cli_is_reproducible_if_workspace_dataset_available(tmp_path: Path):
    if not RAW_TRAIN_ROOT.exists():
        pytest.skip("External train dataset is not available in this workspace.")

    output_a = tmp_path / "run_a"
    output_b = tmp_path / "run_b"

    subprocess.run(
        [
            sys.executable,
            "data/split_dataset.py",
            "--train-root",
            str(RAW_TRAIN_ROOT),
            "--output-dir",
            str(output_a),
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "data/split_dataset.py",
            "--train-root",
            str(RAW_TRAIN_ROOT),
            "--output-dir",
            str(output_b),
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    assert (output_a / "split_v1.json").read_text(encoding="utf-8") == (
        output_b / "split_v1.json"
    ).read_text(encoding="utf-8")
    assert (output_a / "folds_v1.csv").read_text(encoding="utf-8") == (
        output_b / "folds_v1.csv"
    ).read_text(encoding="utf-8")
    assert json.loads((output_a / "split_v1.json").read_text(encoding="utf-8"))["total_samples"] == 290
