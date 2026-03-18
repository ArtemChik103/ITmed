from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.classifier import HipDysplasiaClassifier
from models.keypoint_losses import MaskedMSELoss
from scripts.export_keypoint_backbone import export_keypoint_backbone
from tests.helpers import build_test_raster
from train.keypoint_evaluate import evaluate_keypoint_experiment
from train.keypoint_train import KeypointTrainingConfig, train_keypoint_experiment


def _write_manifest(path: Path, image_paths: list[Path], *, split_name: str) -> Path:
    records = []
    for image_path in image_paths:
        keypoints = [8, 8, 2, 12, 8, 2, 16, 8, 2, 20, 8, 2, 8, 16, 2, 12, 16, 2, 16, 16, 2, 20, 16, 2]
        records.append(
            {
                "sample_id": f"mtddh_kp::Dataset1/Keypoints/{split_name}/{image_path.name}",
                "group_id": f"mtddh_kp::{image_path.stem}",
                "group_name": image_path.stem,
                "relative_path": f"Dataset1/Keypoints/{split_name}/{image_path.name}",
                "path": str(image_path.resolve()),
                "split": split_name.lower(),
                "image_width": 32,
                "image_height": 32,
                "bbox_x": 2.0,
                "bbox_y": 2.0,
                "bbox_w": 24.0,
                "bbox_h": 24.0,
                "num_keypoints": 8,
                "keypoints_json": json.dumps(keypoints),
                "dataset_name": "MTDDH",
                "source": "MTDDH_Keypoints",
                "source_code": "mtddh_kp",
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_masked_mse_loss_is_finite():
    criterion = MaskedMSELoss()
    predictions = torch.rand(2, 8, 16, 16)
    targets = torch.rand(2, 8, 16, 16)
    visibility = torch.ones(2, 8)

    loss = criterion(predictions, targets, visibility)

    assert np.isfinite(float(loss.item()))


def test_keypoint_training_evaluate_and_export_smoke(tmp_path: Path):
    image_paths = [
        build_test_raster(tmp_path / "images" / "sample_a.jpg", pixel_array=np.full((32, 32), 96, dtype=np.uint8)),
        build_test_raster(tmp_path / "images" / "sample_b.jpg", pixel_array=np.full((32, 32), 144, dtype=np.uint8)),
    ]
    train_manifest = _write_manifest(tmp_path / "train.csv", image_paths, split_name="Train")
    val_manifest = _write_manifest(tmp_path / "val.csv", image_paths, split_name="Validation")
    config = KeypointTrainingConfig(
        manifest_path=str(train_manifest.resolve()),
        val_manifest_path=str(val_manifest.resolve()),
        experiment="synthetic_mtddh_keypoints_v1",
        experiment_dir=str((tmp_path / "checkpoints").resolve()),
        input_size=64,
        heatmap_size=16,
        batch_size=1,
        epochs=1,
        freeze_epochs=0,
        pretrained_weights_path=None,
        amp=False,
        device="cpu",
    )

    experiment_dir = train_keypoint_experiment(config)
    analysis_dir = evaluate_keypoint_experiment(experiment_dir, analysis_dir=tmp_path / "analysis", device_name="cpu")
    exported = export_keypoint_backbone(Path(experiment_dir) / "best.ckpt", tmp_path / "pretrained" / "encoder.pth")

    assert (Path(experiment_dir) / "best.ckpt").exists()
    assert (analysis_dir / "metrics.json").exists()
    assert any((analysis_dir / "visuals").glob("*.jpg"))
    assert exported.exists()

    classifier = HipDysplasiaClassifier(
        architecture="resnet50",
        pretrained=True,
        pretrained_weights_path=exported,
    )
    outputs = classifier(torch.randn(1, 3, 64, 64))
    assert outputs.shape == torch.Size([1])
