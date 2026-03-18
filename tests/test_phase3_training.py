from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.amp import GradScaler
from torch.utils.data import WeightedRandomSampler

from models.classifier import HipDysplasiaClassifier
from models.losses import build_loss
from tests.phase3_helpers import (
    create_phase3_manifest,
    write_split_snapshot,
    write_synthetic_checkpoint,
)
from train.classifier_train import (
    build_train_subset,
    create_dataloader,
    create_optimizer,
    find_optimal_threshold,
    raise_oom_system_exit,
    resolve_pretrained_weights_argument,
    save_json,
    train_one_epoch,
    validate_external_manifest_against_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_one_epoch_smoke_runs_without_device_or_shape_errors(tmp_path: Path):
    _, manifest = create_phase3_manifest(tmp_path, labels=[0, 1, 0, 1])
    sample_ids = manifest["sample_id"].tolist()
    loader = create_dataloader(
        manifest,
        sample_ids=sample_ids,
        image_size=64,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    model = HipDysplasiaClassifier(architecture="resnet50", pretrained=False)
    criterion = build_loss("bce")
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=False)

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device=torch.device("cpu"),
        scaler=scaler,
        amp_enabled=False,
        gradient_accumulation=1,
        input_size=64,
        batch_size=2,
    )

    assert np.isfinite(loss)


def test_find_optimal_threshold_stays_within_expected_bounds():
    y_true = np.array([0, 0, 1, 1, 1], dtype=int)
    probabilities = np.array([0.10, 0.30, 0.55, 0.80, 0.95], dtype=np.float32)

    threshold, metrics, sweep = find_optimal_threshold(y_true, probabilities)

    assert 0.05 <= threshold <= 0.95
    assert metrics["threshold"] == threshold
    assert sweep


def test_create_dataloader_uses_weighted_sampler_for_hard_negatives(tmp_path: Path):
    _, manifest = create_phase3_manifest(tmp_path, labels=[0, 0, 1, 1])
    sample_ids = manifest["sample_id"].tolist()
    weighted_sample_id = sample_ids[0]

    loader = create_dataloader(
        manifest,
        sample_ids=sample_ids,
        image_size=64,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        sample_weight_map={weighted_sample_id: 3.0},
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_oom_handling_message_is_explicit():
    with pytest.raises(SystemExit) as exc_info:
        raise_oom_system_exit(RuntimeError("CUDA out of memory"), input_size=384, batch_size=4)

    message = str(exc_info.value)
    assert "--input-size" in message
    assert "--batch-size" in message


def test_resolve_pretrained_weights_argument_supports_auto_and_torchvision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    local_weights = tmp_path / "resnet50_imagenet1k_v2.pth"
    local_weights.write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules["train.classifier_train"].LOCAL_PRETRAINED_WEIGHTS,
        "resnet50",
        local_weights,
    )

    assert resolve_pretrained_weights_argument("auto", "resnet50") == str(local_weights.resolve())
    assert resolve_pretrained_weights_argument("torchvision", "resnet50") is None
    assert resolve_pretrained_weights_argument("none", "resnet50") is None


def test_evaluate_cli_saves_metrics_and_model_manifest(tmp_path: Path):
    manifest_path, manifest = create_phase3_manifest(tmp_path, labels=[0, 1, 0, 1])
    holdout_sample_ids = manifest["sample_id"].tolist()
    experiment_dir = tmp_path / "experiment"
    save_json(
        experiment_dir / "experiment_config.json",
        {
            "experiment": "synthetic_resnet50_cv_v1",
            "manifest_path": str(manifest_path.resolve()),
            "architecture": "resnet50",
            "input_size": 64,
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "amp": False,
        },
    )
    write_split_snapshot(
        experiment_dir / "split_snapshot.json",
        manifest_path=manifest_path,
        holdout_sample_ids=holdout_sample_ids,
    )
    write_synthetic_checkpoint(experiment_dir / "fold_0" / "best.pt", architecture="resnet50", threshold=0.5)

    subprocess.run(
        [sys.executable, "train/evaluate.py", "--experiment-dir", str(experiment_dir)],
        cwd=REPO_ROOT,
        check=True,
    )

    metrics_path = experiment_dir / "holdout" / "metrics.json"
    holdout_predictions_path = experiment_dir / "holdout" / "predictions.csv"
    holdout_group_predictions_path = experiment_dir / "holdout" / "group_predictions.csv"
    manifest_output = experiment_dir / "model_manifest.json"
    object_level_summary_path = experiment_dir / "object_level_summary.json"
    assert metrics_path.exists()
    assert holdout_predictions_path.exists()
    assert holdout_group_predictions_path.exists()
    assert manifest_output.exists()
    assert object_level_summary_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {"sensitivity", "specificity", "accuracy", "f1", "roc_auc", "pr_auc", "confusion_matrix"} <= set(
        metrics["metrics"]
    )


def test_build_train_subset_appends_external_samples_without_touching_val_manifest(tmp_path: Path):
    _, base_manifest = create_phase3_manifest(tmp_path / "base", labels=[0, 1])
    _, extra_manifest = create_phase3_manifest(tmp_path / "extra", labels=[0, 1])
    extra_manifest = extra_manifest.assign(sample_id=lambda frame: frame["sample_id"].map(lambda value: f"mtddh::{value}"))

    train_subset, external_sample_ids = build_train_subset(
        base_manifest,
        train_sample_ids=base_manifest["sample_id"].tolist(),
        extra_manifest=extra_manifest,
        extra_train_policy="normal_only",
    )

    assert len(train_subset) == len(base_manifest) + 1
    assert len(external_sample_ids) == 1
    assert all(sample_id.startswith("mtddh::") for sample_id in external_sample_ids)


def test_validate_external_manifest_against_split_rejects_overlap(tmp_path: Path):
    _, manifest = create_phase3_manifest(tmp_path, labels=[0, 1])
    overlapping_sample_id = manifest["sample_id"].tolist()[0]
    extra_manifest = manifest.iloc[[0]].copy()
    extra_manifest["sample_id"] = overlapping_sample_id
    split_payload = {
        "holdout_sample_ids": [overlapping_sample_id],
        "folds": [{"fold": 0, "val_sample_ids": []}],
    }

    with pytest.raises(ValueError):
        validate_external_manifest_against_split(extra_manifest, split_payload)
