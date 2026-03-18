"""Evaluate the Phase 3 ensemble on the holdout split."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_manifest
from models.classifier import load_classifier_from_checkpoint
from train.aggregation import AGGREGATION_METHODS, build_group_prediction_table, merge_predictions_with_manifest
from train.classifier_train import (
    compute_binary_metrics,
    create_dataloader,
    find_optimal_threshold,
    load_json,
    resolve_device,
    save_json,
)


@torch.no_grad()
def _predict_checkpoint(
    checkpoint_path: str | Path,
    loader,
    *,
    device: torch.device,
    input_size: int,
    batch_size: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model, checkpoint = load_classifier_from_checkpoint(checkpoint_path, device=device)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []

    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        logits = model(inputs)
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
        sample_ids.extend(batch["sample_id"])

    return {
        "fold": int(checkpoint.get("fold", -1)),
        "threshold": float(checkpoint.get("best_threshold", 0.5)),
        "probabilities": np.concatenate(probabilities, axis=0).reshape(-1),
        "targets": np.concatenate(targets, axis=0).reshape(-1),
        "sample_ids": sample_ids,
        "input_size": input_size,
        "batch_size": batch_size,
        "amp_enabled": amp_enabled,
    }


def _build_prediction_frame(
    *,
    sample_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "sample_id": [str(sample_id) for sample_id in sample_ids],
            "target": np.asarray(targets, dtype=int).reshape(-1),
            "probability": np.asarray(probabilities, dtype=float).reshape(-1),
        }
    )
    return merge_predictions_with_manifest(predictions, manifest)


def _load_validation_predictions(
    *,
    fold_dir: Path,
    fold_spec: dict[str, Any],
    manifest: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    cached_path = fold_dir / "val_predictions.csv"
    if cached_path.exists():
        cached_predictions = pd.read_csv(cached_path)
        return merge_predictions_with_manifest(cached_predictions, manifest)

    val_sample_ids = list(fold_spec.get("val_sample_ids", []))
    if not val_sample_ids:
        raise ValueError(
            f"Fold {fold_spec.get('fold')} has no validation sample ids and no cached val_predictions.csv."
        )

    loader = create_dataloader(
        manifest,
        sample_ids=val_sample_ids,
        image_size=int(config["input_size"]),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        preprocessing_profile=str(config.get("preprocessing_profile", "default")),
    )
    prediction = _predict_checkpoint(
        fold_dir / "best.pt",
        loader,
        device=device,
        input_size=int(config["input_size"]),
        batch_size=int(config["batch_size"]),
        amp_enabled=bool(config.get("amp", True) and device.type == "cuda"),
    )
    cached_predictions = pd.DataFrame(
        {
            "sample_id": prediction["sample_ids"],
            "target": prediction["targets"].astype(int),
            "probability": prediction["probabilities"].astype(float),
        }
    )
    cached_predictions.to_csv(cached_path, index=False)
    return merge_predictions_with_manifest(cached_predictions, manifest)


def _compute_object_level_summary(
    *,
    experiment_dir: Path,
    manifest: pd.DataFrame,
    split_payload: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    holdout_prediction_frame: pd.DataFrame,
    fold_predictions: list[dict[str, Any]],
    group_key: str,
) -> tuple[Path, Path]:
    holdout_dir = experiment_dir / "holdout"
    holdout_group_table = build_group_prediction_table(
        holdout_prediction_frame,
        group_key=group_key,
        methods=AGGREGATION_METHODS,
    )

    fold_specs_by_id = {int(fold_spec["fold"]): fold_spec for fold_spec in split_payload["folds"]}
    per_fold_reports: dict[int, dict[str, Any]] = {}

    for fold_prediction in fold_predictions:
        fold = int(fold_prediction["fold"])
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_spec = fold_specs_by_id[fold]
        validation_frame = _load_validation_predictions(
            fold_dir=fold_dir,
            fold_spec=fold_spec,
            manifest=manifest,
            config=config,
            device=device,
        )
        validation_group_table = build_group_prediction_table(
            validation_frame,
            group_key=group_key,
            methods=AGGREGATION_METHODS,
        )
        validation_group_table.to_csv(fold_dir / "val_group_predictions.csv", index=False)

        method_payloads: dict[str, Any] = {}
        for method in AGGREGATION_METHODS:
            threshold, metrics, _ = find_optimal_threshold(
                validation_group_table["target"].to_numpy(dtype=int),
                validation_group_table[f"probability_{method}"].to_numpy(dtype=np.float32),
                policy=str(config.get("threshold_policy", "max_sensitivity")),
                sensitivity_floor=float(config.get("sensitivity_floor", 0.90)),
            )
            method_payloads[method] = {
                "threshold": float(threshold),
                "group_count": int(len(validation_group_table)),
                "negative_groups": int((validation_group_table["target"] == 0).sum()),
                "positive_groups": int((validation_group_table["target"] == 1).sum()),
                "metrics": metrics,
            }

        per_fold_reports[fold] = {"fold": fold, "group_key": group_key, "methods": method_payloads}
        save_json(fold_dir / "object_level_metrics.json", per_fold_reports[fold])

    methods_payload: dict[str, Any] = {}
    for method in AGGREGATION_METHODS:
        fold_method_reports = [per_fold_reports[int(prediction["fold"])]["methods"][method] for prediction in fold_predictions]
        method_threshold = float(
            np.clip(
                np.mean([fold_report["threshold"] for fold_report in fold_method_reports]),
                0.05,
                0.95,
            )
        )
        holdout_metrics = compute_binary_metrics(
            holdout_group_table["target"].to_numpy(dtype=int),
            holdout_group_table[f"probability_{method}"].to_numpy(dtype=np.float32),
            threshold=method_threshold,
        )
        holdout_group_table[f"prediction_{method}"] = (
            holdout_group_table[f"probability_{method}"].to_numpy(dtype=np.float32) >= method_threshold
        ).astype(int)

        methods_payload[method] = {
            "cv_summary": {
                "group_key": group_key,
                "mean_sensitivity": float(np.mean([report["metrics"]["sensitivity"] for report in fold_method_reports])),
                "mean_specificity": float(np.mean([report["metrics"]["specificity"] for report in fold_method_reports])),
                "mean_accuracy": float(np.mean([report["metrics"]["accuracy"] for report in fold_method_reports])),
                "mean_f1": float(np.mean([report["metrics"]["f1"] for report in fold_method_reports])),
                "mean_threshold": method_threshold,
                "folds": [
                    {
                        "fold": int(fold_report["fold"]),
                        "threshold": float(report["threshold"]),
                        "group_count": int(report["group_count"]),
                        "negative_groups": int(report["negative_groups"]),
                        "positive_groups": int(report["positive_groups"]),
                        "metrics": report["metrics"],
                    }
                    for fold_report, report in zip(
                        [per_fold_reports[int(prediction["fold"])] for prediction in fold_predictions],
                        fold_method_reports,
                        strict=True,
                    )
                ],
            },
            "holdout": {
                "group_key": group_key,
                "group_count": int(len(holdout_group_table)),
                "negative_groups": int((holdout_group_table["target"] == 0).sum()),
                "positive_groups": int((holdout_group_table["target"] == 1).sum()),
                "threshold": method_threshold,
                "metrics": holdout_metrics,
            },
        }

    holdout_group_predictions_path = holdout_dir / "group_predictions.csv"
    holdout_group_table.to_csv(holdout_group_predictions_path, index=False)

    summary_payload = {
        "experiment": config["experiment"],
        "generated_at": datetime.now(UTC).isoformat(),
        "group_key": group_key,
        "methods": methods_payload,
    }
    summary_path = experiment_dir / "object_level_summary.json"
    save_json(summary_path, summary_payload)
    return summary_path, holdout_group_predictions_path


def evaluate_experiment(experiment_dir: str | Path, *, group_key: str = "group_id") -> tuple[Path, Path, Path]:
    """Run holdout evaluation and write metrics and model manifest artifacts."""
    experiment_dir = Path(experiment_dir).resolve()
    config = load_json(experiment_dir / "experiment_config.json")
    split_payload = load_json(experiment_dir / "split_snapshot.json")
    manifest = load_manifest(config["manifest_path"])

    holdout_sample_ids = list(split_payload["holdout_sample_ids"])
    device = resolve_device(config.get("device", "auto"))
    loader = create_dataloader(
        manifest,
        sample_ids=holdout_sample_ids,
        image_size=int(config["input_size"]),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        preprocessing_profile=str(config.get("preprocessing_profile", "default")),
    )

    fold_predictions = []
    for fold_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        fold_predictions.append(
            _predict_checkpoint(
                fold_dir / "best.pt",
                loader,
                device=device,
                input_size=int(config["input_size"]),
                batch_size=int(config["batch_size"]),
                amp_enabled=bool(config.get("amp", True) and device.type == "cuda"),
            )
        )

    if not fold_predictions:
        raise ValueError(f"No fold checkpoints found in '{experiment_dir}'.")

    sample_ids = fold_predictions[0]["sample_ids"]
    targets = fold_predictions[0]["targets"].astype(int)
    probabilities = np.mean([prediction["probabilities"] for prediction in fold_predictions], axis=0)
    ensemble_threshold = float(np.clip(np.mean([prediction["threshold"] for prediction in fold_predictions]), 0.05, 0.95))
    holdout_metrics = compute_binary_metrics(targets, probabilities, threshold=ensemble_threshold)
    holdout_prediction_frame = _build_prediction_frame(
        sample_ids=sample_ids,
        targets=targets,
        probabilities=probabilities,
        manifest=manifest,
    )

    holdout_dir = experiment_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_prediction_frame.to_csv(holdout_dir / "predictions.csv", index=False)

    metrics_payload = {
        "experiment": config["experiment"],
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "holdout_samples": len(sample_ids),
        "ensemble_threshold": ensemble_threshold,
        "metrics": holdout_metrics,
        "sample_ids": sample_ids,
    }
    metrics_path = holdout_dir / "metrics.json"
    save_json(metrics_path, metrics_payload)

    model_manifest = {
        "schema_version": 1,
        "plugin": "hip_dysplasia",
        "experiment": config["experiment"],
        "generated_at": datetime.now(UTC).isoformat(),
        "architecture": config["architecture"],
        "preprocessing_profile": str(config.get("preprocessing_profile", "default")),
        "input_size": int(config["input_size"]),
        "ensemble_threshold": ensemble_threshold,
        "cv_threshold_mean": ensemble_threshold,
        "selection_metric": str(config.get("threshold_policy", "max_sensitivity")),
        "folds": [
            {
                "fold": int(prediction["fold"]),
                "checkpoint": str((experiment_dir / f"fold_{int(prediction['fold'])}" / "best.pt").resolve()),
                "threshold": float(prediction["threshold"]),
            }
            for prediction in fold_predictions
        ],
    }
    manifest_path = experiment_dir / "model_manifest.json"
    save_json(manifest_path, model_manifest)
    object_summary_path, _ = _compute_object_level_summary(
        experiment_dir=experiment_dir,
        manifest=manifest,
        split_payload=split_payload,
        config=config,
        device=device,
        holdout_prediction_frame=holdout_prediction_frame,
        fold_predictions=fold_predictions,
        group_key=group_key,
    )
    return metrics_path, manifest_path, object_summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Phase 3 experiment on the holdout split.")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to models/checkpoints/<experiment>",
    )
    parser.add_argument(
        "--group-key",
        default="group_id",
        help="Manifest column used for object-level aggregation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path, manifest_path, object_summary_path = evaluate_experiment(
        args.experiment_dir,
        group_key=args.group_key,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics": str(metrics_path),
                "model_manifest": str(manifest_path),
                "object_level_summary": str(object_summary_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
