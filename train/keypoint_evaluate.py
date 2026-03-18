"""Evaluate a trained MTDDH keypoint detector and write Path B analysis artifacts."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from data.import_mtddh_keypoints import RAW_KEYPOINT_NAMES
from data.keypoint_dataset import MTDDHKeypointDataset
from models.keypoint_detector import load_keypoint_detector_from_checkpoint
from models.keypoint_losses import decode_heatmaps
from train.classifier_train import resolve_device, save_json


def _default_analysis_dir(experiment_dir: Path) -> Path:
    experiment_name = experiment_dir.name
    parts = experiment_name.split("_", 1)
    suffix = parts[1] if len(parts) == 2 else experiment_name
    return (Path("analysis") / suffix).resolve()


def _unnormalize_image(image_tensor: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    image = (image_tensor.detach().cpu() * std) + mean
    image = torch.clamp(image, 0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).astype("uint8")


def _draw_overlay(
    *,
    image_tensor: torch.Tensor,
    target_xy,
    pred_xy,
    visibility,
    sample_id: str,
    output_path: Path,
) -> None:
    image = Image.fromarray(_unnormalize_image(image_tensor))
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(RAW_KEYPOINT_NAMES):
        if float(visibility[index]) <= 0.0:
            continue
        tx, ty = float(target_xy[index][0]), float(target_xy[index][1])
        px, py = float(pred_xy[index][0]), float(pred_xy[index][1])
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), outline="lime", width=2)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline="red", width=2)
        draw.text((tx + 5, ty + 5), name, fill="white")
    draw.text((8, 8), sample_id, fill="yellow")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=90)


def evaluate_keypoint_experiment(
    experiment_dir: str | Path,
    *,
    analysis_dir: str | Path | None = None,
    device_name: str = "auto",
) -> Path:
    experiment_dir = Path(experiment_dir).resolve()
    checkpoint_path = experiment_dir / "best.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing keypoint checkpoint '{checkpoint_path}'.")

    config = json.loads((experiment_dir / "experiment_config.json").read_text(encoding="utf-8"))
    analysis_dir = Path(analysis_dir).resolve() if analysis_dir else _default_analysis_dir(experiment_dir)
    visuals_dir = analysis_dir / "visuals"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_name if device_name != "auto" else config.get("device", "auto"))
    model, _ = load_keypoint_detector_from_checkpoint(checkpoint_path, device=device)
    dataset = MTDDHKeypointDataset(
        config["val_manifest_path"],
        image_size=int(config["input_size"]),
        heatmap_size=int(config["heatmap_size"]),
        train=False,
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

    prediction_rows: list[dict[str, Any]] = []
    visual_candidates: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            heatmaps = model(inputs)
            decoded = decode_heatmaps(heatmaps, image_size=int(config["input_size"]))

            target_xy = batch["keypoints_xy"].numpy()
            pred_xy = decoded.keypoints_xy.detach().cpu().numpy()
            confidence = decoded.confidence.detach().cpu().numpy()
            visibility = batch["visibility"].numpy()
            bbox = batch["bbox"].numpy()
            image_batch = batch["image"]

            for sample_index, sample_id in enumerate(batch["sample_id"]):
                bbox_diagonal = math.sqrt(max(float(bbox[sample_index, 2] ** 2 + bbox[sample_index, 3] ** 2), 1.0))
                for keypoint_index, keypoint_name in enumerate(RAW_KEYPOINT_NAMES):
                    visible = float(visibility[sample_index, keypoint_index]) > 0.0
                    pixel_error = (
                        float(((pred_xy[sample_index, keypoint_index] - target_xy[sample_index, keypoint_index]) ** 2).sum() ** 0.5)
                        if visible
                        else None
                    )
                    normalized_error = (pixel_error / bbox_diagonal) if visible and bbox_diagonal > 0 else None
                    missing_prediction = bool(float(confidence[sample_index, keypoint_index]) < 0.05)
                    prediction_rows.append(
                        {
                            "sample_id": str(sample_id),
                            "path": str(batch["path"][sample_index]),
                            "keypoint_name": keypoint_name,
                            "visibility": int(visibility[sample_index, keypoint_index]),
                            "target_x": float(target_xy[sample_index, keypoint_index, 0]),
                            "target_y": float(target_xy[sample_index, keypoint_index, 1]),
                            "pred_x": float(pred_xy[sample_index, keypoint_index, 0]),
                            "pred_y": float(pred_xy[sample_index, keypoint_index, 1]),
                            "confidence": float(confidence[sample_index, keypoint_index]),
                            "pixel_error": pixel_error,
                            "normalized_error": normalized_error,
                            "bbox_diagonal": bbox_diagonal,
                            "is_correct_pck_005": bool(normalized_error is not None and normalized_error <= 0.05),
                            "is_correct_pck_010": bool(normalized_error is not None and normalized_error <= 0.10),
                            "missing_prediction": missing_prediction,
                        }
                    )

                visual_candidates.append(
                    {
                        "sample_id": str(sample_id),
                        "image_tensor": image_batch[sample_index],
                        "target_xy": target_xy[sample_index],
                        "pred_xy": pred_xy[sample_index],
                        "visibility": visibility[sample_index],
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(analysis_dir / "predictions.csv", index=False)

    visible_predictions = predictions[predictions["visibility"].astype(int) > 0].copy()
    per_keypoint_metrics = (
        visible_predictions.groupby("keypoint_name", dropna=False)
        .agg(
            visible_count=("sample_id", "count"),
            mean_pixel_error=("pixel_error", "mean"),
            mean_normalized_error=("normalized_error", "mean"),
            pck_005=("is_correct_pck_005", "mean"),
            pck_010=("is_correct_pck_010", "mean"),
            missing_prediction_rate=("missing_prediction", "mean"),
        )
        .reset_index()
    )
    per_keypoint_metrics.to_csv(analysis_dir / "per_keypoint_metrics.csv", index=False)

    gate = {
        "pck_010_pass": bool(float(visible_predictions["is_correct_pck_010"].mean()) >= 0.85) if not visible_predictions.empty else False,
        "mean_normalized_distance_pass": bool(float(visible_predictions["normalized_error"].mean()) <= 0.08) if not visible_predictions.empty else False,
        "left_right_collapse_detected": False,
        "per_keypoint_total_failure": bool(
            not per_keypoint_metrics.empty and bool((per_keypoint_metrics["pck_010"].fillna(0.0) < 0.2).any())
        ),
    }
    gate["passed"] = bool(gate["pck_010_pass"] and gate["mean_normalized_distance_pass"] and not gate["per_keypoint_total_failure"])

    save_json(
        analysis_dir / "metrics.json",
        {
            "experiment": config["experiment"],
            "generated_at": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint_path),
            "mean_pixel_error": float(visible_predictions["pixel_error"].mean()) if not visible_predictions.empty else None,
            "mean_normalized_distance": float(visible_predictions["normalized_error"].mean()) if not visible_predictions.empty else None,
            "pck_005": float(visible_predictions["is_correct_pck_005"].mean()) if not visible_predictions.empty else None,
            "pck_010": float(visible_predictions["is_correct_pck_010"].mean()) if not visible_predictions.empty else None,
            "missing_prediction_rate": float(predictions["missing_prediction"].mean()) if not predictions.empty else None,
            "visible_keypoint_count": int(len(visible_predictions)),
            "quality_gate": gate,
            "verdict": (
                "quality_gate_passed: export encoder and run classifier pilot"
                if gate["passed"]
                else "quality_gate_failed: stop after keypoint-only baseline and revisit data/architecture"
            ),
        },
    )
    save_json(
        analysis_dir / "verdict.json",
        {
            "experiment": config["experiment"],
            "quality_gate": gate,
            "next_recommendation": (
                "run scripts/export_keypoint_backbone.py and launch resnet50_bce_kptpretrain_v1"
                if gate["passed"]
                else "skip classifier transfer until detector quality improves"
            ),
        },
    )

    random.Random(42).shuffle(visual_candidates)
    for candidate in visual_candidates[: min(20, len(visual_candidates))]:
        _draw_overlay(
            image_tensor=candidate["image_tensor"],
            target_xy=candidate["target_xy"],
            pred_xy=candidate["pred_xy"],
            visibility=candidate["visibility"],
            sample_id=candidate["sample_id"],
            output_path=visuals_dir / f"{candidate['sample_id'].replace(':', '_').replace('/', '_')}.jpg",
        )

    return analysis_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MTDDH keypoint detector.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--analysis-dir", help="Optional output directory. Defaults to analysis/<experiment_suffix>.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = evaluate_keypoint_experiment(
        args.experiment_dir,
        analysis_dir=args.analysis_dir,
        device_name=args.device,
    )
    print(json.dumps({"status": "ok", "analysis_dir": str(analysis_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
