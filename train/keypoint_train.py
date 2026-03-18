"""Train a heatmap-based MTDDH keypoint detector."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.keypoint_dataset import MTDDHKeypointDataset
from models.keypoint_detector import KeypointDetector
from models.keypoint_losses import MaskedMSELoss, decode_heatmaps
from train.classifier_train import (
    is_out_of_memory_error,
    resolve_device,
    resolve_pretrained_weights_argument,
    save_json,
    set_seed,
)


@dataclass(slots=True)
class KeypointTrainingConfig:
    manifest_path: str
    val_manifest_path: str
    experiment: str
    experiment_dir: str
    architecture: str = "resnet50"
    input_size: int = 384
    heatmap_size: int = 96
    sigma: float = 2.0
    batch_size: int = 2
    epochs: int = 20
    freeze_epochs: int = 2
    learning_rate_head: float = 1e-3
    learning_rate_finetune: float = 5e-5
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    pretrained_weights_path: str | None = None
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def raise_keypoint_oom_system_exit(exc: BaseException, *, input_size: int, batch_size: int) -> None:
    raise SystemExit(
        "Keypoint training aborted due to out-of-memory. "
        f"Reduce --batch-size below {batch_size} before lowering --input-size below {input_size}."
    ) from exc


def create_keypoint_dataloader(
    manifest: str | Path,
    *,
    image_size: int,
    heatmap_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = MTDDHKeypointDataset(
        manifest,
        image_size=image_size,
        heatmap_size=heatmap_size,
        train=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _build_optimizer(model: KeypointDetector, *, learning_rate: float, weight_decay: float) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _compute_val_metrics(
    model: KeypointDetector,
    loader: DataLoader,
    criterion: MaskedMSELoss,
    *,
    device: torch.device,
    amp_enabled: bool,
    input_size: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    visible_errors: list[float] = []
    visible_normalized_errors: list[float] = []
    pck_005: list[float] = []
    pck_010: list[float] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["heatmaps"].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)

            try:
                with autocast(device_type="cuda", enabled=amp_enabled):
                    predictions = model(inputs)
                    loss = criterion(predictions, targets, visibility)
            except RuntimeError as exc:
                if is_out_of_memory_error(exc):
                    raise_keypoint_oom_system_exit(exc, input_size=input_size, batch_size=batch_size)
                raise

            decoded = decode_heatmaps(predictions, image_size=input_size)
            target_xy = batch["keypoints_xy"].to(device, non_blocking=True)
            pixel_error = torch.linalg.norm(decoded.keypoints_xy - target_xy, dim=-1)
            bbox_diagonal = torch.sqrt(torch.clamp((bbox[:, 2] ** 2) + (bbox[:, 3] ** 2), min=1.0))
            normalized_error = pixel_error / bbox_diagonal[:, None]
            visible_mask = visibility > 0
            if torch.any(visible_mask):
                visible_errors.extend(pixel_error[visible_mask].detach().cpu().tolist())
                visible_normalized_errors.extend(normalized_error[visible_mask].detach().cpu().tolist())
                pck_005.extend((normalized_error[visible_mask] <= 0.05).detach().cpu().float().tolist())
                pck_010.extend((normalized_error[visible_mask] <= 0.10).detach().cpu().float().tolist())

            loss_total += float(loss.item()) * int(inputs.shape[0])
            sample_count += int(inputs.shape[0])

    return {
        "loss": loss_total / max(sample_count, 1),
        "mean_pixel_error": sum(visible_errors) / max(len(visible_errors), 1) if visible_errors else None,
        "mean_normalized_distance": sum(visible_normalized_errors) / max(len(visible_normalized_errors), 1)
        if visible_normalized_errors
        else None,
        "pck_005": sum(pck_005) / max(len(pck_005), 1) if pck_005 else None,
        "pck_010": sum(pck_010) / max(len(pck_010), 1) if pck_010 else None,
    }


def train_keypoint_experiment(config: KeypointTrainingConfig) -> Path:
    set_seed(config.seed)
    experiment_dir = Path(config.experiment_dir).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "experiment_config.json", config.to_dict())

    device = resolve_device(config.device)
    train_loader = create_keypoint_dataloader(
        config.manifest_path,
        image_size=config.input_size,
        heatmap_size=config.heatmap_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = create_keypoint_dataloader(
        config.val_manifest_path,
        image_size=config.input_size,
        heatmap_size=config.heatmap_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = KeypointDetector(
        num_keypoints=8,
        pretrained=True,
        pretrained_weights_path=config.pretrained_weights_path,
    ).to(device)
    model.freeze_encoder(config.freeze_epochs > 0)
    criterion = MaskedMSELoss()
    optimizer = _build_optimizer(
        model,
        learning_rate=config.learning_rate_head,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    amp_enabled = bool(config.amp and device.type == "cuda")
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_path = experiment_dir / "best.ckpt"

    for epoch in range(config.epochs):
        if epoch == config.freeze_epochs and config.freeze_epochs > 0:
            model.freeze_encoder(False)
            optimizer = _build_optimizer(
                model,
                learning_rate=config.learning_rate_finetune,
                weight_decay=config.weight_decay,
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["heatmaps"].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)
            sample_count += int(inputs.shape[0])

            try:
                with autocast(device_type="cuda", enabled=amp_enabled):
                    predictions = model(inputs)
                    loss = criterion(predictions, targets, visibility)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                if is_out_of_memory_error(exc):
                    raise_keypoint_oom_system_exit(exc, input_size=config.input_size, batch_size=config.batch_size)
                raise

            running_loss += float(loss.detach().item()) * int(inputs.shape[0])

        train_loss = running_loss / max(sample_count, 1)
        val_metrics = _compute_val_metrics(
            model,
            val_loader,
            criterion,
            device=device,
            amp_enabled=amp_enabled,
            input_size=config.input_size,
            batch_size=config.batch_size,
        )
        epoch_payload = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            **val_metrics,
        }
        history.append(epoch_payload)
        save_json(experiment_dir / "history.json", {"epochs": history})

        metric_for_selection = float(val_metrics["mean_normalized_distance"] or float("inf"))
        if metric_for_selection < best_score:
            best_score = metric_for_selection
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "training_config": config.to_dict(),
                    "metrics": epoch_payload,
                },
                checkpoint_path,
            )

    save_json(
        experiment_dir / "train_summary.json",
        {
            "experiment": config.experiment,
            "generated_at": datetime.now(UTC).isoformat(),
            "device": str(device),
            "best_checkpoint": str(checkpoint_path),
            "history": history,
        },
    )
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-50 heatmap keypoint detector on MTDDH Dataset1.")
    parser.add_argument("--manifest", required=True, help="Path to mtddh_keypoints_train.csv")
    parser.add_argument("--val-manifest", required=True, help="Path to mtddh_keypoints_val.csv")
    parser.add_argument("--experiment", required=True, help="Experiment name under models/checkpoints/")
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--heatmap-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--learning-rate-head", type=float, default=1e-3)
    parser.add_argument("--learning-rate-finetune", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-weights",
        default="auto",
        help="ImageNet initialization source: auto, torchvision, none, or a local checkpoint path.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = KeypointTrainingConfig(
        manifest_path=str(Path(args.manifest).resolve()),
        val_manifest_path=str(Path(args.val_manifest).resolve()),
        experiment=args.experiment,
        experiment_dir=str((Path("models") / "checkpoints" / args.experiment).resolve()),
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        learning_rate_head=args.learning_rate_head,
        learning_rate_finetune=args.learning_rate_finetune,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.disable_amp,
        pretrained_weights_path=resolve_pretrained_weights_argument(args.pretrained_weights, "resnet50"),
        device=args.device,
    )

    experiment_dir = train_keypoint_experiment(config)
    print(json.dumps({"status": "ok", "experiment_dir": str(experiment_dir), "experiment": config.experiment}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
