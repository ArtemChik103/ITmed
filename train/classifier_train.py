"""Phase 3 classifier training entry point with 5-fold CV."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import HipDysplasiaDataset, load_manifest
from models.classifier import HipDysplasiaClassifier
from models.losses import build_loss
from core.preprocessor import PREPROCESSING_PROFILES

LOCAL_PRETRAINED_WEIGHTS = {
    "resnet34": Path("models/pretrained/resnet34_imagenet1k_v1.pth"),
    "resnet50": Path("models/pretrained/resnet50_imagenet1k_v2.pth"),
    "densenet121": Path("models/pretrained/densenet121_imagenet1k_v1.pth"),
    "convnext_tiny": Path("models/pretrained/convnext_tiny_imagenet1k_v1.pth"),
    "swin_t": Path("models/pretrained/swin_t_imagenet1k_v1.pth"),
    "efficientnet_b4": Path("models/pretrained/efficientnet_b4_imagenet1k_v1.pth"),
    "radimagenet_resnet50": Path("models/pretrained/radimagenet_resnet50.pth"),
}



@dataclass(slots=True)
class TrainingConfig:
    """Serializable training configuration."""

    manifest_path: str
    split_path: str
    experiment: str
    experiment_dir: str
    architecture: str = "resnet50"
    preprocessing_profile: str = "default"
    input_size: int = 384
    batch_size: int = 4
    gradient_accumulation: int = 4
    epochs: int = 6
    freeze_epochs: int = 1
    learning_rate_head: float = 1e-3
    learning_rate_finetune: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    loss_name: str = "focal"
    label_smoothing: float = 0.0
    use_swa: bool = False
    swa_start_epoch: int = 8
    mixup_prob: float = 0.0
    threshold_policy: str = "max_sensitivity"
    sensitivity_floor: float = 0.90
    save_val_predictions: bool = False
    use_pos_weight: bool = False
    hard_negative_manifest_path: str | None = None
    hard_negative_weight: float = 1.0
    extra_train_manifest_path: str | None = None
    extra_train_policy: str = "all_confident"
    extra_train_weight: float = 1.0
    pretrained_weights_path: str | None = None
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str | None) -> torch.device:
    """Resolve a device preference into a concrete torch device."""
    if device_name and device_name != "auto":
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Return True for CUDA or CPU OOM errors."""
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "cuda oom" in text)


def raise_oom_system_exit(exc: BaseException, *, input_size: int, batch_size: int) -> None:
    """Raise an explicit CLI-level OOM failure with remediation guidance."""
    raise SystemExit(
        "Training aborted due to out-of-memory. "
        f"Reduce --input-size below {input_size} or --batch-size below {batch_size}."
    ) from exc


def build_pos_weight(labels: pd.Series) -> torch.Tensor | None:
    """Compute a positive-class weight for BCE-style losses."""
    positive = int(labels.sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        return None
    return torch.tensor([negative / positive], dtype=torch.float32)


def resolve_pretrained_weights_argument(raw_value: str | None, architecture: str) -> str | None:
    """Resolve CLI input into a concrete local checkpoint path or torchvision fallback."""
    if not raw_value:
        return None

    normalized = raw_value.strip().lower()
    if normalized in {"none", "torchvision"}:
        return None

    if normalized == "auto":
        candidate = LOCAL_PRETRAINED_WEIGHTS.get(architecture)
        if candidate is not None and candidate.exists():
            return str(candidate.resolve())
        return None

    return str(Path(raw_value).resolve())


def create_dataloader(
    manifest: pd.DataFrame,
    *,
    sample_ids: list[str] | None,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    preprocessing_profile: str = "default",
    sample_weight_map: dict[str, float] | None = None,
) -> DataLoader:
    """Create a dataloader for a subset of the manifest."""
    if sample_ids is None:
        subset = manifest.copy()
    else:
        subset = manifest[manifest["sample_id"].astype(str).isin({str(sample_id) for sample_id in sample_ids})].copy()
    subset = subset.sort_values(["group_id", "relative_path"]).reset_index(drop=True)
    dataset = HipDysplasiaDataset(
        subset,
        image_size=image_size,
        preprocessing_profile=preprocessing_profile,
        train=shuffle,
    )
    sampler = None
    if shuffle and sample_weight_map:
        sample_weights = (
            subset["sample_id"]
            .astype(str)
            .map(sample_weight_map)
            .fillna(1.0)
            .astype(float)
            .to_numpy()
        )
        if not np.allclose(sample_weights, np.ones_like(sample_weights)):
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute the Phase 3 evaluation metrics for a binary classifier."""
    y_true = y_true.astype(int)
    probabilities = np.nan_to_num(np.asarray(probabilities, dtype=np.float32), nan=0.5)
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = float(accuracy_score(y_true, predictions))
    f1 = float(f1_score(y_true, predictions, zero_division=0))

    roc_auc: float | None
    pr_auc: float | None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
    else:
        roc_auc = None
        pr_auc = None

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_predictions": int(predictions.sum()),
        "negative_predictions": int((predictions == 0).sum()),
    }


def build_threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> list[dict[str, Any]]:
    """Compute metrics across a threshold grid."""
    thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)
    return [
        compute_binary_metrics(y_true, probabilities, threshold=float(threshold))
        for threshold in thresholds
    ]


def compute_roc_convex_hull(sweep: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter threshold sweep points to the upper Pareto ROC convex hull."""
    if not sweep:
        return []

    # Map each point to (fpr, tpr)
    points = []
    for m in sweep:
        tpr = float(m.get("sensitivity", 0.0))
        fpr = round(1.0 - float(m.get("specificity", 1.0)), 6)
        points.append((fpr, tpr, m))

    # Sort primarily by fpr ascending, secondarily by tpr descending
    points.sort(key=lambda item: (item[0], -item[1]))

    # Filter Pareto non-dominated: cannot have higher FPR with lower or equal TPR
    pareto_points = []
    max_tpr_seen = -1.0
    for fpr, tpr, m in points:
        if tpr > max_tpr_seen:
            pareto_points.append((fpr, tpr, m))
            max_tpr_seen = tpr

    # Compute upper convex hull using 2D cross product
    hull: list[tuple[float, float, dict[str, Any]]] = []
    for p in pareto_points:
        while len(hull) >= 2:
            x1, y1, _ = hull[-2]
            x2, y2, _ = hull[-1]
            x3, y3, _ = p
            # Cross product (p2 - p1) x (p3 - p2)
            cp = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            if cp >= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    return [item[2] for item in hull] if hull else sweep


class MultiModelConvexHullOptimizer:
    """Finds the Pareto-optimal mixture weights over multi-model ensembles on the ROC Convex Hull."""

    def __init__(self, sensitivity_floor: float = 0.90) -> None:
        self.sensitivity_floor = sensitivity_floor

    def optimize_ensemble_weights(
        self,
        y_true: np.ndarray,
        predictions_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        predictions_matrix: [N, M] probabilities from M models
        Returns:
            optimal_weights: [M]
            optimal_threshold: float
            best_metrics: dict
        """
        n_samples, n_models = predictions_matrix.shape
        if n_models == 1:
            best_t, best_m, _ = find_optimal_threshold(
                y_true, predictions_matrix[:, 0], policy="pareto_convex_hull", sensitivity_floor=self.sensitivity_floor
            )
            return np.array([1.0]), best_t, best_m

        from scipy.optimize import differential_evolution

        def obj(raw_weights: np.ndarray) -> float:
            w = np.clip(raw_weights, 1e-5, 1.0)
            w = w / np.sum(w)
            blended = predictions_matrix @ w
            roc = float(roc_auc_score(y_true, blended)) if len(np.unique(y_true)) > 1 else 0.5
            pr = float(average_precision_score(y_true, blended)) if len(np.unique(y_true)) > 1 else 0.5
            return -(roc + pr)

        bounds = [(0.0, 1.0) for _ in range(n_models)]
        res = differential_evolution(obj, bounds, seed=42, maxiter=40, popsize=10)
        optimal_weights = res.x / np.sum(res.x)

        blended_prob = predictions_matrix @ optimal_weights
        best_threshold, best_metrics, _ = find_optimal_threshold(
            y_true,
            blended_prob,
            policy="pareto_convex_hull",
            sensitivity_floor=self.sensitivity_floor,
        )
        return optimal_weights, best_threshold, best_metrics


def _threshold_score(
    metrics: dict[str, Any],
    *,
    policy: str,
    sensitivity_floor: float,
) -> tuple[float, ...]:
    threshold = float(metrics["threshold"])
    if policy in {"pareto_convex_hull", "roc_convex_hull"}:
        meets_floor = 1.0 if float(metrics["sensitivity"]) >= sensitivity_floor else 0.0
        youden = float(metrics["sensitivity"]) + float(metrics["specificity"]) - 1.0
        return (
            meets_floor,
            youden,
            float(metrics["specificity"]),
            float(metrics.get("f1", 0.0)),
            float(metrics["sensitivity"]),
            -abs(threshold - 0.5),
        )

    if policy == "cost_sensitive":
        cm = metrics.get("confusion_matrix", {})
        fn = float(cm.get("fn", 0))
        fp = float(cm.get("fp", 0))
        tp = float(cm.get("tp", 0))
        tn = float(cm.get("tn", 0))
        total_pos = max(tp + fn, 1.0)
        total_neg = max(tn + fp, 1.0)
        max_cost = 5.0 * total_pos + 1.0 * total_neg
        cost = 5.0 * fn + 1.0 * fp
        clinical_utility = 1.0 - (cost / max_cost)
        balanced_acc = (float(metrics["sensitivity"]) + float(metrics["specificity"])) / 2.0
        return (
            clinical_utility,
            balanced_acc,
            float(metrics["sensitivity"]),
            float(metrics["f1"]),
            -abs(threshold - 0.5),
        )

    if policy == "max_specificity_under_sensitivity_floor":
        meets_floor = 1.0 if float(metrics["sensitivity"]) >= sensitivity_floor else 0.0
        return (
            meets_floor,
            float(metrics["specificity"]),
            float(metrics["accuracy"]),
            float(metrics["f1"]),
            float(metrics["sensitivity"]),
            -abs(threshold - 0.5),
        )

    if policy in {"balanced_accuracy", "youden"}:
        balanced_acc = (float(metrics["sensitivity"]) + float(metrics["specificity"])) / 2.0
        return (
            balanced_acc,
            float(metrics["f1"]),
            float(metrics["accuracy"]),
            float(metrics["sensitivity"]),
            float(metrics["specificity"]),
            -abs(threshold - 0.5),
        )

    if policy == "max_f1":
        return (
            float(metrics["f1"]),
            float(metrics["accuracy"]),
            float(metrics["sensitivity"]),
            float(metrics["specificity"]),
            -abs(threshold - 0.5),
        )

    return (
        float(metrics["sensitivity"]),
        float(metrics["f1"]),
        float(metrics["specificity"]),
        float(metrics["accuracy"]),
        -abs(threshold - 0.5),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    policy: str = "max_sensitivity",
    sensitivity_floor: float = 0.90,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Select a threshold in [0.05, 0.95] under the requested policy."""
    sweep = build_threshold_sweep(y_true, probabilities)
    candidate_pool = compute_roc_convex_hull(sweep) if policy in {"pareto_convex_hull", "roc_convex_hull"} else sweep
    if not candidate_pool:
        candidate_pool = sweep

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, probabilities, threshold=best_threshold)
    best_score = _threshold_score(
        best_metrics,
        policy=policy,
        sensitivity_floor=sensitivity_floor,
    )

    for metrics in candidate_pool:
        score = _threshold_score(
            metrics,
            policy=policy,
            sensitivity_floor=sensitivity_floor,
        )
        if score > best_score:
            best_score = score
            best_threshold = float(metrics["threshold"])
            best_metrics = metrics

    return best_threshold, best_metrics, sweep



def load_hard_negative_groups(
    hard_negative_manifest_path: str | None,
) -> dict[int, set[str]]:
    """Load per-fold hard-negative group ids from disk."""
    if not hard_negative_manifest_path:
        return {}

    payload = load_json(hard_negative_manifest_path)
    folds = payload.get("folds", [])
    result: dict[int, set[str]] = {}
    for fold_payload in folds:
        fold = int(fold_payload["fold"])
        result[fold] = {str(group_id) for group_id in fold_payload.get("hard_negative_group_ids", [])}
    return result


def select_extra_train_rows(
    extra_manifest: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    """Filter external samples according to the configured extra-train policy."""
    normalized_policy = str(policy)
    if normalized_policy == "normal_only":
        subset = extra_manifest[extra_manifest["label"].astype(int) == 0]
    elif normalized_policy == "pathology_only":
        subset = extra_manifest[extra_manifest["label"].astype(int) == 1]
    elif normalized_policy == "all_confident":
        subset = extra_manifest
    else:
        raise ValueError(f"Unsupported extra train policy: {policy}")

    return subset.sort_values(["group_id", "relative_path"]).reset_index(drop=True).copy()


def validate_external_manifest_against_split(
    extra_manifest: pd.DataFrame,
    split_payload: dict[str, Any],
) -> None:
    """Ensure external samples cannot collide with validation or holdout ids."""
    protected_sample_ids = {str(sample_id) for sample_id in split_payload.get("holdout_sample_ids", [])}
    for fold_spec in split_payload.get("folds", []):
        protected_sample_ids.update(str(sample_id) for sample_id in fold_spec.get("val_sample_ids", []))

    overlapping = sorted(set(extra_manifest["sample_id"].astype(str)).intersection(protected_sample_ids))
    if overlapping:
        raise ValueError(
            "External manifest overlaps with protected validation or holdout samples: "
            + ", ".join(overlapping[:10])
        )


def build_train_subset(
    manifest: pd.DataFrame,
    *,
    train_sample_ids: list[str],
    extra_manifest: pd.DataFrame | None = None,
    extra_train_policy: str = "all_confident",
) -> tuple[pd.DataFrame, set[str]]:
    """Build a fold-specific train subset with optional external samples appended."""
    base_subset = manifest[manifest["sample_id"].astype(str).isin({str(sample_id) for sample_id in train_sample_ids})].copy()
    external_sample_ids: set[str] = set()
    if extra_manifest is None or extra_manifest.empty:
        return base_subset.sort_values(["group_id", "relative_path"]).reset_index(drop=True), external_sample_ids

    extra_subset = select_extra_train_rows(extra_manifest, policy=extra_train_policy)
    if extra_subset.empty:
        return base_subset.sort_values(["group_id", "relative_path"]).reset_index(drop=True), external_sample_ids

    external_sample_ids = set(extra_subset["sample_id"].astype(str))
    combined = pd.concat([base_subset, extra_subset], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first")
    return combined.sort_values(["group_id", "relative_path"]).reset_index(drop=True), external_sample_ids


def create_optimizer(model: HipDysplasiaClassifier, *, learning_rate: float, weight_decay: float) -> Optimizer:
    """Create an optimizer over trainable parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    model: HipDysplasiaClassifier,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
    gradient_accumulation: int,
    input_size: int,
    batch_size: int,
    mixup_prob: float = 0.0,
) -> float:
    """Run a single training epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(loader):
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        sample_count += int(inputs.shape[0])

        if mixup_prob > 0.0 and random.random() < mixup_prob and inputs.shape[0] > 1:
            lam = float(np.random.beta(0.4, 0.4))
            perm = torch.randperm(inputs.shape[0])
            if random.random() < 0.5:
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                targets = lam * targets + (1.0 - lam) * targets[perm]
            else:
                h, w = inputs.shape[-2], inputs.shape[-1]
                cut_rat = np.sqrt(1.0 - lam)
                cut_w = int(w * cut_rat)
                cut_h = int(h * cut_rat)
                cx = np.random.randint(w)
                cy = np.random.randint(h)
                bbx1 = np.clip(cx - cut_w // 2, 0, w)
                bby1 = np.clip(cy - cut_h // 2, 0, h)
                bbx2 = np.clip(cx + cut_w // 2, 0, w)
                bby2 = np.clip(cy + cut_h // 2, 0, h)
                inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[perm, :, bby1:bby2, bbx1:bbx2]
                actual_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
                targets = actual_lam * targets + (1.0 - actual_lam) * targets[perm]

        try:
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
                scaled_loss = loss / gradient_accumulation

            scaler.scale(scaled_loss).backward()
        except RuntimeError as exc:
            if is_out_of_memory_error(exc):
                raise_oom_system_exit(exc, input_size=input_size, batch_size=batch_size)
            raise

        if (step + 1) % gradient_accumulation == 0 or (step + 1) == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.detach().item()) * int(inputs.shape[0])

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def predict_loader(
    model: HipDysplasiaClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    input_size: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run forward passes and collect probabilities, labels, and sample ids."""
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    sample_ids: list[str] = []

    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].detach().cpu().numpy()

        try:
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(inputs)
        except RuntimeError as exc:
            if is_out_of_memory_error(exc):
                raise_oom_system_exit(exc, input_size=input_size, batch_size=batch_size)
            raise

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        all_probabilities.append(probabilities)
        all_targets.append(targets)
        sample_ids.extend(batch["sample_id"])

    return {
        "sample_ids": sample_ids,
        "probabilities": np.concatenate(all_probabilities, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
    }


def _snapshot_split(config: TrainingConfig, split_payload: dict[str, Any], output_dir: Path) -> None:
    snapshot = dict(split_payload)
    snapshot["manifest_path"] = str(Path(config.manifest_path).resolve())
    save_json(output_dir / "split_snapshot.json", snapshot)


def save_validation_predictions(
    path: str | Path,
    *,
    sample_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Persist per-sample validation probabilities for later recalibration."""
    dataframe = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": targets.astype(int),
            "probability": probabilities.astype(float),
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def train_fold(
    *,
    fold_spec: dict[str, Any],
    fold_dir: Path,
    manifest: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
    hard_negative_groups_by_fold: dict[int, set[str]],
    extra_train_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train one fold and persist its best checkpoint and metrics."""
    train_sample_ids = list(fold_spec["train_sample_ids"])
    val_sample_ids = list(fold_spec["val_sample_ids"])
    fold = int(fold_spec["fold"])
    hard_negative_group_ids = hard_negative_groups_by_fold.get(fold, set())
    train_subset, external_sample_ids = build_train_subset(
        manifest,
        train_sample_ids=train_sample_ids,
        extra_manifest=extra_train_manifest,
        extra_train_policy=config.extra_train_policy,
    )
    sample_weight_map: dict[str, float] | None = None
    if (hard_negative_group_ids and config.hard_negative_weight > 1.0) or (
        external_sample_ids and config.extra_train_weight != 1.0
    ):
        train_weight_frame = train_subset[["sample_id", "group_id"]].copy()
        sample_weight_map = {
            str(row.sample_id): (
                float(config.extra_train_weight)
                if str(row.sample_id) in external_sample_ids and config.extra_train_weight != 1.0
                else float(config.hard_negative_weight)
                if str(row.group_id) in hard_negative_group_ids and config.hard_negative_weight > 1.0
                else 1.0
            )
            for row in train_weight_frame.itertuples(index=False)
        }

    train_loader = create_dataloader(
        train_subset,
        sample_ids=None,
        image_size=config.input_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        preprocessing_profile=config.preprocessing_profile,
        sample_weight_map=sample_weight_map,
    )
    val_loader = create_dataloader(
        manifest,
        sample_ids=val_sample_ids,
        image_size=config.input_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        preprocessing_profile=config.preprocessing_profile,
    )

    model = HipDysplasiaClassifier(
        architecture=config.architecture,
        dropout=config.dropout,
        pretrained=True,
        pretrained_weights_path=config.pretrained_weights_path,
    ).to(device)
    model.freeze_backbone(config.freeze_epochs > 0)

    train_labels = manifest[manifest["sample_id"].isin(train_sample_ids)]["label"]
    if external_sample_ids:
        external_labels = train_subset[train_subset["sample_id"].isin(external_sample_ids)]["label"]
        train_labels = pd.concat([train_labels, external_labels], ignore_index=True)
    pos_weight = build_pos_weight(train_labels) if config.use_pos_weight else None
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = build_loss(config.loss_name, pos_weight=pos_weight, label_smoothing=config.label_smoothing)
    optimizer = create_optimizer(
        model,
        learning_rate=config.learning_rate_head,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    amp_enabled = bool(config.amp and device.type == "cuda")

    scheduler = None
    best_score: tuple[float, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    best_predictions: dict[str, Any] | None = None
    best_threshold_sweep: list[dict[str, Any]] | None = None
    train_history: list[dict[str, Any]] = []
    checkpoint_path = fold_dir / "best.pt"
    swa_model = torch.optim.swa_utils.AveragedModel(model) if config.use_swa else None
    swa_start = max(config.freeze_epochs + 1, config.swa_start_epoch)
    print(f"\n--- Training Fold {fold} (Train: {len(train_sample_ids)}, Val: {len(val_sample_ids)}) ---", flush=True)

    for epoch in range(config.epochs):
        if epoch == config.freeze_epochs and config.freeze_epochs > 0:
            model.freeze_backbone(False)
            optimizer = create_optimizer(
                model,
                learning_rate=config.learning_rate_finetune,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - config.freeze_epochs, 1),
                eta_min=1e-6,
            )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation=config.gradient_accumulation,
            input_size=config.input_size,
            batch_size=config.batch_size,
            mixup_prob=config.mixup_prob,
        )
        if scheduler is not None:
            scheduler.step()

        predictions = predict_loader(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            input_size=config.input_size,
            batch_size=config.batch_size,
        )
        threshold, metrics, threshold_sweep = find_optimal_threshold(
            predictions["targets"].astype(int),
            predictions["probabilities"].astype(np.float32),
            policy=config.threshold_policy,
            sensitivity_floor=config.sensitivity_floor,
        )
        metrics["loss"] = float(train_loss)
        metrics["epoch"] = int(epoch)
        train_history.append({"epoch": int(epoch), "train_loss": float(train_loss), "threshold": threshold})
        print(
            f"Fold {fold} | Epoch {epoch+1:02d}/{config.epochs:02d} | "
            f"Loss: {train_loss:.4f} | Val Sens: {metrics['sensitivity']:.3f} | "
            f"Spec: {metrics['specificity']:.3f} | Acc: {metrics['accuracy']:.3f} | "
            f"F1: {metrics['f1']:.3f} | Thresh: {threshold:.3f}",
            flush=True,
        )

        score = _threshold_score(
            metrics,
            policy=config.threshold_policy,
            sensitivity_floor=config.sensitivity_floor,
        )
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)

        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics
            best_predictions = predictions
            best_threshold_sweep = threshold_sweep
            torch.save(
                {
                    "fold": fold,
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "best_threshold": float(threshold),
                    "metrics": metrics,
                    "training_config": config.to_dict(),
                    "hard_negative_group_ids": sorted(hard_negative_group_ids),
                },
                checkpoint_path,
            )

    if config.use_swa and swa_model is not None:
        class _ImageOnlyLoader:
            def __init__(self, loader: Any) -> None:
                self.loader = loader

            def __iter__(self) -> Any:
                for batch in self.loader:
                    yield batch["image"]

            def __len__(self) -> int:
                return len(self.loader)

        torch.optim.swa_utils.update_bn(_ImageOnlyLoader(train_loader), swa_model, device=device)
        swa_predictions = predict_loader(
            swa_model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            input_size=config.input_size,
            batch_size=config.batch_size,
        )
        swa_threshold, swa_metrics, swa_threshold_sweep = find_optimal_threshold(
            swa_predictions["targets"].astype(int),
            swa_predictions["probabilities"].astype(np.float32),
            policy=config.threshold_policy,
            sensitivity_floor=config.sensitivity_floor,
        )
        swa_score = _threshold_score(
            swa_metrics,
            policy=config.threshold_policy,
            sensitivity_floor=config.sensitivity_floor,
        )
        print(
            f"Fold {fold} | SWA Final Eval | "
            f"Val Sens: {swa_metrics['sensitivity']:.3f} | Spec: {swa_metrics['specificity']:.3f} | "
            f"Acc: {swa_metrics['accuracy']:.3f} | F1: {swa_metrics['f1']:.3f} | Thresh: {swa_threshold:.3f}",
            flush=True,
        )
        if best_score is None or swa_score >= best_score:
            print(f"Fold {fold} | SWA weights accepted as best checkpoint!", flush=True)
            best_score = swa_score
            best_metrics = swa_metrics
            best_predictions = swa_predictions
            best_threshold_sweep = swa_threshold_sweep
            swa_state = {
                k.removeprefix("module."): v
                for k, v in swa_model.module.state_dict().items()
            }
            torch.save(
                {
                    "fold": fold,
                    "epoch": config.epochs,
                    "model_state": swa_state,
                    "model_config": model.config.to_dict(),
                    "best_threshold": float(swa_threshold),
                    "metrics": swa_metrics,
                    "training_config": config.to_dict(),
                    "hard_negative_group_ids": sorted(hard_negative_group_ids),
                    "is_swa": True,
                },
                checkpoint_path,
            )

    if best_metrics is None:
        raise RuntimeError(f"Fold {fold_spec['fold']} produced no metrics.")
    if best_predictions is None or best_threshold_sweep is None:
        raise RuntimeError(f"Fold {fold_spec['fold']} did not retain validation predictions.")

    if config.save_val_predictions:
        save_validation_predictions(
            fold_dir / "val_predictions.csv",
            sample_ids=best_predictions["sample_ids"],
            targets=best_predictions["targets"],
            probabilities=best_predictions["probabilities"],
        )
        save_json(
            fold_dir / "threshold_sweep.json",
            {
                "fold": int(fold_spec["fold"]),
                "policy": config.threshold_policy,
                "sensitivity_floor": float(config.sensitivity_floor),
                "thresholds": best_threshold_sweep,
            },
        )

    fold_result = {
        "fold": fold,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_threshold": float(best_metrics["threshold"]),
        "best_metrics": best_metrics,
        "history": train_history,
        "hard_negative_group_count": int(len(hard_negative_group_ids)),
        "external_train_samples": int(len(external_sample_ids)),
    }
    save_json(fold_dir / "metrics.json", fold_result)
    return fold_result


def run_cross_validation(config: TrainingConfig) -> Path:
    """Run the full cross-validation experiment and persist artifacts."""
    set_seed(config.seed)

    output_dir = Path(config.experiment_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config.manifest_path)
    split_payload = load_json(config.split_path)
    extra_train_manifest = (
        load_manifest(config.extra_train_manifest_path)
        if config.extra_train_manifest_path
        else None
    )
    if extra_train_manifest is not None:
        validate_external_manifest_against_split(extra_train_manifest, split_payload)

    save_json(output_dir / "experiment_config.json", config.to_dict())
    _snapshot_split(config, split_payload, output_dir)

    device = resolve_device(config.device)
    hard_negative_groups_by_fold = load_hard_negative_groups(config.hard_negative_manifest_path)

    print(f"Pre-caching dataset images into RAM ({config.preprocessing_profile}, {config.input_size}px)...", flush=True)
    from data.dataset import build_preprocessor, _PREPROCESSED_IMAGE_CACHE
    from core.image_loader import load_medical_image
    preprocessor = build_preprocessor(config.input_size, profile=config.preprocessing_profile)
    paths_to_cache = set(manifest["path"])
    if extra_train_manifest is not None:
        paths_to_cache.update(extra_train_manifest["path"])
    for img_path in paths_to_cache:
        cache_key = (str(Path(img_path)), config.input_size, config.preprocessing_profile)
        if cache_key not in _PREPROCESSED_IMAGE_CACHE:
            img, meta = load_medical_image(img_path)
            _PREPROCESSED_IMAGE_CACHE[cache_key] = preprocessor.preprocess(img, meta)
    print(f"RAM cache ready ({len(_PREPROCESSED_IMAGE_CACHE)} images loaded into memory).", flush=True)

    fold_results: list[dict[str, Any]] = []
    for fold_spec in split_payload["folds"]:
        fold_dir = output_dir / f"fold_{int(fold_spec['fold'])}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_results.append(
            train_fold(
                fold_spec=fold_spec,
                fold_dir=fold_dir,
                manifest=manifest,
                config=config,
                device=device,
                hard_negative_groups_by_fold=hard_negative_groups_by_fold,
                extra_train_manifest=extra_train_manifest,
            )
        )

    cv_summary = {
        "experiment": config.experiment,
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "fold_count": len(fold_results),
        "mean_sensitivity": float(np.mean([result["best_metrics"]["sensitivity"] for result in fold_results])),
        "mean_specificity": float(np.mean([result["best_metrics"]["specificity"] for result in fold_results])),
        "mean_accuracy": float(np.mean([result["best_metrics"]["accuracy"] for result in fold_results])),
        "mean_f1": float(np.mean([result["best_metrics"]["f1"] for result in fold_results])),
        "mean_threshold": float(np.mean([result["best_threshold"] for result in fold_results])),
        "mean_external_train_samples": float(np.mean([result["external_train_samples"] for result in fold_results])),
        "folds": fold_results,
    }
    save_json(output_dir / "cv_summary.json", cv_summary)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 3 classifier with 5-fold CV.")
    parser.add_argument("--manifest", required=True, help="Path to data/manifests/train_manifest.csv")
    parser.add_argument("--split", required=True, help="Path to data/manifests/split_v1.json")
    parser.add_argument("--experiment", required=True, help="Experiment name under models/checkpoints/")
    parser.add_argument("--architecture", default="resnet50", help="Backbone architecture.")
    parser.add_argument(
        "--preprocessing-profile",
        default="default",
        choices=list(PREPROCESSING_PROFILES),
        help="Deterministic preprocessing profile applied before augmentations.",
    )
    parser.add_argument("--input-size", type=int, default=384, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Total epochs per fold.")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=1,
        help="Epochs with the backbone frozen before full fine-tuning.",
    )
    parser.add_argument("--learning-rate-head", type=float, default=1e-3, help="LR for the classifier head.")
    parser.add_argument(
        "--learning-rate-finetune",
        type=float,
        default=1e-4,
        help="LR after unfreezing the backbone.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Classifier head dropout.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--loss", default="focal", choices=["bce", "focal"], help="Loss function.")
    parser.add_argument(
        "--threshold-policy",
        default="max_sensitivity",
        choices=[
            "max_sensitivity",
            "max_specificity_under_sensitivity_floor",
            "balanced_accuracy",
            "max_f1",
            "youden",
            "cost_sensitive",
            "pareto_convex_hull",
            "roc_convex_hull",
        ],
        help="Policy used to choose the validation threshold.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (e.g. 0.05 to 0.10).",
    )
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Enable Stochastic Weight Averaging on final epochs.",
    )
    parser.add_argument(
        "--swa-start-epoch",
        type=int,
        default=8,
        help="Epoch to begin SWA parameter accumulation.",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.0,
        help="Probability of applying Mixup / CutMix augmentation to a training batch.",
    )
    parser.add_argument(
        "--sensitivity-floor",
        type=float,
        default=0.90,
        help="Minimum sensitivity required by specificity-oriented threshold selection.",
    )
    parser.add_argument(
        "--save-val-predictions",
        action="store_true",
        help="Save validation predictions and threshold sweeps for each fold.",
    )
    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Use class-frequency-based pos_weight for BCE loss.",
    )
    parser.add_argument(
        "--hard-negative-manifest",
        help="Optional JSON file with per-fold hard-negative group ids.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help="Sampling multiplier applied to hard-negative groups.",
    )
    parser.add_argument(
        "--extra-train-manifest",
        help="Optional external manifest appended to the train subset of each fold only.",
    )
    parser.add_argument(
        "--extra-train-policy",
        default="all_confident",
        choices=["normal_only", "pathology_only", "all_confident"],
        help="Subset of external labels to include in fold training.",
    )
    parser.add_argument(
        "--extra-train-weight",
        type=float,
        default=1.0,
        help="Sampling multiplier applied to all external training samples.",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="auto",
        help="ImageNet initialization source: auto, torchvision, none, or a local checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision even when CUDA is available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        manifest_path=str(Path(args.manifest).resolve()),
        split_path=str(Path(args.split).resolve()),
        experiment=args.experiment,
        experiment_dir=str((Path("models") / "checkpoints" / args.experiment).resolve()),
        architecture=args.architecture,
        preprocessing_profile=args.preprocessing_profile,
        input_size=args.input_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        learning_rate_head=args.learning_rate_head,
        learning_rate_finetune=args.learning_rate_finetune,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.disable_amp,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        use_swa=args.use_swa,
        swa_start_epoch=args.swa_start_epoch,
        mixup_prob=args.mixup_prob,
        threshold_policy=args.threshold_policy,
        sensitivity_floor=args.sensitivity_floor,
        save_val_predictions=args.save_val_predictions,
        use_pos_weight=args.use_pos_weight,
        hard_negative_manifest_path=str(Path(args.hard_negative_manifest).resolve())
        if args.hard_negative_manifest
        else None,
        hard_negative_weight=args.hard_negative_weight,
        extra_train_manifest_path=str(Path(args.extra_train_manifest).resolve())
        if args.extra_train_manifest
        else None,
        extra_train_policy=args.extra_train_policy,
        extra_train_weight=args.extra_train_weight,
        pretrained_weights_path=resolve_pretrained_weights_argument(args.pretrained_weights, args.architecture),
        device=args.device,
    )

    try:
        experiment_dir = run_cross_validation(config)
    except BaseException as exc:
        if is_out_of_memory_error(exc):
            raise_oom_system_exit(exc, input_size=config.input_size, batch_size=config.batch_size)
        raise

    print(
        json.dumps(
            {"experiment_dir": str(experiment_dir), "status": "ok", "experiment": config.experiment},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
