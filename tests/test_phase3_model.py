from __future__ import annotations

from pathlib import Path

import torch

from models.classifier import HipDysplasiaClassifier
from models.losses import FocalLoss, build_loss
from plugins.hip_dysplasia.model import REPO_ROOT, resolve_checkpoint_path


def test_resnet50_forward_pass_runs_without_shape_errors():
    model = HipDysplasiaClassifier(architecture="resnet50", pretrained=False)
    inputs = torch.randn(1, 3, 384, 384)

    outputs = model(inputs)

    assert tuple(outputs.shape) == (1,)


def test_bce_and_focal_losses_return_finite_values():
    logits = torch.tensor([0.1, -0.4, 0.9], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    bce_loss = build_loss("bce")(logits, targets)
    focal_loss = FocalLoss()(logits, targets)

    assert torch.isfinite(bce_loss)
    assert torch.isfinite(focal_loss)


def test_resolve_checkpoint_path_maps_windows_manifest_paths_to_repo_root():
    manifest_path = REPO_ROOT / "models" / "checkpoints" / "resnet50_bce_v1" / "model_manifest.json"
    resolved = resolve_checkpoint_path(
        r"C:\Users\pvppv\Desktop\roo\it-med-2026\models\checkpoints\resnet50_bce_v1\fold_0\best.pt",
        manifest_path=manifest_path,
    )

    assert resolved == (REPO_ROOT / "models" / "checkpoints" / "resnet50_bce_v1" / "fold_0" / "best.pt").resolve()
