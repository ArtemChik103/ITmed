from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tests.phase3_helpers import write_synthetic_keypoint_checkpoint


def test_keypoint_runtime_loads_checkpoint_and_runs_forward(tmp_path: Path):
    from plugins.hip_dysplasia.keypoint_runtime import HipDysplasiaKeypointRuntime

    checkpoint_path = write_synthetic_keypoint_checkpoint(tmp_path / "keypoints" / "best.ckpt")
    runtime = HipDysplasiaKeypointRuntime(checkpoint_path, device="cpu")

    prediction = runtime.predict(np.ones((384, 384), dtype=np.float32))

    assert runtime.model_loaded is True
    assert prediction.model_loaded is True
    assert prediction.input_size == 384
    assert prediction.keypoint_names == ["re", "ry", "rc", "rh", "le", "ly", "lc", "lh"]
    assert len(prediction.keypoints_xy) == 8
    assert prediction.score_map is not None
    assert prediction.score_map.shape == (8, 96, 96)


def test_keypoint_runtime_decodes_heatmaps_into_image_coordinates(tmp_path: Path):
    from plugins.hip_dysplasia.keypoint_runtime import HipDysplasiaKeypointRuntime

    class FixedHeatmapModel:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            heatmaps = torch.zeros((1, 8, 96, 96), dtype=torch.float32, device=tensor.device)
            maxima = [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14), (32, 16)]
            for channel, (x_index, y_index) in enumerate(maxima):
                heatmaps[0, channel, y_index, x_index] = 1.0
            return heatmaps

    checkpoint_path = write_synthetic_keypoint_checkpoint(tmp_path / "keypoints" / "best.ckpt")
    runtime = HipDysplasiaKeypointRuntime(checkpoint_path, device="cpu")
    runtime.model = FixedHeatmapModel()

    prediction = runtime.predict(np.zeros((384, 384), dtype=np.float32))

    assert prediction.keypoints_xy[0] == (18.0, 10.0)
    assert prediction.keypoints_xy[4] == (82.0, 42.0)
    assert all(0.0 <= x <= 383.0 and 0.0 <= y <= 383.0 for x, y in prediction.keypoints_xy)


def test_keypoint_runtime_prepare_tensor_and_missing_checkpoint_behavior(tmp_path: Path, monkeypatch):
    from plugins.hip_dysplasia.keypoint_runtime import (
        HipDysplasiaKeypointRuntime,
        prepare_keypoint_image_tensor,
        resolve_keypoint_checkpoint_path,
    )

    tensor = prepare_keypoint_image_tensor(np.zeros((32, 32), dtype=np.float32), input_size=64)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32

    monkeypatch.delenv("HIP_DYSPLASIA_KEYPOINT_CHECKPOINT", raising=False)
    assert resolve_keypoint_checkpoint_path() is None

    missing_path = tmp_path / "missing.ckpt"
    try:
        HipDysplasiaKeypointRuntime(missing_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for a missing keypoint checkpoint.")
