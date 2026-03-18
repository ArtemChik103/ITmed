from __future__ import annotations

import torch

from models.keypoint_detector import KeypointDetector


def test_keypoint_detector_forward_returns_8_heatmaps():
    model = KeypointDetector(pretrained=False, num_keypoints=8)
    inputs = torch.randn(2, 3, 384, 384)
    outputs = model(inputs)

    assert outputs.shape == torch.Size([2, 8, 96, 96])
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
