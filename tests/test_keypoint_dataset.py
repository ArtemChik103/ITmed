from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from data.keypoint_dataset import MTDDHKeypointDataset
from tests.helpers import build_test_raster


def test_keypoint_dataset_returns_expected_shapes_and_heatmaps(tmp_path: Path):
    image_path = build_test_raster(tmp_path / "case_a.jpg", pixel_array=torch.full((64, 64), 128, dtype=torch.uint8).numpy())
    keypoints = [10, 20, 2, 20, 20, 2, 30, 20, 2, 40, 20, 2, 10, 30, 2, 20, 30, 2, 30, 30, 2, 40, 30, 2]
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "mtddh_kp::Dataset1/Keypoints/Train/case_a.jpg",
                "group_id": "mtddh_kp::case_a",
                "group_name": "case_a",
                "relative_path": "Dataset1/Keypoints/Train/case_a.jpg",
                "path": str(image_path.resolve()),
                "split": "train",
                "image_width": 64,
                "image_height": 64,
                "bbox_x": 4.0,
                "bbox_y": 8.0,
                "bbox_w": 40.0,
                "bbox_h": 32.0,
                "num_keypoints": 8,
                "keypoints_json": json.dumps(keypoints),
                "dataset_name": "MTDDH",
                "source": "MTDDH_Keypoints",
                "source_code": "mtddh_kp",
            }
        ]
    )

    dataset = MTDDHKeypointDataset(manifest, image_size=64, heatmap_size=16, train=False)
    item = dataset[0]

    assert item["image"].shape == torch.Size([3, 64, 64])
    assert item["heatmaps"].shape == torch.Size([8, 16, 16])
    assert item["visibility"].shape == torch.Size([8])
    assert item["bbox"].shape == torch.Size([4])
    assert float(item["heatmaps"][0].max()) > 0.0
