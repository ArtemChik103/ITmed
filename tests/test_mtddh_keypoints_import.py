from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from data.import_mtddh_keypoints import build_mtddh_keypoint_manifests


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=128).save(path)


def _build_payload(file_name: str, *, image_id: int, annotation_id: int, num_keypoints: int = 8) -> dict[str, object]:
    keypoints = []
    for index in range(8):
        visibility = 2 if index < num_keypoints else 0
        keypoints.extend([4 + index, 8 + index, visibility])
    return {
        "images": [{"id": image_id, "file_name": file_name, "width": 32, "height": 32}],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [1, 2, 20, 24],
                "segmentation": [[1, 2, 21, 2, 21, 26, 1, 26]],
                "keypoints": keypoints,
                "num_keypoints": num_keypoints,
            }
        ],
        "categories": [{"id": 1, "name": "hip", "supercategory": "hip", "keypoints": ["re", "ry", "rc", "rh", "le", "ly", "lc", "lh"]}],
    }


def test_mtddh_keypoint_import_builds_train_and_val_manifests(tmp_path: Path):
    dataset_root = tmp_path / "MTDDH"
    train_image = dataset_root / "Dataset1" / "Keypoints" / "Train" / "case_a.jpg"
    val_image = dataset_root / "Dataset1" / "Keypoints" / "Validation" / "case_b.jpg"
    _write_image(train_image)
    _write_image(val_image)

    (dataset_root / "Dataset1" / "Keypoints").mkdir(parents=True, exist_ok=True)
    (dataset_root / "Dataset1" / "Keypoints" / "Keypoints_Train.json").write_text(
        json.dumps(_build_payload("case_a.jpg", image_id=1, annotation_id=10)),
        encoding="utf-8",
    )
    (dataset_root / "Dataset1" / "Keypoints" / "Keypoints_Validation.json").write_text(
        json.dumps(_build_payload("case_b.jpg", image_id=2, annotation_id=20)),
        encoding="utf-8",
    )

    summary = build_mtddh_keypoint_manifests(dataset_root=dataset_root, output_dir=tmp_path / "out")

    assert summary["train_rows"] == 1
    assert summary["val_rows"] == 1
    assert "mtddh_kp::Dataset1/Keypoints/Train/case_a.jpg" in (tmp_path / "out" / "mtddh_keypoints_train.csv").read_text(encoding="utf-8")
    assert '"raw_keypoint_names"' in (tmp_path / "out" / "import_summary.json").read_text(encoding="utf-8")


def test_mtddh_keypoint_import_drops_partial_rows_in_strict_mode(tmp_path: Path):
    dataset_root = tmp_path / "MTDDH"
    train_image = dataset_root / "Dataset1" / "Keypoints" / "Train" / "case_a.jpg"
    val_image = dataset_root / "Dataset1" / "Keypoints" / "Validation" / "case_b.jpg"
    _write_image(train_image)
    _write_image(val_image)

    (dataset_root / "Dataset1" / "Keypoints").mkdir(parents=True, exist_ok=True)
    (dataset_root / "Dataset1" / "Keypoints" / "Keypoints_Train.json").write_text(
        json.dumps(_build_payload("case_a.jpg", image_id=1, annotation_id=10, num_keypoints=7)),
        encoding="utf-8",
    )
    (dataset_root / "Dataset1" / "Keypoints" / "Keypoints_Validation.json").write_text(
        json.dumps(_build_payload("case_b.jpg", image_id=2, annotation_id=20)),
        encoding="utf-8",
    )

    summary = build_mtddh_keypoint_manifests(dataset_root=dataset_root, output_dir=tmp_path / "out")

    assert summary["train_rows"] == 0
    assert summary["dropped_rows"]["train"]["insufficient_keypoints"] == 1
