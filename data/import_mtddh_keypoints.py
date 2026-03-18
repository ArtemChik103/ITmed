"""Import MTDDH Dataset1 keypoints into reproducible train/val manifests."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.image_loader import load_medical_image

RAW_KEYPOINT_NAMES = ("re", "ry", "rc", "rh", "le", "ly", "lc", "lh")
REQUIRED_CATEGORY_NAME = "hip"
SUPPORTED_SPLITS = {
    "train": {
        "annotation_name": "Keypoints_Train.json",
        "image_dir": Path("Dataset1") / "Keypoints" / "Train",
    },
    "validation": {
        "annotation_name": "Keypoints_Validation.json",
        "image_dir": Path("Dataset1") / "Keypoints" / "Validation",
    },
}


def _load_coco_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    for key in ("images", "annotations", "categories"):
        if key not in payload:
            raise ValueError(f"'{path}' is missing required top-level key '{key}'.")
    return payload


def _validate_categories(categories: list[dict[str, Any]]) -> None:
    if len(categories) != 1:
        raise ValueError(f"Expected exactly one category in MTDDH keypoints JSON, found {len(categories)}.")
    category = categories[0]
    if str(category.get("name")) != REQUIRED_CATEGORY_NAME:
        raise ValueError(f"Expected category name '{REQUIRED_CATEGORY_NAME}', got '{category.get('name')}'.")
    keypoints = tuple(str(name) for name in category.get("keypoints", []))
    if keypoints != RAW_KEYPOINT_NAMES:
        raise ValueError(
            "Unexpected raw keypoint names. "
            f"Expected {list(RAW_KEYPOINT_NAMES)}, got {list(keypoints)}."
        )


def _is_broken_image(path: Path) -> bool:
    try:
        load_medical_image(path)
    except Exception:
        return True
    return False


def _build_row(
    *,
    dataset_root: Path,
    image_dir: Path,
    split: str,
    image_entry: dict[str, Any],
    annotation_entry: dict[str, Any],
) -> dict[str, Any]:
    file_name = str(image_entry["file_name"])
    image_path = (dataset_root / image_dir / file_name).resolve()
    relative_path = image_path.relative_to(dataset_root).as_posix()
    file_stem = Path(file_name).stem
    bbox = annotation_entry.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return {
        "sample_id": f"mtddh_kp::{relative_path}",
        "group_id": f"mtddh_kp::{file_stem}",
        "group_name": file_stem,
        "relative_path": relative_path,
        "path": str(image_path),
        "split": split,
        "image_width": int(image_entry["width"]),
        "image_height": int(image_entry["height"]),
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_w": float(bbox[2]),
        "bbox_h": float(bbox[3]),
        "num_keypoints": int(annotation_entry.get("num_keypoints", 0)),
        "keypoints_json": json.dumps(annotation_entry.get("keypoints", []), ensure_ascii=False),
        "dataset_name": "MTDDH",
        "source": "MTDDH_Keypoints",
        "source_code": "mtddh_kp",
        "annotation_id": int(annotation_entry["id"]),
        "image_id": int(image_entry["id"]),
        "category_id": int(annotation_entry["category_id"]),
        "segmentation_json": json.dumps(annotation_entry.get("segmentation", []), ensure_ascii=False),
    }


def _build_split_manifest(
    *,
    dataset_root: Path,
    split: str,
    strict_keypoints: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    split_spec = SUPPORTED_SPLITS[split]
    payload = _load_coco_payload(dataset_root / "Dataset1" / "Keypoints" / split_spec["annotation_name"])
    _validate_categories(payload["categories"])

    images_by_id = {int(entry["id"]): entry for entry in payload["images"]}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    rows: list[dict[str, Any]] = []
    dropped = Counter()

    for image_id, image_entry in sorted(images_by_id.items()):
        annotations = annotations_by_image_id.get(image_id, [])
        if not annotations:
            dropped["missing_annotation"] += 1
            continue
        if len(annotations) != 1:
            dropped["multiple_annotations"] += 1
            continue

        annotation = annotations[0]
        num_keypoints = int(annotation.get("num_keypoints", 0))
        if strict_keypoints and num_keypoints < len(RAW_KEYPOINT_NAMES):
            dropped["insufficient_keypoints"] += 1
            continue

        keypoints = annotation.get("keypoints", [])
        if len(keypoints) != len(RAW_KEYPOINT_NAMES) * 3:
            dropped["invalid_keypoint_vector"] += 1
            continue

        image_path = (dataset_root / split_spec["image_dir"] / str(image_entry["file_name"])).resolve()
        if not image_path.exists():
            dropped["missing_image_file"] += 1
            continue
        if _is_broken_image(image_path):
            dropped["broken_image_file"] += 1
            continue

        rows.append(
            _build_row(
                dataset_root=dataset_root,
                image_dir=split_spec["image_dir"],
                split=split,
                image_entry=image_entry,
                annotation_entry=annotation,
            )
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=[
                "sample_id",
                "group_id",
                "group_name",
                "relative_path",
                "path",
                "split",
                "image_width",
                "image_height",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "num_keypoints",
                "keypoints_json",
                "dataset_name",
                "source",
                "source_code",
                "annotation_id",
                "image_id",
                "category_id",
                "segmentation_json",
            ]
        )
    else:
        dataframe = dataframe.sort_values(["group_id", "relative_path"]).reset_index(drop=True)
    return dataframe, {key: int(value) for key, value in sorted(dropped.items())}


def build_mtddh_keypoint_manifests(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    strict_keypoints: bool = True,
) -> dict[str, Any]:
    """Build train/val keypoint manifests from MTDDH Dataset1/Keypoints only."""
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest, train_dropped = _build_split_manifest(
        dataset_root=dataset_root,
        split="train",
        strict_keypoints=strict_keypoints,
    )
    val_manifest, val_dropped = _build_split_manifest(
        dataset_root=dataset_root,
        split="validation",
        strict_keypoints=strict_keypoints,
    )

    train_path = output_dir / "mtddh_keypoints_train.csv"
    val_path = output_dir / "mtddh_keypoints_val.csv"
    train_manifest.to_csv(train_path, index=False)
    val_manifest.to_csv(val_path, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "strict_keypoints": bool(strict_keypoints),
        "raw_keypoint_names": list(RAW_KEYPOINT_NAMES),
        "train_rows": int(len(train_manifest)),
        "val_rows": int(len(val_manifest)),
        "dropped_rows": {
            "train": train_dropped,
            "validation": val_dropped,
        },
        "artifacts": {
            "train_manifest": str(train_path),
            "val_manifest": str(val_path),
        },
    }
    summary_path = output_dir / "import_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MTDDH Dataset1 keypoints into train/val manifests.")
    parser.add_argument("--dataset-root", required=True, help="Path to the unpacked MTDDH dataset root.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for mtddh_keypoints_train.csv and mtddh_keypoints_val.csv",
    )
    parser.add_argument(
        "--allow-partial-keypoints",
        action="store_true",
        help="Keep rows with num_keypoints < 8. By default strict mode drops them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_mtddh_keypoint_manifests(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        strict_keypoints=not args.allow_partial_keypoints,
    )
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
