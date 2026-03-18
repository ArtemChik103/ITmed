"""Dataset helpers for MTDDH Dataset1 keypoint pretraining."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.image_loader import load_medical_image
from core.preprocessor import XRayPreprocessor, get_preprocessor
from data.dataset import grayscale_to_rgb
from data.import_mtddh_keypoints import RAW_KEYPOINT_NAMES
from data.keypoint_augmentations import (
    get_keypoint_eval_augmentations,
    get_keypoint_train_augmentations,
)

REQUIRED_KEYPOINT_MANIFEST_COLUMNS = {
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
}


def load_keypoint_manifest(manifest_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(manifest_path)
    missing_columns = REQUIRED_KEYPOINT_MANIFEST_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Keypoint manifest '{manifest_path}' is missing required columns: {missing}")

    dataframe = dataframe.copy()
    for column in (
        "sample_id",
        "group_id",
        "group_name",
        "relative_path",
        "path",
        "split",
        "dataset_name",
        "source",
        "source_code",
        "keypoints_json",
    ):
        dataframe[column] = dataframe[column].astype(str)
    for column in ("image_width", "image_height", "num_keypoints"):
        dataframe[column] = dataframe[column].astype(int)
    for column in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        dataframe[column] = dataframe[column].astype(float)
    return dataframe.sort_values(["group_id", "relative_path"]).reset_index(drop=True)


def build_keypoint_preprocessor(image_size: int, *, profile: str = "default") -> XRayPreprocessor:
    return get_preprocessor(profile=profile, target_size=(image_size, image_size))


def generate_gaussian_heatmaps(
    *,
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    heatmap_size: int,
    image_size: int,
    sigma: float = 2.0,
) -> np.ndarray:
    num_keypoints = keypoints_xy.shape[0]
    heatmaps = np.zeros((num_keypoints, heatmap_size, heatmap_size), dtype=np.float32)
    stride = image_size / float(max(heatmap_size, 1))

    xs = np.arange(heatmap_size, dtype=np.float32)
    ys = np.arange(heatmap_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for index in range(num_keypoints):
        if visibility[index] <= 0:
            continue
        center_x = float(keypoints_xy[index, 0] / stride)
        center_y = float(keypoints_xy[index, 1] / stride)
        if center_x < 0.0 or center_y < 0.0 or center_x > (heatmap_size - 1.0) or center_y > (heatmap_size - 1.0):
            continue
        squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        heatmaps[index] = np.exp(-squared_distance / (2.0 * sigma * sigma))
    return heatmaps


class MTDDHKeypointDataset(Dataset[dict[str, Any]]):
    """Manifest-backed dataset that returns images and Gaussian heatmap targets."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        sample_ids: Iterable[str] | None = None,
        image_size: int = 384,
        heatmap_size: int | None = None,
        sigma: float = 2.0,
        preprocessing_profile: str = "default",
        train: bool = False,
        transform: Any | None = None,
        preprocessor: XRayPreprocessor | None = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            dataframe = load_keypoint_manifest(manifest)
        else:
            dataframe = manifest.copy()

        if sample_ids is not None:
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            dataframe = dataframe[dataframe["sample_id"].astype(str).isin(sample_id_set)]

        dataframe = dataframe.sort_values(["group_id", "relative_path"]).reset_index(drop=True)
        if dataframe.empty:
            raise ValueError("MTDDHKeypointDataset received an empty dataframe.")

        self._dataframe = dataframe
        self._image_size = int(image_size)
        self._heatmap_size = int(heatmap_size or (image_size // 4))
        self._sigma = float(sigma)
        self._preprocessor = preprocessor or build_keypoint_preprocessor(image_size, profile=preprocessing_profile)
        self._transform = transform or (
            get_keypoint_train_augmentations(target_size=image_size)
            if train
            else get_keypoint_eval_augmentations(target_size=image_size)
        )

    def __len__(self) -> int:
        return len(self._dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._dataframe.iloc[index]
        image_path = Path(str(row["path"]))
        image, metadata = load_medical_image(image_path)
        original_height, original_width = image.shape[:2]
        processed = self._preprocessor.preprocess(image, metadata)
        processed_height, processed_width = processed.shape[:2]
        rgb_image = grayscale_to_rgb(processed)

        raw_keypoints = np.asarray(json.loads(str(row["keypoints_json"])), dtype=np.float32).reshape(len(RAW_KEYPOINT_NAMES), 3)
        keypoints_xy = raw_keypoints[:, :2].copy()
        visibility = raw_keypoints[:, 2].copy()
        bbox = np.asarray([row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]], dtype=np.float32)
        scale_x = processed_width / float(max(original_width, 1))
        scale_y = processed_height / float(max(original_height, 1))
        visible_indices = visibility > 0
        keypoints_xy[visible_indices, 0] *= scale_x
        keypoints_xy[visible_indices, 1] *= scale_y
        bbox[0] *= scale_x
        bbox[1] *= scale_y
        bbox[2] *= scale_x
        bbox[3] *= scale_y

        transformed = self._transform(
            image=rgb_image,
            keypoints_xy=keypoints_xy,
            visibility=visibility,
            bbox=bbox,
        )
        image_tensor = transformed["image"].to(dtype=torch.float32)
        keypoints_xy = transformed["keypoints_xy"].astype(np.float32, copy=False)
        visibility = transformed["visibility"].astype(np.float32, copy=False)
        bbox = transformed["bbox"].astype(np.float32, copy=False)
        heatmaps = generate_gaussian_heatmaps(
            keypoints_xy=keypoints_xy,
            visibility=visibility,
            heatmap_size=self._heatmap_size,
            image_size=self._image_size,
            sigma=self._sigma,
        )

        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps).to(dtype=torch.float32),
            "keypoints_xy": torch.from_numpy(keypoints_xy).to(dtype=torch.float32),
            "visibility": torch.from_numpy((visibility > 0).astype(np.float32)),
            "bbox": torch.from_numpy(bbox).to(dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
            "group_name": str(row["group_name"]),
            "path": str(row["path"]),
            "relative_path": str(row["relative_path"]),
            "split": str(row["split"]),
            "source": str(row["source"]),
            "source_code": str(row["source_code"]),
            "image_size": torch.tensor([self._image_size, self._image_size], dtype=torch.int64),
            "heatmap_size": torch.tensor([self._heatmap_size, self._heatmap_size], dtype=torch.int64),
            "keypoint_names": list(RAW_KEYPOINT_NAMES),
        }
