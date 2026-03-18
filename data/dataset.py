"""Dataset helpers for the Phase 3 hip dysplasia classifier."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.image_loader import load_medical_image
from core.preprocessor import XRayPreprocessor, get_preprocessor
from data.augmentations import get_eval_augmentations, get_train_augmentations

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "group_id",
    "label",
    "relative_path",
    "path",
    "source",
    "class_name",
}


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest CSV."""
    manifest_path = Path(manifest_path)
    dataframe = pd.read_csv(manifest_path)
    missing_columns = REQUIRED_MANIFEST_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest '{manifest_path}' is missing required columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["sample_id"] = dataframe["sample_id"].astype(str)
    dataframe["group_id"] = dataframe["group_id"].astype(str)
    dataframe["path"] = dataframe["path"].astype(str)
    dataframe["relative_path"] = dataframe["relative_path"].astype(str)
    dataframe["source"] = dataframe["source"].astype(str)
    dataframe["class_name"] = dataframe["class_name"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)
    return dataframe


def build_preprocessor(image_size: int, *, profile: str = "default") -> XRayPreprocessor:
    """Create a deterministic preprocessor for classifier inputs."""
    return get_preprocessor(profile=profile, target_size=(image_size, image_size))


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a normalized grayscale image to 3 channels."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {list(image.shape)}")
    return np.repeat(image[..., None], 3, axis=2).astype(np.float32, copy=False)


class HipDysplasiaDataset(Dataset[dict[str, Any]]):
    """Map manifest rows to normalized 3-channel tensors."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        sample_ids: Iterable[str] | None = None,
        image_size: int = 384,
        preprocessing_profile: str = "default",
        train: bool = False,
        transform: Any | None = None,
        preprocessor: XRayPreprocessor | None = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            dataframe = load_manifest(manifest)
        else:
            dataframe = manifest.copy()

        if sample_ids is not None:
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            dataframe = dataframe[dataframe["sample_id"].astype(str).isin(sample_id_set)]

        dataframe = dataframe.sort_values(["group_id", "relative_path"]).reset_index(drop=True)
        if dataframe.empty:
            raise ValueError("HipDysplasiaDataset received an empty dataframe.")

        self._dataframe = dataframe
        self._image_size = image_size
        self._preprocessing_profile = str(preprocessing_profile)
        self._preprocessor = preprocessor or build_preprocessor(image_size, profile=self._preprocessing_profile)
        self._transform = transform or (
            get_train_augmentations(image_size) if train else get_eval_augmentations(image_size)
        )

    def __len__(self) -> int:
        return len(self._dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._dataframe.iloc[index]
        image, metadata = load_medical_image(str(Path(row["path"])))
        processed = self._preprocessor.preprocess(image, metadata)
        rgb_image = grayscale_to_rgb(processed)

        if self._transform is not None:
            transformed = self._transform(image=rgb_image)
            image_tensor = transformed["image"].to(dtype=torch.float32)
        else:
            image_tensor = torch.from_numpy(np.transpose(rgb_image, (2, 0, 1))).to(dtype=torch.float32)

        return {
            "image": image_tensor,
            "target": torch.tensor(float(row["label"]), dtype=torch.float32),
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
            "path": str(row["path"]),
            "relative_path": str(row["relative_path"]),
            "source": str(row["source"]),
            "class_name": str(row["class_name"]),
        }
