from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.image_loader import load_medical_image
from data.dataset import HipDysplasiaDataset
from tests.helpers import build_test_dicom, build_test_raster


def test_load_medical_image_reads_dicom(tmp_path: Path):
    dicom_path = build_test_dicom(tmp_path / "sample.dcm", pixel_spacing=[0.3, 0.4], study_date="20260101")

    image, metadata = load_medical_image(dicom_path)

    assert tuple(image.shape) == (2, 2)
    assert image.dtype == np.float32
    assert metadata["pixel_spacing_mm"] == [0.3, 0.4]
    assert metadata["source_format"] == "dcm"


def test_load_medical_image_reads_raster(tmp_path: Path):
    raster_path = build_test_raster(tmp_path / "sample.png", pixel_array=np.array([[0, 32], [128, 255]], dtype=np.uint8))

    image, metadata = load_medical_image(raster_path)

    assert tuple(image.shape) == (2, 2)
    assert image.dtype == np.float32
    assert metadata["source_format"] == "png"
    assert metadata["pixel_spacing_mm"] == []
    assert metadata["is_grayscale"] is True


def test_dataset_supports_raster_images(tmp_path: Path):
    raster_path = build_test_raster(tmp_path / "normal" / "sample.jpg")
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "mtddh::normal/sample.jpg",
                "group_id": "mtddh::group_001",
                "group_name": "group_001",
                "label": 0,
                "class_name": "normal",
                "source": "MTDDH",
                "source_code": "mtddh",
                "relative_path": "normal/sample.jpg",
                "path": str(raster_path),
            }
        ]
    )

    dataset = HipDysplasiaDataset(manifest, image_size=64, train=False)
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 64, 64)
    assert sample["label"] == 0
