from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.mtddh_audit import run_audit
from tests.helpers import build_test_raster


def test_mtddh_audit_writes_required_outputs_and_exclusions(tmp_path: Path):
    dataset_root = tmp_path / "mtddh"
    normal_path = build_test_raster(
        dataset_root / "normal_bucket" / "case_001" / "img001.png",
        pixel_array=np.array([[0, 10], [20, 30]], dtype=np.uint8),
    )
    pathology_path = build_test_raster(
        dataset_root / "pathology_bucket" / "case_002" / "img002.jpg",
        pixel_array=np.array([[255, 120], [64, 0]], dtype=np.uint8),
    )
    build_test_raster(
        dataset_root / "overlay_bucket" / "case_002" / "img002_overlay.png",
        pixel_array=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    build_test_raster(
        dataset_root / "normal_bucket" / "case_001_copy" / "img001_copy.png",
        pixel_array=np.array([[0, 10], [20, 30]], dtype=np.uint8),
    )
    labels = pd.DataFrame(
        [
            {"image_id": "img001", "diagnosis": "normal"},
            {"image_id": "img002", "diagnosis": "ddh"},
            {"image_id": "img001_copy", "diagnosis": "normal"},
        ]
    )
    labels.to_csv(dataset_root / "labels.csv", index=False)

    output_dir = tmp_path / "audit"
    inventory = run_audit(dataset_root, output_dir=output_dir, local_manifest_path=None)

    assert inventory["total_files"] == 5
    assert (output_dir / "inventory.json").exists()
    assert (output_dir / "file_index.csv").exists()
    assert (output_dir / "label_summary.csv").exists()
    assert (output_dir / "metadata_summary.json").exists()
    assert (output_dir / "duplicates.csv").exists()
    assert (output_dir / "excluded_samples.csv").exists()
    assert (output_dir / "contact_sheets" / "normal.jpg").exists()
    assert (output_dir / "contact_sheets" / "pathology.jpg").exists()

    file_index = pd.read_csv(output_dir / "file_index.csv")
    excluded = pd.read_csv(output_dir / "excluded_samples.csv")
    duplicates = pd.read_csv(output_dir / "duplicates.csv")

    assert int(file_index[file_index["relative_path"] == normal_path.relative_to(dataset_root).as_posix()]["eligible_for_training"].iloc[0]) == 1
    assert int(file_index[file_index["relative_path"] == pathology_path.relative_to(dataset_root).as_posix()]["eligible_for_training"].iloc[0]) == 1
    assert "non_radiograph_role" in set(excluded["exclude_reason"])
    assert "exact_duplicate_mtddh" in set(excluded["exclude_reason"])
    assert "exact_mtddh" in set(duplicates["kind"])
