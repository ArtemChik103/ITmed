from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ddh_binary_audit import run_audit
from tests.helpers import build_test_raster


def test_ddh_binary_audit_writes_required_outputs_and_detects_variants(tmp_path: Path):
    dataset_root = tmp_path / "ddh_binary"
    local_root = tmp_path / "local"

    build_test_raster(
        dataset_root / "224" / "Normal" / "case_001.png",
        pixel_array=np.tile(np.array([[0, 32], [96, 255]], dtype=np.uint8), (32, 48)),
    )
    larger_variant = build_test_raster(
        dataset_root / "331" / "Normal" / "case_001.png",
        pixel_array=np.tile(
            np.array(
                [
                    [0, 0, 32, 32],
                    [0, 32, 96, 96],
                    [32, 96, 255, 255],
                    [32, 96, 255, 255],
                ],
                dtype=np.uint8,
            ),
            (48, 40),
        ),
    )
    build_test_raster(
        dataset_root / "331" / "DDH" / "case_002.png",
        pixel_array=np.tile(np.array([[255, 128], [64, 0]], dtype=np.uint8), (40, 56)),
    )
    build_test_raster(
        dataset_root / "331" / "mild" / "case_003.png",
        pixel_array=np.tile(np.array([[255, 255], [32, 32]], dtype=np.uint8), (28, 28)),
    )
    (dataset_root / "README.md").write_text("# DDH binary dataset\n", encoding="utf-8")

    local_duplicate = build_test_raster(
        local_root / "local_duplicate.png",
        pixel_array=np.tile(np.array([[255, 128], [64, 0]], dtype=np.uint8), (40, 56)),
    )
    local_manifest = pd.DataFrame(
        [
            {
                "sample_id": "local::sample_001",
                "group_id": "local::group_001",
                "label": 1,
                "relative_path": "local_duplicate.png",
                "path": str(local_duplicate.resolve()),
                "source": "local",
                "class_name": "pathology",
            }
        ]
    )
    local_manifest_path = tmp_path / "train_manifest.csv"
    local_manifest.to_csv(local_manifest_path, index=False)

    output_dir = tmp_path / "audit"
    inventory = run_audit(dataset_root, output_dir=output_dir, local_manifest_path=local_manifest_path)

    assert inventory["total_files"] == 5
    assert (output_dir / "inventory.json").exists()
    assert (output_dir / "file_index.csv").exists()
    assert (output_dir / "label_summary.csv").exists()
    assert (output_dir / "metadata_summary.json").exists()
    assert (output_dir / "duplicates.csv").exists()
    assert (output_dir / "excluded_samples.csv").exists()
    assert (output_dir / "verdict.json").exists()
    assert (output_dir / "contact_sheets" / "excluded.jpg").exists()

    file_index = pd.read_csv(output_dir / "file_index.csv")
    excluded = pd.read_csv(output_dir / "excluded_samples.csv")
    duplicates = pd.read_csv(output_dir / "duplicates.csv")
    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))

    canonical_row = file_index[file_index["relative_path"] == larger_variant.relative_to(dataset_root).as_posix()].iloc[0]
    assert canonical_row["exclude_reason"] == "overlay_text_heavy"
    assert inventory["eligible_training_images"] == 0
    assert "same_object_variant" in set(excluded["exclude_reason"])
    assert "exact_duplicate_local_train" in set(excluded["exclude_reason"])
    assert "ambiguous_label" in set(excluded["exclude_reason"])
    assert "same_object_variant" in set(duplicates["kind"])
    assert verdict["gates"]["explicit_binary_label"] is False
    assert verdict["gates"]["no_local_duplicate_leakage"] is False
