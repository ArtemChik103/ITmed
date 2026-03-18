from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.import_mtddh import build_mtddh_manifest


def test_mtddh_import_builds_deterministic_manifest(tmp_path: Path):
    dataset_root = tmp_path / "mtddh"
    dataset_root.mkdir(parents=True, exist_ok=True)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    file_index = pd.DataFrame(
        [
            {
                "is_image": 1,
                "eligible_for_training": 1,
                "label": 0,
                "relative_path": "bucket_a/case_001/image_001.png",
                "absolute_path": str((dataset_root / "bucket_a" / "case_001" / "image_001.png").resolve()),
                "annotation_group_id": "patient_001",
                "case_folder": "case_001",
                "parent_folder": "case_001",
                "file_stem": "image_001",
                "original_id": "image_001",
                "dataset_name": "MTDDH",
                "file_type": "png",
                "view": "ap",
                "age_months": 6,
                "label_confidence": 1.0,
                "label_source": "table:labels.csv",
            },
            {
                "is_image": 1,
                "eligible_for_training": 1,
                "label": 1,
                "relative_path": "bucket_b/case_002/image_002.jpg",
                "absolute_path": str((dataset_root / "bucket_b" / "case_002" / "image_002.jpg").resolve()),
                "annotation_group_id": "",
                "case_folder": "case_002",
                "parent_folder": "case_002",
                "file_stem": "image_002",
                "original_id": "image_002",
                "dataset_name": "MTDDH",
                "file_type": "jpg",
                "view": "pelvis",
                "age_months": None,
                "label_confidence": 0.9,
                "label_source": "folder_name",
            },
            {
                "is_image": 1,
                "eligible_for_training": 0,
                "label": 0,
                "relative_path": "bucket_c/case_003/image_003.jpg",
                "absolute_path": str((dataset_root / "bucket_c" / "case_003" / "image_003.jpg").resolve()),
                "annotation_group_id": "",
                "case_folder": "case_003",
                "parent_folder": "case_003",
                "file_stem": "image_003",
                "original_id": "image_003",
                "dataset_name": "MTDDH",
                "file_type": "jpg",
                "view": "pelvis",
                "age_months": None,
                "label_confidence": 0.9,
                "label_source": "folder_name",
            },
        ]
    )
    file_index.to_csv(audit_dir / "file_index.csv", index=False)

    manifest = build_mtddh_manifest(
        dataset_root=dataset_root,
        audit_dir=audit_dir,
        output_manifest=tmp_path / "mtddh_manifest.csv",
    )

    assert len(manifest) == 2
    assert manifest["sample_id"].tolist()[0] == "mtddh::bucket_a/case_001/image_001.png"
    assert manifest["group_id"].tolist()[0] == "mtddh::patient_001"
    assert manifest["group_id"].tolist()[1] == "mtddh::case_002"
    assert set(manifest["source_code"]) == {"mtddh"}
