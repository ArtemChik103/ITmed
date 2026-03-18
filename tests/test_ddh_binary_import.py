from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.import_ddh_binary import build_ddh_binary_manifest


def test_ddh_binary_import_builds_manifest_from_positive_audit(tmp_path: Path):
    dataset_root = tmp_path / "ddh_binary"
    dataset_root.mkdir(parents=True, exist_ok=True)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    (audit_dir / "verdict.json").write_text(
        json.dumps({"training_decision": "worth_continuing", "verdict": "worth_fine_tuning"}, ensure_ascii=False),
        encoding="utf-8",
    )
    file_index = pd.DataFrame(
        [
            {
                "is_image": 1,
                "eligible_for_training": 1,
                "label": 0,
                "relative_path": "331/Normal/case_001.png",
                "absolute_path": str((dataset_root / "331" / "Normal" / "case_001.png").resolve()),
                "annotation_group_id": "",
                "inferred_object_id": "case_001",
                "parent_folder": "Normal",
                "file_stem": "case_001",
                "original_id": "case_001",
                "dataset_name": "A dataset of DDH x-ray images",
                "file_type": "png",
                "label_confidence": 1.0,
                "label_source": "folder_name",
            },
            {
                "is_image": 1,
                "eligible_for_training": 1,
                "label": 1,
                "relative_path": "331/DDH/case_002.png",
                "absolute_path": str((dataset_root / "331" / "DDH" / "case_002.png").resolve()),
                "annotation_group_id": "",
                "inferred_object_id": "case_002",
                "parent_folder": "DDH",
                "file_stem": "case_002",
                "original_id": "case_002",
                "dataset_name": "A dataset of DDH x-ray images",
                "file_type": "png",
                "label_confidence": 0.95,
                "label_source": "folder_name",
            },
        ]
    )
    file_index.to_csv(audit_dir / "file_index.csv", index=False)

    manifest = build_ddh_binary_manifest(
        dataset_root=dataset_root,
        audit_dir=audit_dir,
        output_manifest=tmp_path / "ddh_binary_manifest.csv",
    )

    assert len(manifest) == 2
    assert manifest["sample_id"].tolist()[0] == "ddh_binary_ext::331/Normal/case_001.png"
    assert set(manifest["source_code"]) == {"ddh_binary_ext"}
    assert manifest["group_id"].tolist()[0].startswith("ddh_binary_ext::")


def test_ddh_binary_import_rejects_negative_audit_verdict(tmp_path: Path):
    dataset_root = tmp_path / "ddh_binary"
    dataset_root.mkdir(parents=True, exist_ok=True)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    (audit_dir / "verdict.json").write_text(
        json.dumps(
            {"training_decision": "audit_only_do_not_fine_tune", "verdict": "audit-only verdict: do not fine-tune"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "is_image": 1,
                "eligible_for_training": 1,
                "label": 0,
                "relative_path": "331/Normal/case_001.png",
                "absolute_path": str((dataset_root / "331" / "Normal" / "case_001.png").resolve()),
                "annotation_group_id": "",
                "inferred_object_id": "case_001",
                "parent_folder": "Normal",
                "file_stem": "case_001",
                "original_id": "case_001",
                "dataset_name": "A dataset of DDH x-ray images",
                "file_type": "png",
                "label_confidence": 1.0,
                "label_source": "folder_name",
            }
        ]
    ).to_csv(audit_dir / "file_index.csv", index=False)

    with pytest.raises(ValueError, match="does not allow fine-tuning"):
        build_ddh_binary_manifest(
            dataset_root=dataset_root,
            audit_dir=audit_dir,
            output_manifest=tmp_path / "ddh_binary_manifest.csv",
        )
