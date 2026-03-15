from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data_quality_check import build_markdown_report, save_report, scan_root
from scripts.add_dcm_extensions import rename_extensionless_dicoms
from scripts.verify_id_format import collect_test_ids
from tests.helpers import build_test_dicom


def test_verify_id_format_handles_mixed_folder_and_file_layout(tmp_path: Path):
    (tmp_path / "1OGQ64").mkdir()
    (tmp_path / "1OGQ64" / "26022616").mkdir()
    build_test_dicom(tmp_path / "1OGQ64" / "26022616" / "00000001")
    build_test_dicom(tmp_path / "28v1xk.dcm")
    (tmp_path / "KTjfqT.txt").write_text("ignored", encoding="utf-8")

    result = collect_test_ids(tmp_path)

    assert result["total_ids"] == 2
    assert result["valid_format"] is True
    assert sorted(result["ids"]) == ["1OGQ64", "28v1xk"]
    assert result["ignored_items"] == ["KTjfqT.txt"]


def test_verify_id_format_matches_real_workspace_layout_if_available():
    test_root = Path(__file__).resolve().parents[2] / "test_done"
    if not test_root.exists():
        pytest.skip("External test dataset is not available in this workspace.")

    result = collect_test_ids(test_root)

    assert result["total_ids"] == 19
    assert result["valid_format"] is True
    assert "KTjfqT.txt" in result["ignored_items"]


def test_data_quality_scan_and_report_generation(tmp_path: Path):
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    train_root.mkdir()
    test_root.mkdir()

    build_test_dicom(train_root / "sample_train.dcm", pixel_spacing=[0.2, 0.2], patient_id="PAT-001")
    build_test_dicom(test_root / "sample_test.dcm", imager_pixel_spacing=[0.2, 0.2], series_date="20240111")
    (test_root / "ignored.txt").write_text("ignore me", encoding="utf-8")

    train_summary = scan_root(train_root)
    test_summary = scan_root(test_root)

    assert train_summary["candidate_files"] == 1
    assert train_summary["valid_files"] == 1
    assert test_summary["candidate_files"] == 1
    assert test_summary["warning_files"] == 1

    markdown = build_markdown_report(train_summary, test_summary)
    output_path = save_report(
        train_summary=train_summary,
        test_summary=test_summary,
        output_path=tmp_path / "report.md",
        output_format="markdown",
    )

    assert "# Data Quality Report" in markdown
    assert output_path.exists()


def test_add_dcm_extensions_renames_only_extensionless_dicoms(tmp_path: Path):
    extensionless_dicom = build_test_dicom(tmp_path / "00000001")
    extensionless_text = tmp_path / "notes"
    extensionless_text.write_text("not a dicom", encoding="utf-8")
    existing_dicom = build_test_dicom(tmp_path / "already_named.dcm")

    summary = rename_extensionless_dicoms(tmp_path)

    assert summary["candidate_files"] == 2
    assert summary["renamed_count"] == 1
    assert (tmp_path / "00000001.dcm").exists()
    assert not extensionless_dicom.exists()
    assert extensionless_text.exists()
    assert existing_dicom.exists()
    assert summary["skipped_non_dicom_count"] == 1


def test_add_dcm_extensions_dry_run_does_not_modify_files(tmp_path: Path):
    extensionless_dicom = build_test_dicom(tmp_path / "preview_only")

    summary = rename_extensionless_dicoms(tmp_path, dry_run=True)

    assert summary["renamed_count"] == 1
    assert extensionless_dicom.exists()
    assert not (tmp_path / "preview_only.dcm").exists()
