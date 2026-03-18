from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.data_quality_check import build_markdown_report, save_report, scan_root
from scripts.add_dcm_extensions import rename_extensionless_dicoms
from scripts.export_test_done_reports import main as export_test_done_reports_main
from scripts.generate_submission import main as generate_submission_main
from scripts.submission_common import collect_test_objects
from scripts.verify_id_format import collect_test_ids, verify_submission_format
from tests.helpers import build_test_dicom
from tests.phase3_helpers import write_synthetic_checkpoint, write_synthetic_model_manifest


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


def test_collect_test_objects_resolves_folder_and_file_objects(tmp_path: Path):
    folder_object = tmp_path / "ABC123"
    (folder_object / "26022616").mkdir(parents=True)
    build_test_dicom(folder_object / "26022616" / "00000001.dcm")
    build_test_dicom(folder_object / "26022616" / "00000002.dcm")
    build_test_dicom(tmp_path / "XYZ789.dcm")

    objects = collect_test_objects(tmp_path)

    assert [test_object.object_id for test_object in objects] == ["ABC123", "XYZ789"]
    assert len(objects[0].dicom_paths) == 2
    assert len(objects[1].dicom_paths) == 1


def test_generate_submission_writes_headerless_csv_and_detailed_output(tmp_path: Path):
    test_root = tmp_path / "test_done"
    test_root.mkdir()
    object_dir = test_root / "AAA111"
    (object_dir / "series").mkdir(parents=True)
    build_test_dicom(object_dir / "series" / "00000001.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")
    build_test_dicom(test_root / "BBB222.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")

    checkpoint_path = write_synthetic_checkpoint(tmp_path / "runtime" / "fold_0" / "best.pt")
    manifest_path = write_synthetic_model_manifest(
        tmp_path / "runtime" / "model_manifest.json",
        checkpoint_paths=[checkpoint_path],
        input_size=64,
        threshold=0.5,
    )

    output_csv = tmp_path / "predictions.csv"
    detailed_csv = tmp_path / "predictions_detailed.csv"
    exit_code = generate_submission_main(
        [
            "--test-root",
            str(test_root),
            "--output",
            str(output_csv),
            "--detailed-output",
            str(detailed_csv),
            "--manifest-path",
            str(manifest_path),
            "--local-runtime",
        ]
    )

    assert exit_code == 0
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("AAA111,")
    assert "object_id" in detailed_csv.read_text(encoding="utf-8")


def test_verify_submission_format_checks_csv_and_screenshots(tmp_path: Path):
    build_test_dicom(tmp_path / "QWE123.dcm")
    build_test_dicom(tmp_path / "RTY456.dcm")

    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text("QWE123,1\nRTY456,0\n", encoding="utf-8")

    screenshots_dir = tmp_path / "screens"
    screenshots_dir.mkdir()
    Image.new("RGB", (32, 32), "white").save(screenshots_dir / "QWE123.jpg")
    Image.new("RGB", (32, 32), "white").save(screenshots_dir / "RTY456.jpg")

    result = verify_submission_format(
        test_root=tmp_path,
        csv_path=csv_path,
        screenshots_dir=screenshots_dir,
        base_result=collect_test_ids(tmp_path),
    )

    assert result["submission_valid"] is True
    assert result["missing_csv_ids"] == []
    assert result["missing_screenshots"] == []


def test_verify_submission_format_checks_csv_without_screenshots(tmp_path: Path):
    build_test_dicom(tmp_path / "AAA111.dcm")
    build_test_dicom(tmp_path / "BBB222.dcm")

    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text("AAA111,1\nBBB222,0\n", encoding="utf-8")

    result = verify_submission_format(
        test_root=tmp_path,
        csv_path=csv_path,
        base_result=collect_test_ids(tmp_path),
        check_sorted=True,
    )

    assert result["submission_valid"] is True
    assert result["screenshot_check_enabled"] is False
    assert result["missing_screenshots"] == []


def test_export_test_done_reports_writes_final_artifacts(tmp_path: Path):
    test_root = tmp_path / "test_done"
    test_root.mkdir()
    object_dir = test_root / "AAA111" / "series"
    object_dir.mkdir(parents=True)
    build_test_dicom(object_dir / "00000001.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")
    build_test_dicom(test_root / "BBB222.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")

    checkpoint_path = write_synthetic_checkpoint(tmp_path / "runtime" / "fold_0" / "best.pt")
    manifest_path = write_synthetic_model_manifest(
        tmp_path / "runtime" / "model_manifest.json",
        checkpoint_paths=[checkpoint_path],
        input_size=64,
        threshold=0.5,
    )

    output_dir = tmp_path / "deliverables" / "results_test_done"
    predictions_output = tmp_path / "deliverables" / "predictions.csv"
    zip_output = tmp_path / "deliverables" / "results_test_done.zip"
    exit_code = export_test_done_reports_main(
        [
            "--test-root",
            str(test_root),
            "--output-dir",
            str(output_dir),
            "--predictions-output",
            str(predictions_output),
            "--zip-output",
            str(zip_output),
            "--manifest-path",
            str(manifest_path),
            "--keypoint-checkpoint",
            "",
        ]
    )

    assert exit_code == 0
    assert predictions_output.exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "reports" / "AAA111.json").exists()
    assert (output_dir / "reports" / "BBB222.txt").exists()
    assert (output_dir / "README_results.txt").exists()
    assert (output_dir / "verification.json").exists()
    assert zip_output.exists()
