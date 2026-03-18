"""Export final test_done deliverables as CSV, JSON, TXT and ZIP artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dicom_loader import load_dicom
from core.dicom_validator import DICOMValidator
from core.plugin_manager import AnalysisResult, PluginManager
from plugins.hip_dysplasia import HipDysplasiaPlugin
from scripts.submission_common import collect_test_objects, load_default_aggregation_method
from scripts.verify_id_format import collect_test_ids, verify_submission_format
from train.aggregation import aggregate_probability

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "models" / "checkpoints" / "resnet50_bce_v1" / "model_manifest.json"
DEFAULT_KEYPOINT_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "resnet50_mtddh_keypoints_v1" / "best.ckpt"


@dataclass(slots=True)
class ImageReport:
    path: Path
    result: AnalysisResult

    @property
    def runtime_model_loaded(self) -> bool:
        return bool(int(round(float(self.result.metrics.get("runtime_model_loaded", 0.0)))))

    @property
    def keypoint_model_loaded(self) -> bool:
        return bool(int(round(float(self.result.metrics.get("keypoint_model_loaded", 0.0)))))

    @property
    def geometry_available(self) -> bool:
        return bool(int(round(float(self.result.metrics.get("geometry_available", 0.0)))))

    @property
    def threshold(self) -> float:
        return float(self.result.metrics.get("model_threshold", 0.5))

    @property
    def probability(self) -> float:
        return float(self.result.metrics.get("model_probability", self.result.confidence))

    @property
    def keypoint_count(self) -> int:
        return int(round(float(self.result.metrics.get("keypoint_count", len(self.result.keypoints)))))

    def to_dict(self, *, relative_to: Path) -> dict[str, object]:
        return {
            "dicom_file": str(self.path.relative_to(relative_to)).replace("\\", "/"),
            "disease_detected": bool(self.result.disease_detected),
            "confidence": round(float(self.result.confidence), 6),
            "threshold": round(self.threshold, 6),
            "runtime_model_loaded": int(self.runtime_model_loaded),
            "keypoint_model_loaded": int(self.keypoint_model_loaded),
            "keypoint_count": self.keypoint_count,
            "geometry_available": int(self.geometry_available),
            "processing_time_ms": int(self.result.processing_time_ms),
            "validation_warnings": list(self.result.validation_warnings),
            "message": self.result.message,
            "metadata": self.result.metadata,
            "keypoints": [[round(float(x), 3), round(float(y), 3)] for x, y in self.result.keypoints],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final test_done deliverables.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test_done root")
    parser.add_argument(
        "--output-dir",
        default="deliverables/results_test_done",
        help="Directory for structured results",
    )
    parser.add_argument(
        "--predictions-output",
        default="deliverables/predictions.csv",
        help="Headerless id,class output path",
    )
    parser.add_argument(
        "--zip-output",
        default="deliverables/results_test_done.zip",
        help="ZIP archive path",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST.relative_to(REPO_ROOT)),
        help="Classifier manifest path",
    )
    parser.add_argument(
        "--keypoint-checkpoint",
        default=str(DEFAULT_KEYPOINT_CHECKPOINT.relative_to(REPO_ROOT)),
        help="Optional keypoint checkpoint path",
    )
    parser.add_argument(
        "--aggregation-method",
        default=load_default_aggregation_method(),
        help="Object-level aggregation method",
    )
    return parser.parse_args(argv)


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path if path.exists() else None


def _build_manager(*, manifest_path: Path | None, keypoint_checkpoint: Path | None) -> PluginManager:
    if manifest_path is not None:
        os.environ["HIP_DYSPLASIA_MODEL_MANIFEST"] = str(manifest_path)
    if keypoint_checkpoint is not None:
        os.environ["HIP_DYSPLASIA_KEYPOINT_CHECKPOINT"] = str(keypoint_checkpoint)
    manager = PluginManager()
    manager.register(HipDysplasiaPlugin())
    return manager


def _analyze_dicom(
    *,
    manager: PluginManager,
    validator: DICOMValidator,
    path: Path,
) -> ImageReport:
    image, metadata = load_dicom(str(path))
    validation_report = validator.validate(image, metadata)
    if not validation_report.valid:
        first_error = validation_report.errors[0]
        raise ValueError(f"Invalid DICOM '{path}': {first_error.message}")

    result = manager.analyze(
        "hip_dysplasia",
        image,
        metadata,
        mode="education",
        validation_warnings=validation_report.warning_messages(),
    )
    return ImageReport(path=path, result=result)


def _representative_report(image_reports: list[ImageReport]) -> ImageReport:
    return max(image_reports, key=lambda report: (report.probability, report.keypoint_count, report.path.name))


def _object_message(*, runtime_loaded: bool, aggregation_method: str, geometry_available: bool) -> str:
    if runtime_loaded:
        message = (
            "Classifier-first runtime выполнил итоговую агрегацию на уровне объекта "
            f"методом '{aggregation_method}'."
        )
    else:
        message = (
            "Для агрегации на уровне объекта использован fallback-режим. "
            "Результат сохраняет работоспособность pipeline, но не является диагностическим."
        )
    if not geometry_available:
        message = (
            f"{message} Количественная geometry автоматически не рассчитана, "
            "потому что семантика raw keypoints пока не валидирована для клинического использования."
        )
    return message


def _build_object_report(
    *,
    object_id: str,
    image_reports: list[ImageReport],
    aggregation_method: str,
    test_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    representative = _representative_report(image_reports)
    aggregated_probability = aggregate_probability(
        [report.probability for report in image_reports],
        method=aggregation_method,
    )
    threshold = representative.threshold
    runtime_loaded = all(report.runtime_model_loaded for report in image_reports)
    keypoint_model_loaded = any(report.keypoint_model_loaded for report in image_reports)
    keypoint_count = max(report.keypoint_count for report in image_reports)
    geometry_available = any(report.geometry_available for report in image_reports)
    disease_detected = runtime_loaded and aggregated_probability >= threshold
    report_class = int(disease_detected)
    warnings = [
        warning
        for report in image_reports
        for warning in report.result.validation_warnings
    ]
    unique_warnings = list(dict.fromkeys(warnings))
    message = _object_message(
        runtime_loaded=runtime_loaded,
        aggregation_method=aggregation_method,
        geometry_available=geometry_available,
    )
    processed_dicoms = [
        str(report.path.relative_to(test_root)).replace("\\", "/")
        for report in image_reports
    ]
    keypoints = [
        [round(float(x), 3), round(float(y), 3)]
        for x, y in representative.result.keypoints
    ]

    object_report = {
        "id": object_id,
        "class": report_class,
        "disease_detected": bool(disease_detected),
        "confidence": round(float(aggregated_probability), 6),
        "threshold": round(float(threshold), 6),
        "runtime_model_loaded": int(runtime_loaded),
        "keypoint_model_loaded": int(keypoint_model_loaded),
        "keypoint_count": int(keypoint_count),
        "geometry_available": int(geometry_available),
        "message": message,
        "metadata": representative.result.metadata,
        "keypoints": keypoints,
        "processed_dicoms": processed_dicoms,
        "validation_warnings": unique_warnings,
        "aggregation_method": aggregation_method,
        "representative_dicom": processed_dicoms[processed_dicoms.index(str(representative.path.relative_to(test_root)).replace("\\", "/"))],
        "images": [report.to_dict(relative_to=test_root) for report in image_reports],
    }

    summary_row = {
        "id": object_id,
        "class": report_class,
        "confidence": round(float(aggregated_probability), 6),
        "threshold": round(float(threshold), 6),
        "runtime_model_loaded": int(runtime_loaded),
        "keypoint_model_loaded": int(keypoint_model_loaded),
        "keypoint_count": int(keypoint_count),
        "geometry_available": int(geometry_available),
        "num_images": len(image_reports),
        "aggregation_method": aggregation_method,
        "representative_dicom": object_report["representative_dicom"],
        "report_file_json": f"reports/{object_id}.json",
        "report_file_txt": f"reports/{object_id}.txt",
    }
    return object_report, summary_row


def _build_text_report(report: dict[str, object]) -> str:
    label = "Патология выявлена" if report["class"] == 1 else "Признаки патологии не выявлены"
    runtime_status = "обученная модель загружена" if report["runtime_model_loaded"] else "использован fallback-режим"
    keypoint_status = (
        f"доступно {report['keypoint_count']} ориентиров"
        if report["keypoint_model_loaded"]
        else "keypoint-модель недоступна"
    )
    geometry_status = (
        "количественная geometry доступна"
        if report["geometry_available"]
        else "количественная geometry автоматически не рассчитана"
    )
    lines = [
        f"id: {report['id']}",
        f"Заключение: {label}.",
        f"confidence: {report['confidence']}",
        f"threshold: {report['threshold']}",
        f"runtime: {runtime_status}",
        f"keypoints: {keypoint_status}",
        f"geometry: {geometry_status}",
        f"message: {report['message']}",
        "Processed DICOM files:",
    ]
    lines.extend(f"- {path}" for path in report["processed_dicoms"])
    if report["validation_warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report["validation_warnings"])
    if not report["geometry_available"]:
        lines.append(
            "Примечание: количественные геометрические метрики не рассчитаны автоматически, "
            "потому что семантика raw keypoints пока не валидирована для клинического использования."
        )
    return "\n".join(lines) + "\n"


def _write_predictions_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = [f"{object_id},{class_value}" for object_id, class_value in sorted(rows)]
    path.write_text("\n".join(sorted_rows) + ("\n" if sorted_rows else ""), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "class",
        "confidence",
        "threshold",
        "runtime_model_loaded",
        "keypoint_model_loaded",
        "keypoint_count",
        "geometry_available",
        "num_images",
        "aggregation_method",
        "representative_dicom",
        "report_file_json",
        "report_file_txt",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["id"])):
            writer.writerow(row)


def _write_results_readme(path: Path, *, object_count: int, aggregation_method: str) -> None:
    lines = [
        "results_test_done",
        "",
        "Структура архива:",
        "- predictions.csv: итоговый headerless файл id,class",
        "- summary.csv: расширенная сводка по каждому объекту",
        "- reports/{id}.json: структурированный отчет по объекту",
        "- reports/{id}.txt: краткий человекочитаемый отчет по объекту",
        "",
        f"Количество объектов: {object_count}",
        f"Метод агрегации на уровне объекта: {aggregation_method}",
        "Keypoints используются только как explainability layer.",
        "Количественная geometry автоматически не рассчитывается, если она не валидирована.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_zip(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=str(path.relative_to(source_dir)).replace("\\", "/"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = Path(args.test_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    predictions_output = Path(args.predictions_output).resolve()
    zip_output = Path(args.zip_output).resolve()
    manifest_path = _resolve_optional_path(args.manifest_path)
    keypoint_checkpoint = _resolve_optional_path(args.keypoint_checkpoint)
    aggregation_method = str(args.aggregation_method).strip().lower() or load_default_aggregation_method()

    manager = _build_manager(
        manifest_path=manifest_path,
        keypoint_checkpoint=keypoint_checkpoint,
    )
    validator = DICOMValidator()
    objects = collect_test_objects(test_root)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[tuple[str, int]] = []
    summary_rows: list[dict[str, object]] = []

    for test_object in objects:
        image_reports = [
            _analyze_dicom(manager=manager, validator=validator, path=path)
            for path in test_object.dicom_paths
        ]
        object_report, summary_row = _build_object_report(
            object_id=test_object.object_id,
            image_reports=image_reports,
            aggregation_method=aggregation_method,
            test_root=test_root,
        )
        prediction_rows.append((test_object.object_id, int(object_report["class"])))
        summary_rows.append(summary_row)

        json_path = reports_dir / f"{test_object.object_id}.json"
        txt_path = reports_dir / f"{test_object.object_id}.txt"
        json_path.write_text(
            json.dumps(object_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        txt_path.write_text(_build_text_report(object_report), encoding="utf-8")

    _write_predictions_csv(predictions_output, prediction_rows)
    _write_predictions_csv(output_dir / "predictions.csv", prediction_rows)
    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    _write_results_readme(
        output_dir / "README_results.txt",
        object_count=len(objects),
        aggregation_method=aggregation_method,
    )
    _create_zip(output_dir, zip_output)

    base_result = collect_test_ids(test_root)
    verification = verify_submission_format(
        test_root=test_root,
        csv_path=predictions_output,
        base_result=base_result,
        check_sorted=True,
    )
    (output_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not verification.get("submission_valid", False):
        raise SystemExit("Predictions CSV failed verification.")

    print(f"Saved predictions: {predictions_output}")
    print(f"Saved structured results: {output_dir}")
    print(f"Saved ZIP archive: {zip_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
