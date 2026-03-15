"""Dataset quality checks for training and test DICOM collections."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dicom_loader import load_dicom
from core.dicom_validator import DICOMValidator

IGNORED_SUFFIXES = {".jpg", ".jpeg", ".png", ".txt", ".csv", ".md"}


def iter_dicom_candidates(root: Path) -> list[Path]:
    """Return DICOM-like files, including extensionless files used in the dataset."""
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        candidates.append(path)
    return sorted(candidates)


def scan_root(root: Path, validator: DICOMValidator | None = None) -> dict[str, Any]:
    """Scan a root directory and return a quality summary."""
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    validator = validator or DICOMValidator()
    candidates = iter_dicom_candidates(root)
    modality_counts: Counter[str] = Counter()
    pixel_spacing_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    error_files: list[dict[str, str]] = []

    valid_files = 0
    warning_files = 0

    for path in candidates:
        try:
            image, metadata = load_dicom(str(path))
        except Exception as exc:
            error_files.append({"file": str(path), "error": str(exc)})
            continue

        report = validator.validate(image, metadata)
        if report.valid:
            valid_files += 1
        else:
            error_files.append({"file": str(path), "error": report.errors[0].message})
            continue

        if report.warnings:
            warning_files += 1
            for warning in report.warnings:
                warning_counts[warning.code] += 1

        modality_counts[str(metadata.get("modality") or "UNKNOWN")] += 1
        pixel_spacing_counts[str(metadata.get("pixel_spacing_source") or "UNKNOWN")] += 1

    return {
        "root": str(root),
        "candidate_files": len(candidates),
        "valid_files": valid_files,
        "warning_files": warning_files,
        "error_files": len(error_files),
        "modalities": dict(modality_counts),
        "pixel_spacing_sources": dict(pixel_spacing_counts),
        "warning_counts": dict(warning_counts),
        "errors": error_files[:20],
    }


def build_markdown_report(train_summary: dict[str, Any], test_summary: dict[str, Any]) -> str:
    """Render a markdown report."""
    sections: list[str] = ["# Data Quality Report", ""]
    for title, summary in (("Training", train_summary), ("Test", test_summary)):
        sections.extend(
            [
                f"## {title}",
                "",
                f"- Root: `{summary['root']}`",
                f"- Candidate files: `{summary['candidate_files']}`",
                f"- Valid files: `{summary['valid_files']}`",
                f"- Files with warnings: `{summary['warning_files']}`",
                f"- Files with errors: `{summary['error_files']}`",
                "",
                "### Modalities",
                "",
            ]
        )
        if summary["modalities"]:
            for key, value in summary["modalities"].items():
                sections.append(f"- `{key}`: {value}")
        else:
            sections.append("- none")

        sections.extend(["", "### Pixel spacing sources", ""])
        if summary["pixel_spacing_sources"]:
            for key, value in summary["pixel_spacing_sources"].items():
                sections.append(f"- `{key}`: {value}")
        else:
            sections.append("- none")

        sections.extend(["", "### Warning counts", ""])
        if summary["warning_counts"]:
            for key, value in summary["warning_counts"].items():
                sections.append(f"- `{key}`: {value}")
        else:
            sections.append("- none")

        sections.extend(["", "### Sample errors", ""])
        if summary["errors"]:
            for item in summary["errors"]:
                sections.append(f"- `{item['file']}`: {item['error']}")
        else:
            sections.append("- none")
        sections.append("")

    return "\n".join(sections).strip() + "\n"


def save_report(
    *,
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
    output_path: Path,
    output_format: str,
) -> Path:
    """Persist the requested report format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        payload = {"training": train_summary, "test": test_summary}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        output_path.write_text(
            build_markdown_report(train_summary, test_summary),
            encoding="utf-8",
        )

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dataset quality report.")
    parser.add_argument("--train-root", default="../train", help="Path to the training dataset")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset")
    parser.add_argument(
        "--output",
        default="docs/data_quality_report.md",
        help="Output report path",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output report format",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validator = DICOMValidator()
    train_summary = scan_root(Path(args.train_root), validator)
    test_summary = scan_root(Path(args.test_root), validator)
    output_path = save_report(
        train_summary=train_summary,
        test_summary=test_summary,
        output_path=Path(args.output),
        output_format=args.format,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
