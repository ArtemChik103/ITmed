"""Verification helpers for mixed test-set ID layout."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# The hackathon TЗ uses the top-level object name as the ID.
# Real workspace data contains both 5-char and 6-char alphanumeric IDs.
ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


def collect_test_ids(test_root: Path) -> dict[str, Any]:
    """Collect IDs from a mixed test root containing folders and single DICOM files."""
    if not test_root.exists():
        raise FileNotFoundError(f"Path does not exist: {test_root}")

    ids: list[str] = []
    folder_ids: list[str] = []
    file_ids: list[str] = []
    ignored_items: list[str] = []

    for item in sorted(test_root.iterdir(), key=lambda path: path.name.lower()):
        if item.is_dir():
            ids.append(item.name)
            folder_ids.append(item.name)
            continue

        if item.is_file() and item.suffix.lower() in {".dcm", ".dicom"}:
            ids.append(item.stem)
            file_ids.append(item.stem)
            continue

        ignored_items.append(item.name)

    valid_ids = [object_id for object_id in ids if ID_PATTERN.fullmatch(object_id)]
    invalid_ids = [object_id for object_id in ids if object_id not in valid_ids]

    return {
        "test_root": str(test_root),
        "total_ids": len(ids),
        "valid_format": len(valid_ids) == len(ids),
        "ids": ids,
        "folder_ids": folder_ids,
        "file_ids": file_ids,
        "ignored_items": ignored_items,
        "invalid_ids": invalid_ids,
        "sample": ids[:5],
    }


def verify_submission_format(
    *,
    csv_path: Path | None = None,
    screenshots_dir: Path | None = None,
    base_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optional submission-format checks for later phases."""
    result = dict(base_result or {})

    if csv_path is None or screenshots_dir is None:
        result["submission_check"] = "skipped"
        return result

    csv_ids: list[str] = []
    if csv_path.exists():
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            object_id = line.split(",", 1)[0].strip()
            csv_ids.append(object_id)

    screenshot_ids = sorted(path.stem for path in screenshots_dir.glob("*.jpg"))

    missing_screenshots = sorted(set(csv_ids) - set(screenshot_ids))
    extra_screenshots = sorted(set(screenshot_ids) - set(csv_ids))

    result.update(
        {
            "submission_check": "completed",
            "csv_count": len(csv_ids),
            "screenshot_count": len(screenshot_ids),
            "missing_screenshots": missing_screenshots,
            "extra_screenshots": extra_screenshots,
            "submission_valid": not missing_screenshots and not extra_screenshots,
        }
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify test-set ID format.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset")
    parser.add_argument("--csv", default=None, help="Optional predictions.csv path")
    parser.add_argument("--screenshots-dir", default=None, help="Optional screenshots directory path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_result = collect_test_ids(Path(args.test_root))
    result = verify_submission_format(
        csv_path=Path(args.csv) if args.csv else None,
        screenshots_dir=Path(args.screenshots_dir) if args.screenshots_dir else None,
        base_result=base_result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
