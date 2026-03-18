"""Verification helpers for mixed test-set ID layout."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.submission_common import collect_test_objects

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


def _parse_submission_csv(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not csv_path.exists():
        return rows

    for line_number, raw_line in enumerate(csv_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            rows.append({"line_number": str(line_number), "object_id": "", "class": "", "raw": raw_line})
            continue

        rows.append(
            {
                "line_number": str(line_number),
                "object_id": parts[0],
                "class": parts[1],
                "raw": raw_line,
            }
        )

    return rows


def verify_submission_format(
    *,
    test_root: Path | None = None,
    csv_path: Path | None = None,
    screenshots_dir: Path | None = None,
    base_result: dict[str, Any] | None = None,
    check_sorted: bool = False,
) -> dict[str, Any]:
    """Optional submission-format checks for later phases."""
    result = dict(base_result or {})

    if csv_path is None:
        result["submission_check"] = "skipped"
        return result

    test_root = test_root or Path(result["test_root"])
    test_objects = collect_test_objects(test_root)
    expected_ids = [test_object.object_id for test_object in test_objects]

    csv_rows = _parse_submission_csv(csv_path)
    csv_ids = [row["object_id"] for row in csv_rows if row["object_id"]]
    screenshot_ids = (
        sorted(path.stem for path in screenshots_dir.glob("*.jpg"))
        if screenshots_dir is not None and screenshots_dir.exists()
        else []
    )

    malformed_csv_rows = [row for row in csv_rows if not row["object_id"] or not row["class"]]
    duplicate_ids = sorted(
        object_id for object_id, count in Counter(csv_ids).items() if count > 1
    )
    invalid_class_rows = [
        {
            "line_number": row["line_number"],
            "object_id": row["object_id"],
            "class": row["class"],
        }
        for row in csv_rows
        if row["class"] not in {"0", "1"}
    ]
    missing_csv_ids = sorted(set(expected_ids) - set(csv_ids))
    extra_csv_ids = sorted(set(csv_ids) - set(expected_ids))
    screenshot_check_enabled = screenshots_dir is not None
    missing_screenshots = (
        sorted(set(csv_ids) - set(screenshot_ids))
        if screenshot_check_enabled
        else []
    )
    extra_screenshots = (
        sorted(set(screenshot_ids) - set(csv_ids))
        if screenshot_check_enabled
        else []
    )
    sorted_ok = csv_ids == sorted(csv_ids)
    submission_valid = not (
        malformed_csv_rows
        or invalid_class_rows
        or duplicate_ids
        or missing_csv_ids
        or extra_csv_ids
        or (screenshot_check_enabled and (missing_screenshots or extra_screenshots))
        or (check_sorted and not sorted_ok)
    )

    result.update(
        {
            "submission_check": "completed",
            "expected_count": len(expected_ids),
            "csv_count": len(csv_ids),
            "screenshot_check_enabled": screenshot_check_enabled,
            "screenshot_count": len(screenshot_ids),
            "missing_csv_ids": missing_csv_ids,
            "extra_csv_ids": extra_csv_ids,
            "duplicate_csv_ids": duplicate_ids,
            "malformed_csv_rows": malformed_csv_rows,
            "invalid_class_rows": invalid_class_rows,
            "missing_screenshots": missing_screenshots,
            "extra_screenshots": extra_screenshots,
            "sorted_ok": sorted_ok,
            "sorted_check_enabled": bool(check_sorted),
            "submission_valid": submission_valid,
        }
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify test-set ID format.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset")
    parser.add_argument("--csv", default=None, help="Optional predictions.csv path")
    parser.add_argument("--screenshots-dir", default=None, help="Optional screenshots directory path")
    parser.add_argument(
        "--check-sorted",
        action="store_true",
        help="Require submission CSV IDs to be sorted lexicographically",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_result = collect_test_ids(Path(args.test_root))
    result = verify_submission_format(
        test_root=Path(args.test_root),
        csv_path=Path(args.csv) if args.csv else None,
        screenshots_dir=Path(args.screenshots_dir) if args.screenshots_dir else None,
        base_result=base_result,
        check_sorted=args.check_sorted,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("submission_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
