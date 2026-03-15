"""Rename extensionless DICOM files by appending the .dcm suffix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import pydicom
except ModuleNotFoundError:  # pragma: no cover - host fallback for one-off filesystem ops
    pydicom = None


def has_dicom_magic(path: Path) -> bool:
    """Return True when the file contains the standard DICOM preamble."""
    try:
        with path.open("rb") as file_obj:
            header = file_obj.read(132)
    except OSError:
        return False

    return len(header) >= 132 and header[128:132] == b"DICM"


def is_probably_dicom(path: Path) -> bool:
    """Return True when the file can be parsed as a DICOM dataset."""
    if has_dicom_magic(path):
        return True

    if pydicom is None:
        return False

    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "SOPClassUID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Modality",
                "Rows",
                "Columns",
            ],
        )
    except Exception:
        return False

    return any(
        getattr(dataset, field_name, None) is not None
        for field_name in (
            "SOPClassUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Modality",
            "Rows",
            "Columns",
        )
    )


def rename_extensionless_dicoms(
    root: Path,
    *,
    suffix: str = ".dcm",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rename extensionless DICOM files under the given root."""
    renamed: list[dict[str, str]] = []
    skipped_non_dicom: list[str] = []
    skipped_collisions: list[dict[str, str]] = []
    candidates = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix:
            continue

        candidates += 1
        target = path.with_name(f"{path.name}{suffix}")

        if target.exists():
            skipped_collisions.append({"source": str(path), "target": str(target)})
            continue

        if not is_probably_dicom(path):
            skipped_non_dicom.append(str(path))
            continue

        renamed.append({"source": str(path), "target": str(target)})
        if not dry_run:
            path.rename(target)

    return {
        "root": str(root),
        "candidate_files": candidates,
        "renamed_count": len(renamed),
        "renamed": renamed,
        "skipped_non_dicom_count": len(skipped_non_dicom),
        "skipped_non_dicom": skipped_non_dicom,
        "collision_count": len(skipped_collisions),
        "collisions": skipped_collisions,
        "dry_run": dry_run,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append .dcm to extensionless DICOM files.")
    parser.add_argument(
        "roots",
        nargs="*",
        default=["../train", "../test_done"],
        help="Dataset roots to normalize",
    )
    parser.add_argument(
        "--suffix",
        default=".dcm",
        help="Suffix to append to extensionless DICOM files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without changing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []

    for root_arg in args.roots:
        root = Path(root_arg).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")

        summaries.append(
            rename_extensionless_dicoms(
                root,
                suffix=args.suffix,
                dry_run=args.dry_run,
            )
        )

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
