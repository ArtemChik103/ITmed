"""Export a small deterministic subset of objects for local demo preparation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.submission_common import collect_test_objects, select_representative_dicoms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a few test objects into a local demo folder.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset root")
    parser.add_argument(
        "--output-dir",
        default="frontend/data/demo_cases",
        help="Where demo objects and manifest should be exported",
    )
    parser.add_argument("--limit", type=int, default=3, help="How many objects to export")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects = collect_test_objects(test_root)[: max(1, args.limit)]
    manifest_rows: list[dict[str, object]] = []

    for test_object in objects:
        destination = output_dir / test_object.object_id
        destination.mkdir(parents=True, exist_ok=True)
        for dicom_path in test_object.dicom_paths:
            shutil.copy2(dicom_path, destination / dicom_path.name)

        manifest_rows.append(
            {
                "object_id": test_object.object_id,
                "source_path": str(test_object.source_path),
                "num_images": len(test_object.dicom_paths),
                "representative_files": [path.name for path in select_representative_dicoms(test_object.dicom_paths)],
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(objects)} demo objects to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
