"""Create a minimal Phase 3 bundle for Google Colab runs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_kaggle_bundle import build_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a minimal Phase 3 bundle for Google Colab.")
    parser.add_argument(
        "--output-dir",
        default="build/colab_phase3_bundle",
        help="Directory where the bundle will be created.",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="Include local ImageNet checkpoints from models/pretrained.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a zip archive next to the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary = build_bundle(output_dir, include_pretrained=bool(args.include_pretrained))
    archive_path = None
    if args.archive:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)

    payload = {
        "status": "ok",
        "output_dir": str(output_dir),
        "archive_path": archive_path,
        "file_count": summary["file_count"],
        "include_pretrained": summary["include_pretrained"],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
