"""Generate static JPG screenshots for each submission object."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.submission_common import (
    ApiRuntimeAnalyzer,
    LocalRuntimeAnalyzer,
    collect_test_objects,
    load_dicom_preview,
    render_prediction_screenshot,
    save_jpeg,
    select_representative_dicoms,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JPG screenshots for test_done objects.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset root")
    parser.add_argument(
        "--output-dir",
        default="submissions/screenshots",
        help="Directory for generated JPG screenshots",
    )
    parser.add_argument("--plugin-type", default="hip_dysplasia", help="Plugin identifier")
    parser.add_argument("--api-url", default=None, help="Use HTTP API instead of local runtime")
    parser.add_argument("--local-runtime", action="store_true", help="Force direct local runtime inference")
    parser.add_argument("--manifest-path", default=None, help="Optional model manifest for local runtime")
    return parser.parse_args(argv)


def _build_analyzer(args: argparse.Namespace):
    if args.api_url and not args.local_runtime:
        return ApiRuntimeAnalyzer(api_url=args.api_url, plugin_type=args.plugin_type)
    return LocalRuntimeAnalyzer(plugin_type=args.plugin_type, manifest_path=args.manifest_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = _build_analyzer(args)
    objects = collect_test_objects(test_root)

    for test_object in objects:
        prediction = analyzer.predict_object(test_object)
        preview_paths = select_representative_dicoms(test_object.dicom_paths, limit=4)
        preview_images = [load_dicom_preview(path)[0] for path in preview_paths]
        screenshot = render_prediction_screenshot(prediction, preview_images)
        save_jpeg(screenshot, output_dir / f"{test_object.object_id}.jpg")

    print(f"Saved {len(objects)} screenshots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
