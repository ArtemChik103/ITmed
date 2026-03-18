"""Generate submission CSVs for the mixed-layout `test_done` dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.submission_common import (
    ApiRuntimeAnalyzer,
    LocalRuntimeAnalyzer,
    collect_test_objects,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions.csv for test_done.")
    parser.add_argument("--test-root", default="../test_done", help="Path to the test dataset root")
    parser.add_argument("--output", default="submissions/predictions.csv", help="Headerless output CSV path")
    parser.add_argument(
        "--detailed-output",
        default="submissions/predictions_detailed.csv",
        help="Detailed CSV path with probabilities and runtime info",
    )
    parser.add_argument("--plugin-type", default="hip_dysplasia", help="Plugin identifier")
    parser.add_argument("--api-url", default=None, help="Use HTTP API instead of local runtime")
    parser.add_argument(
        "--local-runtime",
        action="store_true",
        help="Force direct local runtime inference",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional model manifest for local runtime. If omitted, env/default resolution is used.",
    )
    return parser.parse_args(argv)


def _build_analyzer(args: argparse.Namespace):
    if args.api_url and not args.local_runtime:
        return ApiRuntimeAnalyzer(api_url=args.api_url, plugin_type=args.plugin_type)
    return LocalRuntimeAnalyzer(plugin_type=args.plugin_type, manifest_path=args.manifest_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_root = Path(args.test_root)
    output_path = Path(args.output)
    detailed_output_path = Path(args.detailed_output)

    objects = collect_test_objects(test_root)
    analyzer = _build_analyzer(args)

    csv_rows: list[tuple[str, int]] = []
    detailed_rows: list[dict[str, object]] = []

    for test_object in objects:
        prediction = analyzer.predict_object(test_object)
        csv_rows.append((prediction.object_id, prediction.csv_class()))
        detailed_rows.append(prediction.detailed_row())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_csv_rows = [f"{object_id},{class_value}" for object_id, class_value in sorted(csv_rows)]
    output_path.write_text(
        "\n".join(sorted_csv_rows) + ("\n" if sorted_csv_rows else ""),
        encoding="utf-8",
    )

    detailed_frame = pd.DataFrame(detailed_rows)
    if not detailed_frame.empty:
        detailed_frame = detailed_frame.sort_values("object_id").reset_index(drop=True)
    detailed_output_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_frame.to_csv(detailed_output_path, index=False)

    print(f"Saved {len(sorted_csv_rows)} predictions to {output_path}")
    print(f"Saved detailed predictions to {detailed_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
