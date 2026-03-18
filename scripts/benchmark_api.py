"""Measure API latency on real DICOM inputs."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pandas as pd

from scripts.submission_common import collect_test_objects


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /api/v1/analyze on real DICOM files.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--dataset-root",
        default="../test_done",
        help="Root with DICOM objects; recursive DICOM files are benchmarked",
    )
    parser.add_argument("--plugin-type", default="hip_dysplasia", help="Plugin identifier")
    parser.add_argument("--mode", default="doctor", choices=["doctor", "education"], help="API mode")
    parser.add_argument("--sample-count", type=int, default=10, help="Maximum number of DICOMs to benchmark")
    parser.add_argument(
        "--output-json",
        default="submissions/benchmark_summary.json",
        help="Path to summary JSON",
    )
    parser.add_argument(
        "--output-csv",
        default="submissions/benchmark_requests.csv",
        help="Path to per-request CSV log",
    )
    return parser.parse_args(argv)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    objects = collect_test_objects(dataset_root)
    dicom_paths = [
        path
        for test_object in objects
        for path in test_object.dicom_paths
    ][: max(1, args.sample_count)]

    records: list[dict[str, object]] = []
    with httpx.Client(timeout=120.0) as client:
        for path in dicom_paths:
            started_at = time.perf_counter()
            response = client.post(
                f"{args.api_url.rstrip('/')}/api/v1/analyze",
                params={"plugin_type": args.plugin_type, "mode": args.mode},
                files={"file": (path.name, path.read_bytes(), "application/dicom")},
            )
            latency_ms = (time.perf_counter() - started_at) * 1000.0

            payload = {}
            runtime_model_loaded = None
            if response.headers.get("content-type", "").startswith("application/json"):
                payload = response.json()
                runtime_model_loaded = int(
                    round(float((payload.get("metrics") or {}).get("runtime_model_loaded", 0.0)))
                )

            records.append(
                {
                    "path": str(path),
                    "status_code": int(response.status_code),
                    "success": int(response.is_success),
                    "latency_ms": round(latency_ms, 3),
                    "runtime_model_loaded": runtime_model_loaded,
                    "message": payload.get("message") if payload else None,
                }
            )

    latencies = [float(record["latency_ms"]) for record in records if record["success"]]
    success_rate = (sum(int(record["success"]) for record in records) / len(records)) if records else 0.0
    by_mode: dict[str, dict[str, float | int]] = {}
    for runtime_value in (0, 1):
        mode_latencies = [
            float(record["latency_ms"])
            for record in records
            if record["success"] and record["runtime_model_loaded"] == runtime_value
        ]
        label = "runtime_model" if runtime_value == 1 else "fallback_mode"
        by_mode[label] = {
            "count": len(mode_latencies),
            "mean_latency_ms": round(statistics.mean(mode_latencies), 3) if mode_latencies else 0.0,
            "median_latency_ms": round(statistics.median(mode_latencies), 3) if mode_latencies else 0.0,
            "p95_latency_ms": round(_percentile(mode_latencies, 95), 3) if mode_latencies else 0.0,
            "max_latency_ms": round(max(mode_latencies), 3) if mode_latencies else 0.0,
        }

    summary = {
        "api_url": args.api_url,
        "plugin_type": args.plugin_type,
        "mode": args.mode,
        "dataset_root": str(dataset_root),
        "sample_count": len(records),
        "success_rate": round(success_rate, 4),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 95), 3) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "by_runtime_mode": by_mode,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(output_csv, index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved request log to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
