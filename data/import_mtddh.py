"""Build an MTDDH training manifest from audit outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text.lower() != "nan":
            return text
    return None


def _build_group_token(row: pd.Series) -> str:
    return _first_non_empty(
        row.get("annotation_group_id"),
        row.get("case_folder"),
        row.get("parent_folder"),
        row.get("file_stem"),
        row.get("original_id"),
    ) or Path(str(row["relative_path"])).stem


def build_mtddh_manifest(
    *,
    dataset_root: str | Path,
    audit_dir: str | Path,
    output_manifest: str | Path,
    label_policy: str = "strict",
) -> pd.DataFrame:
    """Create a project-compatible MTDDH manifest from audit summaries."""
    if label_policy != "strict":
        raise ValueError("Only --label-policy strict is currently supported.")

    dataset_root = Path(dataset_root).resolve()
    audit_dir = Path(audit_dir).resolve()
    output_manifest = Path(output_manifest).resolve()

    file_index = pd.read_csv(audit_dir / "file_index.csv")
    eligible = file_index[
        (file_index["is_image"].fillna(0).astype(int) == 1)
        & (file_index["eligible_for_training"].fillna(0).astype(int) == 1)
        & file_index["label"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError(f"No training-eligible MTDDH rows were found in '{audit_dir}'.")

    eligible["relative_path"] = eligible["relative_path"].astype(str)
    eligible["path"] = eligible["absolute_path"].astype(str)
    eligible["label"] = eligible["label"].astype(int)
    eligible["class_name"] = eligible["label"].map({0: "normal", 1: "pathology"})
    eligible["group_token"] = eligible.apply(_build_group_token, axis=1)
    eligible["sample_id"] = eligible["relative_path"].map(lambda value: f"mtddh::{value}")
    eligible["group_id"] = eligible["group_token"].map(lambda value: f"mtddh::{value}")
    eligible["group_name"] = eligible["group_token"].astype(str)
    eligible["source"] = "MTDDH"
    eligible["source_code"] = "mtddh"
    if "dataset_name" in eligible.columns:
        eligible["dataset_name"] = eligible["dataset_name"].fillna("MTDDH")
    else:
        eligible["dataset_name"] = "MTDDH"
    eligible["is_external"] = 1
    eligible["path"] = eligible["path"].map(lambda value: str(Path(value).resolve()))

    manifest_columns = [
        "sample_id",
        "group_id",
        "group_name",
        "label",
        "class_name",
        "source",
        "source_code",
        "relative_path",
        "path",
        "dataset_name",
        "file_type",
        "view",
        "age_months",
        "is_external",
        "label_confidence",
        "label_source",
        "original_id",
    ]
    manifest = eligible[manifest_columns].sort_values(["group_id", "relative_path"]).reset_index(drop=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "audit_dir": str(audit_dir),
        "output_manifest": str(output_manifest),
        "label_policy": label_policy,
        "rows": int(len(manifest)),
        "normal_rows": int((manifest["label"] == 0).sum()),
        "pathology_rows": int((manifest["label"] == 1).sum()),
    }
    (output_manifest.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MTDDH audit outputs into a training manifest.")
    parser.add_argument("--dataset-root", required=True, help="Path to the unpacked MTDDH dataset root.")
    parser.add_argument(
        "--audit-dir",
        default="analysis/mtddh_audit",
        help="Directory produced by analysis/mtddh_audit.py",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Where to write data/manifests/mtddh_manifest.csv",
    )
    parser.add_argument(
        "--label-policy",
        default="strict",
        choices=["strict"],
        help="Import policy for labels resolved by the audit pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_mtddh_manifest(
        dataset_root=args.dataset_root,
        audit_dir=args.audit_dir,
        output_manifest=args.output_manifest,
        label_policy=args.label_policy,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(manifest)),
                "output_manifest": str(Path(args.output_manifest).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
