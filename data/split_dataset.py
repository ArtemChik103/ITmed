"""Create deterministic manifest and split artifacts for Phase 3 training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

SOURCE_SPECS = {
    "Норма": {"label": 0, "class_name": "normal", "source_code": "normal_main"},
    "норма_отдельные снимки": {
        "label": 0,
        "class_name": "normal",
        "source_code": "normal_single",
    },
    "Патология": {"label": 1, "class_name": "pathology", "source_code": "pathology_main"},
    "патология_отдельные файлы": {
        "label": 1,
        "class_name": "pathology",
        "source_code": "pathology_single",
    },
}


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    group_id: str
    group_name: str
    label: int
    class_name: str
    source: str
    source_code: str
    relative_path: str
    path: str


def _derive_group_name(source: str, relative_to_source: Path, path: Path) -> str:
    if source in {"Норма", "Патология"}:
        return relative_to_source.parts[0] if relative_to_source.parts else path.stem
    return path.stem


def build_manifest(train_root: str | Path) -> pd.DataFrame:
    """Scan the raw dataset directory and build a manifest dataframe."""
    train_root = Path(train_root).resolve()
    records: list[ManifestRecord] = []

    for source, spec in SOURCE_SPECS.items():
        source_root = train_root / source
        if not source_root.exists():
            continue

        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".dcm":
                continue

            relative_to_source = path.relative_to(source_root)
            relative_path = path.relative_to(train_root).as_posix()
            group_name = _derive_group_name(source, relative_to_source, path)
            group_id = f"{spec['source_code']}::{group_name}"
            sample_id = f"{spec['source_code']}::{relative_path}"

            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    group_id=group_id,
                    group_name=group_name,
                    label=int(spec["label"]),
                    class_name=str(spec["class_name"]),
                    source=source,
                    source_code=str(spec["source_code"]),
                    relative_path=relative_path,
                    path=str(path),
                )
            )

    if not records:
        raise ValueError(f"No DICOM files found under '{train_root}'.")

    dataframe = pd.DataFrame(asdict(record) for record in records)
    return dataframe.sort_values(["group_id", "relative_path"]).reset_index(drop=True)


def _build_group_table(manifest: pd.DataFrame) -> pd.DataFrame:
    group_table = (
        manifest.groupby("group_id", as_index=False)
        .agg(label=("label", "first"), sample_count=("sample_id", "count"))
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    return group_table


def create_split(
    manifest: pd.DataFrame,
    *,
    seed: int,
    holdout_ratio: float = 0.15,
    n_folds: int = 5,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Create deterministic holdout and CV fold assignments without group leakage."""
    group_table = _build_group_table(manifest)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(group_table["group_id"], group_table["label"]))

    train_groups = group_table.iloc[train_idx].sort_values("group_id").reset_index(drop=True)
    holdout_groups = group_table.iloc[holdout_idx].sort_values("group_id").reset_index(drop=True)

    group_to_samples = (
        manifest.groupby("group_id")["sample_id"].apply(lambda items: sorted(items.tolist())).to_dict()
    )

    fold_assignments: dict[str, int] = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, object]] = []
    train_group_ids = train_groups["group_id"].tolist()
    train_group_labels = train_groups["label"].tolist()

    for fold, (_, val_idx) in enumerate(skf.split(train_group_ids, train_group_labels)):
        val_group_ids = sorted(train_groups.iloc[val_idx]["group_id"].tolist())
        for group_id in val_group_ids:
            fold_assignments[group_id] = fold

        train_group_ids_fold = sorted(set(train_group_ids).difference(val_group_ids))
        train_sample_ids = sorted(
            sample_id
            for group_id in train_group_ids_fold
            for sample_id in group_to_samples[group_id]
        )
        val_sample_ids = sorted(
            sample_id for group_id in val_group_ids for sample_id in group_to_samples[group_id]
        )

        folds.append(
            {
                "fold": fold,
                "train_group_ids": train_group_ids_fold,
                "val_group_ids": val_group_ids,
                "train_sample_ids": train_sample_ids,
                "val_sample_ids": val_sample_ids,
                "train_positive": int(manifest[manifest["sample_id"].isin(train_sample_ids)]["label"].sum()),
                "val_positive": int(manifest[manifest["sample_id"].isin(val_sample_ids)]["label"].sum()),
                "train_samples": len(train_sample_ids),
                "val_samples": len(val_sample_ids),
            }
        )

    holdout_group_ids = sorted(holdout_groups["group_id"].tolist())
    holdout_sample_ids = sorted(
        sample_id for group_id in holdout_group_ids for sample_id in group_to_samples[group_id]
    )

    assignments = manifest[["sample_id", "group_id", "label", "relative_path", "source"]].copy()
    assignments["fold"] = assignments["group_id"].map(fold_assignments).fillna(-1).astype(int)
    assignments["split"] = assignments["fold"].map(lambda fold: "holdout" if fold == -1 else "trainval")
    assignments = assignments.sort_values(["fold", "group_id", "relative_path"]).reset_index(drop=True)

    split_payload: dict[str, object] = {
        "version": "split_v1",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "n_folds": n_folds,
        "total_samples": int(len(manifest)),
        "total_groups": int(len(group_table)),
        "positive_samples": int(manifest["label"].sum()),
        "negative_samples": int((manifest["label"] == 0).sum()),
        "holdout_group_ids": holdout_group_ids,
        "holdout_sample_ids": holdout_sample_ids,
        "folds": folds,
    }
    return split_payload, assignments


def write_artifacts(
    manifest: pd.DataFrame,
    split_payload: dict[str, object],
    assignments: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Persist manifest, split JSON, and fold CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "train_manifest.csv"
    split_path = output_dir / "split_v1.json"
    folds_path = output_dir / "folds_v1.csv"

    manifest.to_csv(manifest_path, index=False)
    assignments.to_csv(folds_path, index=False)
    split_path.write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, split_path, folds_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Phase 3 dataset splits.")
    parser.add_argument(
        "--train-root",
        required=True,
        help="Path to the raw training dataset root containing normal/pathology folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest and split artifacts will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout and fold splits.")
    parser.add_argument("--holdout-ratio", type=float, default=0.15, help="Holdout ratio at group level.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.train_root)
    split_payload, assignments = create_split(
        manifest,
        seed=args.seed,
        holdout_ratio=args.holdout_ratio,
        n_folds=args.n_folds,
    )
    manifest_path, split_path, folds_path = write_artifacts(
        manifest,
        split_payload,
        assignments,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "split": str(split_path),
                "folds": str(folds_path),
                "samples": int(len(manifest)),
                "holdout_samples": int(len(split_payload["holdout_sample_ids"])),
                "folds_count": int(args.n_folds),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
