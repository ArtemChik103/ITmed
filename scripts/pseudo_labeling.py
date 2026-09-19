"""Generate high-confidence pseudo-labels from unlabeled datasets for semi-supervised training."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.image_loader import load_medical_image
from plugins.hip_dysplasia.model import HipDysplasiaEnsemble, resolve_model_manifest_path


def discover_unlabeled_images(unlabeled_root: Path) -> list[Path]:
    """Scan directory recursively for supported medical image files."""
    valid_extensions = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp"}
    images: list[Path] = []
    if not unlabeled_root.exists():
        return images

    for path in sorted(unlabeled_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            images.append(path)
    return images


def generate_pseudo_labels(
    image_paths: list[Path],
    *,
    ensemble: HipDysplasiaEnsemble,
    high_threshold: float = 0.90,
    low_threshold: float = 0.10,
    use_tta: bool = True,
    use_dual_hip: bool = True,
) -> list[dict[str, Any]]:
    """Predict probabilities and filter high-confidence normal / pathological cases."""
    preprocessor = ensemble.build_preprocessor()
    pseudo_records: list[dict[str, Any]] = []

    for idx, path in enumerate(image_paths):
        try:
            raw_image, metadata = load_medical_image(path)
            processed = preprocessor.preprocess(raw_image, metadata)
            prediction = ensemble.predict(processed, tta=use_tta, dual_hip=use_dual_hip)
            prob = float(prediction.probability)

            if prob >= high_threshold:
                label = 1
                class_name = "pathology"
            elif prob <= low_threshold:
                label = 0
                class_name = "normal"
            else:
                continue

            sample_id = f"pseudo_{idx:05d}_{path.stem}"
            # Extract group id from parent folder name or stem
            group_id = path.parent.name if path.parent.name != "" else path.stem

            pseudo_records.append({
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "relative_path": str(path.name),
                "label": label,
                "class_name": class_name,
                "group_id": group_id,
                "confidence": prob,
                "source": "pseudo_label",
                "left_hip_probability": prediction.left_hip_probability,
                "right_hip_probability": prediction.right_hip_probability,
                "symmetry_index": prediction.symmetry_index,
            })
        except Exception as err:
            print(f"Warning: Failed to process {path}: {err}", file=sys.stderr)

    return pseudo_records


def export_pseudo_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save pseudo-labeled records to a CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        df = pd.DataFrame(columns=[
            "sample_id", "path", "relative_path", "label", "class_name",
            "group_id", "confidence", "source", "left_hip_probability",
            "right_hip_probability", "symmetry_index"
        ])
    else:
        df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Exported {len(records)} pseudo-labeled samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for semi-supervised training.")
    parser.add_argument("--unlabeled-root", default="../test_done", help="Path to unlabeled images directory.")
    parser.add_argument("--manifest", default=None, help="Path to model manifest JSON.")
    parser.add_argument("--output", default="data/manifests/train_pseudo_manifest.csv", help="Output CSV path.")
    parser.add_argument("--high-threshold", type=float, default=0.90, help="Confidence threshold for pathology.")
    parser.add_argument("--low-threshold", type=float, default=0.10, help="Confidence threshold for normal.")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation.")
    parser.add_argument("--no-dual-hip", action="store_true", help="Disable dual-hip joint analysis.")

    args = parser.parse_args()
    manifest_path = resolve_model_manifest_path(args.manifest)
    if manifest_path is None:
        raise FileNotFoundError("Could not resolve model manifest.")

    ensemble = HipDysplasiaEnsemble(manifest_path)
    unlabeled_root = Path(args.unlabeled_root)
    images = discover_unlabeled_images(unlabeled_root)
    print(f"Discovered {len(images)} unlabeled candidate images in {unlabeled_root}")

    records = generate_pseudo_labels(
        images,
        ensemble=ensemble,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        use_tta=not args.no_tta,
        use_dual_hip=not args.no_dual_hip,
    )

    export_pseudo_manifest(records, Path(args.output))


if __name__ == "__main__":
    main()
