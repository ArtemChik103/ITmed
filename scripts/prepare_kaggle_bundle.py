"""Create a minimal Phase 3 bundle for notebook-based training runs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_FILES = [
    Path("README.md"),
    Path("requirements-train-colab.txt"),
    Path("requirements-train.txt"),
    Path("requirements-train-kaggle.txt"),
    Path("docs/colab_checklist.md"),
    Path("docs/colab_notebook_cells.md"),
    Path("docs/colab_phase3.md"),
    Path("docs/external_ddh_datasets.md"),
    Path("docs/phase3_training.md"),
    Path("docs/kaggle_checklist.md"),
    Path("docs/kaggle_notebook_cells.md"),
    Path("docs/kaggle_phase3.md"),
]

INCLUDE_DIRS = [
    Path("core"),
    Path("data"),
    Path("models"),
    Path("plugins"),
    Path("scripts"),
    Path("train"),
]

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "analysis",
    "frontend",
    "presentation",
    "submissions",
    "tests",
    "weights",
    "checkpoints",
}

EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILE_NAMES = {".DS_Store"}


def should_skip(path: Path, *, include_pretrained: bool) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if not include_pretrained and Path("models") in path.parents and path.name == "pretrained":
        return True
    if not include_pretrained and "pretrained" in parts:
        return True
    if path.suffix in EXCLUDED_FILE_SUFFIXES or path.name in EXCLUDED_FILE_NAMES:
        return True
    return False


def copy_tree(source: Path, destination: Path, *, include_pretrained: bool) -> list[Path]:
    copied: list[Path] = []
    for item in source.rglob("*"):
        relative_path = item.relative_to(PROJECT_ROOT)
        if should_skip(relative_path, include_pretrained=include_pretrained):
            continue
        target_path = destination / relative_path
        if item.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target_path)
        copied.append(relative_path)
    return copied


def build_bundle(output_dir: Path, *, include_pretrained: bool) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []

    for relative_file in INCLUDE_FILES:
        source = PROJECT_ROOT / relative_file
        if not source.exists():
            continue
        target = output_dir / relative_file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_files.append(relative_file)

    for relative_dir in INCLUDE_DIRS:
        source = PROJECT_ROOT / relative_dir
        if not source.exists():
            continue
        copied_files.extend(copy_tree(source, output_dir, include_pretrained=include_pretrained))

    summary = {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "include_pretrained": include_pretrained,
        "file_count": len(copied_files),
        "files": [str(path).replace("\\", "/") for path in sorted(set(copied_files))],
    }
    summary_path = output_dir / "bundle_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a minimal Phase 3 bundle for notebook workflows.")
    parser.add_argument(
        "--output-dir",
        default="build/kaggle_phase3_bundle",
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
