"""Unit tests for the pseudo-labeling utility."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.pseudo_labeling import discover_unlabeled_images, export_pseudo_manifest, generate_pseudo_labels


def test_discover_unlabeled_images(tmp_path: Path) -> None:
    # Create test directory with various file extensions
    (tmp_path / "img1.dcm").write_bytes(b"dummy")
    (tmp_path / "img2.png").write_bytes(b"dummy")
    (tmp_path / "img3.jpg").write_bytes(b"dummy")
    (tmp_path / "notes.txt").write_text("not an image")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "img4.jpeg").write_bytes(b"dummy")

    found = discover_unlabeled_images(tmp_path)
    assert len(found) == 4
    suffixes = {p.suffix.lower() for p in found}
    assert suffixes == {".dcm", ".png", ".jpg", ".jpeg"}


def test_export_pseudo_manifest(tmp_path: Path) -> None:
    records = [
        {
            "sample_id": "pseudo_001",
            "path": "/fake/path/img1.dcm",
            "relative_path": "img1.dcm",
            "label": 1,
            "class_name": "pathology",
            "group_id": "patient_01",
            "confidence": 0.95,
            "source": "pseudo_label",
            "left_hip_probability": 0.96,
            "right_hip_probability": 0.40,
            "symmetry_index": 0.44,
        }
    ]
    out_csv = tmp_path / "pseudo.csv"
    export_pseudo_manifest(records, out_csv)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.iloc[0]["sample_id"] == "pseudo_001"
    assert df.iloc[0]["label"] == 1
    assert df.iloc[0]["confidence"] == 0.95
