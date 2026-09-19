"""Unit tests for batch processing component."""
from __future__ import annotations

import io
import zipfile

from frontend.components.batch_processing import extract_files_from_zip


def test_extract_files_from_zip():
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("test1.dcm", b"DICOM_DATA_1")
        zf.writestr("subfolder/test2.png", b"PNG_DATA_2")
        zf.writestr("ignore.txt", b"TEXT_DATA")
        zf.writestr("__MACOSX/._test1.dcm", b"MAC_DATA")
        zf.writestr("empty_dir/", b"")

    zip_bytes = zip_buf.getvalue()
    extracted = extract_files_from_zip(zip_bytes)

    assert len(extracted) == 2
    filenames = [name for name, _ in extracted]
    assert "test1.dcm" in filenames
    assert "test2.png" in filenames
    assert "ignore.txt" not in filenames
