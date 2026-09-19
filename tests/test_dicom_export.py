"""Unit tests for DICOM Secondary Capture export."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image
import pydicom

from frontend.utils.dicom_export import create_dicom_secondary_capture


def test_create_dicom_secondary_capture():
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    metadata = {
        "patient_id": "TEST-PAT-001",
        "patient_name": "IVANOV^IVAN",
        "study_instance_uid": "1.2.3.4.5.6.7890",
        "study_date": "20260315",
    }
    clinical_result = {
        "disease_detected": True,
        "disease_probability": 0.89,
        "right_tonnis_grade": 2,
        "left_tonnis_grade": 1,
        "right_reimers_index_pct": 34.5,
        "left_reimers_index_pct": 18.2,
        "right_acetabular_angle_deg": 31.0,
        "left_acetabular_angle_deg": 22.5,
        "right_peak_stress_mpa": 3.8,
        "left_peak_stress_mpa": 2.1,
    }

    dcm_bytes = create_dicom_secondary_capture(img, metadata=metadata, clinical_result=clinical_result)
    assert len(dcm_bytes) > 0

    # Parse back with pydicom
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    assert ds.PatientID == "TEST-PAT-001"
    assert ds.StudyInstanceUID == "1.2.3.4.5.6.7890"
    assert ds.Modality == "SC"
    assert ds.Rows == 256
    assert ds.Columns == 256
    assert ds.SamplesPerPixel == 3
    assert ds.PhotometricInterpretation == "RGB"
    assert "Tonnis Grade: Right=2, Left=1" in ds.ImageComments
    assert "Reimers Index: Right=34.5%, Left=18.2%" in ds.ImageComments
    assert ds.pixel_array.shape == (256, 256, 3)
