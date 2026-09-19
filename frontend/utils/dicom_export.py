"""Export annotated radiographic images as valid DICOM Secondary Capture (.dcm) files."""
from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def create_dicom_secondary_capture(
    image: Image.Image | np.ndarray,
    metadata: dict[str, Any] | None = None,
    clinical_result: dict[str, Any] | None = None,
) -> bytes:
    """Create a DICOM Secondary Capture Image Storage dataset and return serialized bytes.

    Args:
        image: PIL Image or numpy array (RGB or Grayscale).
        metadata: Optional dictionary with existing DICOM tags (PatientID, StudyInstanceUID, etc.).
        clinical_result: Optional clinical analysis result with metrics to embed in comments.

    Returns:
        bytes: Valid DICOM file bytes compatible with PACS / Orthanc / DICOM viewers.
    """
    metadata = metadata or {}
    clinical_result = clinical_result or {}

    # 1. Normalize image to 8-bit RGB array
    if isinstance(image, Image.Image):
        rgb_img = image.convert("RGB")
        pixel_array = np.array(rgb_img, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            norm = image.astype(np.float32)
            if norm.max() > norm.min():
                norm = (norm - norm.min()) / (norm.max() - norm.min()) * 255.0
            pixel_array = np.repeat(norm.astype(np.uint8)[..., None], 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                pixel_array = image[..., :3].astype(np.uint8)
            elif image.shape[2] == 1:
                pixel_array = np.repeat(image.astype(np.uint8), 3, axis=-1)
            else:
                pixel_array = image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image shape for DICOM SC: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    height, width, channels = pixel_array.shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels for RGB Secondary Capture, got {channels}")

    # 2. File Meta Information
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 0  # recalculated on save
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.9.7126.1"
    file_meta.ImplementationVersionName = "ITMED_2026_AI"

    # 3. Main Dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Identification & SOP Class
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8 for full Cyrillic support
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.Modality = "SC"
    ds.ConversionType = "WSD"  # Workstation
    ds.ImageType = ["DERIVED", "SECONDARY", "OTHER"]

    # Dates & Times
    now_date = time.strftime("%Y%m%d")
    now_time = time.strftime("%H%M%S")
    ds.ContentDate = now_date
    ds.ContentTime = now_time
    ds.InstanceCreationDate = now_date
    ds.InstanceCreationTime = now_time
    ds.StudyDate = str(metadata.get("study_date") or now_date).replace("-", "")
    ds.StudyTime = str(metadata.get("study_time") or now_time).replace(":", "")

    # Patient & Study Information
    ds.PatientID = str(metadata.get("patient_id") or "PAT-DDH-2026")
    ds.PatientName = str(metadata.get("patient_name") or "PEDIATRIC^PATIENT")
    ds.PatientBirthDate = str(metadata.get("patient_birth_date") or "20250101").replace("-", "")
    ds.PatientSex = str(metadata.get("patient_sex") or "O")

    # Hierarchy: preserve StudyInstanceUID to link with original study in PACS
    ds.StudyInstanceUID = str(metadata.get("study_instance_uid") or generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = str(metadata.get("study_id") or "1")
    ds.SeriesNumber = int(metadata.get("series_number") or 100)
    ds.InstanceNumber = 1

    # Descriptions & Equipment
    ds.StudyDescription = str(metadata.get("study_description") or "Рентгенография таза обзорная")
    ds.SeriesDescription = "IT+Med AI Protocol & Overlays (Secondary Capture)"
    ds.Manufacturer = "IT+Med Pediatric Radiomics"
    ds.ManufacturerModelName = "HipDysplasia-Ensemble-v1"
    ds.SoftwareVersions = "2026.1.0"
    ds.InstitutionName = "Детская ортопедическая клиника ИТ+Мед"
    ds.InstitutionalDepartmentName = "Отделение лучевой диагностики"

    # Image Plane & Pixel Attributes
    ds.Rows = height
    ds.Columns = width
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0  # interleaved RGBRGBRGB...
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0  # unsigned integer

    # Structured Comments & Findings
    findings: list[str] = ["IT+Med AI Clinical Protocol:"]
    if clinical_result:
        metrics = clinical_result.get("metrics") or {}
        is_pathology = bool(clinical_result.get("disease_detected", False))
        confidence_val = float(clinical_result.get("confidence", metrics.get("model_probability", clinical_result.get("disease_probability", 0.0))))
        findings.append(f"Status: {'Патология (ДТБС)' if is_pathology else 'Норма'}")
        findings.append(f"Confidence: {confidence_val * 100:.1f}%")

        # Bilateral Tonnis
        r_tonnis = int(round(float(metrics.get("right_tonnis_grade", metrics.get("tonnis_grade", clinical_result.get("right_tonnis_grade", clinical_result.get("tonnis_grade", 0)))))))
        l_tonnis = int(round(float(metrics.get("left_tonnis_grade", clinical_result.get("left_tonnis_grade", 0)))))
        findings.append(f"Tonnis Grade: Right={r_tonnis}, Left={l_tonnis}")

        # Bilateral Reimers
        r_reimers = float(metrics.get("right_reimers_index_pct", metrics.get("reimers_index_pct", clinical_result.get("right_reimers_index_pct", clinical_result.get("reimers_index_pct", 0.0)))))
        l_reimers = float(metrics.get("left_reimers_index_pct", clinical_result.get("left_reimers_index_pct", 0.0)))
        findings.append(f"Reimers Index: Right={r_reimers:.1f}%, Left={l_reimers:.1f}%")

        # Bilateral Acetabular Angle
        r_angle = float(metrics.get("right_acetabular_angle_deg", metrics.get("acetabular_angle_deg", clinical_result.get("right_acetabular_angle_deg", clinical_result.get("acetabular_angle_deg", 0.0)))))
        l_angle = float(metrics.get("left_acetabular_angle_deg", clinical_result.get("left_acetabular_angle_deg", 0.0)))
        findings.append(f"Acetabular Angle: Right={r_angle:.1f} deg, Left={l_angle:.1f} deg")

        # Bilateral Stress
        r_stress = float(metrics.get("right_peak_stress_mpa", metrics.get("peak_contact_stress_mpa", clinical_result.get("right_peak_stress_mpa", clinical_result.get("peak_stress_mpa", 0.0)))))
        l_stress = float(metrics.get("left_peak_stress_mpa", clinical_result.get("left_peak_stress_mpa", 0.0)))
        findings.append(f"Peak Stress: Right={r_stress:.2f} MPa, Left={l_stress:.2f} MPa")

    ds.ImageComments = " | ".join(findings)

    # Pixel Data
    ds.PixelData = pixel_array.tobytes()

    # Serialize to bytes buffer
    buffer = io.BytesIO()
    ds.save_as(buffer, write_like_original=False)
    return buffer.getvalue()
