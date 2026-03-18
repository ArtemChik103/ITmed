"""Shared helpers for synthetic DICOM-based tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def build_test_dicom(
    path: Path,
    *,
    pixel_array: np.ndarray | None = None,
    pixel_spacing: list[float] | None = None,
    imager_pixel_spacing: list[float] | None = None,
    shared_pixel_spacing: list[float] | None = None,
    per_frame_pixel_spacing: list[float] | None = None,
    pixel_aspect_ratio: list[int] | None = None,
    patient_id: str | None = None,
    study_instance_uid: str | None = None,
    study_date: str | None = None,
    series_date: str | None = None,
    acquisition_date: str | None = None,
    content_date: str | None = None,
    modality: str = "DX",
    photometric_interpretation: str = "MONOCHROME2",
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
) -> Path:
    array = pixel_array if pixel_array is not None else np.array([[1, 2], [3, 4]], dtype=np.uint16)

    path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_instance_uid or generate_uid()
    ds.Modality = modality
    ds.Rows, ds.Columns = array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric_interpretation
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = array.astype(np.uint16).tobytes()

    if patient_id is not None:
        ds.PatientID = patient_id
    if study_date is not None:
        ds.StudyDate = study_date
    if series_date is not None:
        ds.SeriesDate = series_date
    if acquisition_date is not None:
        ds.AcquisitionDate = acquisition_date
    if content_date is not None:
        ds.ContentDate = content_date
    if pixel_spacing is not None:
        ds.PixelSpacing = pixel_spacing
    if imager_pixel_spacing is not None:
        ds.ImagerPixelSpacing = imager_pixel_spacing
    if pixel_aspect_ratio is not None:
        ds.PixelAspectRatio = pixel_aspect_ratio
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept

    if shared_pixel_spacing is not None:
        pixel_measures = Dataset()
        pixel_measures.PixelSpacing = shared_pixel_spacing
        shared_group = Dataset()
        shared_group.PixelMeasuresSequence = Sequence([pixel_measures])
        ds.SharedFunctionalGroupsSequence = Sequence([shared_group])

    if per_frame_pixel_spacing is not None:
        pixel_measures = Dataset()
        pixel_measures.PixelSpacing = per_frame_pixel_spacing
        frame_group = Dataset()
        frame_group.PixelMeasuresSequence = Sequence([pixel_measures])
        ds.PerFrameFunctionalGroupsSequence = Sequence([frame_group])

    ds.save_as(str(path), write_like_original=False)
    return path


def build_test_raster(
    path: Path,
    *,
    pixel_array: np.ndarray | None = None,
) -> Path:
    array = pixel_array if pixel_array is not None else np.array([[16, 64], [128, 240]], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array)
    image.save(path)
    return path
