"""
DICOM → preprocessed frames
----------------------------
Converts a multiframe (or single-frame) DICOM to a list of float32 numpy
arrays ready for EfficientNetB0 preprocessing.

Processing steps (matches training pipeline in CNN_remade_07-22-25.ipynb and
test_pas_predict.py):
  1. pydicom.dcmread → pixel_array
  2. Apply VOI-LUT (window/level)
  3. Invert MONOCHROME1
  4. Normalize to float32
  5. Crop: top=65, left=66, right=150 pixels removed
  6. Resize to (256, 256) via PIL BILINEAR
  7. Ensure 3-channel RGB
"""
from __future__ import annotations

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

# Crop constants (pixels removed from each edge during training)
_CROP_TOP = 65
_CROP_LEFT = 66
_CROP_RIGHT = 150   # pixels removed from the RIGHT side
_TARGET = (256, 256)


def _apply_voi_and_photometric(frame: np.ndarray, ds) -> np.ndarray:
    """Apply VOI-LUT windowing and correct photometric inversion."""
    try:
        frame = apply_voi_lut(frame, ds)
    except Exception:
        frame = frame.astype(np.float32)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    frame = frame.astype(np.float32)
    if photometric == "MONOCHROME1":
        fmin, fmax = np.nanmin(frame), np.nanmax(frame)
        frame = fmax - (frame - fmin)
    return frame.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize float array to uint8 [0, 255]."""
    img = img.astype(np.float32)
    imin, imax = np.nanmin(img), np.nanmax(img)
    if imax <= imin:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - imin) / (imax - imin)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def _crop(img: np.ndarray) -> np.ndarray:
    """Remove border pixels matching the training crop."""
    h, w = img.shape[:2]
    top = _CROP_TOP
    left = _CROP_LEFT
    bottom = h          # no bottom crop
    right = w - _CROP_RIGHT
    # Guard against degenerate images
    if bottom <= top or right <= left:
        return img
    return img[top:bottom, left:right]


def _resize_to_tensor(img: np.ndarray, size: tuple = _TARGET) -> np.ndarray:
    """Crop → uint8 → PIL resize → 3-channel float32."""
    img = _crop(img)
    u8 = _to_uint8(img)
    pil = Image.fromarray(u8)
    # PIL resize takes (width, height); size is (H, W)
    pil = pil.resize((size[1], size[0]), resample=Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def load_dicom_frames(dicom_path: str) -> tuple[list[np.ndarray], object]:
    """
    Load a DICOM and return:
      - frames: list of float32 arrays shaped (H, W) before crop/resize
      - ds: the pydicom Dataset (for metadata)
    """
    ds = pydicom.dcmread(dicom_path)
    pix = ds.pixel_array
    if pix.ndim == 2:
        pix = pix[np.newaxis, ...]
    frames = [_apply_voi_and_photometric(pix[i], ds) for i in range(pix.shape[0])]
    return frames, ds


def preprocess_frames(frames: list[np.ndarray]) -> np.ndarray:
    """
    Apply crop + resize to a list of raw frames.
    Returns float32 array shaped (N, 256, 256, 3) — NOT yet preprocess_input'd.
    EfficientNet preprocess_input is applied in cnn_predictor.py.
    """
    tensors = [_resize_to_tensor(f) for f in frames]
    return np.stack(tensors, axis=0)


def dicom_info(ds) -> dict:
    """Extract lightweight metadata from a pydicom Dataset."""
    declared = int(getattr(ds, "NumberOfFrames", 1))
    return {
        "declared_frames": declared,
        "photometric": getattr(ds, "PhotometricInterpretation", "UNKNOWN"),
        "modality": getattr(ds, "Modality", "UNKNOWN"),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
    }
