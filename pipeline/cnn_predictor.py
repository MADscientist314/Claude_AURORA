"""
EfficientNetB0 CNN inference
-----------------------------
Loads best_model_copy.h5 once at startup and runs per-frame inference.

The model was saved with Keras 3.9.0 (backend: tensorflow).
Loading requires the standalone `keras>=3.0` package with KERAS_BACKEND=tensorflow.

Key detail: the model includes internal Rescaling/Normalization layers so
preprocess_input does NOT need to be applied manually — the model handles
its own preprocessing internally.
"""
from __future__ import annotations

import os

# Must be set before any keras/tensorflow import
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

_BATCH_SIZE = 32


def load_cnn_model(model_path: str):
    """
    Load an EfficientNetB0 .h5 model saved with Keras 3.x.

    Uses keras.saving.load_model (standalone Keras 3) which is the only
    loader compatible with models saved under Keras >= 3.0.
    """
    import keras
    return keras.saving.load_model(model_path, compile=False)


def predict_frames(preprocessed_batch: np.ndarray, model) -> np.ndarray:
    """
    Run inference on a (N, 256, 256, 3) float32 array.

    The batch should be crop/resize'd (from dicom_processor) but NOT
    preprocessed with efficientnet.preprocess_input — the model's own
    internal Rescaling layers handle that.

    Returns a 1-D float32 array of per-frame PAS probabilities.
    """
    probs = model.predict(preprocessed_batch, batch_size=_BATCH_SIZE, verbose=0).ravel()
    return probs.astype(np.float32)


def aggregate_probs(probs: np.ndarray, method: str = "mean") -> float:
    """Aggregate per-frame probabilities to a single patient-level score."""
    if probs.size == 0:
        return float("nan")
    method = method.lower()
    if method == "mean":
        return float(np.nanmean(probs))
    if method == "max":
        return float(np.nanmax(probs))
    if method == "median":
        return float(np.nanmedian(probs))
    raise ValueError(f"Unknown aggregation method: {method!r}")
