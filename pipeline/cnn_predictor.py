"""
EfficientNetB0 CNN inference
-----------------------------
Loads best_model_copy.h5 once at startup and runs per-frame inference.

Key detail: the model was trained with
  tensorflow.keras.applications.efficientnet.preprocess_input
not a simple /255 normalization.  preprocess_input is applied here, after the
crop/resize step in dicom_processor.py.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

try:
    import keras  # standalone Keras v3 fallback
    _HAS_STANDALONE_KERAS = True
except Exception:
    _HAS_STANDALONE_KERAS = False

_BATCH_SIZE = 32


def load_cnn_model(model_path: str):
    """
    Robust loader for legacy .h5 EfficientNetB0 weights.
    Tries tf.keras first, then standalone keras if available.
    """
    last_err: Exception | None = None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        last_err = e

    if _HAS_STANDALONE_KERAS:
        try:
            from keras.saving import load_model as kload
            return kload(model_path, compile=False, safe_mode=False)
        except Exception as e2:
            last_err = e2

    raise last_err  # type: ignore[misc]


def predict_frames(preprocessed_batch: np.ndarray, model) -> np.ndarray:
    """
    Run EfficientNet inference on a (N, 256, 256, 3) float32 array.

    preprocessed_batch must already be crop/resize'd (from dicom_processor).
    preprocess_input is applied here before model.predict.

    Returns 1-D float32 array of per-frame PAS probabilities.
    """
    batch = effnet_preprocess(preprocessed_batch.copy())
    probs = model.predict(batch, batch_size=_BATCH_SIZE, verbose=0).ravel()
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
