"""
GradCAM (Gradient-weighted Class Activation Mapping)
------------------------------------------------------
Produces spatial heatmaps showing which regions of each ultrasound frame
drove the CNN's PAS prediction.

Implementation matches CNN_model_9.0_mdj.ipynb:
  - Target layer: 'top_conv' (EfficientNetB0 last conv layer)
  - JET colormap, alpha=0.4 blend
  - Per-frame, independent heatmaps
"""
from __future__ import annotations

import base64
import io
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

_GRADCAM_LAYER = "top_conv"
_ALPHA = 0.4
_MAX_FRAMES = 200
_JPEG_QUALITY = 75


def build_grad_model(cnn_model, layer_name: str = _GRADCAM_LAYER):
    """Create a sub-model that exposes [last_conv_output, prediction]."""
    import keras
    last_conv = cnn_model.get_layer(layer_name)
    return keras.Model(
        inputs=cnn_model.inputs,
        outputs=[last_conv.output, cnn_model.output],
    )


def compute_heatmap(grad_model, img_tensor: np.ndarray) -> np.ndarray:
    """
    Compute GradCAM heatmap for a single frame.

    img_tensor : float32 (1, 256, 256, 3) — crop/resize'd, NOT preprocess_input'd
                 (model handles preprocessing internally via Rescaling layers)
    Returns    : float32 (h, w) heatmap in [0, 1]
    """
    x = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv_out, preds = grad_model(x)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_frame(
    frame_float32: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = _ALPHA,
) -> np.ndarray:
    """
    Alpha-blend a JET heatmap over the original frame.

    frame_float32 : float32 (256, 256, 3) in [0, 255] range (from preprocess_frames)
    heatmap       : float32 (h, w) in [0, 1]
    Returns       : uint8 (256, 256, 3) RGB overlay
    """
    # Convert frame to uint8 RGB
    frame_u8 = np.clip(frame_float32, 0, 255).astype(np.uint8)
    if frame_u8.ndim == 2:
        frame_u8 = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)

    H, W = frame_u8.shape[:2]

    # Resize heatmap to match frame
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8).resize((W, H), Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_pil).astype(np.float32) / 255.0

    # Apply JET colormap
    jet = cm.get_cmap("jet")
    colored = (jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Alpha blend
    overlay = (frame_u8 * (1.0 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)
    return overlay


def frame_to_b64_jpeg(rgb: np.ndarray, quality: int = _JPEG_QUALITY) -> str:
    """Encode a uint8 RGB array to a base64 JPEG string."""
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_to_b64_jpeg(
    heatmap: np.ndarray, H: int, W: int, quality: int = _JPEG_QUALITY
) -> str:
    """
    Apply JET colormap to heatmap and encode as base64 JPEG (no frame blend).

    heatmap : float32 (h, w) in [0, 1] from compute_heatmap()
    Returns : base64 JPEG string of the colored heatmap
    """
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8).resize((W, H), Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_pil).astype(np.float32) / 255.0
    jet = cm.get_cmap("jet")
    colored = (jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    return frame_to_b64_jpeg(colored, quality)


def compute_gradcam_all(
    cnn_model,
    preprocessed_frames: np.ndarray,
    max_frames: int = _MAX_FRAMES,
) -> list[dict]:
    """
    Compute GradCAM heatmaps for up to max_frames, sampled uniformly.

    preprocessed_frames : float32 (N, 256, 256, 3) from dicom_processor.preprocess_frames()
    Returns             : list of {"frame_idx": int, "raw_b64": str, "heatmap_b64": str}
                          so the frontend can composite at variable opacity.
    """
    n = len(preprocessed_frames)

    # Uniform sampling if scan exceeds max_frames
    if n <= max_frames:
        indices = list(range(n))
    else:
        indices = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
        indices = sorted(set(indices))

    grad_model = build_grad_model(cnn_model)
    results = []

    for idx in indices:
        frame_arr = preprocessed_frames[idx]          # (256, 256, 3) float32
        frame_u8  = np.clip(frame_arr, 0, 255).astype(np.uint8)
        if frame_u8.ndim == 2:
            frame_u8 = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
        H, W = frame_u8.shape[:2]

        raw_b64 = frame_to_b64_jpeg(frame_u8)

        frame_batch = preprocessed_frames[idx : idx + 1]  # (1, 256, 256, 3)
        heatmap = compute_heatmap(grad_model, frame_batch)
        heatmap_b64 = heatmap_to_b64_jpeg(heatmap, H, W)

        results.append({
            "frame_idx": int(idx),
            "raw_b64":    raw_b64,
            "heatmap_b64": heatmap_b64,
        })

    return results


def compute_gradcam_generator(
    cnn_model,
    preprocessed_frames: np.ndarray,
    max_frames: int = _MAX_FRAMES,
):
    """
    Generator version of compute_gradcam_all — yields one frame at a time so
    the caller can stream progress updates.

    Each yield is a dict with keys:
        frame_idx, raw_b64, heatmap_b64, progress (1-based), total
    """
    n = len(preprocessed_frames)

    if n <= max_frames:
        indices = list(range(n))
    else:
        indices = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
        indices = sorted(set(indices))

    grad_model = build_grad_model(cnn_model)
    total = len(indices)

    for progress, idx in enumerate(indices, start=1):
        frame_arr = preprocessed_frames[idx]
        frame_u8  = np.clip(frame_arr, 0, 255).astype(np.uint8)
        if frame_u8.ndim == 2:
            frame_u8 = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
        H, W = frame_u8.shape[:2]

        raw_b64     = frame_to_b64_jpeg(frame_u8)
        frame_batch = preprocessed_frames[idx : idx + 1]
        heatmap     = compute_heatmap(grad_model, frame_batch)
        heatmap_b64 = heatmap_to_b64_jpeg(heatmap, H, W)

        yield {
            "frame_idx":   int(idx),
            "raw_b64":     raw_b64,
            "heatmap_b64": heatmap_b64,
            "progress":    progress,
            "total":       total,
        }
