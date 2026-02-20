"""
AURORA FastAPI Web Application
-------------------------------
Serves the clinical dashboard and handles DICOM prediction requests.

Startup:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /          → static/index.html
    POST /predict   → multipart form → JSON prediction result
"""
from __future__ import annotations

import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.dicom_processor import dicom_info, load_dicom_frames, preprocess_frames
from pipeline.cnn_predictor import aggregate_probs, load_cnn_model, predict_frames
from pipeline.ensemble_predictor import load_ensemble, predict_ensemble

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CNN_MODEL_PATH = MODELS_DIR / "best_model_copy.h5"
ENSEMBLE_PKL_PATH = MODELS_DIR / "ensemble.pkl"
STATIC_DIR = BASE_DIR / "static"

# Global model state (loaded once at startup)
_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan — load models once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models…")
    if CNN_MODEL_PATH.exists():
        _state["cnn_model"] = load_cnn_model(str(CNN_MODEL_PATH))
        print(f"  CNN loaded: {CNN_MODEL_PATH.name}")
    else:
        print(f"  WARNING: CNN model not found at {CNN_MODEL_PATH}")
        _state["cnn_model"] = None

    if ENSEMBLE_PKL_PATH.exists():
        _state["ensemble_bundle"] = load_ensemble(str(ENSEMBLE_PKL_PATH))
        print(f"  Ensemble loaded: {ENSEMBLE_PKL_PATH.name}")
    else:
        print(f"  WARNING: Ensemble pkl not found at {ENSEMBLE_PKL_PATH}")
        _state["ensemble_bundle"] = None

    print("Models ready.")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AURORA — PAS Detection",
    description="CNN + ensemble model for Placenta Accreta Spectrum detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (CSS, JS) but NOT index.html via mount
# (index.html is served explicitly via GET / below)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cnn_loaded": _state.get("cnn_model") is not None,
        "ensemble_loaded": _state.get("ensemble_bundle") is not None,
    }


@app.post("/predict")
async def predict(
    dicom_file: UploadFile = File(...),
    num_prior_cs: int = Form(...),
    previa: str = Form(...),
    patient_id: str = Form(""),
    agg_method: str = Form("mean"),
):
    """
    Accept a DICOM upload + clinical inputs and return prediction JSON.

    Form fields:
        dicom_file   : .dcm file (≤ 50 MB)
        num_prior_cs : number of prior cesarean sections (int)
        previa       : 'yes' or 'no'
        patient_id   : optional patient identifier string
        agg_method   : 'mean' | 'max' | 'median'  (default: 'mean')
    """
    cnn_model = _state.get("cnn_model")
    ensemble_bundle = _state.get("ensemble_bundle")

    if cnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model not loaded. Copy best_model_copy.h5 to the models/ directory.",
        )

    # Validate upload size (50 MB)
    MAX_BYTES = 50 * 1024 * 1024
    contents = await dicom_file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    tmp_path: str | None = None
    try:
        # Write to temp file so pydicom can read it
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # ---- DICOM processing ----
        frames, ds = load_dicom_frames(tmp_path)
        info = dicom_info(ds)
        num_frames = len(frames)
        batch = preprocess_frames(frames)

        # ---- CNN inference ----
        per_frame_probs = predict_frames(batch, cnn_model)
        cnn_prob = aggregate_probs(per_frame_probs, method=agg_method)
        cnn_pred = int(cnn_prob >= 0.5)

        # ---- Ensemble inference ----
        ensemble_result: dict = {}
        if ensemble_bundle is not None:
            ensemble_result = predict_ensemble(
                bundle=ensemble_bundle,
                number_prior_cs=num_prior_cs,
                previa=previa,
                cnn_prob=cnn_prob,
            )

        # ---- Risk level ----
        final_prob = ensemble_result.get("ensemble_prob", cnn_prob)
        risk_level = "HIGH" if final_prob >= 0.5 else "LOW"

        return JSONResponse(
            content={
                "patient_id": patient_id or None,
                "cnn": {
                    "per_frame_prob": [round(float(p), 4) for p in per_frame_probs],
                    "cnn_prob": round(float(cnn_prob), 4),
                    "cnn_pred": cnn_pred,
                    "agg_method": agg_method,
                },
                "ensemble": ensemble_result,
                "dicom_info": {**info, "num_frames": num_frames},
                "risk_level": risk_level,
                "num_prior_cs": num_prior_cs,
                "previa": previa,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}\n{tb}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
