# AURORA — PAS Detection Pipeline & Dashboard

**Research use only.** AI-assisted detection of Placenta Accreta Spectrum (PAS)
from obstetric ultrasound DICOM files using an EfficientNetB0 CNN combined with
an ensemble model (logistic regression + random forest + gradient boosting) that
incorporates clinical risk factors.

---

## Directory Structure

```
Claude_AURORA/
├── app.py                     # FastAPI web application
├── requirements.txt
├── pipeline/
│   ├── dicom_processor.py     # DICOM → preprocessed frames
│   ├── cnn_predictor.py       # EfficientNetB0 inference
│   └── ensemble_predictor.py  # Ensemble (CNN + prior CS + previa)
├── static/
│   ├── index.html             # Single-page dashboard
│   ├── style.css
│   └── app.js
└── models/                    # ← copy model files here (gitignored)
    └── .gitkeep
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy model files

```bash
cp ../AURORA/best_model_copy.h5  models/
cp ../AURORA/ensemble.pkl        models/
```

Both files are **gitignored** (large binaries). They must be present before
starting the server.

### 3. Run the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser.

---

## API

### `GET /`
Returns the clinical dashboard (static/index.html).

### `GET /health`
Returns model load status.

### `POST /predict`
Multipart form fields:

| Field          | Type   | Required | Description                         |
|----------------|--------|----------|-------------------------------------|
| `dicom_file`   | file   | yes      | `.dcm` file (≤ 50 MB)               |
| `num_prior_cs` | int    | yes      | Number of prior cesarean sections   |
| `previa`       | string | yes      | `"yes"` or `"no"`                   |
| `patient_id`   | string | no       | Patient identifier for display      |
| `agg_method`   | string | no       | `mean` (default) / `max` / `median` |

**Response JSON:**

```json
{
  "patient_id": "PAS499",
  "cnn": {
    "per_frame_prob": [0.42, 0.51, ...],
    "cnn_prob": 0.47,
    "cnn_pred": 0,
    "agg_method": "mean"
  },
  "ensemble": {
    "ensemble_prob": 0.28,
    "ensemble_pred": 0,
    "log_prob": 0.21,
    "rf_prob": 0.31,
    "gb_prob": 0.33
  },
  "dicom_info": {
    "num_frames": 388,
    "declared_frames": 388,
    "photometric": "RGB",
    "modality": "US",
    "rows": 480,
    "columns": 640
  },
  "risk_level": "LOW",
  "num_prior_cs": 1,
  "previa": "no"
}
```

---

## Pipeline Details

### DICOM Processing
- VOI-LUT windowing applied via `pydicom`
- MONOCHROME1 photometric inversion handled
- **Crop**: 65 px top, 66 px left, 150 px right (matches training data)
- Resize to 256×256 via PIL BILINEAR
- Single-channel frames stacked to 3-channel RGB

### CNN Model
- EfficientNetB0, trained on cropped grayscale→RGB ultrasound frames
- `efficientnet.preprocess_input` applied (not simple /255)
- Per-frame sigmoid outputs aggregated (default: mean)

### Ensemble Model (`ensemble.pkl`)
- Logistic Regression + Random Forest + Gradient Boosting
- Features: `[number_prior_cs, previa_bin, cnn_prob]`
- Trained on validation split (n=17 patients)
- Test-set performance: sensitivity 100%, specificity 75%, accuracy 88%

---

## Disclaimer

This tool is for **investigational/research use only**. It is not FDA-cleared
and must not be used as the sole basis for clinical decisions.
