"""
Ensemble predictor (CNN probability + clinical risk factors)
-------------------------------------------------------------
Loads ensemble.pkl which contains three sklearn models trained on validation
data from CNN_manuscript_model_evaluation_02182026.ipynb:

    bundle = {
        "log_model": LogisticRegression,
        "rf_model":  RandomForestClassifier,
        "gb_model":  GradientBoostingClassifier,
    }

Feature matrix (CRITICAL column order matching training):
    X = [[number_prior_cs, previa_bin, cnn_prob]]

    previa_bin: "y"/"yes" → 1,  "n"/"no" → 0

Ensemble probability = mean(log_prob, rf_prob, gb_prob)
"""
from __future__ import annotations

import pickle
import numpy as np


def load_ensemble(pkl_path: str) -> dict:
    """Load ensemble.pkl and return the bundle dict."""
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    expected = {"log_model", "rf_model", "gb_model"}
    missing = expected - set(bundle.keys())
    if missing:
        raise ValueError(f"ensemble.pkl is missing keys: {missing}")
    return bundle


def _previa_to_bin(previa: str) -> int:
    """Convert 'yes'/'y'/'no'/'n' (case-insensitive) to 1 or 0."""
    v = previa.strip().lower()
    if v in ("yes", "y"):
        return 1
    if v in ("no", "n"):
        return 0
    raise ValueError(f"Unrecognized previa value: {previa!r}. Use 'yes' or 'no'.")


def predict_ensemble(
    bundle: dict,
    number_prior_cs: int,
    previa: str,
    cnn_prob: float,
    threshold: float = 0.5,
) -> dict:
    """
    Run ensemble inference for a single patient.

    Parameters
    ----------
    bundle          : dict from load_ensemble()
    number_prior_cs : number of prior cesarean sections (int)
    previa          : 'yes' / 'no'
    cnn_prob        : aggregated CNN probability (float, 0–1)
    threshold       : decision threshold for binary prediction

    Returns
    -------
    dict with keys:
        ensemble_prob  : float
        ensemble_pred  : int (0 or 1)
        log_prob       : float
        rf_prob        : float
        gb_prob        : float
    """
    previa_bin = _previa_to_bin(previa)
    X = [[int(number_prior_cs), previa_bin, float(cnn_prob)]]

    log_prob = float(bundle["log_model"].predict_proba(X)[0][1])
    rf_prob  = float(bundle["rf_model"].predict_proba(X)[0][1])
    gb_prob  = float(bundle["gb_model"].predict_proba(X)[0][1])

    ensemble_prob = (log_prob + rf_prob + gb_prob) / 3.0
    ensemble_pred = int(ensemble_prob >= threshold)

    return {
        "ensemble_prob": ensemble_prob,
        "ensemble_pred": ensemble_pred,
        "log_prob": log_prob,
        "rf_prob": rf_prob,
        "gb_prob": gb_prob,
    }
