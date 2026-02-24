import os
import sys
import joblib
import numpy as np

from text_preprocess import normalize_texts


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def load_production_artifacts():
    model_path = resource_path(os.path.join("models", "production", "model.joblib"))
    vec_path = resource_path(os.path.join("models", "production", "vectorizer.joblib"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def predict_texts(texts, min_words: int = 3):
    """
    texts: list[str]
    return: list[dict] in same order as inputs
      - if too short after preprocessing => label=None + message
    """
    model, vectorizer = load_production_artifacts()

    texts_norm, kept_idx = normalize_texts(texts, min_words=min_words)

    # On ne prédit que pour les textes gardés
    kept_texts = [texts_norm[i] for i in kept_idx]

    results = [{"label": None, "confidence": None, "proba": None, "note": "too_short_after_preprocess"} for _ in texts]

    if len(kept_texts) == 0:
        return results

    X = vectorizer.transform(kept_texts)

    # LightGBM exige float32/float64 sur sparse
    if hasattr(X, "dtype") and X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for row, orig_i in enumerate(kept_idx):
            p = proba[row]
            conf = float(np.max(p))
            results[orig_i] = {
                "label": int(y_pred[row]),
                "confidence": conf,
                "proba": [float(x) for x in p],
                "note": "ok",
            }
    else:
        for row, orig_i in enumerate(kept_idx):
            results[orig_i] = {"label": int(y_pred[row]), "confidence": None, "proba": None, "note": "ok"}

    return results