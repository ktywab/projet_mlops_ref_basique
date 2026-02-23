import os
import logging
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz


# -----------------------------
# Logging setup
# -----------------------------
def setup_logger(log_dir="logs", log_file="feature_engineering.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("feature_engineering")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# -----------------------------
# Params / config
# -----------------------------
def load_config(params_path="params.yaml"):
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    fe = params.get("feature_engineering", {})

    cfg = {
        "train_path": "data/processed/train_processed.csv",
        "test_path": "data/processed/test_processed.csv",
        "features_dir": "data/features",
        "vec_dir": "models/vectorizers",
        "text_col": fe.get("text_col", "text_norm"),
        "label_col": fe.get("label_col", "label"),
        "methods": fe.get("methods", ["bow", "tfidf"]),
        "min_df": int(fe.get("min_df", 5)),
        "max_df": float(fe.get("max_df", 0.9)),
        "token_pattern": fe.get("token_pattern", r"\b[a-z]{3,15}\b"),
        "ngram_range": tuple(fe.get("ngram_range", [1, 2])),
        "sublinear_tf": bool(fe.get("sublinear_tf", True)),
        "norm": fe.get("norm", "l2"),
    }

    return cfg


# -----------------------------
# Data loading
# -----------------------------
def load_processed_data(cfg):
    train_df = pd.read_csv(cfg["train_path"])
    test_df = pd.read_csv(cfg["test_path"])

    logger.info(f"Train processed shape: {train_df.shape}")
    logger.info(f"Test processed shape: {test_df.shape}")

    for col in (cfg["text_col"], cfg["label_col"]):
        if col not in train_df.columns:
            raise KeyError(f"Missing column '{col}' in train data")
        if col not in test_df.columns:
            raise KeyError(f"Missing column '{col}' in test data")

    train_df[cfg["text_col"]] = train_df[cfg["text_col"]].fillna("")
    test_df[cfg["text_col"]] = test_df[cfg["text_col"]].fillna("")

    return (
        train_df[cfg["text_col"]].values,
        train_df[cfg["label_col"]].values,
        test_df[cfg["text_col"]].values,
        test_df[cfg["label_col"]].values,
    )


# -----------------------------
# Vectorizers
# -----------------------------
def build_vectorizer(method, cfg):
    common = dict(
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        token_pattern=cfg["token_pattern"],
        ngram_range=cfg["ngram_range"],
    )

    if method == "bow":
        return CountVectorizer(**common)

    if method == "tfidf":
        return TfidfVectorizer(
            **common,
            sublinear_tf=cfg["sublinear_tf"],
            norm=cfg["norm"],
        )

    raise ValueError(f"Unknown method '{method}'")


def featurize_and_save(method, vectorizer, X_train, X_test, cfg):
    X_train_mat = vectorizer.fit_transform(X_train)
    X_test_mat = vectorizer.transform(X_test)

    logger.info(
        f"{method.upper()} train: {X_train_mat.shape} | nnz={X_train_mat.nnz}"
    )
    logger.info(
        f"{method.upper()} test: {X_test_mat.shape} | nnz={X_test_mat.nnz}"
    )

    save_npz(
        os.path.join(cfg["features_dir"], f"X_train_{method}.npz"),
        X_train_mat,
    )
    save_npz(
        os.path.join(cfg["features_dir"], f"X_test_{method}.npz"),
        X_test_mat,
    )

    joblib.dump(
        vectorizer,
        os.path.join(cfg["vec_dir"], f"{method}_vectorizer.joblib"),
    )


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        cfg = load_config()

        logger.info(
            f"Config loaded | methods={cfg['methods']} | "
            f"min_df={cfg['min_df']} | max_df={cfg['max_df']} | "
            f"ngram_range={cfg['ngram_range']}"
        )

        os.makedirs(cfg["features_dir"], exist_ok=True)
        os.makedirs(cfg["vec_dir"], exist_ok=True)

        X_train, y_train, X_test, y_test = load_processed_data(cfg)

        # labels communs à toutes les features
        np.save(os.path.join(cfg["features_dir"], "y_train.npy"), y_train)
        np.save(os.path.join(cfg["features_dir"], "y_test.npy"), y_test)

        for method in cfg["methods"]:
            vectorizer = build_vectorizer(method, cfg)
            featurize_and_save(method, vectorizer, X_train, X_test, cfg)

        logger.info("Feature engineering terminé (modulaire, sans classes).")

    except Exception:
        logger.exception("Feature engineering échoué.")
        raise


if __name__ == "__main__":
    main()