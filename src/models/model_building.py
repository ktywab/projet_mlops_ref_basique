import os
import logging
import numpy as np
import yaml
import joblib

from scipy.sparse import load_npz
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_dir="logs", log_file="model_building.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("model_building")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_feature_paths(features_type: str):
    return (
        os.path.join("data", "features", f"X_train_{features_type}.npz"),
        os.path.join("data", "features", f"X_test_{features_type}.npz"),
        os.path.join("data", "features", "y_train.npy"),
        os.path.join("data", "features", "y_test.npy"),
    )


def load_data(features_type: str):
    X_train_path, X_test_path, y_train_path, y_test_path = get_feature_paths(features_type)

    X_train = load_npz(X_train_path)
    X_test = load_npz(X_test_path)

    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    return X_train, X_test, y_train, y_test


def ensure_float_for_lgbm(X_train, X_test):
    """
    LightGBM exige que les valeurs dans la matrice sparse soient float32/float64.
    BoW est souvent en int64 => on cast en float32.
    """
    if X_train.dtype not in (np.float32, np.float64):
        logger.info(f"Casting X_train from {X_train.dtype} -> float32 (LightGBM compatibility)")
        X_train = X_train.astype(np.float32)

    if X_test.dtype not in (np.float32, np.float64):
        logger.info(f"Casting X_test from {X_test.dtype} -> float32 (LightGBM compatibility)")
        X_test = X_test.astype(np.float32)

    return X_train, X_test


def build_xgb(xgb_params: dict, num_class: int, random_state: int):
    # On passe UNIQUEMENT les params demandés (le reste = defaults XGBoost)
    kwargs = dict(
        learning_rate=float(xgb_params.get("learning_rate", 0.2)),
        n_estimators=int(xgb_params.get("n_estimators", 800)),
        reg_alpha=float(xgb_params.get("reg_alpha", 0.0)),
        reg_lambda=float(xgb_params.get("reg_lambda", 1.0)),
        n_jobs=int(xgb_params.get("n_jobs", -1)),
        random_state=random_state,
        tree_method="hist",
        eval_metric="mlogloss" if num_class > 2 else "logloss",
    )

    if num_class > 2:
        kwargs["objective"] = "multi:softprob"
        kwargs["num_class"] = num_class
    else:
        kwargs["objective"] = "binary:logistic"

    return XGBClassifier(**kwargs)


def build_lgbm(lgbm_params: dict, num_class: int, random_state: int):
    # On passe UNIQUEMENT les params demandés (le reste = defaults LightGBM)
    objective = "multiclass" if num_class > 2 else "binary"

    kwargs = dict(
        objective=objective,
        learning_rate=float(lgbm_params.get("learning_rate", 0.2)),
        n_estimators=int(lgbm_params.get("n_estimators", 800)),
        reg_alpha=float(lgbm_params.get("reg_alpha", 0.0)),
        reg_lambda=float(lgbm_params.get("reg_lambda", 0.0)),
        n_jobs=int(lgbm_params.get("n_jobs", -1)),
        random_state=random_state,
    )

    return LGBMClassifier(**kwargs)


def save_model(model, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)


def main():
    try:
        params = load_params("params.yaml")

        mb = params.get("model_building", {})
        features_type = mb.get("features_type", "bow")
        random_state = int(mb.get("random_state", 42))

        xgb_params = params.get("xgboost", {})
        lgbm_params = params.get("lightgbm", {})

        logger.info(f"Loading features: {features_type}")
        X_train, X_test, y_train, y_test = load_data(features_type)
        logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
        logger.info(f"X_train dtype: {X_train.dtype} | X_test dtype: {X_test.dtype}")

        num_class = len(np.unique(y_train))
        logger.info(f"Detected num_class={num_class}")

        # Train XGBoost (XGBoost accepte int/float, pas besoin de cast)
        logger.info(
            f"Training XGBoost | lr={xgb_params.get('learning_rate')} | "
            f"n_estimators={xgb_params.get('n_estimators')} | "
            f"reg_alpha={xgb_params.get('reg_alpha')} | reg_lambda={xgb_params.get('reg_lambda')}"
        )
        xgb_model = build_xgb(xgb_params, num_class=num_class, random_state=random_state)
        xgb_model.fit(X_train, y_train)

        # Train LightGBM (nécessite float32/float64)
        X_train_lgbm, X_test_lgbm = ensure_float_for_lgbm(X_train, X_test)

        logger.info(
            f"Training LightGBM | lr={lgbm_params.get('learning_rate')} | "
            f"n_estimators={lgbm_params.get('n_estimators')} | "
            f"reg_alpha={lgbm_params.get('reg_alpha')} | reg_lambda={lgbm_params.get('reg_lambda')}"
        )
        lgbm_model = build_lgbm(lgbm_params, num_class=num_class, random_state=random_state)
        lgbm_model.fit(X_train_lgbm, y_train)

        # Save models
        save_model(xgb_model, os.path.join("models", f"xgb_{features_type}_model.joblib"))
        save_model(lgbm_model, os.path.join("models", f"lgbm_{features_type}_model.joblib"))

        logger.info("Model building terminé (2 modèles sauvegardés).")

    except Exception:
        logger.exception("Model building échoué.")
        raise


if __name__ == "__main__":
    main()