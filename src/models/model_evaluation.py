import os
import json
import logging
import numpy as np
import yaml
import joblib
import mlflow

import argparse
import subprocess
import time
import webbrowser
import sys

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_dir="logs", log_file="model_evaluation.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("model_evaluation")
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


# -----------------------------
# Params
# -----------------------------
def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_test_paths(features_type: str):
    return (
        os.path.join("data", "features", f"X_test_{features_type}.npz"),
        os.path.join("data", "features", "y_test.npy"),
    )


def load_test_data(features_type: str):
    X_test_path, y_test_path = get_test_paths(features_type)
    X_test = load_npz(X_test_path)
    y_test = np.load(y_test_path, allow_pickle=True)
    return X_test, y_test


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    # AUC (multi-classe) nécessite predict_proba
    try:
        y_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            metrics["auc_ovr_macro"] = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )
            metrics["auc_ovr_weighted"] = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            )
    except Exception as e:
        metrics["auc_error"] = str(e)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm


def save_confusion_matrix(cm, labels, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# MLflow logging
# -----------------------------
def log_run_to_mlflow(
    model_name: str,
    model,
    params_to_log: dict,
    metrics: dict,
    report: dict,
    cm: np.ndarray,
    artifacts_dir: str,
):
    os.makedirs(artifacts_dir, exist_ok=True)

    report_path = os.path.join(artifacts_dir, f"{model_name}_classification_report.json")
    cm_path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_labels = [str(i) for i in range(cm.shape[0])]
    save_confusion_matrix(cm, cm_labels, cm_path)

    with mlflow.start_run(run_name=model_name):
        for k, v in params_to_log.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

        # log model as joblib artifact (simple/robuste)
        model_path = os.path.join(artifacts_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)


# -----------------------------
# Auto-start MLflow UI (optional)
# -----------------------------
def start_mlflow_ui(tracking_uri="file:./mlruns", host="127.0.0.1", port=5000):
    """
    Lance MLflow UI en local + ouvre le navigateur par défaut.
    NOTE: à éviter dans un stage DVC (le serveur reste actif). Utiliser via --serve-mlflow.
    """
    url = f"http://{host}:{port}"

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--host", host,
        "--port", str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.getcwd(),
        shell=False,
    )

    time.sleep(1.2)
    webbrowser.open(url)

    return proc, url


def main(serve_mlflow: bool = False):
    try:
        params = load_params("params.yaml")

        # features
        features_type = params.get("model_building", {}).get("features_type", "bow")

        # mlflow
        mlflow_cfg = params.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
        experiment_name = mlflow_cfg.get("experiment_name", "sentiment-model-evaluation")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking_uri={tracking_uri} | experiment={experiment_name}")
        logger.info(f"Loading test data: {features_type}")

        X_test, y_test = load_test_data(features_type)

        # modèles
        xgb_path = os.path.join("models", f"xgb_{features_type}_model.joblib")
        lgbm_path = os.path.join("models", f"lgbm_{features_type}_model.joblib")

        logger.info(f"Loading models: {xgb_path} | {lgbm_path}")
        xgb_model = joblib.load(xgb_path)
        lgbm_model = joblib.load(lgbm_path)

        # LightGBM veut float
        if X_test.dtype not in (np.float32, np.float64):
            X_test_float = X_test.astype(np.float32)
        else:
            X_test_float = X_test

        artifacts_dir = os.path.join("reports", "mlflow_artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # ---- XGBoost eval ----
        logger.info("Evaluating XGBoost...")
        xgb_metrics, xgb_report, xgb_cm = compute_metrics(xgb_model, X_test, y_test)

        xgb_params = params.get("xgboost", {})
        xgb_params_to_log = {
            "model": "xgboost",
            "features_type": features_type,
            **{f"xgb_{k}": v for k, v in xgb_params.items()},
        }

        log_run_to_mlflow(
            model_name=f"xgb_{features_type}",
            model=xgb_model,
            params_to_log=xgb_params_to_log,
            metrics=xgb_metrics,
            report=xgb_report,
            cm=xgb_cm,
            artifacts_dir=artifacts_dir,
        )

        logger.info(f"XGB metrics: {xgb_metrics}")

        # ---- LightGBM eval ----
        logger.info("Evaluating LightGBM...")
        lgbm_metrics, lgbm_report, lgbm_cm = compute_metrics(lgbm_model, X_test_float, y_test)

        lgbm_params = params.get("lightgbm", {})
        lgbm_params_to_log = {
            "model": "lightgbm",
            "features_type": features_type,
            **{f"lgbm_{k}": v for k, v in lgbm_params.items()},
        }

        log_run_to_mlflow(
            model_name=f"lgbm_{features_type}",
            model=lgbm_model,
            params_to_log=lgbm_params_to_log,
            metrics=lgbm_metrics,
            report=lgbm_report,
            cm=lgbm_cm,
            artifacts_dir=artifacts_dir,
        )

        logger.info(f"LGBM metrics: {lgbm_metrics}")

        # Résumé local
        summary = {"xgb": xgb_metrics, "lgbm": lgbm_metrics}
        os.makedirs("reports", exist_ok=True)
        with open(os.path.join("reports", "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Model evaluation terminé + runs MLflow créés.")

        # ---- lancer MLflow UI + navigateur ----
        if serve_mlflow:
            proc, url = start_mlflow_ui(tracking_uri=tracking_uri, host="127.0.0.1", port=5000)
            logger.info(f"MLflow UI lancé: {url}")
            logger.info("Pour arrêter MLflow UI, ferme le process Python (ou CTRL+C dans le terminal).")

    except Exception:
        logger.exception("Model evaluation échoué.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-mlflow", action="store_true", help="Lance mlflow ui et ouvre le navigateur")
    args = parser.parse_args()
    main(serve_mlflow=args.serve_mlflow)