import os
import json
import shutil
import logging
import yaml
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow.lightgbm


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_dir="logs", log_file="register_model.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("register_model")
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
# Helpers
# -----------------------------
def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_best_model(metrics_summary_path: str, metric_key: str):
    with open(metrics_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # summary = {"xgb": {...}, "lgbm": {...}}
    if "xgb" not in summary or "lgbm" not in summary:
        raise KeyError("metrics_summary.json doit contenir les clés 'xgb' et 'lgbm'.")

    xgb_score = summary["xgb"].get(metric_key)
    lgbm_score = summary["lgbm"].get(metric_key)

    if xgb_score is None or lgbm_score is None:
        raise KeyError(
            f"Metric '{metric_key}' absente dans metrics_summary.json. "
            f"Disponibles xgb={list(summary['xgb'].keys())}, lgbm={list(summary['lgbm'].keys())}"
        )

    best = "xgb" if xgb_score >= lgbm_score else "lgbm"
    best_score = xgb_score if best == "xgb" else lgbm_score

    return best, float(best_score), summary


def promote_to_production(best: str, features_type: str):
    """
    Copie le meilleur modèle + le vectorizer BoW dans models/production/
    """
    prod_dir = os.path.join("models", "production")
    os.makedirs(prod_dir, exist_ok=True)

    # modèle entraîné
    model_src = os.path.join("models", f"{best}_{features_type}_model.joblib")
    if not os.path.exists(model_src):
        raise FileNotFoundError(f"Modèle introuvable: {model_src}")

    model_dst = os.path.join(prod_dir, "model.joblib")
    shutil.copy2(model_src, model_dst)

    # vectorizer (BoW)
    # ton feature_engineering sauvegarde: models/vectorizers/bow_vectorizer.joblib
    # si chez toi c'est un autre nom, adapte ici
    vec_src = os.path.join("models", "vectorizers", "bow_vectorizer.joblib")
    if not os.path.exists(vec_src):
        # fallback si tu as utilisé count_vectorizer.joblib
        alt = os.path.join("models", "vectorizers", "count_vectorizer.joblib")
        if os.path.exists(alt):
            vec_src = alt
        else:
            raise FileNotFoundError(
                "Vectorizer BoW introuvable. Attendu: models/vectorizers/bow_vectorizer.joblib "
                "ou models/vectorizers/count_vectorizer.joblib"
            )

    vec_dst = os.path.join(prod_dir, "vectorizer.joblib")
    shutil.copy2(vec_src, vec_dst)

    return model_dst, vec_dst


def register_in_mlflow_registry(
    best: str,
    features_type: str,
    model_registry_name: str,
    tracking_uri: str,
    experiment_name: str,
    metric_key: str,
    best_score: float,
):
    """
    Crée un run MLflow "register_model" et enregistre le modèle dans le registry.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Charger le modèle depuis models/
    model_path = os.path.join("models", f"{best}_{features_type}_model.joblib")
    model = joblib.load(model_path)

    # Run de registration
    with mlflow.start_run(run_name=f"register_{best}_{features_type}") as run:
        run_id = run.info.run_id

        # log params/metric utiles
        mlflow.log_param("selected_model", best)
        mlflow.log_param("features_type", features_type)
        mlflow.log_param("selection_metric", metric_key)
        mlflow.log_metric(f"selected_{metric_key}", best_score)

        # Log du modèle avec un flavor adapté
        # - XGBoost sklearn API -> mlflow.sklearn.log_model marche bien
        # - LightGBM sklearn -> mlflow.lightgbm.log_model marche bien
        if best == "xgb":
            mlflow.sklearn.log_model(model, artifact_path="model")
        else:
            mlflow.lightgbm.log_model(model, artifact_path="model")

        model_uri = f"runs:/{run_id}/model"

        # Enregistrement dans le registry
        # Si ton backend store ne supporte pas le registry, ça peut lever une exception.
        mv = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

        # Optionnel: tagger la version
        client.set_model_version_tag(
            name=model_registry_name,
            version=mv.version,
            key="features_type",
            value=features_type,
        )
        client.set_model_version_tag(
            name=model_registry_name,
            version=mv.version,
            key="selected_metric",
            value=metric_key,
        )

        return {
            "registered_model_name": model_registry_name,
            "registered_version": mv.version,
            "model_uri": model_uri,
            "run_id": run_id,
        }


def main():
    try:
        params = load_params("params.yaml")

        # Quel type de features ?
        features_type = params.get("model_building", {}).get("features_type", "bow")

        # Quel metric choisir ?
        metric_key = params.get("register_model", {}).get("metric_key", "auc_ovr_macro")

        # Où lire le résumé metrics ?
        metrics_summary_path = os.path.join("reports", "metrics_summary.json")

        best, best_score, summary = choose_best_model(metrics_summary_path, metric_key)
        logger.info(f"Best model: {best} | {metric_key}={best_score}")

        # Promote (copie dans models/production)
        model_dst, vec_dst = promote_to_production(best, features_type)
        logger.info(f"Promoted model -> {model_dst}")
        logger.info(f"Promoted vectorizer -> {vec_dst}")

        # MLflow config
        mlflow_cfg = params.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        experiment_name = mlflow_cfg.get("experiment_name", "sentiment-model-evaluation")
        registry_model_name = mlflow_cfg.get("registry_model_name", "sentiment-classifier")

        # Register in MLflow Model Registry
        logger.info(
            f"Registering into MLflow Model Registry | name={registry_model_name} | tracking_uri={tracking_uri}"
        )

        registry_info = register_in_mlflow_registry(
            best=best,
            features_type=features_type,
            model_registry_name=registry_model_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            metric_key=metric_key,
            best_score=best_score,
        )

        # Sauvegarde une preuve locale (utile pour DVC/Git)
        os.makedirs("reports", exist_ok=True)
        out = {
            "selection_metric": metric_key,
            "best_model": best,
            "best_score": best_score,
            "metrics_summary": summary,
            "production": {
                "model_path": model_dst,
                "vectorizer_path": vec_dst,
            },
            "mlflow_registry": registry_info,
        }

        with open(os.path.join("reports", "register_summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        logger.info("Register_model terminé (promotion + registry).")

    except Exception as e:
        logger.exception("Register_model échoué.")
        # Hint utile si registry ne marche pas
        msg = str(e).lower()
        if "model registry" in msg or "registry" in msg:
            logger.error(
                "Si tu utilises encore 'file:./mlruns', passe à un backend sqlite pour activer le Model Registry: "
                "mlflow.tracking_uri: sqlite:///mlflow.db"
            )
        raise


if __name__ == "__main__":
    main()