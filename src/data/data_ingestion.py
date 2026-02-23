import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


# -----------------------------
# Logging setup
# -----------------------------
def setup_logger(log_dir: str = "logs", log_file: str = "data_ingestion.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("data_ingestion")
    logger.setLevel(logging.INFO)

    # Évite les doublons si relancé (notebook / dvc)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
################################################################
# Mise en place de l'objet logger comme globale pour ce fichier
################################################################
logger = setup_logger()


# -----------------------------
# Data functions
# -----------------------------
def load_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded. Shape={df.shape}")
        return df
    except pd.errors.ParserError:
        logger.exception(f"Erreur parsing CSV: {data_path}")
        raise
    except FileNotFoundError:
        logger.exception(f"Fichier introuvable: {data_path}")
        raise
    except Exception:
        logger.exception("Erreur inattendue pendant le chargement des données.")
        raise


def preprocess_basic_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preprocessing data...")

        df = df.copy()

        # Suppression de la colonne inutile
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
            logger.info("Column 'Unnamed: 0' dropped.")

        # Vérification présence colonne label
        if "label" not in df.columns:
            raise KeyError("label")

        tri_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        df["label"] = df["label"].map(tri_map)

        # Vérifier si des valeurs sont devenues NaN après le mapping
        n_missing = int(df["label"].isna().sum())
        if n_missing > 0:
            raise ValueError(f"{n_missing} valeurs de 'label' non présentes dans tri_map -> NaN.")

        logger.info(f"Label distribution after mapping:\n{df['label'].value_counts(dropna=False)}")
        logger.info("Preprocessing done.")
        return df

    except Exception:
        logger.exception("Erreur pendant le preprocessing.")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_path = os.path.join(data_path, "raw")
        os.makedirs(raw_path, exist_ok=True)

        train_fp = os.path.join(raw_path, "train.csv")
        test_fp = os.path.join(raw_path, "test.csv")

        train_data.to_csv(train_fp, index=False)
        test_data.to_csv(test_fp, index=False)

        logger.info(f"Saved train data to: {train_fp} (shape={train_data.shape})")
        logger.info(f"Saved test data to: {test_fp} (shape={test_data.shape})")

    except Exception:
        logger.exception("Erreur pendant la sauvegarde des données.")
        raise


def main():
    try:
        # -----------------------------
        # Read params.yaml in main()
        # -----------------------------
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        test_size = float(params["data_ingestion"]["test_size"])
        logger.info(f"Loaded params: data_ingestion.test_size={test_size}")

        # -----------------------------
        # Load data (CSV FILE, not folder)
        # -----------------------------
        df = load_data(
            data_path=r"C:\Users\abotsi\Desktop\Projet_pro\projet_mlops_ref_basique\data\text.csv"
        )

        # -----------------------------
        # Preprocess + split + save
        # -----------------------------
        final_df = preprocess_basic_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        save_data(train_data, test_data, data_path=r"C:\Users\abotsi\Desktop\Projet_pro\projet_mlops_ref_basique\data")

        logger.info("Data ingestion terminé avec succès.")

    except Exception:
        logger.exception("Data ingestion échoué.")
        raise


if __name__ == "__main__":
    main()
