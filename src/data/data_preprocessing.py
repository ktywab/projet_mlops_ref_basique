import os
import re
import logging
import numpy as np
import pandas as pd
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# Logging setup (même style que ingestion)
# -----------------------------
def setup_logger(log_dir: str = "logs", log_file: str = "data_preprocessing.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("data_preprocessing")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# -----------------------------
# NLTK ensure
# -----------------------------
def ensure_nltk():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


# -----------------------------
# Normalization helpers (ton bloc "selon ça")
# -----------------------------
ensure_nltk()

_lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet", "nor"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
PUNCT_PATTERN = re.compile(r"[%s]" % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""))


def removing_urls(text: str) -> str:
    return URL_PATTERN.sub("", text)


def removing_punctuations(text: str) -> str:
    text = PUNCT_PATTERN.sub(" ", text)
    text = text.replace("؛", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lower_case(text: str) -> str:
    return text.lower()


def removing_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)


def remove_stop_words(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def lemmatization(text: str) -> str:
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def remove_long_tokens(text: str, max_len: int = 20) -> str:
    return " ".join([w for w in text.split() if len(w) <= max_len])


def normalized_sentence(sentence) -> str:
    if sentence is None or (isinstance(sentence, float) and np.isnan(sentence)):
        return ""

    sentence = str(sentence)

    sentence = removing_urls(sentence)
    sentence = removing_punctuations(sentence)
    sentence = lower_case(sentence)
    sentence = removing_numbers(sentence)
    sentence = remove_long_tokens(sentence, max_len=20)
    sentence = remove_stop_words(sentence)
    sentence = lemmatization(sentence)

    return sentence


def normalize_df(df: pd.DataFrame, text_col: str, out_col: str, min_words: int) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Colonne texte '{text_col}' introuvable. Colonnes dispo: {list(df.columns)}")

    df = df.copy()
    df[out_col] = df[text_col].apply(normalized_sentence)

    df.loc[df[out_col].str.split().str.len() < min_words, out_col] = np.nan
    df = df.dropna(subset=[out_col])

    return df


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        # Params (optionnel, mais propre)
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        # valeurs par défaut si tu n'ajoutes pas preprocess dans params.yaml
        preprocess_params = params.get("preprocess", {})
        text_col = preprocess_params.get("text_col", "text")
        out_col = preprocess_params.get("out_col", "text_norm")
        min_words = int(preprocess_params.get("min_words", 3))

        logger.info(f"Params preprocess: text_col={text_col}, out_col={out_col}, min_words={min_words}")

        # Inputs depuis stage précédent
        train_in = os.path.join("data", "raw", "train.csv")
        test_in = os.path.join("data", "raw", "test.csv")

        logger.info("Loading raw train/test...")
        train_df = pd.read_csv(train_in)
        test_df = pd.read_csv(test_in)
        logger.info(f"Train raw shape: {train_df.shape}")
        logger.info(f"Test raw shape: {test_df.shape}")

        logger.info("Normalizing train...")
        train_processed = normalize_df(train_df, text_col=text_col, out_col=out_col, min_words=min_words)

        logger.info("Normalizing test...")
        test_processed = normalize_df(test_df, text_col=text_col, out_col=out_col, min_words=min_words)

        logger.info(f"Train processed shape: {train_processed.shape}")
        logger.info(f"Test processed shape: {test_processed.shape}")

        # Outputs
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        train_out = os.path.join(processed_dir, "train_processed.csv")
        test_out = os.path.join(processed_dir, "test_processed.csv")

        train_processed.to_csv(train_out, index=False)
        test_processed.to_csv(test_out, index=False)

        logger.info(f"Saved: {train_out}")
        logger.info(f"Saved: {test_out}")
        logger.info("Data preprocessing terminé.")

    except Exception:
        logger.exception("Data preprocessing échoué.")
        raise


if __name__ == "__main__":
    main()
