import os
import re
import sys
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def resource_path(relative_path: str) -> str:
    """
    Support PyInstaller:
    - en dev: chemin normal (racine projet)
    - en exe: fichiers copiés dans sys._MEIPASS
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# ---- NLTK data path (IMPORTANT pour EXE) ----
# On force NLTK à chercher les corpus dans un dossier embarqué (nltk_data/)
_NLTK_DATA_DIR = resource_path("nltk_data")
if os.path.isdir(_NLTK_DATA_DIR) and _NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DATA_DIR)


# ---- Regex patterns ----
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
PUNCT_PATTERN = re.compile(r"[%s]" % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""))


# ---- Stopwords + lemmatizer (comme ton notebook) ----
# Stopwords: on garde négations/contrastes
_STOPWORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet", "nor"}
_LEMMATIZER = WordNetLemmatizer()


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
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(tokens)


def lemmatization(text: str) -> str:
    tokens = text.split()
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def remove_long_tokens(text: str, max_len: int = 20) -> str:
    return " ".join([w for w in text.split() if len(w) <= max_len])


def normalized_sentence(sentence: str) -> str:
    # Gestion NaN/None
    if sentence is None:
        return ""
    if isinstance(sentence, float) and np.isnan(sentence):
        return ""

    sentence = str(sentence)

    sentence = removing_urls(sentence)
    sentence = removing_punctuations(sentence)
    sentence = lower_case(sentence)
    sentence = removing_numbers(sentence)
    sentence = remove_long_tokens(sentence)
    sentence = remove_stop_words(sentence)
    sentence = lemmatization(sentence)

    return sentence.strip()


def normalize_texts(texts, min_words: int = 3):
    """
    Applique normalized_sentence à une liste.
    Filtre les textes trop courts (comme ton normalize_df).
    Retourne (texts_norm, kept_idx)
    """
    out = []
    kept_idx = []
    for i, t in enumerate(texts):
        norm = normalized_sentence(t)
        if len(norm.split()) >= min_words:
            out.append(norm)
            kept_idx.append(i)
        else:
            out.append("")  # placeholder
    return out, kept_idx