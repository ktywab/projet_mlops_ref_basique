########################################################
######## à exécuter UNE FOIS avant PyInstaller #########
########################################################


import os
import nltk

def main():
    target_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    os.makedirs(target_dir, exist_ok=True)

    # Téléchargement local (build-time)
    nltk.download("stopwords", download_dir=target_dir)
    nltk.download("wordnet", download_dir=target_dir)

    print(f"NLTK data downloaded to: {target_dir}")

if __name__ == "__main__":
    main()