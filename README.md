# projet_mlops_ref_basique
Ce projet sert de référence pour les bonnes manières d'utiliser la technique mlops avec un projet fait de A à Z comme référence surtout coté dev et en local (pas pour sur cloud)


# commande pour ouvrir mlflow avec sqlite
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Commande bash pour faire le packaging avec pyinstaller
pyinstaller --noconsole --onefile ^
Plus ?   --name SentimentApp ^
Plus ?   --hidden-import=lightgbm ^
Plus ?   --hidden-import=lightgbm.basic ^
Plus ?   --hidden-import=lightgbm.sklearn ^
Plus ?   --add-data "models\production;models\production" ^
Plus ?   --add-data "src\app\nltk_data;nltk_data" ^
Plus ?   src\app\gui.py