import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier

# Chemin racine du projet
root_dir = os.path.dirname(os.path.abspath(__file__))

# Chargement des données
data_path = os.path.join(root_dir, "data", "BD_avec_regles_paiement_latest.xlsx")
df = pd.read_excel(data_path)

# Préparation des données (à adapter selon ton dataset réel)
X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

# Entraînement du modèle
model = LGBMClassifier()
model.fit(X, y)

# Sauvegarde dans le dossier assets
assets_dir = os.path.join(root_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)
joblib.dump(model, os.path.join(assets_dir, "model_lgbm_multi.pkl"))
joblib.dump(list(X.columns), os.path.join(assets_dir, "model_lgbm_multi_features.pkl"))
print(f"✅ Modèle et features sauvegardés dans {assets_dir}")