# Outil d’Analyse & Prédiction des Retards de Paiement

## Structure du projet

```
data-app/
│
├── app.py
|
├── modules/
│   ├── data_processing.py
│   ├── eda_visuals.py
│   ├── ml_predict.py
│   ├── ai_assistant.py
|   └── check_categories.py
├── assets/
│   ├── logo.png
│   ├── style.css
│   └── model_lgbm_multi.pkl
│   └── model_lgbm_multi_features.pkl
├── data/
│   └── BD_avec_regles_paiement_latest.xlsx
│   └── updated_BD V2.xlsx
├── utils/
│   └── utils.py
├── notebooks/
│   ├── week1.py
│   ├── week2.py
│   └── semaine3.py
├── scripts/
│   └── train_model.py
├── requirements.txt
├── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement du modèle (manuel)

```bash
python train_model.py
```

## Lancement de l’application

```bash
streamlit run app.py
```

## Bonnes pratiques

- Toujours exécuter depuis la racine du projet (`data-app/`)
- Tous les chemins sont robustes grâce à `os.path`
- Les modules (`modules/`) contiennent toute la logique réutilisable
