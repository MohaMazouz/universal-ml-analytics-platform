import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "assets/model_lgbm_multi.pkl"
FEATURES_PATH = "assets/model_lgbm_multi_features.pkl"

class PaymentDelayAI:
    def __init__(self, multi_class_classifier_model=None, feature_columns=None):
        self.ml_multi_classifier = multi_class_classifier_model
        self.feature_columns = feature_columns
        self.category_names = {
            0: "Aucun retard (ML)",
            1: "Est en retard (ML)",
            2: "Retard exagere (ML)"
        }

    def predict_payment_behavior(self, df):
        df_featured = self.create_advanced_features(df.copy())
        X_pred = self.preprocess_features(df_featured)
        if X_pred is None:
            st.error("Erreur: Impossible de preparer les features pour la prediction")
            return df_featured
        pred_cat_num = self.ml_multi_classifier.predict(X_pred)
        df_featured['ML_Prediction_Num'] = pred_cat_num
        df_featured['ML_Prediction'] = df_featured['ML_Prediction_Num'].map(self.category_names)
        # Risque financier (optionnel)
        if ' T.T.C ' in df_featured.columns:
            risk_factor = df_featured['ML_Prediction_Num'].map({0: 0.05, 1: 0.2, 2: 0.5}).fillna(0)
            df_featured['amount_at_risk_prediction'] = df_featured[' T.T.C '] * risk_factor
        else:
            df_featured['amount_at_risk_prediction'] = 0
        return df_featured

    def preprocess_features(self, df):
        if self.feature_columns is None:
            st.error("Erreur: feature_columns non defini")
            return None
        common_cols = [col for col in df.columns if col in self.feature_columns]
        X_pred = df[common_cols].copy()
    # NaN numeriques -> mediane
        num_cols = X_pred.select_dtypes(include=np.number).columns
        for col in num_cols:
            X_pred[col] = X_pred[col].fillna(X_pred[col].median())
        # NaN cat -> 'Missing'
        cat_cols = X_pred.select_dtypes(include='object').columns
        for col in cat_cols:
            X_pred[col] = X_pred[col].fillna('Missing')
        X_pred = pd.get_dummies(X_pred, columns=cat_cols, dummy_na=False)
        X_pred = X_pred.reindex(columns=self.feature_columns, fill_value=0)
        return X_pred

    def create_advanced_features(self, df):
        if 'echeance' in df.columns:
            df['echeance'] = pd.to_datetime(df['echeance'], errors='coerce')
        df["Date d'Emission"] = pd.to_datetime(df["Date d'Emission"], errors='coerce')
        df['days_since_invoice'] = (pd.Timestamp.now() - df["Date d'Emission"]).dt.days
        df['days_to_due'] = (df['échéance'] - pd.Timestamp.now()).dt.days
        df['invoice_month'] = df["Date d'Emission"].dt.month
        df['due_day_of_week'] = df['échéance'].dt.dayofweek
        # Rolling features client
        cols_rolling = ['Code Client', 'Est_En_Retard', 'Jours_Retard', ' T.T.C ']
        if all(c in df.columns for c in cols_rolling):
            df_sorted = df.sort_values(by=['Code Client', "Date d'Emission"]).copy()
            client_feats = df_sorted.groupby('Code Client').rolling(window=5)[
                ['Est_En_Retard', 'Jours_Retard', ' T.T.C ']].agg(['mean', 'std', 'max', 'sum'])
            client_feats.columns = ['_'.join(x) for x in client_feats.columns]
            client_feats = client_feats.reset_index().rename(columns={'level_1': 'original_index'})
            df = df.reset_index().rename(columns={'index': 'original_index'})
            df = df.merge(client_feats.drop(columns=['Code Client']), on='original_index', how='left').drop('original_index', axis=1)
            rename_map = {
                'Est_En_Retard_mean': 'client_delay_mean_5',
                'Est_En_Retard_std': 'client_delay_std_5',
                'Jours_Retard_mean': 'client_avg_delay_5',
                'Jours_Retard_max': 'client_max_delay_5',
                ' T.T.C _mean': 'client_avg_ttc_5',
                ' T.T.C _sum': 'client_sum_ttc_5'
            }
            df.rename(columns=rename_map, inplace=True)
        if ' Caution ' in df.columns and ' T.T.C ' in df.columns:
            df['caution_utilization_rate'] = df[' T.T.C '] / df[' Caution '].replace(0, np.inf)
            df['caution_buffer'] = df[' Caution '] - df[' T.T.C ']
        df['payment_regularity'] = 0.8
        df['client_risk_trend'] = 0.2
        return df

# ----------- Partie ENTRAINEMENT -----------
def train_model(df):
    # 1. Construction de la cible
    def new_cat(row):
        if row.get('Est_Retard_Exagéré', 0) == 1:
            return 2
        elif row.get('Est_En_Retard', 0) == 1:
            return 1
        else:
            return 0
    df['nouvelle_categorie_retard'] = df.apply(new_cat, axis=1)

    # 2. Feature engineering
    temp_ai = PaymentDelayAI()
    df_fe = temp_ai.create_advanced_features(df.copy())

    # 3. Sélection features
    feature_cols = [
        'days_since_invoice', 'days_to_due', 'invoice_month', 'due_day_of_week',
        'client_delay_mean_5', 'client_delay_std_5', 'client_avg_delay_5',
        'client_max_delay_5', 'client_avg_ttc_5', 'client_sum_ttc_5',
        'caution_utilization_rate', 'caution_buffer', 'payment_regularity',
        'client_risk_trend', ' T.T.C ', ' H.T ', ' T.V.A ', ' T.R ',
        'Code Client', 'Client', 'Catégorie_Règle'
    ]
    feature_cols = [c for c in feature_cols if c in df_fe.columns]
    X_multi = df_fe[feature_cols].copy()
    y_multi = df_fe['nouvelle_categorie_retard']

    # 4. Remplacement des NaN
    for c in X_multi.select_dtypes(include=np.number).columns:
        X_multi[c].fillna(X_multi[c].median(), inplace=True)
    for c in X_multi.select_dtypes(include='object').columns:
        X_multi[c].fillna('Missing', inplace=True)

    # 5. One-hot encoding
    X_multi = pd.get_dummies(X_multi, columns=X_multi.select_dtypes(include='object').columns, dummy_na=False)
    # Supprimer colonnes liées à 'Catégorie_Règle' si présentes
    cols_to_drop = [col for col in X_multi.columns if col.startswith('Catégorie_Règle_')]
    X_multi.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 6. Split et SMOTE
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    smote = BorderlineSMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_multi, y_train_multi)

    # 7. Entraînement LightGBM
    lgb_multi = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, n_jobs=-1)
    lgb_multi.fit(X_train_bal, y_train_bal)

    # 8. Évaluation rapide
    y_pred = lgb_multi.predict(X_test_multi)
    acc = accuracy_score(y_test_multi, y_pred)
    report = classification_report(y_test_multi, y_pred)
    cm = confusion_matrix(y_test_multi, y_pred)
    st.success(f"Modèle entraîné avec {acc:.2%} de précision sur le test.")
    st.text("Matrice de confusion :\n" + str(cm))
    st.text("Rapport de classification :\n" + report)

    # 9. Sauvegarde
    os.makedirs('assets', exist_ok=True)
    joblib.dump(lgb_multi, MODEL_PATH)
    joblib.dump(X_train_multi.columns.tolist(), FEATURES_PATH)
    st.success("Modèle et features sauvegardés dans assets/.")

    return lgb_multi, X_train_multi.columns.tolist()

# ----------- Partie PRÉDICTION -----------
@st.cache_resource(show_spinner="Chargement du modèle ML…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Modèle non trouvé. Merci d'entraîner d'abord.")
        return None, None
    model = joblib.load(MODEL_PATH)
    if os.path.exists(FEATURES_PATH):
        features = joblib.load(FEATURES_PATH)
    else:
        features = None
    return model, features

def run_prediction(df):
    model, feature_cols = load_model()
    if model is None or feature_cols is None:
        st.warning("Modèle non disponible. Merci de l'entraîner d'abord.")
        return df
    payment_ai = PaymentDelayAI(multi_class_classifier_model=model, feature_columns=feature_cols)
    df_pred = payment_ai.predict_payment_behavior(df)
    return df_pred