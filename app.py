import streamlit as st
import pandas as pd
import io
import os

# Import modules métier
from modules import data_processing, eda_visuals, ml_predict, ai_assistant

st.set_page_config(
    page_title="Outil d’Analyse & Prédiction des Retards de Paiement",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

css_path = os.path.join(BASE_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ========== MAPPING AUTOMATIQUE DES COLONNES ==========
COLUMN_MAPPING = {
    "Client Name": "Client",
    "Nom Client": "Client",
    "Amount TTC": " T.T.C ",
    "Montant TTC": " T.T.C ",
    "Days Late": "Jours_Retard",
    "Retard (jours)": "Jours_Retard",
    "Due Date": "échéance",
    "Date d'échéance": "échéance",
    # Ajoute ici toutes les variantes possibles de tes fichiers historiques
}

def harmonize_columns(df, mapping):
    """Renomme automatiquement les colonnes du DataFrame selon le mapping fourni."""
    cols = [mapping.get(col, col) for col in df.columns]
    df.columns = cols
    return df

# Sidebar
logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=180)
st.sidebar.title("Navigation")
PAGES = [
    "Vue d'ensemble",
    "Analyse détaillée",
    "Prédictions ML",
    "AI Insights & Relances"
]
page = st.sidebar.radio("Aller à", PAGES)

# Initialisation des états
for k in ["df_raw", "df_processed", "ml_preds"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ======================= PAGE 1 : Vue d'ensemble =======================
if page == "Vue d'ensemble":
    st.header("Vue d’ensemble")
    st.write("Importez votre fichier Excel de factures (brut ou déjà nettoyé).")
    uploaded_file = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df_raw = pd.read_excel(uploaded_file)
            df_raw = harmonize_columns(df_raw, COLUMN_MAPPING)  # Mapping automatique
            st.session_state["df_raw"] = df_raw
            st.success("Fichier importé avec succès ! (colonnes harmonisées)")
            st.dataframe(df_raw.head(20))
            st.write(f"**Dimensions** : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")

            # Pipeline DATA immédiatement après import
            with st.spinner("Nettoyage et préparation des données..."):
                df_processed = data_processing.clean_and_prepare(df_raw)
            if isinstance(df_processed, pd.DataFrame) and not df_processed.empty:
                st.session_state["df_processed"] = df_processed
                st.success("Traitement terminé. Dataset prêt !")
                st.dataframe(df_processed.head(20))
                towrite = io.BytesIO()
                df_processed.to_excel(towrite, index=False, engine="openpyxl")
                towrite.seek(0)
                st.download_button(
                    label="Télécharger la version traitée",
                    data=towrite,
                    file_name="BD_avec_regles_paiement_latest.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("Le traitement a retourné un dataframe vide. Vérifiez votre fichier source.")
        except Exception as e:
            st.error(f"Erreur lors de l’import ou du traitement du fichier : {e}")

    if st.session_state["df_processed"] is not None:
        st.subheader("Indicateurs clés (extrait)")
        try:
            kpi_dict, _ = data_processing.generate_kpis(st.session_state["df_processed"])
            st.json(kpi_dict)
        except Exception as e:
            st.warning(f"Erreur dans l'extraction des KPIs : {e}")
        st.markdown("---")

# ======================= PAGE 2 : Analyse détaillée =======================
elif page == "Analyse détaillée":
    st.header("Analyse détaillée & Visualisations")
    df = st.session_state.get("df_processed")
    if df is not None and not df.empty:
        try:
            eda_visuals.display_eda(df)
        except Exception as e:
            st.error(f"Erreur dans l’analyse ou la visualisation : {e}")
    else:
        st.warning("Merci d’importer un fichier dans l’onglet Vue d’ensemble.")

# ======================= PAGE 3 : Prédictions ML (Entraînement + Prédiction) =======================
elif page == "Prédictions ML":
    st.header("Prédictions ML sur les retards de paiement")
    df = st.session_state.get("df_processed")
    if df is not None and not df.empty:
        st.info("Vous pouvez (ré)entraîner le modèle sur vos données, puis lancer la prédiction.")

        # --- BOUTONS côte à côte ---
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("🚀 Entraîner le modèle (sur ces données)"):
                with st.spinner("Entraînement du modèle en cours..."):
                    ml_predict.train_model(df)
                st.success("Modèle réentraîné et sauvegardé.")
        with col2:
            predict_clicked = st.button("🔎 Prédire sur ces données")

        # --- OUTPUT PLEINE LARGEUR ---
        if predict_clicked:
            with st.spinner("Prédiction en cours..."):
                preds = ml_predict.run_prediction(df)
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                st.session_state["ml_preds"] = preds

                # --- OUTPUT ML PREDICTION DETAILLÉ ---
                st.subheader("Exemple prédictions :")
                preview_cols = [
                    'N° Facture', 'Est_En_Retard', 'Jours_Retard',
                    'ML_Prediction_Num', 'ML_Prediction',
                    'amount_at_risk_prediction'
                ]
                existing_cols = [c for c in preview_cols if c in preds.columns]
                st.dataframe(preds[existing_cols].head(10), use_container_width=True)

                st.subheader("Distribution des prédictions :")
                if 'ML_Prediction' in preds.columns:
                    st.write(preds['ML_Prediction'].value_counts())

                st.subheader("💰 Aperçu des prédictions :")
                amount_cols = [
                    'N° Facture', 'Est_En_Retard', 'Jours_Retard', ' T.T.C ',
                    'ML_Prediction_Num', 'ML_Prediction', 'amount_at_risk_prediction'
                ]
                amt_exist_cols = [c for c in amount_cols if c in preds.columns]
                st.dataframe(preds[amt_exist_cols].head(10), use_container_width=True)

                st.subheader("🎯 Échantillons par catégorie prédite :")
                for cat_num, cat_name in ml_predict.PaymentDelayAI().category_names.items():
                    st.markdown(f"**--- Catégorie {cat_num}: {cat_name} ---**")
                    mask = preds['ML_Prediction_Num'] == cat_num
                    sample = preds[mask][['N° Facture', 'Jours_Retard', ' T.T.C ', 'amount_at_risk_prediction']].head(3)
                    if not sample.empty:
                        st.dataframe(sample, use_container_width=True)
                    else:
                        st.write("Aucun échantillon pour cette catégorie.")

                # Statistiques moyennes par catégorie
                st.subheader("🔍 Statistiques moyennes par catégorie :")
                num_stats = {}
                for cat, group in preds.groupby('ML_Prediction'):
                    num_stats[cat] = {
                        "Montant moyen (€)": group[' T.T.C '].mean() if ' T.T.C ' in group else None,
                        "Montant max (€)": group[' T.T.C '].max() if ' T.T.C ' in group else None,
                        "Retard moyen (jours)": group['Jours_Retard'].mean() if 'Jours_Retard' in group else None,
                        "Nb factures": len(group)
                    }
                st.write(pd.DataFrame(num_stats).T)

                # Clients à haut risque (catégorie "Retard Exagere (ML)")
                st.subheader("🔴 Clients à haut risque (Retard exagéré) :")
                high_risk_mask = (preds['ML_Prediction_Num'] == 2)
                if high_risk_mask.any():
                    high_risk_cols = [
                        'Code Client', 'Client', 'N° Facture', ' T.T.C ',
                        'Jours_Retard', 'amount_at_risk_prediction'
                    ]
                    high_risk_exist = [c for c in high_risk_cols if c in preds.columns]
                    st.dataframe(preds[high_risk_mask][high_risk_exist].sort_values('amount_at_risk_prediction', ascending=False).head(10), use_container_width=True)
                else:
                    st.write("Aucun client à haut risque détecté.")

                # Téléchargement
                towrite = io.BytesIO()
                preds.to_excel(towrite, index=False, engine="openpyxl")
                towrite.seek(0)
                st.download_button(
                    label="Télécharger les prédictions",
                    data=towrite,
                    file_name="predictions_retards_paiement.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error("La prédiction a retourné un dataframe vide.")
    else:
        st.warning("Merci d’importer un fichier dans l’onglet Vue d’ensemble.")

# ======================= PAGE 4 : AI Insights & Relances =======================
elif page == "AI Insights & Relances":
    st.header("AI Insights & Générateur de relances clients")
    df = st.session_state.get("df_processed")
    preds = st.session_state.get("ml_preds")
    if df is not None and preds is not None and not preds.empty:
        try:
            ai_assistant.display_chatbot_interface(df, preds)
        except Exception as e:
            st.error(f"Erreur lors du calcul des insights ou de la génération des mails : {e}")
    else:
        st.warning("Merci de générer les prédictions ML avant d’accéder à cet onglet.")

st.markdown(
    """
    <hr style="height:2px;border:none;color:#333;background-color:#e0e0e0;" />
    <div style='text-align: right; color: #888; font-size: 14px;'>
        Powered by Data Science | Stage 2025 | Interface type BCG
    </div>
    """, unsafe_allow_html=True
)