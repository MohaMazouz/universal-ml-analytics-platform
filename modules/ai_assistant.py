import streamlit as st
import pandas as pd
import unicodedata

def generate_mail(client_name, classif, client_df=None):
    """
    Génère un mail intelligent et personnalisé selon la classe de risque prédite et le contexte client.
    - client_df : DataFrame des factures du client (pour personnalisation avancée)
    """
    montant_total = None
    max_retard = None
    nb_factures = None
    oldest_due = None

    # Extraction d'infos personnalisées si possible
    if client_df is not None and not client_df.empty:
        if ' T.T.C ' in client_df.columns:
            montant_total = client_df[' T.T.C '].sum()
        if 'Jours_Retard' in client_df.columns:
            max_retard = client_df['Jours_Retard'].max()
        nb_factures = len(client_df)
        if 'échéance' in client_df.columns:
            oldest_due = client_df['échéance'].min()
            if hasattr(oldest_due, "strftime"):
                oldest_due = oldest_due.strftime('%d/%m/%Y')
            else:
                oldest_due = str(oldest_due)

    # Construction dynamique du contexte
    contexte = ""
    if montant_total is not None:
        contexte += f"\nMontant total en retard : {montant_total:,.2f} €"
    if max_retard is not None:
        contexte += f"\nRetard maximum constaté : {max_retard} jours"
    if nb_factures is not None:
        contexte += f"\nNombre de factures impayées : {nb_factures}"
    if oldest_due is not None:
        contexte += f"\nÉchéance la plus ancienne : {oldest_due}"

    # Ton et texte adaptés selon la catégorie
    classif_norm = unicodedata.normalize('NFKD', classif).encode('ASCII', 'ignore').decode('utf-8').lower()
    if "exagere" in classif_norm:
        objet = "Relance urgente – Retards de paiement"
        texte = (
            f"Cher {client_name},\n\n"
            "Nous constatons plusieurs retards de paiement importants sur votre compte."
            f"{contexte}\n"
            "Merci de régulariser votre situation dans les plus brefs délais afin d’éviter toute pénalité contractuelle. "
            "Notre équipe reste à votre disposition pour tout accompagnement ou pour étudier un éventuel échéancier.\n\n"
            "Cordialement,\nL'équipe Finance"
        )
    elif "retard" in classif_norm:
        objet = "Rappel – Factures en retard"
        texte = (
            f"Cher {client_name},\n\n"
            "Nous vous contactons car certaines de vos factures présentent un retard de paiement."
            f"{contexte}\n"
            "Nous vous remercions de bien vouloir procéder au règlement dans les meilleurs délais.\n\n"
            "Cordialement,\nL'équipe Finance"
        )
    else:
        objet = "Merci de votre ponctualité"
        texte = (
            f"Cher {client_name},\n\n"
            "Nous vous remercions pour votre régularité dans le règlement de vos factures."
            " Notre équipe reste à votre disposition pour toute question ou demande particulière.\n\n"
            "Cordialement,\nL'équipe Finance"
        )

    return f"Objet : {objet}\n\n{texte}"

def display_chatbot_interface(df, preds):
    st.subheader("💬 Assistant : Insights & Relances personnalisées")
    tab1, tab2 = st.tabs(["🔎 Insights simples", "📧 Génération de mails clients"])

    with tab1:
        st.markdown("Voici quelques insights calculés automatiquement sur vos données :")
        st.write(insight_summary(df, preds))

    with tab2:
        if "Client" in df.columns and "ML_Prediction" in preds.columns:
            merged = df.copy()
            merged["ML_Prediction"] = preds["ML_Prediction"]
            clients = merged["Client"].dropna().unique().tolist()
            selected_client = st.selectbox("Choisissez un client", clients)
            if selected_client:
                classif = (
                    merged.loc[merged["Client"] == selected_client, "ML_Prediction"]
                    .mode()
                    .values[0]
                )
                client_df = merged[merged["Client"] == selected_client]
                mail = generate_mail(selected_client, classif, client_df)
                st.text_area("Mail généré :", value=mail, height=280)
                st.download_button(
                    label="Télécharger le mail",
                    data=mail,
                    file_name=f"mail_relance_{selected_client}.txt"
                )
                st.caption("✉️ Ce mail est généré automatiquement, personnalisé selon la situation et sans IA lourde.")
        else:
            st.warning("Impossible de générer les mails : colonne 'Client' ou prédiction manquante.")

def insight_summary(df, preds):
    try:
        merged = df.copy()
        merged["ML_Prediction"] = preds["ML_Prediction"]
        total = len(merged)
        def normalize(s):
            if not isinstance(s, str):
                return ""
            return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
        ml_pred_norm = merged["ML_Prediction"].astype(str).apply(normalize)
        n_on_time = (ml_pred_norm == "aucun retard (ml)").sum()
        n_late = (ml_pred_norm == "est en retard (ml)").sum()
        n_very_late = (ml_pred_norm == "retard exagere (ml)").sum()
        risk_clients = (
            merged[ml_pred_norm == "retard exagere (ml)"]["Client"]
            .value_counts()
            .head(3)
            .to_dict()
        )
        insight = (
            f"- Nombre total de factures analysées : **{total}**\n"
            f"- Factures sans retard : **{n_on_time}**\n"
            f"- Factures en retard : **{n_late}**\n"
            f"- Factures en retard exagéré : **{n_very_late}**\n\n"
        )
        if risk_clients:
            insight += "Clients principaux en retard exagéré :\n"
            for cli, nb in risk_clients.items():
                insight += f"  - {cli} ({nb} factures)\n"
        else:
            insight += "Aucun client avec retard exagéré identifié."
        return insight
    except Exception as e:
        return f"Erreur dans la génération des insights : {e}"