import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def evolution_retard_moyen_et_taux(df):
    st.subheader("📈 Évolution du retard moyen et du taux de retard par mois")
    st.caption("Visualisez la courbe du retard moyen (en jours) et du taux de retard (%), global ou par client.")

    # Sélecteur client
    clients = ["Tous"] + sorted(df["Client"].dropna().unique())
    selected_client = st.selectbox("Sélectionnez un client pour filtrer :", clients, key="client_evo_retard")

    # Sous-ensemble selon le client choisi
    if selected_client != "Tous":
        df_ = df[df["Client"] == selected_client].copy()
    else:
        df_ = df.copy()

    # Vérification des colonnes nécessaires
    required_cols = ["Date d'Emission", "Jours_Retard", "Est_En_Retard"]
    if not all(col in df_.columns for col in required_cols):
        st.warning("Colonnes nécessaires manquantes.")
        return

    # Préparation des données temporelles
    df_ = df_.dropna(subset=["Date d'Emission"])
    df_["Mois_Emission"] = pd.to_datetime(df_["Date d'Emission"]).dt.to_period('M').astype(str)

    # Agrégation par mois
    grouped = df_.groupby("Mois_Emission").agg(
        Retard_moyen=("Jours_Retard", "mean"),
        Taux_retard=("Est_En_Retard", "mean"),
        Nb_factures=("Jours_Retard", "count")
    ).reset_index()
    grouped["Taux_retard"] = grouped["Taux_retard"] * 100  # passage en %

    # Plotly : double axe Y
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grouped['Mois_Emission'],
        y=grouped['Retard_moyen'],
        name="Retard moyen (jours)",
        yaxis="y1",
        mode='lines+markers',
        line=dict(color="#FF5733", width=3),
        marker=dict(symbol="circle", size=7)
    ))

    fig.add_trace(go.Scatter(
        x=grouped['Mois_Emission'],
        y=grouped['Taux_retard'],
        name="Taux de retard (%)",
        yaxis="y2",
        mode='lines+markers',
        line=dict(color="#0074D9", width=3, dash='dot'),
        marker=dict(symbol="diamond", size=8)
    ))

    fig.update_layout(
        title=f"Évolution Retard moyen et Taux de retard {'- ' + selected_client if selected_client != 'Tous' else '(global)'}",
        xaxis=dict(title="Mois"),
        yaxis=dict(title="Retard moyen (jours)", side="left", showgrid=False, color="#FF5733"),
        yaxis2=dict(title="Taux de retard (%)", overlaying="y", side="right", range=[0, 100], color="#0074D9"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=20, r=40, t=60, b=40),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistiques complémentaires
    st.caption(f"Nombre de factures par mois ({'client : ' + selected_client if selected_client != 'Tous' else 'global'}) :")
    st.dataframe(grouped[["Mois_Emission", "Nb_factures"]])


def display_eda(df):
    st.title("Analyse exploratoire des retards de paiement")
    st.caption("Visualisations dynamiques pour explorer votre portefeuille de factures.")

    # --------- Option de filtrage : toutes factures ou seulement les impayées -------------
    show_only_unpaid = st.radio(
        "Afficher les retards sur :",
        options=["Toutes les factures", "Seulement les impayées"],
        index=0,
        horizontal=True
    )
    if 'Encaissement' in df.columns and show_only_unpaid == "Seulement les impayées":
        data_plot = df[df['Encaissement'] != 'OUI']
    else:
        data_plot = df.copy()

    # --------- Diagnostic : diversité des catégories (avant et après filtre) -------------
    st.markdown("**Diagnostic des catégories de retard**")
    st.write("Catégories présentes dans TOUTES les factures :", df['Catégorie_Règle'].value_counts())
    st.write("Catégories présentes dans ce sous-ensemble :", data_plot['Catégorie_Règle'].value_counts())
    if data_plot['Catégorie_Règle'].nunique() == 1:
        st.warning(f"Attention : une seule catégorie détectée dans ce sous-ensemble ({data_plot['Catégorie_Règle'].unique()[0]}).")

    st.divider()

    # --------- Distribution des jours de retard (toutes catégories) -------------
    st.subheader("📊 Distribution des jours de retard (toutes catégories)")
    st.caption("Histogramme interactif montrant la répartition des jours de retard par catégorie de retard sur l'ensemble des factures sélectionnées.")
    if "Jours_Retard" in data_plot.columns and "Catégorie_Règle" in data_plot.columns and not data_plot.empty:
        fig = px.histogram(
            data_plot,
            x="Jours_Retard",
            color="Catégorie_Règle",
            nbins=50,
            title="Distribution interactive des jours de retard",
            labels={"Jours_Retard": "Jours de retard", "count": "Nombre de factures"},
            marginal="box",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonnes nécessaires non trouvées ou sous-ensemble vide.")

    st.divider()

    # --------- Top 10 clients par encours TTC --------------
    st.subheader("🏆 Top 10 clients par encours (Total TTC)")
    st.caption("Bar chart présentant les 10 clients ayant le plus grand encours TTC.")
    for col in ["Code Client", "Client", " T.T.C "]:
        if col not in df.columns:
            st.info("Pas assez d'informations clients pour ce graphique.")
            return
    top_clients = df.groupby(['Code Client', 'Client'])[' T.T.C '].sum().nlargest(10).reset_index()
    if not top_clients.empty:
        fig2 = px.bar(
            top_clients,
            x='Client',
            y=' T.T.C ',
            title="Top 10 clients par encours TTC",
            labels={'Client': 'Client', ' T.T.C ': 'Total TTC (€)'},
            color=' T.T.C ',
            color_continuous_scale='Blues'
        )
        fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Aucune donnée client à afficher.")

    st.divider()

    # --------- Répartition des clients par niveau de risque (pie chart) -------------
    if 'Classification' in df.columns and 'Code Client' in df.columns:
        st.subheader("🧩 Répartition des clients par niveau de risque")
        st.caption("Pie chart des clients classés selon leur niveau de risque (normal, surveillance, haut risque, blocage).")
        pie_data = df.groupby('Classification')['Code Client'].nunique().reset_index()
        pie_data = pie_data.rename(columns={'Code Client': 'Nb Clients'})
        fig_pie = px.pie(
            pie_data, names='Classification', values='Nb Clients',
            title="Distribution des Clients par Risque",
            color='Classification',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hole=0.3
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text='Niveau de risque')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("🔎 Interprétez ce graphique pour cibler vos actions de relance et de gestion du risque.")

    st.divider()

    # --------- Evolution du retard moyen & taux de retard (nouveau) -------------
    if all(col in df.columns for col in ["Date d'Emission", "Jours_Retard", "Est_En_Retard", "Client"]):
        evolution_retard_moyen_et_taux(df)

    st.divider()

    # --------- Cascade de recouvrement (waterfall) ----------
    if all(col in df.columns for col in [" T.T.C ", "Encaissement"]):
        st.subheader("💧 Cascade de recouvrement")
        st.caption("Waterfall chart illustrant la décomposition du total facturé en parts payées et impayées.")
        total = df[" T.T.C "].sum()
        paye = df[df['Encaissement'] == 'OUI'][" T.T.C "].sum()
        impaye = total - paye
        fig_waterfall = go.Figure(go.Waterfall(
            x=["Total Facturé", "Payé", "Impayé"],
            y=[total, -paye, -impaye],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            textposition="outside"
        ))
        fig_waterfall.update_layout(
            title="Cascade de Recouvrement",
            margin=dict(l=20, r=20, t=40, b=20),
            waterfallgap=0.4
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.divider()

    # --------- Taux de retard (gauge) --------------
    if 'Est_En_Retard' in df.columns:
        st.subheader("⏰ Taux global de retard")
        st.caption("Indicateur du pourcentage de factures en retard sur l'ensemble du portefeuille.")
        taux = df['Est_En_Retard'].mean() * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=taux,
            title={'text': "Taux de Retard (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF4136"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ECC40"},
                    {'range': [40, 75], 'color': "#FFDC00"},
                    {'range': [75, 100], 'color': "#FF4136"}
                ],
            }
        ))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # --------- Analyse de risque dynamique -----------------
    seuil_haut = 75
    seuil_moyen = 40
    seuil_min = 5
    if "N° Facture" in df.columns and "Est_En_Retard" in df.columns and "Client" in df.columns:
        st.subheader("📋 Analyse dynamique du risque client")
        st.caption(f"Profilage automatique : clients avec au moins {seuil_min} factures, classement haut risque ≥{seuil_haut}%, moyen risque ≥{seuil_moyen}%")
        client_profiles = df.groupby("Client").agg(
            Nb_factures_total=('N° Facture', 'count'),
            Nb_factures_retard=('Est_En_Retard', 'sum'),
            Retard_moyen=('Jours_Retard', 'mean'),
            CA_total=(' T.T.C ', 'sum')
        ).reset_index()
        client_profiles['Taux_retard'] = 100 * client_profiles['Nb_factures_retard'] / client_profiles['Nb_factures_total']
        filtered = client_profiles[client_profiles['Nb_factures_total'] >= seuil_min]
        st.markdown("**Clients analysés (filtrage automatique)**")
        with st.expander("Tableau détaillé des profils clients (filtrage dynamique)", expanded=False):
            st.dataframe(filtered)

        st.markdown(f"**Clients haut risque (≥{seuil_haut}% en retard)**")
        st.dataframe(filtered[filtered['Taux_retard'] >= seuil_haut])

        st.markdown(f"**Clients risque modéré ({seuil_moyen}%–{seuil_haut-1}% en retard)**")
        st.dataframe(filtered[(filtered['Taux_retard'] >= seuil_moyen) & (filtered['Taux_retard'] < seuil_haut)])

        st.markdown(f"**Clients faible risque (<{seuil_moyen}% en retard)**")
        st.dataframe(filtered[filtered['Taux_retard'] < seuil_moyen])

    st.markdown("---")
    st.caption("💡 Astuce UX : Tous les graphiques sont affichés verticalement et séparés pour une navigation fluide et sans scroll horizontal.")