import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner="Nettoyage/processing en cours…")
def clean_and_prepare(df_raw):
    df = df_raw.copy()
    # Nettoyage des montants (format FR)
    montant_cols = [' H.T ', ' T.V.A ', ' T.R ', ' T.T.C ', ' Caution ', ' Montant ']
    for col in montant_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(',', '.', regex=False)
                .str.replace(' ', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Conversion dates
    date_cols = ["Date d'Emission", 'échéance', 'Date Encaissement']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Suppression factures TTC <= 0
    if ' T.T.C ' in df.columns:
        df = df[df[' T.T.C '] > 0]
    # Logique date émission < échéance
    if "Date d'Emission" in df.columns and 'échéance' in df.columns:
        df = df[(df["Date d'Emission"].notna()) & (df["échéance"].notna())]
        df = df[df["Date d'Emission"] <= df["échéance"]]
    # Application règles métiers retards
    df = apply_payment_delay_rules(df)
    return df

def apply_payment_delay_rules(df):
    today = pd.Timestamp.now()
    def calculate_delay_status(row):
        echeance = row.get('échéance', pd.NaT)
        if pd.isna(echeance): return 'Échéance manquante', 0, 'Indéterminé'
        if row.get('Encaissement', '') == 'OUI' and pd.notna(row.get('Date Encaissement', pd.NaT)):
            days_late = (row['Date Encaissement'] - echeance).days
        else:
            days_late = (today - echeance).days
        if days_late < 0:
            return 'Payée avant échéance', days_late, 'Aucun retard'
        elif days_late <= 30:
            return 'Dans les délais', days_late, 'Pas de retard'
        elif days_late <= 60:
            return 'En retard', days_late, 'Retard'
        else:
            return 'Retard exagéré', days_late, 'Retard exagéré'
    df[['Statut_Détaillé', 'Jours_Retard', 'Catégorie_Règle']] = df.apply(
        calculate_delay_status, axis=1, result_type='expand'
    )
    df['Est_En_Retard'] = df['Catégorie_Règle'].apply(lambda x: 0 if x in ['Pas de retard', 'Aucun retard'] else 1)
    df['Est_Retard_Exagéré'] = (df['Catégorie_Règle'] == 'Retard exagéré').astype(int)
    return df

def analyze_cautions(df):
    # Analyse du respect des limites de caution selon vos règles
    if not all(col in df.columns for col in ['Code Client', 'Client', ' T.T.C ', ' Caution ', 'Est_En_Retard', 'N° Facture', 'Jours_Retard', 'Date d\'Emission', 'Catégorie_Règle']):
        return pd.DataFrame()
    client_encours = df.groupby(['Code Client', 'Client']).agg({
        ' T.T.C ': 'sum',
        ' Caution ': 'first',
        'Est_En_Retard': 'sum',
        'N° Facture': 'count',
        'Jours_Retard': 'mean',
        'Date d\'Emission': 'min',
        'Catégorie_Règle': lambda x: list(x.unique())
    }).round(2)
    client_encours.columns = ['Encours_Total', 'Caution', 'Nb_Retards',
                             'Nb_Factures', 'Retard_Moyen', 'Première_Facture', 'Types_Retard']
    client_encours['Caution_Disponible'] = client_encours['Caution'].fillna(0)
    client_encours['Dépassement'] = np.maximum(0, client_encours['Encours_Total'] - client_encours['Caution_Disponible'])
    def classify_client_situation(row):
        score_risque = 0
        if row['Retard_Moyen'] > 60:
            score_risque += 5
        elif row['Retard_Moyen'] > 30:
            score_risque += 3
        if row['Dépassement'] > 0:
            if row['Caution_Disponible'] > 0:
                score_risque += 4
            else:
                score_risque += 2
        if score_risque >= 7:
            return ' BLOCAGE IMMÉDIAT'
        elif score_risque >= 5:
            return ' HAUT RISQUE'
        elif score_risque >= 3:
            return ' SURVEILLANCE'
        else:
            return ' NORMAL'
    client_encours['Classification'] = client_encours.apply(classify_client_situation, axis=1)
    return client_encours

def identify_priority_actions(df, client_analysis):
    actions = {'urgentes': [], 'importantes': [], 'surveillance': []}
    # 1. Actions URGENTES (blocage immédiat)
    if not client_analysis.empty:
        blocage_clients = client_analysis[client_analysis['Classification'] == ' BLOCAGE IMMÉDIAT']
        for _, client in blocage_clients.iterrows():
            actions['urgentes'].append({
                'client': client.name[1],
                'code': client.name[0],
                'encours': client['Encours_Total'],
                'depassement': client['Dépassement'],
                'action': 'BLOQUER - Suspension livraisons immédiate'
            })
    # 2. Actions IMPORTANTES (retards exagérés sans blocage)
    if 'Catégorie_Règle' in df.columns and 'Encaissement' in df.columns:
        retards_exageres = df[
            (df['Catégorie_Règle'] == 'Retard exagéré') & (df['Encaissement'] != 'OUI')
        ]
        for _, facture in retards_exageres.head(10).iterrows():
            actions['importantes'].append({
                'facture': facture['N° Facture'],
                'client': facture['Client'],
                'montant': facture[' T.T.C '] if ' T.T.C ' in facture else None,
                'jours_retard': facture['Jours_Retard'],
                'action': 'RELANCE DIRECTE - Contact téléphonique direction'
            })
    # 3. SURVEILLANCE (approche des seuils)
    if 'Jours_Retard' in df.columns:
        approche_60j = df[
            (df['Jours_Retard'] >= 45) & (df['Jours_Retard'] <= 60) &
            (df['Encaissement'] != 'OUI')
        ]
        for _, facture in approche_60j.head(10).iterrows():
            actions['surveillance'].append({
                'facture': facture['N° Facture'],
                'client': facture['Client'],
                'montant': facture[' T.T.C '] if ' T.T.C ' in facture else None,
                'jours_retard': facture['Jours_Retard'],
                'action': 'PRÉVENTIF - Relance avant passage retard exagéré'
            })
    return actions

def generate_kpis(df):
    kpis = {}
    kpis['general'] = {
        'total_factures': len(df),
        'total_ttc': round(df[' T.T.C '].sum(), 2) if ' T.T.C ' in df.columns else None,
        'nb_clients': df['Code Client'].nunique() if 'Code Client' in df.columns else None,
        'montant_moyen_facture': round(df[' T.T.C '].mean(), 2) if ' T.T.C ' in df.columns else None
    }
    kpis['retards'] = {
        'factures_dans_delais': int((df['Catégorie_Règle'] == 'Pas de retard').sum()) if 'Catégorie_Règle' in df.columns else None,
        'factures_retard_simple': int((df['Catégorie_Règle'] == 'Retard').sum()) if 'Catégorie_Règle' in df.columns else None,
        'factures_retard_exagere': int((df['Catégorie_Règle'] == 'Retard exagéré').sum()) if 'Catégorie_Règle' in df.columns else None,
        'taux_retard_global': float(df['Est_En_Retard'].mean()*100) if 'Est_En_Retard' in df.columns else None
    }
    factures_impayees = df[df['Encaissement'] != 'OUI'] if 'Encaissement' in df.columns else df
    kpis['temporel'] = {
        'factures_impayees': len(factures_impayees),
        'montant_impaye': round(factures_impayees[' T.T.C '].sum(), 2) if ' T.T.C ' in factures_impayees.columns else None,
        'retard_moyen_jours': round(factures_impayees['Jours_Retard'].mean(), 1) if 'Jours_Retard' in factures_impayees.columns else None,
        'plus_ancien_impaye': float(factures_impayees['Jours_Retard'].max()) if 'Jours_Retard' in factures_impayees.columns and not factures_impayees.empty else None
    }
    return kpis, factures_impayees