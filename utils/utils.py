import pandas as pd

COLUMN_MAPPING = {
    "Client Name": "Client",
    "Nom Client": "Client",
    "Amount TTC": " T.T.C ",
    "Montant TTC": " T.T.C ",
    "Days Late": "Jours_Retard",
    "Retard (jours)": "Jours_Retard",
    "Due Date": "échéance",
    "Date d'échéance": "échéance",
    # ... ajoute d'autres variantes ici au besoin
}

def harmonize_columns(df, mapping):
    # Remplace les noms connus, garde les autres inchangés
    cols = [mapping.get(col, col) for col in df.columns]
    df.columns = cols
    return df