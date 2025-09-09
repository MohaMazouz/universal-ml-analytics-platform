import pandas as pd

EXCEL_PATH = "data/BD_avec_regles_paiement_latest.xlsx"
df = pd.read_excel(EXCEL_PATH)

def check_categorie_regle(df):
    print("\n=== Vérification de la colonne Catégorie_Règle ===")
    if 'Catégorie_Règle' not in df.columns:
        print("❌ La colonne 'Catégorie_Règle' n'existe pas dans le dataframe.")
        return
    print("Valeurs uniques dans 'Catégorie_Règle' :", df['Catégorie_Règle'].unique())
    print("\nEffectifs par catégorie :\n", df['Catégorie_Règle'].value_counts())
    print("\nExemple d'échantillon :")
    print(df[['Catégorie_Règle', 'Jours_Retard', 'Encaissement']].head(15))

if __name__ == "__main__":
    check_categorie_regle(df)