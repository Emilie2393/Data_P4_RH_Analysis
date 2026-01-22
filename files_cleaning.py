import pandas as pd

class FilesCleaning():

    def __init__(self):
        self.extrait_eval_df = pd.read_csv("./raw_data/extrait_eval.csv")
        self.extrait_sirh_df = pd.read_csv("./raw_data/extrait_sirh.csv")
        self.extrait_sondage_df = pd.read_csv("./raw_data/extrait_sondage.csv")
        self.df = None
        

    def doc_analysis(self):

        self.df = {
            "eval": self.extrait_eval_df,
            "sirh": self.extrait_sirh_df,
            "sondage": self.extrait_sondage_df,
        }

        all_csv = {}

        for name, df in self.df.items():
            results = {}
            for col in df.columns:
                # Calcul des pourcentages de chaque valeur
                column = df[col].value_counts(dropna=False, normalize=True).round(2).reset_index()
                column.columns = ["valeur", "pourcentage (%)"]

                # Ajouter type et % null
                column["type"] = str(df[col].dtype)
                column["null_%"] = round(df[col].isna().mean() * 100, 2)

                results[col] = column

            all_csv[name] = results

        # Impression simple et propre
        for file, res in all_csv.items():
            print(f"\n----- Fichier : {file} -----")
            for col, dist in res.items():
                print(f"\nColonne: {col} - Type: {dist['type'].iloc[0]} - % null: {dist['null_%'].iloc[0]}")
                print(dist[["valeur", "pourcentage (%)"]].to_string(index=False))


run = FilesCleaning()
run.doc_analysis()
        