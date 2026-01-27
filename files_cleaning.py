import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score

class FilesCleaning():

    def __init__(self):
        self.extrait_eval_df = pd.read_csv("./raw_data/extrait_eval.csv")
        self.extrait_sirh_df = pd.read_csv("./raw_data/extrait_sirh.csv")
        self.extrait_sondage_df = pd.read_csv("./raw_data/extrait_sondage.csv")
        self.df = None
        self.y = None
        self.preprocess = None
        
        

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
                column = df[col].value_counts(dropna=False, normalize=True).round(3).reset_index()
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
                print(dist[["valeur", "pourcentage (%)"]].head(15).to_string(index=False))
        
        # Fusion des csv
        self.extrait_eval_df["eval_number"] = self.extrait_eval_df["eval_number"].astype(str).str.extract(r"(\d+)").astype("Int64")

        mask_equal = (
            (self.extrait_sirh_df["id_employee"] == self.extrait_eval_df["eval_number"]) &
            (self.extrait_sirh_df["id_employee"] == self.extrait_sondage_df["code_sondage"])
        )

        print(f"\n---- Les colonnes suspectées d'être l'id des salariés sont elles égales? :\n, {mask_equal.value_counts()} \
                sur {len(self.extrait_sirh_df)}, {len(self.extrait_eval_df)}, {len(self.extrait_sondage_df)} lignes")

        self.df = self.extrait_sirh_df.merge(
            self.extrait_eval_df,
            left_on="id_employee",
            right_on="eval_number",
            how="inner"
        ).merge(
            self.extrait_sondage_df,
            left_on="id_employee",
            right_on="code_sondage",
            how="inner"
        )

        # suppression des colonnes id inutiles
        self.df = self.df.drop(columns=["eval_number", "code_sondage"])
        # suppression des features inutiles
        self.df = self.df.drop(columns=["ayant_enfants", "nombre_employee_sous_responsabilite", "nombre_heures_travailless"])

        self.df.to_csv(
            "extrait_rh.csv",
            index=False,
            encoding="utf-8"
        )
        
        # graphiques
        target = "a_quitte_l_entreprise"
        print("\n---- Description de la cible:\n", self.df[target].describe())

        years_features = [
            col for col in self.df.columns
            if col != target and ("annees" in col.lower() or "annes" in col.lower() or "annee" in col.lower())
        ]

        num_features = [
            col for col in self.df.columns
            if col != target and ("note" in col.lower() or "satisfaction" in col.lower() or "niveau" in col.lower())
        ]

        features = {
            "années": years_features, 
            "data": num_features
            }

        df_oui = self.df[self.df[target] == "Oui"]

        output_dir = "Graph"
        os.makedirs(output_dir, exist_ok=True)

        for name, feature in features.items():

            df_long = df_oui.melt(
                value_vars=feature,
                var_name="feature",
                value_name=f"{name}"
            )

            plt.figure(figsize=(12, 8))
            sns.boxplot(
                data=df_long,
                x="feature",
                y=f"{name}"
            )

            plt.title(f"Distribution des {name} – employés ayant quitté l'entreprise")
            plt.xlabel("Features")
            plt.ylabel(f"{name}")
            plt.xticks(rotation=10)
            plt.savefig(f"{output_dir}/{name}_boxplot.png", dpi=500, bbox_inches="tight")
    
    def data_cleaning(self):

        df_num = self.df.select_dtypes(include='number')
        # corr_spearman = df_num.corr(method='spearman')
        # plt.figure(figsize=(30, 10))
        # sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", center=0)
        # plt.xticks(rotation=40, ha="right")
        # plt.title("Spearman correlation")
        # plt.savefig("Graph/spearman_corr.png", dpi=500, bbox_inches="tight")

        sns.pairplot(
        self.df[["annees_dans_l_entreprise", "annees_dans_le_poste_actuel", "annes_sous_responsable_actuel"]],
        diag_kind="hist",
        plot_kws={"alpha": 0.6, "s": 30})

        plt.savefig(f"Graph/relation_pairplot.png", dpi=500, bbox_inches="tight")

        self.df = self.df.drop(columns=["niveau_hierarchique_poste", "annees_dans_le_poste_actuel", "annes_sous_responsable_actuel"])

        self.df["augementation_salaire_precedente"] = (
            self.df["augementation_salaire_precedente"]
            .str.strip()
            .str.replace("%", "", regex=False)
            .astype("int64")
        )

        self.y = self.df["a_quitte_l_entreprise"].map({"Non": 0, "Oui": 1})

        binary_features = ["genre", "heure_supplementaires"]
        nominal_features = ["frequence_deplacement", "domaine_etude", "poste", "departement", "statut_marital", "genre"]
        numerical_features = self.df.select_dtypes(include="number").columns

        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("bin", OrdinalEncoder(), binary_features),
                ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_features)
            ],
            remainder="passthrough"
        )

        self.X = self.df.drop(columns=["a_quitte_l_entreprise"])

    def first_modelisation(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
        )

        models = {
            'Dummy': Pipeline([('preprocess', self.preprocess),
                               ('model', DummyClassifier(strategy='most_frequent', random_state=42))]),
            'Logistic': Pipeline([('preprocess', self.preprocess),
                                  ('model', LogisticRegression(max_iter=1000, random_state=42))]),
            'RandomForest': Pipeline([('preprocess', self.preprocess),
                                      ('model', RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42))])
        }

        def evaluate_model(name, model, X_train, y_train, X_test, y_test):
            print(f"\n=== Modèle : {name} ===")
            # Entrainement du modèle
            model.fit(X_train, y_train)
            # Prediction de l'entrainement
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # Pour chaque jeu train/test y_true pour jeu train, y_pred pour jeu test
            for dataset, y_true, y_pred in [('Train', y_train, y_train_pred),
                                            ('Test', y_test, y_test_pred)]:
                print(f"\n--- {dataset} ---")
                cm = confusion_matrix(y_true, y_pred)
                print("Matrice de confusion :\n", cm)
                print("\nClassification report :\n", classification_report(y_true, y_pred, digits=3))
                print(f"Precision: {precision_score(y_true, y_pred):.3f} | Recall: {recall_score(y_true, y_pred):.3f}")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Prédit")
                plt.ylabel("Réel")
                plt.title(f"{name} - {dataset}")
                plt.show()

        for name, model in models.items():
            evaluate_model(name, model, X_train, y_train, X_test, y_test)


run = FilesCleaning()
run.doc_analysis()
run.data_cleaning()
run.first_modelisation()
        