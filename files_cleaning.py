import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, precision_recall_curve
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import shap

class FilesCleaning():

    def __init__(self):
        self.extrait_eval_df = pd.read_csv("./raw_data/extrait_eval.csv")
        self.extrait_sirh_df = pd.read_csv("./raw_data/extrait_sirh.csv")
        self.extrait_sondage_df = pd.read_csv("./raw_data/extrait_sondage.csv")
        self.df = None
        self.y = None
        self.preprocess = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.pipe = None
        
        

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
            plt.close()
    
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
        plt.close()

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
                print("\nClassification report :\n", classification_report(y_true, y_pred, digits=3, zero_division=0))
                print(f"Precision: {precision_score(y_true, y_pred):.3f} | Recall: {recall_score(y_true, y_pred):.3f}")
                if "Dummy" not in name:
                    plt.figure(figsize=(5, 4)) 
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.xlabel("Prédit")
                    plt.ylabel("Réel")
                    plt.title(f"{name} - {dataset}")
                    plt.savefig(f"Graph/{name}_{dataset}_confusion_matrix.png", dpi=500, bbox_inches="tight")
                    plt.close()

        for name, model in models.items():
            evaluate_model(name, model, X_train, y_train, X_test, y_test)
        
    def classification_models(self, n_estimators, graph, new_features=None, class_weight=None, resample=False):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        scoring = {
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1"
                }
 
        if resample:
            pipe = Pipeline([
                        ("preprocessing", self.preprocess),
                        ("resample", SMOTE(sampling_strategy=0.6, random_state=42)),
                        ("model", RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, max_features='sqrt', random_state=42))
                    ])
        else:
            pipe = Pipeline([
                    ("preprocessing", self.preprocess),
                    ("model", RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, max_features='sqrt', random_state=42))
                ])

        cv_results = cross_validate(
                    estimator=pipe,
                    X=X_train,
                    y=y_train,
                    cv=5,
                    scoring=scoring,
                    return_train_score=True,
                    error_score="raise"
                )

        # Résultats de la cv
        train_scores = {k: v for k, v in cv_results.items() if "train" in k}
        test_scores  = {k: v for k, v in cv_results.items() if "test" in k}

        print("\n---- Train initial")
        for k, v in train_scores.items():
            print(f"{k}: {v.mean():.3f} écart-type {v.std():.3f}")

        print("\n---- Test initial")
        for k, v in test_scores.items():
            print(f"{k}: {v.mean():.3f} écart-type {v.std():.3f}")

        pipe.fit(X_train, y_train)
        self.pipe = pipe
        y_test_proba = pipe.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        # Tracé
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, marker='.', label="Précision-Rappel")
        plt.xlabel("Rappel (part des départs correctement détectés)")
        plt.ylabel("Précision (part des alertes réellement fondées)")
        plt.title("Courbe Précision-Rappel")
        plt.grid(True)
        # plt.legend()
        plt.savefig(f"Graph/Courbe_{graph}_précision_rappel.png", dpi=500, bbox_inches="tight")
        plt.close()

        target_recall = 0.90
        min_precision = 0.40

        # seuil precision supérieur à 0.40
        valid = precision[:-1] >= min_precision

        if valid.sum() == 0:
            raise ValueError("Aucun seuil ne respecte la précision minimale")

        # récupère les seuils valides
        filtered_recalls = recall[:-1][valid]
        filtered_thresholds = thresholds[valid]
        # récupère le rappel le plus proche de ma target_recall
        idx = np.argmin(np.abs(filtered_recalls - target_recall))
        # impute ce seuil à ma variable
        threshold = filtered_thresholds[idx]

        if new_features:
            print("---- Features ajoutés:\n", new_features)

        print(f"\nSeuil choisi : {threshold:.3f}")
        print(f"Recall obtenu avec ce seuil : {filtered_recalls[idx]:.3f}")
        print(f"Precision associée avec ce seuil : {precision[:-1][valid][idx]:.3f}")

        # prédiction finale du modèle en fonction du seuil appliqué
        y_pred = (y_test_proba >= threshold).astype(int)
        # Df des valeurs y comparées aux prédictions et probabilités associées
        df_decision = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_test_proba
        })

        df_decision["prediction_type"] = np.select(
            [
                (df_decision.y_true == 1) & (df_decision.y_pred == 1),
                (df_decision.y_true == 0) & (df_decision.y_pred == 1),
                (df_decision.y_true == 0) & (df_decision.y_pred == 0),
                (df_decision.y_true == 1) & (df_decision.y_pred == 0),
            ],
            ["True Positive", "False Positive", "True Negative", "False Negative"],
            default="Unknown"
        )
        print(df_decision["prediction_type"].value_counts(dropna=False))

        g = sns.displot(
            data=df_decision,
            x="y_proba",
            hue="prediction_type",
            kind="kde",
            fill=True,
            alpha=0.4,
            common_norm=False
        )

        for ax in g.axes.flat:
            ax.axvline(threshold, color="black", linestyle="--")
        g.set_axis_labels("Probabilité prédite de départ", "Densité")
        g.fig.suptitle("Distribution des probabilités par type de prédiction", y=0.98)
        plt.savefig(f"Graph/Proba_{graph}.png", dpi=500, bbox_inches="tight")
        plt.show()

    
    def classification_test(self):

        self.classification_models(200, "sansfeature")

        self.X["carriere_stagnante"] = (
            (self.df["annees_depuis_la_derniere_promotion"] >= 3) &
            (self.df["augementation_salaire_precedente"] < 12)
        ).astype(int)

        self.classification_models(200, "feat_cweight", ["carriere_stagnante"], class_weight={0:1, 1:2})

        self.classification_models(200,"feat_cweight_resample", ["carriere_stagnante"], class_weight={0:1, 1:2}, resample=True) 
    
    def features_results(self, pipe, X_train, X_test, y_train, y_test):

        model = pipe.named_steps["model"]

        # récupération des données transformées
        X_train_transformed = pipe.named_steps["preprocessing"].transform(X_train)
        X_test_transformed  = pipe.named_steps["preprocessing"].transform(X_test)
        feature_names = pipe.named_steps["preprocessing"].get_feature_names_out()
        fi_native = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
        print("\n--- Feature importance native (arbre)")
        print(fi_native.head(10))

        perm = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=20,
            random_state=42,
            scoring="recall"
        )

        fi_perm = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
        print("\n--- Permutation Importance")
        print(fi_perm.head(10))

    def run_script(self):

        self.doc_analysis()
        self.data_cleaning()
        self.first_modelisation()
        self.classification_test()
        self.features_results(self.pipe, self.X_train, self.X_test, self.y_train, self.y_test)


RH_prediction_analysis = FilesCleaning()
RH_prediction_analysis.run_script()


