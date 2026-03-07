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

    def _describe_dataframes(self):
        """Affiche la distribution, le type et le taux de nullité de chaque colonne par source."""
        sources = {
            "eval": self.extrait_eval_df.copy(),
            "sirh": self.extrait_sirh_df.copy(),
            "sondage": self.extrait_sondage_df.copy(),
        }
        for file, df in sources.items():
            print(f"\n----- Fichier : {file} -----")
            for col in df.columns:
                column = df[col].value_counts(dropna=False, normalize=True).round(3).reset_index()
                column.columns = ["valeur", "pourcentage (%)"]
                column["type"] = str(df[col].dtype)
                column["null_%"] = round(df[col].isna().mean() * 100, 2)
                print(f"\nColonne: {col} - Type: {column['type'].iloc[0]} - % null: {column['null_%'].iloc[0]}")
                print(column[["valeur", "pourcentage (%)"]].head(15).to_string(index=False))

    def _merge_sources(self):
        """
        Fusionne les trois sources sur l'identifiant employé et exporte `extrait_rh.csv`.
        Modifie `self.df` avec le DataFrame fusionné et nettoyé.
        """
        eval_df = self.extrait_eval_df.copy()
        extracted = eval_df["eval_number"].astype(str).str.extract(r"(\d+)")
        if extracted.isna().all().all():
            raise ValueError("Aucune valeur numérique extractible dans 'eval_number'.")
        eval_df["eval_number"] = extracted.astype("Int64")

        mask_equal = (
            (self.extrait_sirh_df["id_employee"] == eval_df["eval_number"]) &
            (self.extrait_sirh_df["id_employee"] == self.extrait_sondage_df["code_sondage"])
        )
        print(
            f"\n---- Colonnes suspectées d'être l'id salarié — égalité :\n"
            f"{mask_equal.value_counts()} "
            f"sur {len(self.extrait_sirh_df)}, {len(eval_df)}, {len(self.extrait_sondage_df)} lignes"
        )

        self.df = (
            self.extrait_sirh_df
            .merge(eval_df, left_on="id_employee", right_on="eval_number", how="inner")
            .merge(self.extrait_sondage_df, left_on="id_employee", right_on="code_sondage", how="inner")
            .drop(columns=["eval_number", "code_sondage"])
            .drop(columns=["ayant_enfants", "nombre_employee_sous_responsabilite", "nombre_heures_travailless"])
        )

        self.df.to_csv("extrait_rh.csv", index=False, encoding="utf-8")

    def _plot_departures(self):
        """Génère et sauvegarde les boxplots des features pour les employés ayant quitté l'entreprise."""
        TARGET = "a_quitte_l_entreprise"
        OUTPUT_DIR = "Graph"

        print("\n---- Description de la cible:\n", self.df[TARGET].describe())

        years_features = [col for col in self.df.columns if col != TARGET and any(k in col.lower() for k in ("annees", "annes", "annee"))]
        num_features   = [col for col in self.df.columns if col != TARGET and any(k in col.lower() for k in ("note", "satisfaction", "niveau"))]
        features = {"années": years_features, "data": num_features}
        self.departs = self.df[self.df[TARGET] == "Oui"]
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for name, feature in features.items():
            df_long = self.departs.melt(value_vars=feature, var_name="feature", value_name=name)
            print(df_long.head(20))

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=df_long, x="feature", y=name, ax=ax)
            ax.set_title(f"Distribution des {name} – employés ayant quitté l'entreprise")
            ax.set_xlabel("Features")
            ax.set_ylabel(name)
            plt.xticks(rotation=10)
            fig.savefig(f"{OUTPUT_DIR}/{name}_boxplot.png", dpi=500, bbox_inches="tight")
            plt.close(fig)


    def doc_analysis(self):
        """
        Analyse, fusionne et visualise les données RH des trois sources (eval, sirh, sondage).

        - Calcule la distribution, le type et le taux de nullité de chaque colonne.
        - Vérifie la cohérence des clés de jointure entre les sources.
        - Fusionne les DataFrames sur l'identifiant employé (inner join) et exporte `extrait_rh.csv`.
        - Génère des boxplots (années, notes/satisfaction) pour les employés ayant quitté l'entreprise.

        Side effects:
            Modifie `self.df` | Crée `extrait_rh.csv` | Crée `Graph/*.png`
        """
        self._describe_dataframes()
        self._merge_sources()
        self._plot_departures()
    
    def data_cleaning(self):
        """
        Nettoie et prépare les données pour la modélisation.

        Étapes : corrélation de Spearman, suppression des colonnes redondantes,
        nettoyage de 'augmentation_salaire_precedente', encodage de la cible,
        et définition du préprocesseur (StandardScaler, OrdinalEncoder, OneHotEncoder).

        Returns:
            tuple: X (pd.DataFrame) features, y (pd.Series) cible encodée (0=Non, 1=Oui).
        """

        df_num = self.df.select_dtypes(include='number')
        corr_spearman = df_num.corr(method='spearman')
        plt.figure(figsize=(30, 10))
        sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", center=0)
        plt.xticks(rotation=40, ha="right")
        plt.title("Spearman correlation")
        plt.savefig("Graph/Spearman_corr.png", dpi=500, bbox_inches="tight")
        plt.close()

        sns.pairplot(
            self.df[["annees_dans_l_entreprise", "annees_dans_le_poste_actuel", "annes_sous_responsable_actuel"]],
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 30})
        plt.savefig(f"Graph/Relations_pairplot.png", dpi=500, bbox_inches="tight")
        plt.close()
        # Nettoyage des colonnes
        self.df = self.df.drop(columns=["niveau_hierarchique_poste", "annees_dans_le_poste_actuel", "annes_sous_responsable_actuel"])
        self.df["augementation_salaire_precedente"] = (
            self.df["augementation_salaire_precedente"]
            .str.strip()
            .str.replace("%", "", regex=False)
            .astype("int64")
        )
        # Transformation binaire
        y = self.df["a_quitte_l_entreprise"].map({"Non": 0, "Oui": 1})
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

        self.X  = self.df.drop(columns=["a_quitte_l_entreprise"])
        self.y = y

        return self.X.copy(), y

    def first_modelisation(self, X, y):
        """
        Compare trois modèles baseline : DummyClassifier, LogisticRegression et RandomForest.
        Affiche matrices de confusion, classification report et scores précision/recall
        sur les jeux train et test.

        Args:
            X : Features (pd.DataFrame).
            y : Cible encodée (pd.Series, 0=Non / 1=Oui).

        Side effects:
            Crée Graph/{model}_{Train|Test}_confusion_matrix.png (x4, Dummy exclu)
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
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
        
    def classification_models(self, n_estimators, graph, class_weight=None, resample=False):
        """
        Entraîne un RandomForest, sélectionne le seuil optimal (recall≥0.90, precision≥0.40)
        et génère les courbes précision-rappel et distributions de probabilités.

        Args:
            n_estimators : nombre d'arbres | graph : suffixe des PNG
            class_weight : pondération des classes | resample : active SMOTE si True
        
        Side effects:
            Modifie self.X_train, self.X_test, self.y_train, self.y_test
            Crée Graph/Courbe_{graph}_précision_rappel.png | Crée Graph/Proba_{graph}.png

        Returns:
            Pipeline entraîné.
        """

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
        # Entrainement et init des variables
        pipe.fit(X_train, y_train)
        y_test_proba = pipe.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        # Tracé
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, marker='.', label="Précision-Rappel")
        plt.xlabel("Rappel (part des départs correctement détectés)")
        plt.ylabel("Précision (part des alertes réellement fondées)")
        plt.title("Courbe Précision-Rappel")
        plt.grid(True)
        plt.savefig(f"Graph/Courbe_{graph}_précision_rappel.png", dpi=500, bbox_inches="tight")
        plt.close()
        # Cibles à atteindre
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
        print(f"----- Configuration : {graph}")
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
        g.figure.suptitle("Distribution des probabilités par type de prédiction", y=0.98)
        plt.savefig(f"Graph/Proba_{graph}.png", dpi=500, bbox_inches="tight")
        plt.close()

        return pipe
    
    def classification_test(self):
        """
        Compare trois configurations de RandomForest : baseline, class_weight, et SMOTE.
        Ajoute la feature 'carriere_stagnante' entre le baseline et les modèles pondérés.

        Returns:
            Pipeline de la configuration finale (class_weight + SMOTE).
        """

        self.classification_models(200, "sansfeature")

        self.X["carriere_stagnante"] = (
            (self.df["annees_depuis_la_derniere_promotion"] >= 3) &
            (self.df["augementation_salaire_precedente"] < 12)
        ).astype(int)

        self.classification_models(400, "newfeature_cweight", class_weight={0:1, 1:2})

        selected_pipe = self.classification_models(400, "newfeature_cweight_resample", class_weight={0:1, 1:2}, resample=True) 
        
        return selected_pipe
    
    def features_results(self, pipe, X_test, y_test):
        """
        Analyse l'importance des features via importance native, permutation (recall) et SHAP (probabilité).
        Génère beeswarm, scatter, waterfall et boxplots des départs.

        Args:
            pipe   : Pipeline entraîné (preprocessing + RandomForest).
            X_test : Features du jeu de test.
            y_test : Cible du jeu de test.

        Side effects:
            Crée Graph/Shap_beeswarm_test.png
            Crée Graph/Shap_vs_permutation_test.png
            Crée Graph/Shap_scatter_{feature}.png (x5)
            Crée Graph/Waterfall_class0.png et Waterfall_class1.png
            Crée Graph/Boxplot_final.png
        """

        # Récupération du modèle entraîné depuis le pipeline
        model = pipe.named_steps["model"]
        # Récupération des noms des features après preprocessing
        feature_names = pipe.named_steps["preprocessing"].get_feature_names_out()
        # Importance calculée directement par le modèle d'arbre
        fi_native = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        print("\n--- Feature importance native (arbre)")
        print(fi_native.head(10))
        # Proportions de la classe positive
        print("\nTaux réel de classe Oui dans le test :", y_test.mean())
        proba_test = pipe.predict_proba(X_test)[:, 1]
        print("\nProbabilité moyenne prédite :", proba_test.mean())
        X_test_transformed  = pipe.named_steps["preprocessing"].transform(X_test)
        # Evaluation sur le jeu de données inconnu
        perm = permutation_importance(
            model,
            X_test_transformed,
            y_test,
            n_repeats=20,
            random_state=42,
            scoring="recall"
        )
        # Conversion en Series pour affichage trié
        importances = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
        print("\n--- Permutation Importance")
        print(importances.head(10))
        # Conversion en DataFrame pour conserver les noms de colonnes
        X_test_shap = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
        explainer = shap.TreeExplainer(
            model,
            data=X_test_shap,
            feature_perturbation="interventional",
            model_output="probability"
        )
        # Calcul des SHAP values sur le set test, classes positives
        shap_values_test_pos = explainer(X_test_shap)[:, :, 1]
        shap.summary_plot(
            shap_values_test_pos,
            X_test_shap,
            max_display=5,
            show=False
        )
        plt.savefig("Graph/Shap_beeswarm_test.png", dpi=300)
        plt.close()
        # Valeurs SHAP moyennes par feature
        shap_values_abs = np.abs(shap_values_test_pos.values)
        shap_importance_test = shap_values_abs.mean(axis=0)
        # Serie pour comparaison avec données permutation
        shap_importance_test_series = pd.Series(shap_importance_test, index=feature_names).sort_values(ascending=False)
        print("\n--- SHAP Importance (Test Set, numpy)")
        print(shap_importance_test_series.head(5))
        comparison = pd.DataFrame({
            "Permutation": importances,
            "SHAP": shap_importance_test_series
        }).sort_values("Permutation", ascending=False)
        print("\n--- Comparaison Permutation vs SHAP (Top 5)")
        print(comparison.head(5))
        comparison.head(5).plot(kind="bar")
        plt.title("Permutation vs SHAP Importance (Top 5)")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Graph/Shap_vs_permutation_test.png", dpi=300)
        plt.close()

        # Top 5 features SHAP (test set)
        top5_features = shap_importance_test_series.head(5).index.tolist()
        # Scatter plots automatiques
        for feature in top5_features:
            shap.plots.scatter(shap_values_test_pos[:, feature],
                               color= shap_values_test_pos[:, "bin__heure_supplementaires"],
                               show=False)
            plt.title(f"SHAP Scatter - {feature}")
            plt.xlabel(f"Contribution de {feature} à la prédiction")
            plt.ylabel("Individus")
            plt.tight_layout()
            plt.savefig(f"Graph/Shap_scatter_{feature}.png", dpi=300)
            plt.close()

        # Indices d'exemple dans le test set
        idx_class0 = y_test[y_test == 0].index[0]
        idx_class1 = y_test[y_test == 1].index[0]
        # Waterfall classe 0
        shap.plots.waterfall(
            shap_values_test_pos[X_test_shap.index.get_loc(idx_class0)],
            max_display=5,
            show=False
        )
        plt.title("Waterfall - Classe 0 (Non départ)")
        plt.savefig("Graph/Waterfall_class0.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
        plt.close()
        # Waterfall classe 1
        shap.plots.waterfall(
            shap_values_test_pos[X_test_shap.index.get_loc(idx_class1)],
            max_display=5,
            show=False
        )
        plt.title("Waterfall - Classe 1 (Départ)")
        plt.savefig("Graph/Waterfall_class1.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
        plt.close()

        fig, axes = plt.subplots(1, 5, figsize=(20, 7))

        for ax, col in zip(axes, ['heure_supplementaires', 'nombre_participation_pee', 'age', 'annees_dans_l_entreprise', 'satisfaction_employee_nature_travail']):
            sns.boxplot(data=self.departs, y=col, ax=ax, medianprops=dict(color='red', linewidth=3))
            ax.set_title(col)
        plt.suptitle("Features principales SHAP - Départs")
        plt.tight_layout()
        plt.savefig("Graph/Boxplot_final.png", dpi=300)
        plt.close()

    def run_script(self):

        self.doc_analysis()
        datasets = self.data_cleaning()
        self.first_modelisation(datasets[0], datasets[1])
        pipe = self.classification_test()
        self.features_results(pipe, self.X_test, self.y_test)


RH_prediction_analysis = FilesCleaning()
RH_prediction_analysis.run_script()


