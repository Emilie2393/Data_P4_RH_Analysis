# Prédiction de l'Attrition RH

Analyse prédictive des départs employés à partir de trois sources de données RH (SIRH, évaluations, sondages).

---

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
# Installer uv si ce n'est pas déjà fait
curl -Ls https://astral.sh/uv/install.sh | sh

# Installer les dépendances depuis le pyproject.toml
uv sync

# Lancer le script
uv run main.py
```

## Dépendances

Déclarées dans `pyproject.toml` :

```toml
[project]
dependencies = [
    "imblearn>=0.0",
    "pandas>=2.3.3",
    "requests>=2.32.5",
    "scikit-learn>=1.8.0",
    "seaborn>=0.13.2",
    "shap>=0.49.1",
]
```

---

## Sources de Données

| Fichier | Clé de jointure | Contenu |
|---|---|---|
| `extrait_sirh` | `id_employee` | Données administratives et salariales |
| `extrait_eval` | `eval_number` | Notes et évaluations de performance |
| `extrait_sondage` | `code_sondage` | Scores de satisfaction et engagement |

---

## Pipeline

### 1. `doc_analysis()`
- Distribution, type et taux de nullité par colonne et par source
- Vérification de la cohérence des clés de jointure
- Fusion inner join sur l'identifiant employé → export `extrait_rh.csv`

### 2. `data_cleaning()`
- Corrélation de Spearman + pairplot pour détection de redondances
- Nettoyage de `augementation_salaire_precedente` (strip, cast int64)
- Encodage de la cible : `Non → 0`, `Oui → 1`
- Préprocesseur `ColumnTransformer` :
  - `StandardScaler` → features numériques
  - `OrdinalEncoder` → `genre`, `heure_supplementaires`
  - `OneHotEncoder` → `poste`, `departement`, `statut_marital`, ...

### 3. `first_modelisation()`
Comparaison de trois modèles baseline sur train/test (sans stratification) :

| Modèle | Objectif |
|---|---|
| `DummyClassifier` | Référence — prédit toujours la classe majoritaire |
| `LogisticRegression` | Baseline linéaire |
| `RandomForest` | Baseline arbre (200 estimateurs) |

Affiche matrices de confusion, classification report et scores précision/recall. Génère `Graph/{model}_{Train|Test}_confusion_matrix.png`.

### 4. `classification_test()`
Trois configurations comparées successivement :

| Config | n_estimators | class_weight | SMOTE |
|---|---|---|---|
| Baseline | 200 | None | Non |
| + class_weight | 400 | {0:1, 1:2} | Non |
| **+ SMOTE** ✓ | 400 | {0:1, 1:2} | Oui (0.6) |

Feature engineerée entre le baseline et les modèles pondérés :
```python
carriere_stagnante = (annees_depuis_promotion >= 3) & (augmentation_precedente < 12%)
```

### 5. `features_results()`
Trois méthodes d'analyse d'importance :

| Méthode | Ce qu'elle mesure |
|---|---|
| Importance native | Réduction d'impureté moyenne (biaisée haute cardinalité) |
| Permutation (recall) | Dégradation du recall si feature perturbée |
| SHAP (probabilité) | Contribution individuelle à `predict_proba()` |

Top 5 features identifiées : `heure_supplementaires`, `annee_experience_totale`, `nombre_participation_pee`, `revenu_mensuel`, `annees_dans_l_entreprise`

---

## Stratégie de Seuil

Le seuil n'est pas fixé à 0.5 — il est sélectionné dynamiquement sur la courbe Précision-Rappel :

- **Recall cible** : ≥ 0.90
- **Précision minimale** : ≥ 0.40

---

## Outputs

```
extrait_rh.csv
Graph/
├── Spearman_corr.png
├── Relations_pairplot.png
├── Courbe_{config}_précision_rappel.png  # x3
├── Proba_{config}.png  # x3
├── Shap_beeswarm_test.png
├── Shap_vs_permutation_test.png
├── Shap_scatter_{feature}.png      # x5
├── Waterfall_class0.png
├── Waterfall_class1.png
└── Boxplot_final.png
```