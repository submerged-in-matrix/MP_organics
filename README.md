<div align="center">

# 🔥 Melting Point Prediction via Two-Level Stacked Ensemble

**A decorrelated ensemble with SHAP-optimized features for organic compound melting points**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![R²](https://img.shields.io/badge/R²-≈_0.83-brightgreen)](.)
[![Dataset](https://img.shields.io/badge/Samples-~3%2C041-blue)](.)

</div>

---

## Motivation

Melting point is one of the most practically important thermophysical properties of a material — it governs processing routes, phase stability, and application suitability. Yet predicting it from molecular structure alone remains difficult because melting is a collective phenomenon sensitive to bond character, geometry, and intermolecular forces simultaneously. No single ML model captures all of these well.

This project tackles that by building a **custom two-level stacking ensemble** that combines diverse base learners into a tuned meta-learner. The work reproduces and **extends** a published ensemble methodology originally applied to halides ([Lobanov et al., 2023](https://link.springer.com/article/10.1134/S1995080223010341)) — validating whether the same stacking strategy transfers to a different materials system (organic compounds) and whether custom stacking order can push accuracy further.

## The Pipeline at a Glance

```
SMILES strings (~3,041 compounds, Citrination)
    │
    ▼
┌─────────────────────────────────┐
│  Featurization                  │
│  RDKit descriptors + Morgan FP  │
│  + custom bond-count features   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  SHAP-Guided Feature Selection  │
│  Top 68 of 135 features         │
│  (swept 30→135, optimum @ 68)   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Level 1 — Decorrelated Base Learners           │
│                                                 │
│  10 models: RF / GB / MLP (custom order)        │
│  Bootstrap sampling + 70% feature subspace      │
│  → out-of-fold predictions                      │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Level 2 — Tuned GB Meta-Learner                │
│                                                 │
│  BayesSearchCV-optimized GradientBoosting       │
│  trained on Level 1 stacked predictions         │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
                   R² ≈ 0.83
                 (~4% over baseline)
```

## Feature Engineering

### SMILES → Feature Matrix

Each compound enters the pipeline as a SMILES string. Featurization produces three categories of descriptors:

- **RDKit molecular descriptors** — physicochemical properties, topological indices
- **Morgan fingerprints** — circular substructure fingerprints capturing local molecular neighborhoods
- **Custom bond-count features** — engineered descriptors counting double, triple, and aromatic bonds per molecule

This yields a 135-dimensional feature matrix.

### SHAP-Guided Feature Selection

Rather than using all 135 features, SHAP values are computed across the tree-based base models (RF and GB) in the ensemble. The resulting mean absolute SHAP importances are ranked and thresholded, then a systematic sweep from 30 to 135 features identifies the **optimum feature set size: 68 features**. This SHAP-guided pruning removes noise dimensions that hurt generalization without discarding informative signals.

The top 20 features by SHAP importance and a feature variance analysis are included in the notebooks.

## The Two-Level Ensemble

### Level 1 — Decorrelated Base Learners

The core idea: if base models all make the same errors, stacking gains nothing. Decorrelation is enforced through:

- **Bootstrap resampling** — each of the 10 base models trains on a different bootstrap sample of the training data
- **Feature subspace sampling** — each model sees only 70% of the 68 selected features (optimized via grid search over subspace fractions 0.4–1.0 and ensemble sizes 6–20)
- **Model diversity** — base learners alternate between RF, GB, and MLP with individually tuned hyperparameters (BayesSearchCV)

The final Level 1 order: `GB → RF → MLP → RF → GB → MLP → GB → RF → MLP → GB`

This specific ordering was selected from **8 tested variants** (4 block arrangements + 4 custom interleaved arrangements), with each variant evaluated on validation R² and MAE. The effect of stacking order on ensemble performance is non-trivial — different orderings change the correlation structure of base predictions feeding into the meta-learner.

### Level 2 — Tuned Gradient Boosting Meta-Learner

The initial prototype used Ridge regression as a simple meta-learner. The final pipeline replaces it with a **BayesSearchCV-tuned Gradient Boosting** stacker, optimizing over `n_estimators`, `max_depth`, `max_features`, `learning_rate`, and `min_samples_split`. This non-linear meta-learner better captures interactions between base model predictions than a linear combiner.

### Ensemble Diagnosis

Pairwise correlation heatmaps (both on raw predictions and on residuals) are computed between all 10 base models to verify that decorrelation is working — i.e., that base models disagree in complementary ways rather than making correlated errors.

## Results

| Metric | Value |
|---|---|
| **R² (out-of-sample)** | **≈ 0.83** |
| Improvement over baseline stacking | **~4%** |
| Optimal feature set | 68 / 135 (SHAP-selected) |
| Optimal ensemble size | 10 base models |
| Optimal feature subspace | 70% |
| Best stacking order | Custom interleaved (GB/RF/MLP) |
| Meta-learner | Tuned GB (BayesSearchCV) |
| Key finding | Ensemble diversity contributed more than individual model complexity |

The ~4% improvement from custom stacking order and tuned meta-learner comes primarily from better decorrelation and a smarter composition strategy — same data, same feature set, better architecture.

## Repository Structure

```
MeltingPoints_Inorganics/
├── src/                # Core pipeline implementation
├── featurize/          # SMILES → feature matrix (RDKit + custom bond-count)
├── optimize/           # BayesSearchCV tuning for base & meta learners
├── data/               # Processed data artifacts
├── dataset/            # Raw dataset (~3,041 compounds from Citrination)
├── notebooks/          # Full walkthrough notebook with all experiments
├── utils/              # Shared utilities
├── env/                # Environment configuration
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| **Base Learners** | scikit-learn (RF, GB, MLP), LightGBM |
| **Meta-Learner** | Gradient Boosting (BayesSearchCV-tuned) |
| **Featurization** | RDKit (molecular descriptors, Morgan FP) + custom bond-count features |
| **Feature Selection** | SHAP (TreeExplainer across ensemble) |
| **Hyperparameter Optimization** | BayesSearchCV (skopt) |
| **Data Source** | Citrination |

## Reference

This project reproduces and extends the methodology from:

> Lobanov et al., "Melting Points Prediction of Halides," *Lobachevskii Journal of Mathematics*, 2023.
> [DOI: 10.1134/S1995080223010341](https://link.springer.com/article/10.1134/S1995080223010341)

The extension validates whether the two-level stacking strategy generalizes from halides to organic compounds — and demonstrates that SHAP-guided feature selection combined with custom stacking order can further improve out-of-sample performance on a different materials system.

---

<div align="center">

📬 sayeed.shahriar@gmail.com · [Portfolio](https://submerged-in-matrix.github.io/projects/mp-ensemble/) · [GitHub](https://github.com/submerged-in-matrix)

</div>
