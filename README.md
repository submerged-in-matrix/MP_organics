<div align="center">

# 🔥 Melting Point Prediction via Two-Level Stacked Ensemble

**A decorrelated, SHAP-optimized ensemble for organic compound melting points**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![R²](https://img.shields.io/badge/R²-0.83-brightgreen)](.)
[![Dataset](https://img.shields.io/badge/Samples-~3%2C041-blue)](.)

</div>

---

## Motivation

Melting point is one of the most practically important thermophysical properties of a material — it governs processing routes, phase stability, and application suitability. Yet predicting it from composition alone remains difficult because melting is a collective phenomenon sensitive to bond character, molecular geometry, and intermolecular forces simultaneously. No single ML model captures all of these well.

This project tackles that by building a **custom two-level stacking ensemble** that combines four diverse base learners, each tuned independently, into a meta-learner that exploits their complementary strengths. The work reproduces and **extends** a published ensemble methodology originally applied to halides ([Lobanov et al., 2023](https://link.springer.com/article/10.1134/S1995080223010341)) — validating whether the same stacking strategy transfers to a different materials system (organic compounds) and whether custom stacking order can push accuracy further.

## The Two-Level Architecture

The key insight is that naive stacking — training a meta-learner on base model outputs — can fail when base models are correlated (i.e., they all make the same mistakes). The solution here is a **two-level design** where the first level explicitly decorrelates the base learners before the second level stacks them:

```
                         ┌─────────────┐
                         │  Raw Data   │
                         │ ~3,041 cpds │
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    │   SMILES → Features   │
                    │  RDKit + custom bond   │
                    │  count descriptors     │
                    └───────────┬───────────┘
                                │
              ┌─────────┬───────┴───────┬─────────┐
              ▼         ▼               ▼         ▼
           ┌─────┐  ┌─────┐        ┌──────┐  ┌─────┐
           │ RF  │  │ XGB │        │ LGBM │  │ MLP │
           └──┬──┘  └──┬──┘        └──┬───┘  └──┬──┘
              │        │              │          │
              │   Each independently tuned       │
              │                                  │
    ══════════╪══════════════════════════════════╪═══════
     Level 1  │     Decorrelation Layer          │
    ══════════╪══════════════════════════════════╪═══════
              │                                  │
              └──────────┬───────────────────────┘
                         │  Cross-validated
                         │  predictions
                         ▼
    ══════════════════════════════════════════════════════
     Level 2       Stacking Meta-Learner
                   (custom stacking order)
    ══════════════════════════════════════════════════════
                         │
                         ▼
                    R² ≈ 0.83
                  (~4% over baseline)
```

**Level 1 — Decorrelation:** The four base learners (RF, XGBoost, LightGBM, MLP) are each tuned via Bayesian optimization. Their cross-validated out-of-fold predictions form the input to Level 2. The decorrelation step ensures the base models contribute complementary — not redundant — signals.

**Level 2 — Stacking:** A meta-learner is trained on the Level 1 outputs. Critically, the **stacking order** (which models feed into which) is treated as a tunable hyperparameter. The hero figure in the repository shows the effect of permuting stacking order on final ensemble performance — demonstrating that order matters and the custom configuration outperforms naive stacking by ~4%.

## Dataset & Featurization

**Source:** ~3,041 organic compounds from Citrination, each with an experimentally measured melting point.

**Featurization pipeline:**

```
SMILES string
    │
    ├── RDKit molecular descriptors
    │     (fingerprints, topological indices, physicochemical properties)
    │
    └── Custom bond-count features
          (engineered to capture bond-type distribution)
    │
    ▼
Feature matrix → SHAP-guided refinement → Final feature set
```

SHAP values are computed not just for post-hoc interpretability, but as an active part of the optimization loop — guiding which features to retain, which models benefit most from which feature subsets, and how the ensemble should be composed.

## SHAP-Guided Optimization Loop

SHAP analysis serves three distinct roles in this project:

1. **Feature selection** — Identifying the top contributing features and pruning noise dimensions that hurt generalization
2. **Model diagnostics** — Understanding *why* each base learner disagrees with the others (pairwise correlation heatmaps of prediction accuracy)
3. **Ensemble composition** — Informing which models to include and how to weight them in the stacking layer

This goes beyond the typical "run SHAP at the end for a pretty plot" workflow. Here, SHAP is woven into the model/feature/ensemble optimization cycle.

## Results

| Metric | Value |
|---|---|
| **R² (out-of-sample)** | **≈ 0.83** |
| Improvement over baseline stacking | **~4%** |
| Base learners | RF, XGBoost, LightGBM, MLP (10 tuned models each) |
| Key finding | Ensemble diversity contributed more than individual model complexity |

The ~4% improvement from custom stacking order is notable because it comes at zero additional computational cost — same models, same features, just a smarter composition strategy.

## Repository Structure

```
MeltingPoints_Inorganics/
├── src/                # Core pipeline implementation
├── featurize/          # SMILES → feature matrix (RDKit + custom descriptors)
├── optimize/           # Bayesian hyperparameter tuning for each base learner
├── data/               # Processed data artifacts
├── dataset/            # Raw dataset (~3,041 compounds from Citrination)
├── notebooks/          # Jupyter notebooks for exploration & visualization
├── utils/              # Shared utilities
├── env/                # Environment configuration
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| **Base Learners** | scikit-learn (RF), XGBoost, LightGBM, MLP |
| **Featurization** | RDKit (molecular descriptors from SMILES) + custom bond-count features |
| **Interpretability** | SHAP (feature importance, model diagnostics, ensemble guidance) |
| **Optimization** | Bayesian hyperparameter tuning |
| **Data Source** | Citrination |

## Reference

This project reproduces and extends the methodology from:

> Lobanov et al., "Melting Points Prediction of Halides," *Lobachevskii Journal of Mathematics*, 2023.
> [DOI: 10.1134/S1995080223010341](https://link.springer.com/article/10.1134/S1995080223010341)

The extension validates whether the two-level stacking strategy generalizes from halides to organic compounds — and demonstrates that custom stacking order can further improve out-of-sample performance on a different materials system.

---

<div align="center">

📬 sayeed.shahriar@gmail.com · [Portfolio](https://submerged-in-matrix.github.io/projects/mp-ensemble/) · [GitHub](https://github.com/submerged-in-matrix)

</div>
