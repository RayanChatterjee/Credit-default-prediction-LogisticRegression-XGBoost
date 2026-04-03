# Credit Default Prediction Pipeline using LogisticRegression and XGBoost
Credit default prediction pipeline — end-to-end ML on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150K borrowers). Covers data cleaning, outlier capping, median imputation, StandardScaler, VIF-based multicollinearity removal, SMOTE resampling, and F1-optimal threshold tuning. Compares Logistic Regression vs XGBoost with SHAP interpretability and post-hoc WoE/IV analysis.
Progresses systematically from raw data to a tuned XGBoost model, with every preprocessing and modelling decision justified and benchmarked.

---

## Problem Statement

Predict whether a borrower will experience **90+ days of financial distress** in the next two years (`SeriousDlqin2yrs`). The dataset is heavily imbalanced (~14:1 non-default to default ratio), making standard accuracy metrics misleading and requiring deliberate handling throughout the pipeline.

---

## Pipeline Overview

```
Raw CSV
  └── Data Cleaning          (NaN removal, median imputation)
  └── Outlier Capping        (RevolvingUtilization clipped at 1.0)
  └── Scaling                (StandardScaler)
  └── Multicollinearity      (VIF analysis → drop 2 features)
  └── Class Imbalance        (SMOTE on train only)
  └── Threshold Tuning       (F1-optimal cutoff via PR curve)
  └── Regularisation Tuning  (GridSearchCV over C)
  └── XGBoost                (scale_pos_weight, GridSearchCV)
  └── Interpretability       (SHAP + post-hoc WoE/IV)
```

---

## Key Design Decisions

### Outlier Capping
`RevolvingUtilizationOfUnsecuredLines` had a max value of **50,708** (valid range: 0–1). 2.2% of rows exceeded 1.0. Values clipped at 1.0 before any modelling.

### VIF-Based Feature Removal
Three "days past due" columns were highly multicollinear (VIF > 10). `NumberOfTime60-89DaysPastDueNotWorse` and `NumberOfTimes90DaysLate` were dropped. Their signal is fully captured by `NumberOfTime30-59DaysPastDueNotWorse`.

### SMOTE Applied Only to Training Set
Synthetic oversampling is fit exclusively on `X_train` — never on the test set — preventing data leakage. The real imbalanced test set is used for all evaluation.

### F1-Optimal Threshold
Default threshold of 0.5 is inappropriate for imbalanced credit data. The precision-recall curve is swept to find the threshold that maximises F1 — independently for Logistic Regression and XGBoost, since their probability distributions differ.

### Post-Hoc IV/WoE (Not Pre-Model Filter)
WoE and IV are computed **after** all modelling as an interpretability layer, not as a feature selection gate. This avoids the zero-inflation trap where count features (e.g. `NumberOfTimes90DaysLate`) score IV≈0 due to quantile binning collapse — despite being confirmed strong predictors by SHAP.

---

## Results

| Model | Stage | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| LogReg | Raw Data | — | — | — | — |
| LogReg | Scaled | ~0.82 | Poor | ~0.50 | Low |
| LogReg | After VIF | ~0.82 | Poor | ~0.50 | Low |
| LogReg | SMOTE | Better | Improved | — | — |
| LogReg | Optimal Threshold | Better | Best LR | — | — |
| LogReg | class_weight=balanced | Similar | Similar | — | — |
| LogReg | GridSearchCV (best C) | Best LR | Best LR | — | — |
| **XGBoost** | **GridSearchCV Tuned** | **Best** | **Best** | **Best** | **Best** |

XGBoost outperforms Logistic Regression on all metrics. Its ability to model non-linear feature interactions and handle class imbalance natively via `scale_pos_weight` makes it better suited for credit default prediction.

---

## Interpretability

### SHAP (SHapley Additive exPlanations)
- Applied to both Logistic Regression (post-VIF) and the tuned XGBoost model
- Identifies feature directions and magnitudes per prediction
- Used to validate that multicollinearity removal fixed nonsensical SHAP directions

### WoE / IV (Post-hoc)
- **Weight of Evidence** per bin shows how strongly each segment separates defaulters from non-defaulters
- **Information Value** ranks features by predictive power
- Manual bins used for zero-inflated count features to prevent IV=0 artifact
- Cross-referenced against SHAP rankings — agreement = high confidence, divergence = investigate

#### IV vs SHAP Cross-Reference Summary
| Feature | IV Strength | SHAP Rank | Verdict |
|---|---|---|---|
| RevolvingUtilization | Suspicious* | 1 | Genuine top feature — outliers inflated IV |
| 30-59 DaysLate | Suspicious* | 2 | Strong predictor, sparse bins inflate IV |
| age | Medium | 3 | Consistent across both methods |
| DebtRatio | Weak | 5 | Consistent |
| MonthlyIncome | Weak | 6 | Consistent |
| NumberRealEstateLoans | Useless | 7 | Zero-inflation — trust SHAP over IV |

*Suspicious IV does not mean reject — it means IV's ceiling was hit. SHAP and domain knowledge override.

---

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
statsmodels
matplotlib
```

Install:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap statsmodels matplotlib
```

---

## Usage

1. Training data:  `cs-training.csv` downloaded from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) - it should be in the same directory as the script.
2. Run the script "Credit_default_final_Github_v3LOG_XGB_v3.ipynb" top to bottom — each section prints conclusions inline.
3. SHAP summary plots and comparison charts render automatically.

---

## File Structure

```
├── Credit_default_final_Github_v3LOG_XGB_v3.ipynb   # Main pipeline script
├── cs-training.csv              # Input data (download from Kaggle)
└── README.md
|___ Credit_default_Documentation_Post # detailed documentation 
```

---

## Concepts Covered

- Median imputation for missing values
- Outlier detection and capping
- StandardScaler normalisation
- Variance Inflation Factor (VIF) for multicollinearity
- SMOTE oversampling (train-only, no leakage)
- Logistic Regression with L2 regularisation
- GridSearchCV with cross-validation
- F1-optimal threshold selection via Precision-Recall curve
- XGBoost with `scale_pos_weight`
- SHAP feature importance
- Weight of Evidence (WoE) and Information Value (IV)
- ROC-AUC vs F1 — when each metric applies
