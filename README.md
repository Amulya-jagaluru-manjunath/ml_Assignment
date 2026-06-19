# ML Assignment — Credit Card Fraud Detection (2025)
## 7 Research Question Notebooks — Mac Mini M4 Setup

---

## Dataset
Download from Kaggle: https://www.kaggle.com/datasets/prince7489/credit-card-fraud-2025
Save as: `credit_card_fraud_2025.csv` in the **same folder** as the notebooks.

---

## Environment Setup (Mac Mini M4, Anaconda)

```bash
# Activate your conda environment (or create one)
conda activate thesis_env      # or: conda create -n ml_assign python=3.10 -y && conda activate ml_assign

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Folder Structure After Running All Notebooks

```
ml_assignment/
├── credit_card_fraud_2025.csv          ← YOUR DATASET (download from Kaggle)
├── requirements.txt
├── RQ1_Baseline_Performance.ipynb
├── RQ2_Model_Comparison.ipynb
├── RQ3_Preprocessing_Effect.ipynb
├── RQ4_Feature_Importance.ipynb
├── RQ5_Metric_Sensitivity.ipynb
├── RQ6_Robustness.ipynb
├── RQ7_Final_Recommendation.ipynb
├── figures/
│   ├── RQ1_baseline_performance.pdf
│   ├── RQ2_model_comparison.pdf
│   ├── RQ3_preprocessing_effect.pdf
│   ├── RQ4_feature_importance.pdf
│   ├── RQ5_metric_sensitivity.pdf
│   ├── RQ6_robustness.pdf
│   └── RQ7_final_recommendation.pdf
└── tables/
    ├── RQ1_baseline_performance.csv
    ├── RQ2_model_comparison.csv
    ├── RQ3_preprocessing_effect.csv
    ├── RQ4_feature_importance.csv
    ├── RQ5_metric_sensitivity.csv
    ├── RQ6_robustness.csv
    └── RQ7_final_recommendation.csv
```

---

## Research Questions & Methodology

| # | RQ | Models | Key Method | Figure Type |
|---|----|----|----|----|
| 1 | Baseline Performance | LR, DTree, k-NN | Train/test + SMOTE | Grouped bar chart |
| 2 | Model Comparison | LR, DTree, RF, XGBoost, SVM | Full comparison + SMOTE | Horizontal bar chart |
| 3 | Effect of Preprocessing | Random Forest | Ablation (4 strategies) | Ablation bar chart |
| 4 | Feature Importance | XGBoost | SHAP + feature gain | Bar + SHAP beeswarm |
| 5 | Metric Sensitivity | All 5 models | Rank under each metric | Bump/slope chart |
| 6 | Robustness | XGBoost | 5-fold, 10-fold CV, noise, missingness | Boxplot + line chart |
| 7 | Final Recommendation | All 4 | Decision matrix (5 criteria, 1–5 scale) | Radar + heatmap |

---

## Expected Runtime (Mac Mini M4, 24 GB RAM)

| Notebook | Estimated Time |
|---|---|
| RQ1 | ~3–5 min |
| RQ2 | ~6–10 min |
| RQ3 | ~4–6 min |
| RQ4 | ~5–8 min (SHAP) |
| RQ5 | ~6–10 min |
| RQ6 | ~15–20 min (15 CV folds total) |
| RQ7 | ~6–10 min |

---

## Notes
- All notebooks use `matplotlib.use('Agg')` — no display required; PDFs are saved directly.
- SMOTE is applied to training data only (never test data) to handle the class imbalance.
- XGBoost uses `tree_method='hist'` for fast training on Apple Silicon.
- For the dataset column `is_fraud` is the binary target (0=legitimate, 1=fraud).
- Run notebooks independently — each one reloads and preprocesses the raw CSV from scratch.
