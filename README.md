# Risk Scorecard ML

A credit risk classification model that predicts loan default probability using the LendingClub dataset. Built with a focus on explainability and business-aligned threshold selection — not just raw accuracy.

## Problem

Predicting which borrowers are likely to default is a heavily imbalanced classification problem. The goal is not just to maximise accuracy, but to identify the right tradeoff between catching defaults (recall) and avoiding false alarms (precision) — a decision with real financial consequences.

## Approach

- **Baseline model:** XGBoost classifier wrapped in a scikit-learn pipeline
- **Hyperparameter tuning:** Optuna with cross-validated AUC objective
- **Explainability:** SHAP values for global feature importance and individual prediction explanation
- **Threshold analysis:** Evaluated multiple thresholds (F1-optimal, KS-optimal, business) rather than defaulting to 0.5

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.74 |
| KS Statistic | 0.349 (threshold: >0.2 usable, >0.4 good) |
| Default Recall (at 0.5) | 69.1% |
| Default Precision (at 0.5) | 35.7% |
| Average Precision Score | 0.439 |

The model correctly flags ~69% of actual defaults. Threshold analysis revealed that the F1-optimal threshold (0.53) improves precision to 37.4% while keeping recall at 64.1% — a more practical operating point for a lender.

## Tech Stack

- Python, Jupyter Notebook
- XGBoost, scikit-learn
- Optuna
- SHAP
- pandas, NumPy, matplotlib, seaborn

## Dataset

Downloaded automatically via the Kaggle API (LendingClub loan data). Requires a `kaggle.json` credentials file in your environment.

## Usage

1. Add your `kaggle.json` to the project directory
2. Open `risk_scorecard_ml.ipynb`
3. Run all cells