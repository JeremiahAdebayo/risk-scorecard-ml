# Risk Scorecard ML

A credit risk classification model that predicts loan default probability
across 2.26M real-world loans. Built with a focus on explainability and
business-aligned threshold selection — not just raw accuracy.

## Live Demo

- **API**: https://credit-risk-api.onrender.com/docs

## Problem

Predicting which borrowers are likely to default is a heavily imbalanced
classification problem. The goal is not just to maximise accuracy, but to
identify the right tradeoff between catching defaults (recall) and avoiding
false alarms (precision) — a decision with real financial consequences.

## Approach

| Stage | Key Decisions |
|-------|--------------|
| **Data cleaning** | Removed 35+ columns (post-outcome leakage, low-variance, high-missing); handled missing data via block pattern analysis |
| **Feature engineering** | 15+ financial ratios, credit behavior flags, leakage-safe target encoding |
| **Modeling** | XGBoost with `scale_pos_weight` for class imbalance |
| **Explainability** | SHAP feature importance + individual prediction waterfall plots |
| **Threshold analysis** | Evaluated F1-optimal, KS-optimal, and business thresholds rather than defaulting to 0.5 |

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.74 |
| KS Statistic | 0.349 (>0.2 usable, >0.4 good) |
| Average Precision | 0.439 |
| Default Recall @ 0.5 | 69.1% |
| Default Precision @ 0.5 | 35.7% |

At the F1-optimal threshold of 0.530, the model catches 64% of defaults
while approving 71% of creditworthy borrowers.

## Tech Stack

- Python, Jupyter Notebook
- XGBoost, scikit-learn, Optuna
- SHAP
- FastAPI, Streamlit
- pandas, NumPy, matplotlib, seaborn

## Dataset

Downloaded automatically via the Kaggle API (LendingClub loan data).
Requires a `kaggle.json` credentials file in your environment.

## Usage

1. Add your `kaggle.json` to the project directory
2. Open `risk_scorecard_ml.ipynb`
3. Run all cells

## API

Send a POST request to `/predict` with loan application details to get
a default probability and lending decision (APPROVE / REVIEW / REJECT).
Full schema and interactive testing available at `/docs`.