import joblib
import os
import pandas as pd
import numpy as np


# ── Load model once when the server starts ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_data = joblib.load(os.path.join(BASE_DIR, 'credit_model.pkl'))

pipeline             = model_data['pipeline']
state_default_rates  = model_data['state_default_rates']
global_default_rate  = model_data['global_default_rate']
threshold            = model_data['threshold']


# ── Feature engineering ───────────────────────────────────────
def engineer_features(data: dict) -> pd.DataFrame:
    """
    Takes raw user inputs, engineers features to match
    what the model was trained on, returns a single-row DataFrame.
    """
    # Compute engineered features
    loan_to_income = data['loan_amnt'] / (data['annual_inc'] + 1)

    state_default_rate = state_default_rates.get(
        data['addr_state'], global_default_rate
    )

    # Build the row the model expects
    row = {
        'sub_grade':           data['sub_grade'],
        'issue_year':          data['issue_year'],
        'term':                data['term'],
        'dti':                 data['dti'],
        'fico_range_low':      data['fico_range_low'],
        'emp_length':          data['emp_length'],
        'mort_acc':            data['mort_acc'],
        'home_ownership':      data['home_ownership'],
        'loan_to_income':      loan_to_income,
        'state_default_rate':  state_default_rate,
    }

    return pd.DataFrame([row])


# ── Decision logic ────────────────────────────────────────────
def get_decision(probability: float):
    """
    Converts a probability into a tiered lending decision.

    - APPROVE : low risk, clear to lend
    - REVIEW  : borderline, needs human review
    - REJECT  : high risk, decline
    """
    if probability < 0.40:
        return "APPROVE", "LOW", (
            "Low default risk. Application meets lending criteria."
        )
    elif probability < threshold:
        return "REVIEW", "MEDIUM", (
            "Moderate default risk. Recommend manual underwriter review "
            "before approval."
        )
    else:
        return "REJECT", "HIGH", (
            "High default risk. Application does not meet "
            "lending criteria at current threshold."
        )


# ── Main prediction function ──────────────────────────────────
def predict(application: dict) -> dict:
    """
    Takes a validated loan application dict,
    returns prediction results.
    """
    # Engineer features
    X = engineer_features(application)

    # Get probability
    probability = float(
        pipeline.predict_proba(X)[:, 1][0]
    )

    # Get decision
    decision, risk_level, message = get_decision(probability)

    return {
        'probability':    round(probability, 4),
        'decision':       decision,
        'threshold_used': threshold,
        'risk_level':     risk_level,
        'message':        message
    }