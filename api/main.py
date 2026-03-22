from fastapi import FastAPI, HTTPException
from api.schema import LoanApplication, PredictionResponse
from api.predict import predict


# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk API",
    description="""
    Predicts the probability of loan default for a given application.
    
    Built on LendingClub data (2007-2018, 2.26M loans).
    Deployment model uses 10 key features and achieves ROC-AUC of 0.727.
    
    ## Decision Tiers
    - **APPROVE** — probability < 0.40, low risk
    - **REVIEW**  — probability 0.40–0.530, borderline
    - **REJECT**  — probability > 0.530, high risk
    """,
    version="1.0.0"
)


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Credit Risk API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_default(application: LoanApplication):
    try:
        result = predict(application.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )