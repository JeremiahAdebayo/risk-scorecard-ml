from pydantic import BaseModel, Field
from typing import Literal


class LoanApplication(BaseModel):
    sub_grade: str = Field(..., example="B3")
    issue_year: int = Field(..., example=2018)
    term: Literal[36, 60] = Field(..., example=36)
    dti: float = Field(..., example=15.5)
    loan_amnt: float = Field(..., example=10000)
    annual_inc: float = Field(..., example=60000)
    addr_state: str = Field(..., example="CA")
    home_ownership: Literal[
        "RENT", "OWN", "MORTGAGE", "OTHER"
    ] = Field(..., example="RENT")
    fico_range_low: int = Field(..., example=700)
    emp_length: Literal[
        "< 1 year", "1 year", "2 years", "3 years",
        "4 years", "5 years", "6 years", "7 years",
        "8 years", "9 years", "10+ years", "Unknown"
    ] = Field(..., example="3 years")
    mort_acc: int = Field(..., example=1)


class PredictionResponse(BaseModel):
    probability: float
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    threshold_used: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    message: str