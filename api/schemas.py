# These are the data models for our API
# Pydantic handles validation automatically - wrong type or missing field = clear error

from pydantic import BaseModel


# What the client sends us
class Transaction(BaseModel):
    step: int              # 1 step = 1 hour
    type: str              # CASH_OUT, TRANSFER, PAYMENT etc.
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


# What we send back after scoring
class PredictionResult(BaseModel):
    transaction_id: int
    prediction: str        # FRAUD or LEGIT
    probability: float
    flagged: bool


# Shape of an alert from the database
class Alert(BaseModel):
    id: int
    transaction_id: int
    probability: float
    step: int
    type: str
    amount: float