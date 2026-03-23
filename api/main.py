# The main FastAPI app - this is the core of our API layer
# Run it with: uvicorn api.main:app --reload

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import Transaction, PredictionResult, Alert
from api.database import init_db, save_alert, get_alerts
from feature_engineering import engineer_features

# Load model once when the app starts - not on every request
model = None
explainer = None
transaction_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - load model and init database
    global model, explainer
    print("[INFO] Loading model...")
    model = joblib.load("models/aml_model.joblib")
    explainer = shap.TreeExplainer(model)
    init_db()
    print("[INFO] API ready")
    yield
    # Shutdown
    print("[INFO] Shutting down")


app = FastAPI(
    title="AML Transaction Monitoring API",
    description="Real-time fraud detection with explainable ML",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    # Quick check to confirm the API is running
    return {"status": "online", "model": "XGBoost", "version": "1.0.0"}


@app.post("/score", response_model=PredictionResult)
def score_transaction(tx: Transaction):
    """
    Score a transaction and return fraud probability.
    If flagged as fraud, saves it to the alerts database.
    """
    global transaction_counter
    transaction_counter += 1

    # Build a single row dataframe matching our training features
    raw = pd.DataFrame([{
        "step": tx.step,
        "type": tx.type,
        "amount": tx.amount,
        "nameOrig": "C_API",
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "nameDest": "C_API",
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "isFraud": 0  # placeholder - we don't know yet
    }])

    # Run feature engineering on this single transaction
    raw = engineer_features(raw)

    # Drop columns the model doesn't use
    drop_cols = ["nameOrig", "nameDest", "type", "isFraud"]
    features = [c for c in raw.columns if c not in drop_cols]
    X = raw[features]

    # Get prediction
    probability = float(model.predict_proba(X)[0][1])
    prediction = "FRAUD" if probability > 0.5 else "LEGIT"
    flagged = probability > 0.5

    # Save to database if fraud
    if flagged:
        save_alert(transaction_counter, probability, tx.step, tx.type, tx.amount)

    return PredictionResult(
        transaction_id=transaction_counter,
        prediction=prediction,
        probability=round(probability, 4),
        flagged=flagged
    )


@app.get("/alerts")
def get_fraud_alerts():
    # Returns the most recent flagged transactions from the database
    rows = get_alerts()
    return [
        {
            "id": r[0],
            "transaction_id": r[1],
            "probability": r[2],
            "step": r[3],
            "type": r[4],
            "amount": r[5]
        }
        for r in rows
    ]


@app.get("/explain/{transaction_id}")
def explain(transaction_id: int, tx: Transaction):
    """
    Returns SHAP explanation for why a transaction was flagged.
    Shows which features pushed the prediction toward fraud.
    """
    raw = pd.DataFrame([{
        "step": tx.step,
        "type": tx.type,
        "amount": tx.amount,
        "nameOrig": "C_API",
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "nameDest": "C_API",
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "isFraud": 0
    }])

    raw = engineer_features(raw)
    drop_cols = ["nameOrig", "nameDest", "type", "isFraud"]
    features = [c for c in raw.columns if c not in drop_cols]
    X = raw[features]

    shap_values = explainer.shap_values(X)
    explanation = dict(zip(X.columns, shap_values[0]))
    explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "transaction_id": transaction_id,
        "prediction": "FRAUD" if model.predict(X)[0] == 1 else "LEGIT",
        "probability": round(float(model.predict_proba(X)[0][1]), 4),
        "shap_explanation": {k: round(float(v), 4) for k, v in explanation.items()}
    }