import pandas as pd
import numpy as np
import joblib
import os
from dataloader import load_paysim
from feature_engineering import engineer_features, get_model_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def train_model(X, y):
    print("[INFO] Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train size : {len(X_train):,}")
    print(f"       Test size  : {len(X_test):,}")

    print("\n[INFO] Applying SMOTE to fix class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"       Before SMOTE — Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
    print(f"       After SMOTE  — Fraud: {y_train_bal.sum():,} | Legit: {(y_train_bal==0).sum():,}")

    print("\n[INFO] Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train_bal, y_train_bal)
    print("[INFO] Training complete.")

    print("\n[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'='*40}")
    print("\n[INFO] Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return model, X_test, y_test


def save_model(model, path="models/aml_model.joblib"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


if __name__ == "__main__":
    df = load_paysim("data/paysim.csv")
    df = engineer_features(df)
    X, y = get_model_features(df)

    model, X_test, y_test = train_model(X, y)
    save_model(model)