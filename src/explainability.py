import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_paysim
from feature_engineering import engineer_features, get_model_features


def load_model(path="models/aml_model.joblib"):
    model = joblib.load(path)
    print(f"[INFO] Model loaded from {path}")
    return model


def explain_transaction(model, X, index=0):
    """
    Explain a single transaction using SHAP values.
    Shows why the model flagged it as fraud or legit.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    row = X.iloc[index]
    sv = shap_values[index]

    print(f"\n{'='*50}")
    print(f"  Transaction #{index} — SHAP Explanation")
    print(f"{'='*50}")
    print(f"  Prediction : {'FRAUD' if model.predict(X.iloc[[index]])[0] == 1 else 'LEGIT'}")
    print(f"  Probability: {model.predict_proba(X.iloc[[index]])[0][1]:.4f}")
    print(f"{'='*50}")
    print(f"  {'Feature':<35} {'Value':>12} {'SHAP':>10}")
    print(f"  {'-'*57}")

    # Sort by absolute SHAP value
    pairs = sorted(zip(X.columns, row.values, sv), key=lambda x: abs(x[2]), reverse=True)
    for feat, val, shap_val in pairs:
        direction = "▲" if shap_val > 0 else "▼"
        print(f"  {feat:<35} {val:>12.4f} {direction}{abs(shap_val):>8.4f}")

    print(f"{'='*50}")
    return shap_values


def plot_feature_importance(model, X, save_path="models/shap_importance.png"):
    """
    Plot global SHAP feature importance and save it.
    """
    print("[INFO] Calculating SHAP values for feature importance...")
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed
    sample = X.sample(n=min(1000, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("Feature Importance — SHAP Values")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")


if __name__ == "__main__":
    df = load_paysim("data/paysim.csv")
    df = engineer_features(df)
    X, y = get_model_features(df)
    model = load_model("models/aml_model.joblib")

    # Explain a known fraud transaction
    fraud_indices = y[y == 1].index.tolist()
    fraud_pos = X.index.get_loc(fraud_indices[0])
    print("\n[INFO] Explaining a FRAUD transaction:")
    explain_transaction(model, X, index=fraud_pos)

    # Explain a known legit transaction
    legit_indices = y[y == 0].index.tolist()
    legit_pos = X.index.get_loc(legit_indices[0])
    print("\n[INFO] Explaining a LEGIT transaction:")
    explain_transaction(model, X, index=legit_pos)

    # Global feature importance plot
    plot_feature_importance(model, X)