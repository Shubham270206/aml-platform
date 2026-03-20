import pandas as pd
import numpy as np
from dataloader import load_paysim


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Engineering features...")

    # 1. How much the origin account balance changed
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    # 2. How much the destination account balance changed
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # 3. Transaction amount vs origin balance (how much of their balance they sent)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # 4. Flag high risk transaction types (fraud only happens in these two)
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    # 5. Origin account was empty BEFORE transaction
    df["orig_balance_zero_before"] = (df["oldbalanceOrg"] == 0).astype(int)

    # 6. Origin account was emptied AFTER transaction
    df["orig_balance_zero_after"] = (df["newbalanceOrig"] == 0).astype(int)

    # 7. Destination account was empty BEFORE transaction
    df["dest_balance_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)

    # 8. Balance difference mismatch (money didn't arrive properly)
    df["balance_mismatch"] = abs(df["balance_diff_orig"] - df["amount"])

    print(f"[INFO] Features engineered. Total columns: {len(df.columns)}")
    print(f"[INFO] New features: balance_diff_orig, balance_diff_dest, "
          f"amount_to_balance_ratio, is_transfer_or_cashout, "
          f"orig_balance_zero_before, orig_balance_zero_after, "
          f"dest_balance_zero_before, balance_mismatch")
    return df


def get_model_features(df: pd.DataFrame):
    """
    Returns X (features) and y (labels) ready for model training.
    Drops columns the model should not see.
    """
    drop_cols = ["nameOrig", "nameDest", "type", "isFraud"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["isFraud"]

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Features used: {list(X.columns)}")
    return X, y


if __name__ == "__main__":
    df = load_paysim("data/paysim.csv")
    df = engineer_features(df)

    X, y = get_model_features(df)

    print(f"\n[INFO] Sample fraud row features:")
    print(X[y == 1].head(2).to_string())