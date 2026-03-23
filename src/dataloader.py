import pandas as pd
import os


def load_paysim(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] PaySim file not found at: {filepath}"
            f"\n[FIX]   Place the CSV inside the data/ folder as data/paysim.csv"
        )

    print(f"[INFO] Loading PaySim from {filepath} ...")
    df = pd.read_csv(filepath)

    expected = {
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing columns: {missing}")

    print(f"[INFO] Loaded {len(df):,} rows x {len(df.columns)} columns")

    df = df.drop(columns=["isFlaggedFraud"])

    fraud_count = df["isFraud"].sum()
    total = len(df)
    fraud_pct = (fraud_count / total) * 100

    print(f"\n[INFO] Class distribution:")
    print(f"       Legit : {total - fraud_count:,} ({100 - fraud_pct:.3f}%)")
    print(f"       Fraud : {fraud_count:,} ({fraud_pct:.3f}%)")
    print(f"       Ratio : 1 fraud per {int(total / fraud_count):,} legit txns")

    print(f"\n[INFO] Transaction types:")
    for tx_type, count in df["type"].value_counts().items():
        fraud_in_type = df[df["type"] == tx_type]["isFraud"].sum()
        print(f"       {tx_type:<12} {count:>8,} txns  |  {fraud_in_type:>5,} fraud")

    return dfgit 


if __name__ == "__main__":
    df = load_paysim("data/paysim.csv")
    print(f"\n[INFO] First 3 rows:")
    print(df.head(3).to_string())