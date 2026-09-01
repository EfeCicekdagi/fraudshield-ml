import pandas as pd
import json
from pathlib import Path
import os

def generate_baseline():
    train_path = Path("data/processed/train.parquet")
    features = [
        "step", "hour_of_day", "day", "type", "amount", "amount_log1p",
        "is_risky_type", "oldbalanceOrg", "orig_oldbalance_is_zero",
        "amount_to_oldbalance_orig_ratio", "orig_account_type", "dest_account_type"
    ]
    baseline = {}
    
    if not train_path.exists():
        print("Train path not found. Generating dummy zero baseline.")
        for f in features:
            if f in ["type", "orig_account_type", "dest_account_type"]:
                baseline[f] = "C" if "account" in f else "PAYMENT"
            else:
                baseline[f] = 0.0
    else:
        print("Reading train data for reference baseline...")
        df = pd.read_parquet(train_path)
        for f in features:
            if df[f].dtype.name in ['category', 'object'] or f == "type":
                baseline[f] = str(df[f].mode()[0])
            else:
                baseline[f] = float(df[f].median())
        
    out_path = Path("artifacts/final/reference_baseline.json")
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=4)
    print(f"Baseline saved to {out_path}.")

if __name__ == "__main__":
    generate_baseline()
