"""
Setup Models Script
Generates the sklearn Random Forest and Isolation Forest models
required by the Red Team and Blue Team dashboard tabs.

Run this ONCE after train_baseline.py (which trains the neural net).

Usage:
    python scripts/setup_models.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, MODEL_DIR, ISOLATION_CONTAMINATION

RANDOM_SEED = 42


def main():
    print("=" * 60)
    print("  Setup: Generating sklearn Model Artifacts")
    print("=" * 60)

    # --- 1. Load data ---
    data_path = os.path.join(DATA_DIR, "combined_traffic.csv")
    if not os.path.exists(data_path):
        print(f"\n[ERROR] {data_path} not found.")
        print("This file should already be in the repo. Check your git clone.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\n[1/4] Loaded combined_traffic.csv: {df.shape}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    # --- 2. Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Attack ratio — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # --- 3. Save train/test CSVs (required by runner.py and train_multiseed.py) ---
    feature_cols = [c for c in df.columns if c != "label"]

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["label"] = y_train
    train_df.to_csv(os.path.join(MODEL_DIR, "train_set.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["label"] = y_test
    test_df.to_csv(os.path.join(MODEL_DIR, "test_set.csv"), index=False)
    print(f"\n[2/4] Saved train_set.csv and test_set.csv to {MODEL_DIR}/")

    # --- 4. Train Random Forest ---
    print("\n[3/4] Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    print(f"      Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print(f"      Saved random_forest.pkl")

    # --- 5. Train Isolation Forest (anomaly detector on benign traffic only) ---
    print("\n[4/4] Training Isolation Forest (anomaly detector)...")
    X_train_benign = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=100,
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_SEED
    )
    iso.fit(X_train_benign)
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    print(f"      Trained on {len(X_train_benign)} benign samples")
    print(f"      Saved isolation_forest.pkl")

    # --- 6. Save feature bounds ---
    clip_values = (X_train.min(axis=0), X_train.max(axis=0))
    joblib.dump(clip_values, os.path.join(MODEL_DIR, "feature_bounds.pkl"))
    print(f"      Saved feature_bounds.pkl")

    print("\n" + "=" * 60)
    print("  All model artifacts generated successfully!")
    print(f"  Models saved to: {MODEL_DIR}/")
    print("  You can now launch the dashboard:")
    print("    streamlit run src/dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
