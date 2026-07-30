"""
CollideX -- Step 5: Collision Probability Prediction Model
=============================================================
Trains a Random Forest classifier on ml_dataset.csv and outputs:
  - collision_model.pkl            (trained model)
  - collision_probabilities.csv    (per-sample probability predictions)

Outputs (3 required):
  future_position_coordinates  (from Step 3)  [future_positions.csv]
  risk_class                   (from Step 4)  [ml_dataset.csv]
  collision_probability        (from Step 5)  [collision_probabilities.csv]
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(SCRIPT_DIR, "..")
DATASET_FILE = os.path.join(ROOT_DIR, "data", "processed", "ml_dataset.csv")
MODEL_DIR    = os.path.join(ROOT_DIR, "models")
MODEL_FILE   = os.path.join(MODEL_DIR, "collision_model.pkl")
PROB_FILE    = os.path.join(ROOT_DIR, "data", "processed", "collision_probabilities.csv")

LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# ===========================================================================
# STEP 5.1 -- Load Dataset
# ===========================================================================
def step_5_1_load(path: str):
    print("\n[Step 5.1] Loading ml_dataset.csv ...")
    df = pd.read_csv(path)
    print(f"  Rows    : {len(df):,}")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Target distribution:")
    dist = df["risk_class"].value_counts().sort_index()
    for cls, count in dist.items():
        pct = count / len(df) * 100
        print(f"    Class {cls} ({LABELS[cls]:11s}): {count:>8,}  ({pct:.1f}%)")

    X = df.drop("risk_class", axis=1)
    y = df["risk_class"]
    return X, y


# ===========================================================================
# STEP 5.2 -- Train / Test Split
# ===========================================================================
def step_5_2_split(X, y):
    print("\n[Step 5.2] Splitting train/test sets (80/20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test  samples: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# STEP 5.3 -- Train Random Forest
# ===========================================================================
def step_5_3_train(X_train, y_train) -> RandomForestClassifier:
    print("\n[Step 5.3] Training RandomForestClassifier ...")
    print("  n_estimators = 200  |  max_depth = 12  |  n_jobs = -1")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance (73/19/7 split)
        random_state=42,
        n_jobs=-1,                 # use all CPU cores
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")
    return model


# ===========================================================================
# STEP 5.4 -- Evaluate Model
# ===========================================================================
def step_5_4_evaluate(model, X_test, y_test):
    print("\n[Step 5.4] Evaluating model on test set ...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc * 100:.2f}%")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=[LABELS[i] for i in sorted(LABELS)],
        digits=4,
    )
    print("\n  Classification Report:")
    for line in report.splitlines():
        print("   ", line)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "  {:15s}".format("") + "".join(f"  {LABELS[i]:12s}" for i in range(3))
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {LABELS[i]:15s}" + "".join(f"  {v:12,}" for v in row)
        print(row_str)

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        print(f"\n  ROC-AUC (macro OVR): {auc:.4f}")
    except Exception:
        pass

    # Top-5 feature importances
    feat_names    = list(X_test.columns)
    importances   = model.feature_importances_
    top5_idx      = np.argsort(importances)[::-1][:5]
    print("\n  Top-5 Feature Importances:")
    for rank, i in enumerate(top5_idx, 1):
        print(f"    {rank}. {feat_names[i]:25s}  {importances[i]:.4f}")

    return y_pred, y_prob, acc


# ===========================================================================
# STEP 5.5 -- Save Probability Predictions
# ===========================================================================
def step_5_5_save_probs(y_prob, output_path: str):
    print("\n[Step 5.5] Saving collision_probabilities.csv ...")
    prob_df = pd.DataFrame(
        y_prob,
        columns=["low_risk_prob", "medium_risk_prob", "high_risk_prob"],
    )
    prob_df["predicted_risk_class"] = np.argmax(y_prob, axis=1)
    prob_df["predicted_risk_label"] = prob_df["predicted_risk_class"].map(LABELS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prob_df.to_csv(output_path, index=False)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved  -> {output_path}")
    print(f"  Rows   : {len(prob_df):,}")
    print(f"  Size   : {size_kb:.1f} KB")
    print("\n  Sample probability predictions (first 5 rows):")
    print(prob_df.head(5).to_string(index=False))
    return prob_df


# ===========================================================================
# STEP 5.6 -- Save Trained Model
# ===========================================================================
def step_5_6_save_model(model, output_path: str):
    print("\n[Step 5.6] Saving trained model ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"  Saved  -> {output_path}")
    print(f"  Size   : {size_mb:.2f} MB")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  CollideX -- Step 5: RandomForest Training Engine")
    print("=" * 62)

    # 5.1 Load
    X, y = step_5_1_load(DATASET_FILE)

    # 5.2 Split
    X_train, X_test, y_train, y_test = step_5_2_split(X, y)

    # 5.3 Train
    model = step_5_3_train(X_train, y_train)

    # 5.4 Evaluate
    y_pred, y_prob, acc = step_5_4_evaluate(model, X_test, y_test)

    # 5.5 Save probabilities
    step_5_5_save_probs(y_prob, PROB_FILE)

    # 5.6 Save model
    step_5_6_save_model(model, MODEL_FILE)

    # Summary banner
    print("\n" + "=" * 62)
    print("  STEP 5 COMPLETE -- CollideX Baseline AI Engine Ready")
    print("=" * 62)
    print(f"\n  Model Accuracy     : {acc * 100:.2f}%")
    print(f"  Trained model      : CollideX/models/collision_model.pkl")
    print(f"  Probability output : CollideX/data/processed/collision_probabilities.csv")
    print("\n  Pipeline outputs now active:")
    print("    [OK] future_positions.csv          -- SGP4 trajectory coordinates")
    print("    [OK] ml_dataset.csv (risk_class)   -- collision risk labels")
    print("    [OK] collision_probabilities.csv   -- per-event risk probabilities")
    print("\n  Ready for Step 6 -- Model Evaluation & Tuning\n")
