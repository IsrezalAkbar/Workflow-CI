"""
modelling.py — Kriteria 3 (MLflow Project)
Training model dengan parameter dari MLproject entry point
Dataset: Wine Quality (hasil preprocessing Kriteria 1)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────
# 1. Parse Arguments dari MLproject
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="Wine Quality — Random Forest Training")
parser.add_argument("--n_estimators",      type=int,   default=200)
parser.add_argument("--max_depth",         type=str,   default="10")
parser.add_argument("--min_samples_split", type=int,   default=2)
parser.add_argument("--min_samples_leaf",  type=int,   default=1)
args = parser.parse_args()

# Konversi max_depth: "None" → None, angka → int
max_depth = None if args.max_depth == "None" else int(args.max_depth)

print("=" * 60)
print("WORKFLOW CI — MLflow Project Training")
print("=" * 60)
print(f"Parameters:")
print(f"  n_estimators      : {args.n_estimators}")
print(f"  max_depth         : {max_depth}")
print(f"  min_samples_split : {args.min_samples_split}")
print(f"  min_samples_leaf  : {args.min_samples_leaf}")

# ─────────────────────────────────────────
# 2. Load Dataset Preprocessing
# ─────────────────────────────────────────
DATA_DIR = "dataset_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"\n✔ Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────
# 3. Training Model
# ─────────────────────────────────────────
with mlflow.start_run():
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ──
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    print(f"\n📊 Hasil Evaluasi:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC AUC  : {roc_auc:.4f}")

    # ── Log Parameters ──
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         str(max_depth))
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("class_weight",      "balanced")
    mlflow.log_param("random_state",      42)
    mlflow.log_param("train_size",        X_train.shape[0])
    mlflow.log_param("test_size",         X_test.shape[0])

    # ── Log Metrics ──
    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    # ── Confusion Matrix ──
    os.makedirs("artifacts", exist_ok=True)
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Good", "Good"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Wine Quality CI", fontsize=13)
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ── Feature Importance ──
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    feat_names  = X_train.columns.tolist()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(range(len(importances)), importances[indices], color="steelblue")
    ax2.set_xticks(range(len(importances)))
    ax2.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha="right")
    ax2.set_title("Feature Importances", fontsize=13)
    plt.tight_layout()
    fi_path = "artifacts/feature_importance.png"
    fig2.savefig(fi_path, dpi=120)
    plt.close()
    mlflow.log_artifact(fi_path)

    # ── Log Model ──
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="WineQuality_CI_Model",
        input_example=X_train.head(3),
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\n✔ MLflow run selesai. Run ID: {run_id}")

print("\n" + "=" * 60)
print("TRAINING SELESAI ✔")
print("=" * 60)
