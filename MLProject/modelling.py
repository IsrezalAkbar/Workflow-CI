import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Argparse ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",     type=int,   default=200)
parser.add_argument("--max_depth",        type=int,   default=10)
parser.add_argument("--min_samples_split",type=int,   default=2)
parser.add_argument("--min_samples_leaf", type=int,   default=1)
args = parser.parse_args()

print("=" * 60)
print("WORKFLOW CI — MLflow Project Training")
print("=" * 60)
print(f"Parameters:")
print(f"  n_estimators      : {args.n_estimators}")
print(f"  max_depth         : {args.max_depth}")
print(f"  min_samples_split : {args.min_samples_split}")
print(f"  min_samples_leaf  : {args.min_samples_leaf}")

# ── Load Data ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_preprocessing")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"✔ Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")

# ── MLflow Experiment ─────────────────────────────────────────
# Gunakan active run yang sudah di-inject oleh `mlflow run`
# Jangan panggil mlflow.start_run() manual saat pakai MLproject
mlflow.set_experiment("Wine_Quality_CI")

with mlflow.start_run(nested=False):
    # Log parameters
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)

    # ── Train ──────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators      = args.n_estimators,
        max_depth         = args.max_depth,
        min_samples_split = args.min_samples_split,
        min_samples_leaf  = args.min_samples_leaf,
        random_state      = 42,
        n_jobs            = -1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    print(f"\n✔ Metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {roc_auc:.4f}")

    # ── Confusion Matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(cm_path)

    # ── Feature Importance ─────────────────────────────────────
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    feat_imp.sort_values().plot(kind="barh", ax=ax2)
    ax2.set_title("Feature Importance")
    fi_path = "feature_importance.png"
    fig2.savefig(fi_path, bbox_inches="tight")
    plt.close(fig2)
    mlflow.log_artifact(fi_path)

    # ── Log Model ──────────────────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model            = model,
        artifact_path       = "model",
        registered_model_name = "WineQualityModel"
    )

    print(f"\n✔ Model logged to MLflow.")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")