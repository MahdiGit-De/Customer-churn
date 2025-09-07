"""
Customer Churn Prediction (IBM Telco Dataset)

This script downloads the IBM Telco Churn dataset (if not present),
preprocesses it, trains a LightGBM model, evaluates performance, and
generates SHAP interpretability plots.

Run:
  python demo4.py

Outputs:
  - outputs/roc_curve.png
  - outputs/confusion_matrix.png
  - outputs/shap_summary.png

Dependencies: pandas, scikit-learn, lightgbm, shap, matplotlib, seaborn
"""

import os
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("LightGBM is required. Please install via `pip install lightgbm`.\n" + str(e))

try:
    import shap
    shap.logger.setLevel("ERROR")  # reduce verbosity
except Exception as e:
    raise SystemExit("SHAP is required. Please install via `pip install shap`.\n" + str(e))


PROJECT_NAME = "Customer Churn Prediction"


def ensure_dirs() -> Tuple[Path, Path, Path]:
    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    return data_dir, out_dir, models_dir


def load_telco_dataset(data_dir: Path) -> pd.DataFrame:
    """Load or download the IBM Telco Churn dataset.

    Tries multiple known mirrors for robustness.
    """
    local_path = data_dir / "telco_churn.csv"
    if local_path.exists():
        return pd.read_csv(local_path)

    urls = [
        # Community mirror (commonly referenced)
        "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        # IBM sample (column names may slightly differ)
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    ]
    last_err = None
    for u in urls:
        try:
            df = pd.read_csv(u)
            df.to_csv(local_path, index=False)
            print(f"Downloaded dataset to {local_path}")
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download dataset. Last error: {last_err}")


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], ColumnTransformer]:
    """Prepare features and target with robust handling for common Telco variants."""
    df = df.copy()

    # Standardize common columns
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Normalize target column name / values
    target_col = None
    for cand in ["Churn", "churn", "CHURN"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        raise ValueError("Could not find target column 'Churn' in dataset.")

    # Convert target to binary 0/1
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = df[target_col].map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(df[target_col])
    if df[target_col].dtype == object:
        # If it wasn't a clean yes/no, try to convert to int
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Fix TotalCharges (often has blanks)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Identify categorical vs numerical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocessors
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ],
        remainder="drop",
    )

    return X, y, cat_cols + num_cols, preprocessor


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, out_dir: Path) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save metrics
    metrics = {"accuracy": acc, "f1": f1, "roc_auc": auc}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Customer Churn")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Customer Churn")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    return clf


def shap_explain(pipeline: Pipeline, X: pd.DataFrame, out_dir: Path, sample_size: int = 800) -> None:
    pre = pipeline.named_steps["preprocess"]
    model: lgb.LGBMClassifier = pipeline.named_steps["model"]

    # Sample a subset for faster SHAP computation
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    X_trans = pre.transform(X_sample)
    # Ensure dense matrix for SHAP
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Feature names from preprocessor
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        # Fallback: generic names
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    booster = model.booster_ if hasattr(model, "booster_") else model
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_trans)

    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_use = shap_values[1]
    else:
        shap_values_to_use = shap_values

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values_to_use, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close()


def main() -> None:
    print(f"=== {PROJECT_NAME} ===")
    data_dir, out_dir, models_dir = ensure_dirs()

    print("Loading dataset ...")
    df = load_telco_dataset(data_dir)
    print(f"Rows: {len(df):,} | Cols: {len(df.columns)}")

    print("Preprocessing ...")
    X, y, cols, pre = preprocess(df)
    print(f"Features: {len(cols)} | Positive rate: {y.mean():.3f}")

    print("Training model ...")
    pipeline = train_and_evaluate(X, y, pre, out_dir)

    # Save LightGBM model
    model: lgb.LGBMClassifier = pipeline.named_steps["model"]
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(models_dir / "lgbm_churn.txt"))
        print(f"Saved model to {models_dir / 'lgbm_churn.txt'}")

    print("Generating SHAP explanations ... (this may take ~seconds)")
    shap_explain(pipeline, X, out_dir)
    print(f"Done. See outputs in '{out_dir}'.")


if __name__ == "__main__":
    main()
