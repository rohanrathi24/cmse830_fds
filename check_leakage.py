"""
check_leakage.py — read-only diagnostic for the stroke prediction project.

This script does NOT modify stroke.py, the .pkl model, or any data file.
It loads the CSV, builds two pipelines that differ ONLY in the order of
SMOTE and train_test_split, and prints the metrics side by side.

Usage:
    python check_leakage.py

Expects healthcare-dataset-stroke-data.csv in the same directory
(or pass a path: python check_leakage.py path/to/file.csv).

Requires: pandas, scikit-learn, imbalanced-learn, xgboost
    pip install pandas scikit-learn imbalanced-learn xgboost
"""

import sys

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

CSV = sys.argv[1] if len(sys.argv) > 1 else "healthcare-dataset-stroke-data.csv"
SEED = 42
TEST_SIZE = 0.22


def load_data(path):
    """Same preprocessing as stroke.py: drop id, median-impute bmi, label-encode."""
    df = pd.read_csv(path).drop("id", axis=1)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    for col in [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def make_model():
    """Exact hyperparameters from stroke.py's tuned XGBoost."""
    return XGBClassifier(
        objective="reg:logistic",
        random_state=SEED,
        colsample_bytree=0.5,
        gamma=0.2,
        learning_rate=0.25,
        max_depth=10,
        min_child_weight=1,
        eval_metric="logloss",
    )


def report(label, y_true, y_pred, y_prob):
    prevalence = y_true.mean()
    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"  test set        {len(y_true)} rows, "
          f"{int(y_true.sum())} strokes ({prevalence * 100:.1f}%)")
    print(f"  accuracy        {accuracy_score(y_true, y_pred):.4f}")
    print(f"  recall          {recall_score(y_true, y_pred):.4f}   <-- strokes caught")
    print(f"  precision       {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1              {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC         {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  PR-AUC          {average_precision_score(y_true, y_prob):.4f}"
          f"   (no-skill baseline {prevalence:.4f})")


def main():
    df = load_data(CSV)
    features = df.loc[:, df.columns != "stroke"]
    target = df["stroke"]

    print(f"\nLoaded {len(df)} rows, "
          f"{int(target.sum())} strokes ({target.mean() * 100:.1f}% positive)")

    # ---- Pipeline A: current stroke.py — SMOTE on ALL data, THEN split ----
    X_a, y_a = SMOTE(sampling_strategy="minority", random_state=SEED).fit_resample(
        features, target
    )
    Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(
        X_a, y_a, test_size=TEST_SIZE, random_state=SEED
    )
    model_a = make_model().fit(Xtr_a, ytr_a)
    report(
        "A. CURRENT stroke.py  (SMOTE before split -- leaks)",
        yte_a,
        model_a.predict(Xte_a),
        model_a.predict_proba(Xte_a)[:, 1],
    )

    # ---- Pipeline B: split FIRST, SMOTE on training data only ----
    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED, stratify=target
    )
    Xtr_bs, ytr_bs = SMOTE(random_state=SEED).fit_resample(Xtr_b, ytr_b)
    model_b = make_model().fit(Xtr_bs, ytr_bs)
    prob_b = model_b.predict_proba(Xte_b)[:, 1]
    report(
        "B. FIXED  (split first, SMOTE on train only)",
        yte_b,
        model_b.predict(Xte_b),
        prob_b,
    )

    # ---- Threshold sweep on the fixed model ----
    print(f"\n{'=' * 62}\nThreshold sweep on the FIXED model\n{'=' * 62}")
    print(f"  {'threshold':<12}{'recall':<10}{'precision':<12}{'F1':<8}")
    for t in [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]:
        pred = (prob_b >= t).astype(int)
        print(f"  {t:<12.2f}{recall_score(yte_b, pred):<10.3f}"
              f"{precision_score(yte_b, pred, zero_division=0):<12.3f}"
              f"{f1_score(yte_b, pred, zero_division=0):<8.3f}")

    print("\nNothing was written to disk. stroke.py and the .pkl are untouched.\n")


if __name__ == "__main__":
    main()
