"""
Visualization for model alerts on a single case.

Usage:
    python -m src.visualize --case 3 --target-recall 0.80
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from src.config import (
    DATA_ROOT, SPO2_THRESHOLD, HORIZON_SEC, WINDOW_SEC,
    DEFAULT_TARGET_RECALL, RESAMPLE_IF_DT_MS, RESAMPLE_BIN_MS, FIG_DIR
)
from src.data_io import case_path, load_case_trend_csv
from src.preprocess import extract_numeric_vitals, estimate_dt_ms, resample_to_1s
from src.features import make_features, feature_columns
from src.labeling import label_future_spo2_drop
from src.train import train_gb


def tune_threshold_by_recall(y_true, y_prob, target_recall):
    p, r, t = precision_recall_curve(y_true, y_prob)
    pr = pd.DataFrame({
        "threshold": np.r_[t, np.nan],
        "precision": p,
        "recall": r
    }).dropna()

    pr = pr[pr["recall"] >= target_recall]
    if len(pr) == 0:
        return 0.5

    return float(pr.sort_values(["precision", "recall"], ascending=False).iloc[0]["threshold"])


def build_case_dataset(case_id):
    path = case_path(DATA_ROOT, case_id)
    df = load_case_trend_csv(path)

    d = extract_numeric_vitals(df)
    dt = estimate_dt_ms(d)

    if dt < RESAMPLE_IF_DT_MS:
        d = resample_to_1s(d, RESAMPLE_BIN_MS)
        dt = estimate_dt_ms(d)

    y = label_future_spo2_drop(d, dt, HORIZON_SEC, SPO2_THRESHOLD)

    dd = make_features(d, dt, WINDOW_SEC)
    dd["y"] = y

    cols = feature_columns()
    dd = dd.dropna(subset=cols + ["y"]).copy()

    return dd, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    args = parser.parse_args()

    test_case = args.case

    cases = [1, 2, 3, 4, 5]

    test_df, cols = build_case_dataset(test_case)

    X_test = test_df[cols]
    y_test = test_df["y"].astype(int)

    train_parts = []
    for c in cases:
        if c == test_case:
            continue
        dfc, _ = build_case_dataset(c)
        train_parts.append(dfc)

    train_df = pd.concat(train_parts, ignore_index=True)

    X_train = train_df[cols]
    y_train = train_df["y"].astype(int)

    model = train_gb(X_train, y_train)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    thr = tune_threshold_by_recall(y_train, prob_train, args.target_recall)

    pred = (prob_test >= thr).astype(int)

    spo2 = test_df["SpO2"].values
    t = np.arange(len(spo2))

    plt.figure(figsize=(15, 6))

    plt.plot(t, spo2, label="SpO2", alpha=0.7)
    plt.axhline(SPO2_THRESHOLD, linestyle="--", label="Threshold")

    true_idx = np.where(y_test == 1)[0]
    plt.scatter(true_idx, spo2[true_idx], color="red", s=10, label="True Drop")

    alert_idx = np.where(pred == 1)[0]
    plt.scatter(alert_idx, spo2[alert_idx], color="green", s=12, label="Model Alerts")

    plt.title(f"Case {test_case} - Predictions")
    plt.legend()

    out = FIG_DIR / f"case{test_case:02d}_alerts.png"
    plt.savefig(out, dpi=200)

    print("Saved:", out)

    plt.show()


if __name__ == "__main__":
    main()