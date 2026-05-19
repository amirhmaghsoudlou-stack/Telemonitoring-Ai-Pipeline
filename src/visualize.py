"""
Visualization for model alerts on a single case.

Run (from project root):
    python -m src.visualize --case 3 --target-recall 0.80

Outputs:
    results/figures/caseXX_alerts.png
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


def tune_threshold_by_recall(y_true, y_prob, target_recall: float) -> float:
    # Choose threshold achieving recall >= target_recall and maximizing precision
    p, r, t = precision_recall_curve(y_true, y_prob)
    pr = pd.DataFrame({"threshold": np.r_[t, np.nan], "precision": p, "recall": r}).dropna()
    pr = pr[pr["recall"] >= target_recall]
    if len(pr) == 0:
        return 0.5
    return float(pr.sort_values(["precision", "recall"], ascending=False).iloc[0]["threshold"])


def build_case_dataset(case_id: int):
    # Load and preprocess one case, then build features + labels
    path = case_path(DATA_ROOT, case_id)
    df = load_case_trend_csv(path)

    d = extract_numeric_vitals(df)
    dt = estimate_dt_ms(d)
    if np.isnan(dt) or dt <= 0:
        return None

    # Optional resampling if too dense
    if dt < RESAMPLE_IF_DT_MS:
        d = resample_to_1s(d, RESAMPLE_BIN_MS)
        dt = estimate_dt_ms(d)
        if np.isnan(dt) or dt <= 0:
            return None

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
    target_recall = args.target_recall

    cases = [1, 2, 3, 4, 5]

    # Build test set
    pack = build_case_dataset(test_case)
    if pack is None:
        print("Case build failed")
        return

    test_df, cols = pack
    X_test = test_df[cols]
    y_test = test_df["y"].astype(int)

    # Build training set from remaining cases
    train_parts = []
    for c in cases:
        if c == test_case:
            continue
        p = build_case_dataset(c)
        if p is None:
            continue
        dfc, _ = p
        # skip single-class training parts
        if dfc["y"].nunique() < 2:
            continue
        train_parts.append(dfc)

    if not train_parts:
        print("No training data")
        return

    train_df = pd.concat(train_parts, ignore_index=True)
    X_train = train_df[cols]
    y_train = train_df["y"].astype(int)

    # Train model
    model = train_gb(X_train, y_train)

    # Predict probabilities
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    # Threshold selection
    thr = tune_threshold_by_recall(y_train, prob_train, target_recall)
    pred = (prob_test >= thr).astype(int)

    # Plot
    spo2 = test_df["SpO2"].to_numpy()
    t = np.arange(len(spo2))

    plt.figure(figsize=(15, 6))
    plt.plot(t, spo2, label="SpO2", alpha=0.7)
    plt.axhline(SPO2_THRESHOLD, linestyle="--", label=f"Threshold ({SPO2_THRESHOLD})")

    true_idx = np.where(y_test.to_numpy() == 1)[0]
    plt.scatter(true_idx, spo2[true_idx], color="red", s=10, label="True drop")

    alert_idx = np.where(pred == 1)[0]
    plt.scatter(alert_idx, spo2[alert_idx], color="green", s=12, label="Model alerts")

    plt.title(f"Case {test_case} — Alerts (target recall={target_recall}, thr={thr:.3f})")
    plt.xlabel("Time index")
    plt.ylabel("SpO2")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / f"case{test_case:02d}_alerts.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)

    # Do not block terminal runs
    plt.close()


if __name__ == "__main__":
    main()