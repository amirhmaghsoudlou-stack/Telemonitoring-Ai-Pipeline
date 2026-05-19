"""
Multi-case evaluation and threshold selection.

Run:
    python -m src.evaluate --cases 1 2 3 4 5 --target-recall 0.80

Outputs:
    results/metrics/summary_cases.json
    results/metrics/summary_by_target.json
"""

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)

from src.config import (
    DATA_ROOT, SPO2_THRESHOLD, HORIZON_SEC, WINDOW_SEC,
    DEFAULT_TARGET_RECALL, RESAMPLE_IF_DT_MS, RESAMPLE_BIN_MS, METRICS_DIR
)
from src.data_io import case_path, load_case_trend_csv
from src.preprocess import extract_numeric_vitals, estimate_dt_ms, resample_to_1s
from src.features import make_features, feature_columns
from src.labeling import label_future_spo2_drop
from src.train import train_gb


def tune_threshold_by_recall(y_true, y_prob, target_recall: float):
    # Choose threshold achieving recall >= target_recall and maximizing precision
    p, r, t = precision_recall_curve(y_true, y_prob)
    pr = pd.DataFrame({"threshold": np.r_[t, np.nan], "precision": p, "recall": r}).dropna()
    pr = pr[pr["recall"] >= target_recall]
    if len(pr) == 0:
        return 0.5
    return float(pr.sort_values(["precision", "recall"], ascending=False).iloc[0]["threshold"])


def eval_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def build_case_dataset(case_id: int):
    path = case_path(DATA_ROOT, case_id)
    df = load_case_trend_csv(path)
    d = extract_numeric_vitals(df)

    dt_ms = estimate_dt_ms(d)
    if np.isnan(dt_ms) or dt_ms <= 0:
        return None

    # Optional resampling if data is too dense
    if dt_ms < RESAMPLE_IF_DT_MS:
        d = resample_to_1s(d, RESAMPLE_BIN_MS)
        dt_ms = estimate_dt_ms(d)
        if np.isnan(dt_ms) or dt_ms <= 0:
            return None

    # Labels + features
    y = label_future_spo2_drop(d, dt_ms, HORIZON_SEC, SPO2_THRESHOLD)
    dd = make_features(d, dt_ms, WINDOW_SEC)
    dd["y"] = y

    cols = feature_columns()
    dd = dd.dropna(subset=cols + ["y"]).copy()

    X = dd[cols]
    y = dd["y"].astype(int)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", type=int, required=True)
    ap.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    args = ap.parse_args()

    cases = args.cases
    target_recall = args.target_recall

    rows = []

    # Leave-one-case-out style evaluation
    for test_case in cases:
        test_pack = build_case_dataset(test_case)
        if test_pack is None:
            rows.append({"test_case": test_case, "status": "build_failed"})
            continue

        X_test, y_test = test_pack
        if y_test.nunique() < 2:
            rows.append({"test_case": test_case, "status": "one_class_test"})
            continue

        # Build training set from remaining cases
        train_Xs, train_ys = [], []
        for train_case in cases:
            if train_case == test_case:
                continue
            pack = build_case_dataset(train_case)
            if pack is None:
                continue
            Xc, yc = pack
            if yc.nunique() < 2:
                continue
            train_Xs.append(Xc)
            train_ys.append(yc)

        if len(train_Xs) == 0:
            rows.append({"test_case": test_case, "status": "no_train"})
            continue

        X_train = pd.concat(train_Xs, ignore_index=True)
        y_train = pd.concat(train_ys, ignore_index=True)
        if y_train.nunique() < 2:
            rows.append({"test_case": test_case, "status": "one_class_train"})
            continue

        model = train_gb(X_train, y_train)
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        thr = tune_threshold_by_recall(y_train, prob_train, target_recall)
        auc = float(roc_auc_score(y_test, prob_test))
        m = eval_at_threshold(y_test, prob_test, thr)

        rows.append({
            "test_case": test_case,
            "status": "ok",
            "target_recall": float(target_recall),
            "threshold": float(thr),
            "roc_auc": auc,
            **m,
        })

    out_df = pd.DataFrame(rows)

    out_cases = METRICS_DIR / "summary_cases.json"
    out_df.to_json(out_cases, orient="records", indent=2)

    ok = out_df[out_df["status"] == "ok"].copy()
    agg = ok[["roc_auc", "precision", "recall", "f1", "FP", "FN"]].mean(numeric_only=True).to_dict() if len(ok) else {}
    agg["n_ok"] = int(len(ok))
    agg["n_total"] = int(len(out_df))
    agg["target_recall"] = float(target_recall)

    out_agg = METRICS_DIR / "summary_by_target.json"
    with open(out_agg, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print("Saved:", out_cases)
    print("Saved:", out_agg)
    print("Aggregate:", agg)


if __name__ == "__main__":
    main()