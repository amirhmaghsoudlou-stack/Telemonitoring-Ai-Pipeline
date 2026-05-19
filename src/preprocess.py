"""Preprocessing: select columns, convert types, sort, estimate dt, optional resampling."""

import pandas as pd


def extract_numeric_vitals(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns for this project
    cols = ["RelativeTimeMilliseconds", "HR", "SpO2"]
    if not all(c in df.columns for c in cols):
        raise KeyError(f"Missing required columns: {cols}")

    d = df[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["RelativeTimeMilliseconds"]).sort_values("RelativeTimeMilliseconds")
    return d


def estimate_dt_ms(d: pd.DataFrame) -> float:
    diffs = d["RelativeTimeMilliseconds"].diff()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan")
    return float(diffs.median())


def resample_to_1s(d: pd.DataFrame, bin_ms: int = 1000) -> pd.DataFrame:
    # Bin time into 1-second buckets and average signals
    dd = d.copy()
    dd["t_bin"] = (dd["RelativeTimeMilliseconds"] // bin_ms) * bin_ms
    out = dd.groupby("t_bin", as_index=False).agg({
        "RelativeTimeMilliseconds": "first",
        "HR": "mean",
        "SpO2": "mean",
    })
    out = out.sort_values("RelativeTimeMilliseconds")
    return out