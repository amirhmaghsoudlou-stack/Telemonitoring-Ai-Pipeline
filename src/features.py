"""
Feature extraction.

Rolling-window statistics on HR and SpO2.
Includes short-term SpO2 features for fast drops.
"""

import pandas as pd


def make_features(d: pd.DataFrame, dt_ms: float, window_sec: int = 60) -> pd.DataFrame:
    w = int((window_sec * 1000) / dt_ms)
    w = max(w, 3)
    short_w = max(3, int(w / 3))

    dd = d.copy()

    # Long-window rolling stats
    dd["HR_mean"] = dd["HR"].rolling(w, min_periods=w).mean()
    dd["HR_std"]  = dd["HR"].rolling(w, min_periods=w).std()
    dd["HR_min"]  = dd["HR"].rolling(w, min_periods=w).min()
    dd["HR_max"]  = dd["HR"].rolling(w, min_periods=w).max()

    dd["SpO2_mean"] = dd["SpO2"].rolling(w, min_periods=w).mean()
    dd["SpO2_std"]  = dd["SpO2"].rolling(w, min_periods=w).std()
    dd["SpO2_min"]  = dd["SpO2"].rolling(w, min_periods=w).min()
    dd["SpO2_max"]  = dd["SpO2"].rolling(w, min_periods=w).max()

    # Slopes (long window)
    dd["HR_slope"] = (dd["HR"] - dd["HR"].shift(w)) / (dd["RelativeTimeMilliseconds"] - dd["RelativeTimeMilliseconds"].shift(w))
    dd["SpO2_slope"] = (dd["SpO2"] - dd["SpO2"].shift(w)) / (dd["RelativeTimeMilliseconds"] - dd["RelativeTimeMilliseconds"].shift(w))

    # Short-term features for fast drops
    dd["SpO2_diff_5"] = dd["SpO2"].diff(5)
    dd["SpO2_min_short"] = dd["SpO2"].rolling(short_w, min_periods=short_w).min()
    dd["SpO2_slope_short"] = (dd["SpO2"] - dd["SpO2"].shift(short_w)) / (dd["RelativeTimeMilliseconds"] - dd["RelativeTimeMilliseconds"].shift(short_w))

    return dd


def feature_columns() -> list:
    return [
        "HR_mean","HR_std","HR_min","HR_max","HR_slope",
        "SpO2_mean","SpO2_std","SpO2_min","SpO2_max","SpO2_slope",
        "SpO2_diff_5","SpO2_min_short","SpO2_slope_short",
    ]