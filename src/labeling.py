"""Label construction for early-warning event prediction."""

import pandas as pd


def label_future_spo2_drop(
    d: pd.DataFrame,
    dt_ms: float,
    horizon_sec: int,
    spo2_threshold: float
) -> pd.Series:
    # Compute min SpO2 in the next horizon window
    h = int((horizon_sec * 1000) / dt_ms)
    h = max(h, 1)
    future_min = d["SpO2"][::-1].rolling(window=h, min_periods=1).min()[::-1]
    y = (future_min < spo2_threshold).astype(int)
    return y