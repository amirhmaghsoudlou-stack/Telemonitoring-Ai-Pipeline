"""
Data IO utilities.

The raw CSV files sometimes contain malformed rows (extra delimiters).
We use pandas' python engine and skip bad lines.

Comments are in English.
"""

import pandas as pd
from pathlib import Path


def case_path(data_root: Path, case_id: int) -> Path:
    cid = f"case{case_id:02d}"
    return data_root / cid / f"uq_vsd_{cid}_trenddata.csv"


def load_case_trend_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        header=0,
        engine="python",
        on_bad_lines="skip",
        index_col=False,
    )
    return df