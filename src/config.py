# Project config (English comments)

from pathlib import Path

# DATA_ROOT must point to the folder that contains: case01, case02, ... case32
DATA_ROOT = Path(r"E:\projects\Telemonitoring-Ai-Pipeline\data\uq-vitalsigns\uuqvitalsignsdata".replace("uuq", "uq"))  # safety replace

# ---- Task definition ----
SPO2_THRESHOLD = 98
HORIZON_SEC = 5 * 60      # 5 minutes
WINDOW_SEC = 60           # 60 seconds

# ---- Threshold tuning ----
DEFAULT_TARGET_RECALL = 0.80

# ---- Resampling ----
# If data is too dense (dt_ms < RESAMPLE_IF_DT_MS), resample to 1-second bins
RESAMPLE_IF_DT_MS = 500
RESAMPLE_BIN_MS = 1000

# ---- Output paths ----
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)