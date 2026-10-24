"""Central configuration: paths, random seed, and business assumptions.

All paths are built relative to the repository root so the project runs
anywhere (locally, in CI, or on Streamlit Community Cloud).
"""
from __future__ import annotations

from pathlib import Path

SEED: int = 42

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_FILE: Path = RAW_DIR / "telco_churn.csv"
CLEAN_FILE: Path = PROCESSED_DIR / "telco_churn_clean.csv"
MODEL_FILE: Path = MODELS_DIR / "churn_model.joblib"
METRICS_FILE: Path = REPORTS_DIR / "metrics.json"

DATA_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TARGET: str = "Churn"
ID_COL: str = "customerID"

# --- Business assumptions (editable in the dashboard) -----------------------
# Customer lifetime value lost when a churner is missed (USD).
DEFAULT_CLV: float = 1_500.0
# Cost of a retention offer (discount / free month / support credit) (USD).
DEFAULT_OFFER_COST: float = 100.0
# Share of contacted would-be churners who stay because of the offer.
DEFAULT_SUCCESS_RATE: float = 0.30

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
