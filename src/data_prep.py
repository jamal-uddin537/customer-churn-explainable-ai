"""Data cleaning for the Telco churn dataset.

Cleaning decisions (each is logged when the script runs):

1. ``TotalCharges`` is stored as text and contains blanks. Every blank row
   has ``tenure == 0`` -- these are brand-new customers who have not been
   billed yet -- so the blank is filled with ``0`` rather than dropped or
   imputed with a mean (which would invent spend that never happened).
2. Exact duplicate rows and duplicate customer IDs are removed (keep first).
3. ``SeniorCitizen`` is 0/1 while every other binary column is Yes/No; it is
   mapped to Yes/No for consistency in the dashboard and encoders.
4. Text columns are stripped of surrounding whitespace.
5. The target ``Churn`` is encoded to 1 (Yes) / 0 (No).
6. Any remaining missing numeric values are filled with the median, and
   categorical values with the mode (the real data has none after step 1;
   this is a safety net for new or synthetic data).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import CLEAN_FILE, ID_COL, RAW_FILE, TARGET  # noqa: E402

logger = logging.getLogger(__name__)

NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the raw CSV exactly as downloaded."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/get_data.py` first.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco DataFrame (see module docstring)."""
    df = df.copy()

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # 1. TotalCharges: text with blanks -> numeric; blanks are tenure-0 customers.
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(tc.isna().sum())
    blank_new = tc.isna() & (df["tenure"] == 0)
    tc[blank_new] = 0.0
    df["TotalCharges"] = tc
    logger.info("TotalCharges: %d blanks found, %d were tenure-0 -> set to 0",
                n_blank, int(blank_new.sum()))

    # 2. Duplicates.
    before = len(df)
    df = df.drop_duplicates()
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=ID_COL, keep="first")
    logger.info("Duplicates removed: %d", before - len(df))

    # 3. SeniorCitizen 0/1 -> Yes/No.
    if "SeniorCitizen" in df.columns and pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    # 6. Safety-net imputation.
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.columns.difference(NUMERIC_COLS + [TARGET]):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 5. Target encoding.
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    logger.info("Clean data: %d rows, churn rate %.1f%%", len(df), 100 * df[TARGET].mean())
    return df.reset_index(drop=True)


def main() -> None:
    """Clean the raw CSV and save it to data/processed/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = clean_data(load_raw())
    df.to_csv(CLEAN_FILE, index=False)
    logger.info("Saved -> %s", CLEAN_FILE.relative_to(CLEAN_FILE.parents[2]))


if __name__ == "__main__":
    main()
