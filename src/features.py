"""Feature engineering and the preprocessing pipeline.

Engineered features:
- ``tenure_group``: tenure bucketed into business-friendly bands.
- ``avg_monthly_spend``: TotalCharges / tenure (MonthlyCharges for new customers).
- ``num_services``: count of subscribed services (phone, lines, add-ons, streaming).
- ``is_month_to_month``: 1 if the customer has no long-term contract.
- ``auto_pay``: 1 if payment is an automatic bank transfer or credit card.
- ``charge_increase``: MonthlyCharges - avg_monthly_spend (positive = bill went up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLS: list[str] = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
TENURE_BINS: list[int] = [-1, 12, 24, 48, 72]
TENURE_LABELS: list[str] = ["0-12m", "13-24m", "25-48m", "49-72m"]

NUMERIC_FEATURES: list[str] = [
    "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
    "num_services", "is_month_to_month", "auto_pay", "charge_increase",
]
CATEGORICAL_FEATURES: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_group",
]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered features added."""
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    return df


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """One-hot encode categoricals; optionally scale numerics (needed for LogReg)."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
