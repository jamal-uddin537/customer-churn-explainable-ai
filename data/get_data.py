"""Download the IBM Telco Customer Churn dataset, or generate a synthetic copy.

Usage:
    python data/get_data.py              # download (falls back to synthetic)
    python data/get_data.py --synthetic  # force synthetic data

The synthetic generator reproduces the real schema and the main churn
patterns (month-to-month contracts, short tenure, electronic check and
fiber optic customers churn more) so the full pipeline still works offline.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_URL, RAW_FILE, SEED  # noqa: E402


def download_dataset(url: str = DATA_URL, timeout: int = 30) -> pd.DataFrame:
    """Download the public IBM Telco churn CSV and return it as a DataFrame."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        content = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(content))


def generate_synthetic(n: int = 7043, seed: int = SEED) -> pd.DataFrame:
    """Generate a realistic synthetic dataset with the Telco schema."""
    rng = np.random.default_rng(seed)
    yn = np.array(["Yes", "No"])

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure_mean = np.select(
        [contract == "Month-to-month", contract == "One year"], [18, 42], default=57
    )
    tenure = np.clip(rng.normal(tenure_mean, 15), 0, 72).round().astype(int)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    has_net = internet != "No"

    def addon(p: float) -> np.ndarray:
        vals = np.where(rng.random(n) < p, "Yes", "No").astype(object)
        vals[~has_net] = "No internet service"
        return vals

    phone = np.where(rng.random(n) < 0.9, "Yes", "No")
    multi = np.where(phone == "Yes", rng.choice(yn, n, p=[0.47, 0.53]), "No phone service")
    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-SYN{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
        "Partner": rng.choice(yn, n, p=[0.48, 0.52]),
        "Dependents": rng.choice(yn, n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": addon(0.37), "OnlineBackup": addon(0.44),
        "DeviceProtection": addon(0.44), "TechSupport": addon(0.37),
        "StreamingTV": addon(0.49), "StreamingMovies": addon(0.5),
        "Contract": contract,
        "PaperlessBilling": rng.choice(yn, n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]),
    })
    base = np.select([internet == "Fiber optic", internet == "DSL"], [70, 45], default=20)
    extras = sum((df[c] == "Yes").astype(int) for c in
                 ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies", "MultipleLines"])
    df["MonthlyCharges"] = np.round(base + 5 * extras + rng.normal(0, 3, n), 2).clip(18.25, 118.75)
    total = df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.95, 1.05, n)
    df["TotalCharges"] = total.round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # mimic real blank values

    logit = (-1.6 + 1.4 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             - 0.035 * tenure + 0.7 * (internet == "Fiber optic")
             + 0.6 * (df["PaymentMethod"] == "Electronic check")
             - 0.4 * (df["TechSupport"] == "Yes") - 0.35 * (df["OnlineSecurity"] == "Yes")
             + 0.3 * df["SeniorCitizen"] + 0.3 * (df["PaperlessBilling"] == "Yes"))
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")
    return df


def main() -> None:
    """Fetch the data (or synthesise it) and write it to data/raw/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="force synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        df, source = generate_synthetic(), "synthetic"
    else:
        try:
            df, source = download_dataset(), "IBM Telco (downloaded)"
        except Exception as exc:  # network errors, timeouts, etc.
            print(f"[warn] download failed ({exc}); generating synthetic data instead")
            df, source = generate_synthetic(), "synthetic"

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    (RAW_FILE.parent / "SOURCE.txt").write_text(source + "\n")
    print(f"Saved {len(df):,} rows from {source} -> {RAW_FILE.relative_to(RAW_FILE.parents[2])}")


if __name__ == "__main__":
    main()
