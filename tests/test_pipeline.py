"""Unit tests for cleaning, features, business logic and recommendations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.get_data import generate_synthetic  # noqa: E402
from src.business import breakeven_probability, campaign_impact, optimal_threshold  # noqa: E402
from src.data_prep import clean_data  # noqa: E402
from src.explain import recommend_actions  # noqa: E402
from src.features import FEATURES, add_features, build_preprocessor  # noqa: E402


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """Small synthetic dataset with the real schema (no network needed)."""
    return generate_synthetic(n=500, seed=1)


def test_clean_fixes_total_charges_and_target(raw: pd.DataFrame) -> None:
    df = clean_data(raw)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df.isna().sum().sum() == 0
    assert (df.loc[df.tenure == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_clean_removes_duplicates(raw: pd.DataFrame) -> None:
    doubled = pd.concat([raw, raw.head(10)], ignore_index=True)
    assert len(clean_data(doubled)) == len(raw)


def test_features_are_created_and_sensible(raw: pd.DataFrame) -> None:
    df = add_features(clean_data(raw))
    assert set(FEATURES) <= set(df.columns)
    assert df["num_services"].between(0, 8).all()
    assert df["is_month_to_month"].isin([0, 1]).all()
    new = df[df.tenure == 0]
    assert np.allclose(new["avg_monthly_spend"], new["MonthlyCharges"])


def test_preprocessor_outputs_numeric_matrix(raw: pd.DataFrame) -> None:
    df = add_features(clean_data(raw))
    Xt = build_preprocessor().fit_transform(df[FEATURES])
    assert Xt.shape[0] == len(df) and np.isfinite(Xt).all()


def test_campaign_impact_math() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    r = campaign_impact(y, p, 0.5, clv=1000, offer_cost=100, success_rate=0.5)
    assert (r.churners_caught, r.wasted_offers, r.churners_missed) == (1, 1, 1)
    assert r.net_savings == pytest.approx(1 * 0.5 * 1000 - 2 * 100)


def test_optimal_threshold_beats_extremes() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    t, best = optimal_threshold(y, p)
    assert 0.05 <= t <= 0.95
    assert best >= campaign_impact(y, p, 0.95).net_savings
    assert best >= campaign_impact(y, p, 0.05).net_savings


def test_breakeven_probability() -> None:
    assert breakeven_probability(1500, 100, 0.3) == pytest.approx(100 / 450)
    assert breakeven_probability(0, 100, 0.3) == 1.0


def test_recommendations_follow_drivers() -> None:
    customer = pd.Series({"Contract": "Month-to-month", "tenure": 3, "PaymentMethod": "Electronic check",
                         "auto_pay": 0, "TechSupport": "No"})
    drivers = pd.DataFrame({"feature": ["Contract", "PaymentMethod"], "impact": [0.8, 0.3]})
    actions = recommend_actions(customer, drivers)
    assert any("contract" in a.lower() for a in actions)
    assert any("automatic payment" in a.lower() for a in actions)
    low = recommend_actions(customer, drivers.assign(impact=[-0.5, -0.2]))
    assert low[0].startswith("Low risk")
