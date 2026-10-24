"""Tests for the saved calibrated model and a smoke test of every dashboard page."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_FILE, PROCESSED_DIR, TARGET  # noqa: E402
from src.features import FEATURES  # noqa: E402
from src.model import CalibratedChurnModel, logit  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def model_and_test() -> tuple:
    test = pd.read_csv(PROCESSED_DIR / "test_set.csv", index_col=0)
    return joblib.load(MODEL_FILE), test


def test_model_is_calibrated(model_and_test: tuple) -> None:
    """Average predicted churn should match the real churn rate (within 3 pts)."""
    model, test = model_and_test
    prob = model.predict_proba(test[FEATURES])[:, 1]
    assert prob.min() >= 0 and prob.max() <= 1
    assert abs(prob.mean() - test[TARGET].mean()) < 0.03


def test_calibration_preserves_ranking() -> None:
    raw = np.array([0.1, 0.4, 0.7, 0.95])
    cal = CalibratedChurnModel(pipe=None, a=0.8, b=-0.9).calibrate(raw)
    assert np.all(np.diff(cal) > 0)
    assert np.allclose(logit(cal), 0.8 * logit(raw) - 0.9)


def test_shap_values_add_up(model_and_test: tuple) -> None:
    """SHAP contributions + base value must reproduce the model's log-odds."""
    from src.explain import ChurnExplainer

    model, test = model_and_test
    sample = test.head(20)
    ce = ChurnExplainer(model, test)
    total = ce.shap_values(sample).sum(axis=1).to_numpy() + ce.base_value
    assert np.allclose(total, logit(model.predict_proba(sample[FEATURES])[:, 1]), atol=1e-3)


@pytest.mark.parametrize("page", ["Overview", "Customer Explorer", "Predict", "Business Impact"])
def test_dashboard_page_runs(page: str) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
