"""Calibrated churn model.

Why calibration?
----------------
Training with balanced class weights is good for *ranking* customers, but it
inflates predicted probabilities: before calibration the model predicted an
average churn rate of 41% on a population where the true rate is 26.5%.
Managers read "80% likely to leave" literally, and the cost model multiplies
probabilities by dollars, so the scores must be honest.

We apply **Platt scaling** on out-of-fold predictions: a 1-D logistic
regression ``p = sigmoid(a * logit(p_raw) + b)``. Because it is a linear map of
log-odds, the ranking (ROC-AUC) is unchanged and SHAP values computed on the raw
model stay exact after multiplying by ``a``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Numerically safe log-odds."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class CalibratedChurnModel:
    """A fitted preprocessing+model pipeline followed by Platt scaling."""

    def __init__(self, pipe: Pipeline, a: float = 1.0, b: float = 0.0) -> None:
        self.pipe = pipe
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def from_oof(cls, pipe: Pipeline, y: np.ndarray, oof_raw: np.ndarray) -> "CalibratedChurnModel":
        """Fit the Platt parameters on out-of-fold raw probabilities."""
        lr = LogisticRegression(C=1e6).fit(logit(oof_raw).reshape(-1, 1), y)
        return cls(pipe, lr.coef_[0, 0], lr.intercept_[0])

    def calibrate(self, raw_prob: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities."""
        z = self.a * logit(raw_prob) + self.b
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated ``[P(stay), P(churn)]`` like scikit-learn."""
        p = self.calibrate(self.pipe.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction at ``threshold``."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
