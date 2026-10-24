"""Business-impact model: turn predictions into dollars.

Campaign logic: every customer the model flags receives a retention offer.

* True positive  (churner, flagged)     -> pay ``offer_cost``; with probability
  ``success_rate`` the customer stays and we keep ``clv``.
* False positive (loyal, flagged)       -> pay ``offer_cost`` for nothing.
* False negative (churner, not flagged) -> customer leaves (same as no campaign).
* True negative                          -> nothing happens.

Net savings versus doing nothing::

    savings = TP * (success_rate * clv - offer_cost) - FP * offer_cost

Maximising savings is equivalent to minimising a cost where a missed churner
costs ``success_rate * clv`` and a wasted offer costs ``offer_cost``. Because a missed
churner is several times more expensive than a wasted offer, the best
threshold trades precision for recall.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.config import DEFAULT_CLV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE


@dataclass
class ImpactResult:
    """Outcome of a retention campaign at a given threshold."""

    threshold: float
    customers_targeted: int
    churners_caught: int
    churners_missed: int
    wasted_offers: int
    customers_saved: float
    campaign_cost: float
    revenue_saved: float
    net_savings: float
    roi: float

    def to_dict(self) -> dict:
        """Return the result as a plain dictionary."""
        return asdict(self)


def campaign_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    clv: float = DEFAULT_CLV,
    offer_cost: float = DEFAULT_OFFER_COST,
    success_rate: float = DEFAULT_SUCCESS_RATE,
) -> ImpactResult:
    """Compute the financial result of targeting customers with ``y_prob >= threshold``."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    saved = tp * success_rate
    cost = (tp + fp) * offer_cost
    revenue = saved * clv
    net = revenue - cost
    return ImpactResult(
        threshold=float(threshold), customers_targeted=tp + fp, churners_caught=tp,
        churners_missed=fn, wasted_offers=fp, customers_saved=round(saved, 1),
        campaign_cost=round(cost, 2), revenue_saved=round(revenue, 2),
        net_savings=round(net, 2), roi=round(net / cost, 3) if cost else 0.0,
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clv: float = DEFAULT_CLV,
    offer_cost: float = DEFAULT_OFFER_COST,
    success_rate: float = DEFAULT_SUCCESS_RATE,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(threshold, net_savings)`` that maximises net savings on a grid."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    savings = [campaign_impact(y_true, y_prob, t, clv, offer_cost, success_rate).net_savings
               for t in grid]
    best = int(np.argmax(savings))
    return float(grid[best]), float(savings[best])


def breakeven_probability(clv: float, offer_cost: float, success_rate: float) -> float:
    """Churn probability above which sending an offer has positive expected value.

    Expected gain of contacting a customer with churn probability p is
    ``p * success_rate * clv - offer_cost`` -> break-even at
    ``offer_cost / (success_rate * clv)``.
    """
    denom = success_rate * clv
    return float(min(1.0, offer_cost / denom)) if denom > 0 else 1.0
