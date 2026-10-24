"""Evaluate the trained model on the held-out test set.

Threshold choice
----------------
A default cut-off of 0.5 treats a missed churner and a wasted offer as equally
bad. They are not: with the default assumptions a missed churner costs
``0.30 * $1,500 = $450`` of expected saved value, while a wasted offer costs
``$100``. We therefore pick the threshold that **maximises net savings on the
out-of-fold training predictions** (never the test set, to avoid optimistic
bias), then report test-set performance at that threshold.

Because the model is calibrated (see ``src/model.py``), the optimal cut-off
lands close to the theoretical break-even ``offer_cost / (success_rate * clv)
= 100 / 450 ≈ 0.22``. The dashboard recomputes the optimum live when the
assumptions change.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.business import campaign_impact, optimal_threshold  # noqa: E402
from src.config import (DEFAULT_CLV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE, FIGURES_DIR,  # noqa: E402
                        METRICS_FILE, MODEL_FILE, MODELS_DIR, PROCESSED_DIR, TARGET)
from src.features import FEATURES  # noqa: E402

logger = logging.getLogger(__name__)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Return ROC-AUC, PR-AUC, precision, recall, F1 and confusion counts."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": round(float(threshold), 2),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def plot_threshold_curve(y_true: np.ndarray, y_prob: np.ndarray, best_t: float, path: Path) -> None:
    """Plot net savings against the decision threshold."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    sav = [campaign_impact(y_true, y_prob, t).net_savings for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, np.array(sav) / 1000, lw=2, color="#2E86AB")
    ax.axvline(best_t, ls="--", color="#DD8452", label=f"Chosen threshold = {best_t:.2f}")
    ax.axvline(0.5, ls=":", color="grey", label="Default 0.5")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Net savings on test set ($k)")
    ax.set_title("Choosing the threshold that maximises savings"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> dict:
    """Evaluate the saved model and write metrics + figures to reports/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    pipe = joblib.load(MODEL_FILE)
    test = pd.read_csv(PROCESSED_DIR / "test_set.csv", index_col=0)
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    X_te, y_te = test[FEATURES], test[TARGET].to_numpy()
    prob = pipe.predict_proba(X_te)[:, 1]
    prob_raw = pipe.pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "pipe") else prob

    best_t, _ = optimal_threshold(oof["y"].to_numpy(), oof["prob"].to_numpy())
    logger.info("Cost-optimal threshold (from out-of-fold train predictions): %.2f", best_t)

    m_default = classification_metrics(y_te, prob, 0.5)
    m_tuned = classification_metrics(y_te, prob, best_t)
    imp_default = campaign_impact(y_te, prob, 0.5).to_dict()
    imp_tuned = campaign_impact(y_te, prob, best_t).to_dict()
    # "Contact everyone" baseline for context.
    imp_all = campaign_impact(y_te, prob, 0.0).to_dict()

    info = json.loads((MODELS_DIR / "model_info.json").read_text())

    # Figures --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fpr, tpr, _ = roc_curve(y_te, prob)
    ax.plot(fpr, tpr, lw=2, color="#2E86AB", label=f"{info['model']} (AUC = {m_tuned['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, (prob >= best_t).astype(int), display_labels=["Stay", "Churn"], cmap="Blues", ax=ax,
        colorbar=False)
    ax.set_title(f"Confusion matrix (threshold {best_t:.2f})")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)
    plot_threshold_curve(y_te, prob, best_t, FIGURES_DIR / "threshold_savings.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p_, lab, c in [(prob_raw, "Before calibration", "#B0B0B0"), (prob, "After calibration", "#2E86AB")]:
        frac, mean_p = calibration_curve(y_te, p_, n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=c, lw=2, label=f"{lab} (Brier {brier_score_loss(y_te, p_):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration: can the probabilities be trusted?"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=130); plt.close(fig)

    metrics = {
        "model": info["model"], "cv_roc_auc": info["cv_roc_auc_tuned"],
        "n_test": int(len(y_te)), "test_churn_rate": round(float(y_te.mean()), 4),
        "mean_predicted_prob": round(float(prob.mean()), 4),
        "mean_predicted_prob_uncalibrated": round(float(prob_raw.mean()), 4),
        "brier_uncalibrated": round(brier_score_loss(y_te, prob_raw), 4),
        "assumptions": {"clv": DEFAULT_CLV, "offer_cost": DEFAULT_OFFER_COST,
                        "success_rate": DEFAULT_SUCCESS_RATE},
        "default_threshold": m_default, "tuned_threshold": m_tuned,
        "impact_default": imp_default, "impact_tuned": imp_tuned, "impact_contact_all": imp_all,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    print("\n=== Test-set results ===")
    print(pd.DataFrame([m_default, m_tuned], index=["t=0.50", f"t={best_t:.2f}"]).to_string())
    print(pd.DataFrame([imp_all, imp_default, imp_tuned],
                       index=["contact all", "t=0.50", f"t={best_t:.2f}"])
          [["customers_targeted", "churners_caught", "wasted_offers", "campaign_cost",
            "revenue_saved", "net_savings", "roi"]].to_string())
    return metrics


if __name__ == "__main__":
    main()
