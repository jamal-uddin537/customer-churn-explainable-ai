"""Explainability (SHAP) and AI-generated retention recommendations.

* ``ChurnExplainer`` wraps a fitted pipeline and returns SHAP values aggregated
  back to the *original* business features (one-hot columns are summed), so
  explanations read "Contract = Month-to-month" rather than "Contract_0".
* ``recommend_actions`` turns a customer's top risk drivers into a short,
  plain-English retention plan using business rules.
* ``llm_recommendation`` optionally rewrites that plan with an LLM when
  ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` is set; otherwise the rule-based
  text is returned unchanged, so the project never depends on a paid API.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import SEED  # noqa: E402
from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

PRETTY: dict[str, str] = {
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "avg_monthly_spend": "Average monthly spend", "num_services": "Number of services",
    "is_month_to_month": "Month-to-month flag", "auto_pay": "Automatic payment",
    "charge_increase": "Recent price increase", "SeniorCitizen": "Senior citizen",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MultipleLines": "Multiple lines", "PhoneService": "Phone service", "tenure_group": "Tenure group",
}


def pretty(name: str) -> str:
    """Human-readable label for a feature name."""
    return PRETTY.get(name, name)


class ChurnExplainer:
    """SHAP explainer for a fitted preprocessing + model pipeline."""

    def __init__(self, pipe: Pipeline, background: pd.DataFrame, n_background: int = 200) -> None:
        self.pipe = pipe
        self.prep = pipe.named_steps["prep"]
        self.model = pipe.named_steps["model"]
        self.out_names: list[str] = list(self.prep.get_feature_names_out())
        self.groups = self._group_columns()
        bg = self.prep.transform(background[FEATURES].sample(
            min(n_background, len(background)), random_state=SEED))
        if hasattr(self.model, "coef_"):
            self.explainer = shap.LinearExplainer(
                self.model, shap.maskers.Independent(bg, max_samples=len(bg)))
            self.kind = "linear"
        else:
            self.explainer = shap.TreeExplainer(self.model, bg, model_output="probability")
            self.kind = "tree"

    def _group_columns(self) -> dict[str, list[int]]:
        """Map each original feature to the indices of its transformed columns."""
        groups: dict[str, list[int]] = {f: [] for f in FEATURES}
        for i, col in enumerate(self.out_names):
            if col in NUMERIC_FEATURES:
                groups[col].append(i)
                continue
            for f in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
                if col.startswith(f + "_"):
                    groups[f].append(i)
                    break
        return groups

    def raw_shap(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values per transformed column (log-odds for linear, prob for tree)."""
        vals = self.explainer.shap_values(self.prep.transform(X[FEATURES]))
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals)
        return vals[..., 1] if vals.ndim == 3 else vals

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values aggregated to original features (rows = customers)."""
        raw = self.raw_shap(X)
        agg = {f: raw[:, idx].sum(axis=1) for f, idx in self.groups.items() if idx}
        return pd.DataFrame(agg, index=X.index)

    @property
    def base_value(self) -> float:
        """Expected model output (log-odds for linear models)."""
        ev = self.explainer.expected_value
        return float(np.ravel(ev)[-1])

    def explain_customer(self, row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Top drivers for one customer: feature, value, SHAP impact and direction."""
        sv = self.shap_values(row).iloc[0]
        order = sv.abs().sort_values(ascending=False).index[:top_n]
        return pd.DataFrame({
            "feature": order,
            "label": [pretty(f) for f in order],
            "value": [row.iloc[0][f] for f in order],
            "impact": sv[order].values,
            "direction": ["raises risk" if v > 0 else "lowers risk" for v in sv[order].values],
        }).reset_index(drop=True)


def format_value(feature: str, value: object) -> str:
    """Format a feature value for plain-language text."""
    if feature in ("MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_increase"):
        return f"${float(value):,.2f}"
    if feature in ("is_month_to_month", "auto_pay"):
        return "yes" if int(value) else "no"
    if feature == "tenure":
        return f"{int(value)} month" + ("" if int(value) == 1 else "s")
    return str(value)


def plain_explanation(drivers: pd.DataFrame, probability: float) -> str:
    """One-paragraph plain-English explanation of a prediction."""
    risk = "HIGH" if probability >= 0.6 else "MEDIUM" if probability >= 0.35 else "LOW"
    up = drivers[drivers.impact > 0]
    down = drivers[drivers.impact < 0]
    parts = [f"Estimated churn probability: {probability:.0%} ({risk} risk)."]
    if len(up):
        lead = "Main reasons for concern" if probability >= 0.35 else "Minor factors adding some risk"
        parts.append(f"{lead}: " + "; ".join(
            f"{r.label} is {format_value(r.feature, r.value)}" for r in up.itertuples()) + ".")
    if len(down):
        parts.append("Factors keeping them loyal: " + "; ".join(
            f"{r.label} is {format_value(r.feature, r.value)}" for r in down.itertuples()) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------- recommendations
@dataclass(frozen=True)
class Rule:
    """A retention action triggered when ``feature`` pushes risk up and ``condition`` holds."""

    feature: str
    condition: object  # callable(customer_row) -> bool
    action: str


RULES: list[Rule] = [
    Rule("Contract", lambda c: c["Contract"] == "Month-to-month",
         "Offer a 1-year contract with 10-15% discount or a free month to lock in loyalty."),
    Rule("is_month_to_month", lambda c: c["Contract"] == "Month-to-month",
         "Offer a 1-year contract with 10-15% discount or a free month to lock in loyalty."),
    Rule("tenure", lambda c: c["tenure"] <= 12,
         "Enrol in the new-customer onboarding programme: welcome call and 30/60/90-day check-ins."),
    Rule("tenure_group", lambda c: c["tenure"] <= 12,
         "Enrol in the new-customer onboarding programme: welcome call and 30/60/90-day check-ins."),
    Rule("PaymentMethod", lambda c: c["PaymentMethod"] == "Electronic check",
         "Incentivise switching to automatic payment (e.g. $5 bill credit)."),
    Rule("auto_pay", lambda c: c["auto_pay"] == 0,
         "Incentivise switching to automatic payment (e.g. $5 bill credit)."),
    Rule("TechSupport", lambda c: c["TechSupport"] == "No",
         "Give a free 3-month tech-support trial to reduce service frustration."),
    Rule("OnlineSecurity", lambda c: c["OnlineSecurity"] == "No",
         "Bundle online security free for 3 months to raise perceived value."),
    Rule("InternetService", lambda c: c["InternetService"] == "Fiber optic",
         "Proactively check fiber service quality and review pricing vs competitors."),
    Rule("MonthlyCharges", lambda c: c["MonthlyCharges"] >= 70,
         "Review the plan: offer a right-sized bundle or loyalty price to reduce bill shock."),
    Rule("charge_increase", lambda c: c["charge_increase"] > 5,
         "Bill recently increased: explain the change and offer a temporary credit."),
    Rule("PaperlessBilling", lambda c: c["PaperlessBilling"] == "Yes",
         "Send a clear monthly usage summary so the customer sees value in every bill."),
    Rule("num_services", lambda c: c["num_services"] <= 2,
         "Cross-sell one useful add-on at a discount; more services = stickier customer."),
    Rule("SeniorCitizen", lambda c: c["SeniorCitizen"] == "Yes",
         "Assign a dedicated support contact and offer a simplified senior plan."),
]


def recommend_actions(customer: pd.Series, drivers: pd.DataFrame, max_actions: int = 3) -> list[str]:
    """Rule-based retention actions for the customer's risk-increasing drivers."""
    actions: list[str] = []
    for feat in drivers.loc[drivers.impact > 0, "feature"]:
        for rule in RULES:
            if rule.feature == feat and rule.condition(customer) and rule.action not in actions:
                actions.append(rule.action)
    if not actions:
        actions.append("Low risk: no special offer needed; keep in the standard loyalty newsletter.")
    return actions[:max_actions]


def recommendation_text(customer: pd.Series, drivers: pd.DataFrame, probability: float) -> str:
    """Rule-based plain-English recommendation (explanation + numbered actions)."""
    actions = recommend_actions(customer, drivers)
    lines = [plain_explanation(drivers, probability), "", "Recommended actions:"]
    lines += [f"{i}. {a}" for i, a in enumerate(actions, 1)]
    return "\n".join(lines)


def llm_recommendation(customer: pd.Series, drivers: pd.DataFrame, probability: float) -> tuple[str, str]:
    """Return ``(text, source)``. Uses an LLM if an API key is set, else rules.

    The LLM only rewrites the rule-based facts into a friendlier message for a
    customer-success agent; it never sees personal identifiers.
    """
    base = recommendation_text(customer, drivers, probability)
    prompt = ("You are a customer-retention analyst. Rewrite the following churn-model output "
              "into a concise, friendly 3-4 sentence briefing for a customer-success agent, "
              "keeping all numbers and the recommended actions:\n\n" + base)
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
            msg = client.messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                                         max_tokens=300, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text, "LLM (Anthropic)"
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI  # optional dependency

            client = OpenAI()
            resp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                                  messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=300)
            return resp.choices[0].message.content, "LLM (OpenAI)"
    except Exception as exc:  # missing package, bad key, network error
        logger.warning("LLM call failed (%s); using rule-based text", exc)
    return base, "Rule-based"


# ---------------------------------------------------------------- report figures
def save_shap_figures(pipe: Pipeline, data: pd.DataFrame, out_dir: Path) -> None:
    """Save global importance, beeswarm and an individual waterfall plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ce = ChurnExplainer(pipe, data)
    sample = data.sample(min(1000, len(data)), random_state=SEED)
    sv = ce.shap_values(sample)

    imp = sv.abs().mean().sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([pretty(f) for f in imp.index], imp.values, color="#2E86AB")
    ax.set_xlabel("Mean |SHAP value| (impact on churn log-odds)")
    ax.set_title("What drives churn overall (global importance)")
    fig.tight_layout(); fig.savefig(out_dir / "shap_importance.png", dpi=130); plt.close(fig)

    raw = ce.raw_shap(sample)
    Xt = pd.DataFrame(ce.prep.transform(sample[FEATURES]), columns=ce.out_names)
    plt.figure()
    shap.summary_plot(raw, Xt, max_display=12, show=False)
    plt.title("SHAP summary (each dot = one customer)")
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm.png", dpi=130); plt.close("all")

    probs = pipe.predict_proba(sample[FEATURES])[:, 1]
    idx = int(np.argmax(probs))
    row = sv.iloc[idx]
    top = row.abs().sort_values(ascending=False).index[:10]
    exp = shap.Explanation(values=row[top].values, base_values=ce.base_value,
                           data=[format_value(f, sample.iloc[idx][f]) for f in top],
                           feature_names=[pretty(f) for f in top])
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(f"Why this customer is high risk (p = {probs[idx]:.0%})")
    plt.tight_layout(); plt.savefig(out_dir / "shap_waterfall_customer.png", dpi=130); plt.close("all")


def main() -> None:
    """Generate SHAP figures and print an example explanation."""
    import joblib

    from src.config import FIGURES_DIR, MODEL_FILE
    from src.data_prep import clean_data, load_raw
    from src.features import add_features

    pipe = joblib.load(MODEL_FILE)
    data = add_features(clean_data(load_raw()))
    save_shap_figures(pipe, data, FIGURES_DIR)
    ce = ChurnExplainer(pipe, data)
    probs = pipe.predict_proba(data[FEATURES])[:, 1]
    for i in [int(np.argmax(probs)), int(np.argmin(probs))]:
        row = data.iloc[[i]]
        drivers = ce.explain_customer(row)
        print(f"\n--- customer {data.iloc[i]['customerID']} ---")
        print(llm_recommendation(row.iloc[0], drivers, float(probs[i]))[0])


if __name__ == "__main__":
    main()
