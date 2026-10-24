"""Customer Churn Prediction — Explainable AI Dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.business import campaign_impact, optimal_threshold  # noqa: E402
from src.config import (DEFAULT_CLV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE, METRICS_FILE,  # noqa: E402
                        MODEL_FILE, PROCESSED_DIR, RAW_FILE, RISK_HIGH, RISK_MEDIUM, TARGET)
from src.data_prep import clean_data, load_raw  # noqa: E402
from src.explain import ChurnExplainer, format_value, llm_recommendation, pretty  # noqa: E402
from src.features import FEATURES, add_features  # noqa: E402

st.set_page_config(page_title="Churn Prediction · Explainable AI", page_icon="📉", layout="wide")
ORANGE, BLUE, GREEN = "#E4572E", "#2E86AB", "#3BB273"


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="Loading model…")
def load_assets() -> tuple:
    """Load data, model, explainer, test predictions and metrics (cached)."""
    if not RAW_FILE.exists():
        from data.get_data import download_dataset, generate_synthetic
        try:
            df = download_dataset()
        except Exception:
            df = generate_synthetic()
        df.to_csv(RAW_FILE, index=False)
    data = add_features(clean_data(load_raw()))
    pipe = joblib.load(MODEL_FILE)
    data["churn_prob"] = pipe.predict_proba(data[FEATURES])[:, 1]
    explainer = ChurnExplainer(pipe, data)
    test = pd.read_csv(PROCESSED_DIR / "test_set.csv", index_col=0)
    test_prob = pipe.predict_proba(test[FEATURES])[:, 1]
    metrics = json.loads(METRICS_FILE.read_text())
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    global_imp = explainer.shap_values(data.sample(800, random_state=0)).abs().mean().sort_values()
    return data, pipe, explainer, test[TARGET].to_numpy(), test_prob, metrics, global_imp, oof


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not render them as LaTeX."""
    return text.replace("$", "\\$")


def risk_band(p: float) -> str:
    """Map probability to a risk label."""
    return "High" if p >= RISK_HIGH else "Medium" if p >= RISK_MEDIUM else "Low"


def gauge(prob: float) -> go.Figure:
    """Plotly gauge for a churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".0f"},
        title={"text": "Churn probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, RISK_MEDIUM * 100], "color": "#d4f1dd"},
                         {"range": [RISK_MEDIUM * 100, RISK_HIGH * 100], "color": "#fde8c8"},
                         {"range": [RISK_HIGH * 100, 100], "color": "#f9d0c4"}]}))
    fig.update_layout(height=260, margin=dict(l=30, r=40, t=50, b=10))
    return fig


def driver_chart(drivers: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of a customer's SHAP drivers."""
    d = drivers.iloc[::-1]
    labels = [f"{r.label} = {format_value(r.feature, r.value)}" for r in d.itertuples()]
    fig = go.Figure(go.Bar(x=d.impact, y=labels, orientation="h",
                           marker_color=[ORANGE if v > 0 else GREEN for v in d.impact]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Why? (SHAP: red raises risk, green lowers it)",
                      xaxis_title="Impact on churn log-odds")
    return fig


def show_explanation(row: pd.DataFrame, prob: float) -> None:
    """Gauge, SHAP drivers and recommendation for one customer."""
    drivers = explainer.explain_customer(row, top_n=8)
    c1, c2 = st.columns([1, 1.6])
    c1.plotly_chart(gauge(prob), width="stretch")
    c1.markdown(f"**Risk level:** {risk_band(prob)}  \n"
                f"<small>Calibrated: of customers scored ~{prob:.0%}, about {prob:.0%} actually leave.</small>",
                unsafe_allow_html=True)
    c2.plotly_chart(driver_chart(drivers), width="stretch")
    text, source = llm_recommendation(row.iloc[0], drivers.head(5), prob)
    st.markdown(f"#### 💡 AI retention recommendation <small>({source})</small>", unsafe_allow_html=True)
    st.info(md(text))


data, pipe, explainer, y_test, p_test, metrics, global_imp, oof = load_assets()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📉 Churn Intelligence")
page = st.sidebar.radio("Navigate", ["Overview", "Customer Explorer", "Predict", "Business Impact"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{metrics['model']}** · Test ROC-AUC **{metrics['tuned_threshold']['roc_auc']:.3f}**")
st.sidebar.caption("Data: IBM Telco Customer Churn (7,043 customers)")

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Customer Churn Overview")
    st.caption("Who is likely to leave, why, and what it is worth to act.")
    tuned, imp = metrics["tuned_threshold"], metrics["impact_tuned"]
    lost_mrr = data.loc[data[TARGET] == 1, "MonthlyCharges"].sum()
    k = st.columns(5)
    k[0].metric("Customers", f"{len(data):,}")
    k[1].metric("Churn rate", f"{data[TARGET].mean():.1%}")
    k[2].metric("Monthly revenue lost", f"${lost_mrr/1000:,.0f}k")
    k[3].metric("Churners caught (recall)", f"{tuned['recall']:.0%}")
    k[4].metric("Net savings / campaign*", f"${imp['net_savings']/1000:,.0f}k")
    st.caption(md(f"*On {metrics['n_test']:,} unseen test customers at threshold {tuned['threshold']:.2f}, "
               f"assuming CLV ${DEFAULT_CLV:,.0f}, offer ${DEFAULT_OFFER_COST:.0f}, "
               f"{DEFAULT_SUCCESS_RATE:.0%} success rate."))

    c1, c2 = st.columns(2)
    by_contract = data.groupby("Contract")[TARGET].mean().mul(100).reset_index()
    c1.plotly_chart(px.bar(by_contract, x="Contract", y=TARGET, text_auto=".0f", color_discrete_sequence=[ORANGE],
                           title="Churn rate by contract (%)").update_layout(yaxis_title="%"),
                    width="stretch")
    by_ten = data.groupby("tenure_group")[TARGET].mean().mul(100).reset_index()
    c2.plotly_chart(px.bar(by_ten, x="tenure_group", y=TARGET, text_auto=".0f", color_discrete_sequence=[ORANGE],
                           title="Churn rate by tenure (%)").update_layout(yaxis_title="%", xaxis_title="Tenure"),
                    width="stretch")
    c3, c4 = st.columns(2)
    gi = global_imp.tail(10)
    c3.plotly_chart(px.bar(x=gi.values, y=[pretty(f) for f in gi.index], orientation="h",
                           color_discrete_sequence=[BLUE], title="Top churn drivers (mean |SHAP|)")
                    .update_layout(xaxis_title="Impact", yaxis_title=""), width="stretch")
    bands = data["churn_prob"].map(risk_band).value_counts().reindex(["High", "Medium", "Low"])
    c4.plotly_chart(px.pie(values=bands.values, names=bands.index, hole=0.5, title="Current customers by risk level",
                           color=bands.index, color_discrete_map={"High": ORANGE, "Medium": "#F5B841", "Low": GREEN}),
                    width="stretch")

elif page == "Customer Explorer":
    st.title("Customer Explorer")
    st.caption("Filter the customer base, find who to call first, and see why each customer is at risk.")
    active = data[data[TARGET] == 0] if st.toggle("Show only current (non-churned) customers", True) else data
    f = st.columns(4)
    contract = f[0].multiselect("Contract", sorted(data.Contract.unique()))
    internet = f[1].multiselect("Internet service", sorted(data.InternetService.unique()))
    payment = f[2].multiselect("Payment method", sorted(data.PaymentMethod.unique()))
    risk = f[3].multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    tmin, tmax = st.slider("Tenure (months)", 0, 72, (0, 72))
    view = active.assign(risk=active.churn_prob.map(risk_band))
    for col, sel in [("Contract", contract), ("InternetService", internet), ("PaymentMethod", payment),
                     ("risk", risk)]:
        if sel:
            view = view[view[col].isin(sel)]
    view = view[view.tenure.between(tmin, tmax)].sort_values("churn_prob", ascending=False)
    mrr = view.MonthlyCharges.sum()
    st.markdown(md(f"**{len(view):,} customers** match · **${mrr:,.0f}** monthly revenue at stake"))
    cols = ["customerID", "churn_prob", "risk", "tenure", "Contract", "InternetService",
            "PaymentMethod", "MonthlyCharges", "TechSupport"]
    table = view[cols].assign(churn_prob=view.churn_prob * 100).rename(columns={"churn_prob": "Churn probability"})
    st.dataframe(table, width="stretch",
                 height=320, hide_index=True,
                 column_config={"Churn probability": st.column_config.ProgressColumn(format="%.0f%%",
                                                                                     min_value=0, max_value=100)})
    st.download_button("⬇️ Download call list (CSV)", view[cols].to_csv(index=False), "churn_call_list.csv")
    if len(view):
        cid = st.selectbox("Explain a customer", view.customerID.head(200))
        row = view[view.customerID == cid][FEATURES]
        show_explanation(row, float(view.loc[view.customerID == cid, "churn_prob"].iloc[0]))

elif page == "Predict":
    st.title("Predict Churn for a Customer")
    st.caption("Enter customer details to get a churn probability, an explanation and a retention plan.")
    with st.form("predict"):
        a, b, c = st.columns(3)
        tenure = a.slider("Tenure (months)", 0, 72, 5)
        contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = a.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = a.selectbox("Paperless billing", ["Yes", "No"])
        internet = b.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=1.0)
        addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                  "StreamingMovies"]
        chosen = b.multiselect("Add-on services", addons, default=["StreamingTV"], format_func=pretty)
        phone = c.selectbox("Phone service", ["Yes", "No"])
        multi = c.selectbox("Multiple lines", ["No", "Yes"])
        senior = c.selectbox("Senior citizen", ["No", "Yes"])
        partner = c.selectbox("Partner", ["No", "Yes"])
        dependents = c.selectbox("Dependents", ["No", "Yes"])
        gender = c.selectbox("Gender", ["Female", "Male"])
        st.form_submit_button("Predict churn", type="primary")
    no_net = "No internet service"
    cust = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi if phone == "Yes" else "No phone service", "InternetService": internet,
        **{s: (no_net if internet == "No" else ("Yes" if s in chosen else "No")) for s in addons},
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
    }
    row = add_features(pd.DataFrame([cust]))[FEATURES]
    prob = float(pipe.predict_proba(row)[:, 1][0])
    show_explanation(row, prob)
    thr = metrics["tuned_threshold"]["threshold"]
    if prob >= thr:
        st.success(f"Decision at the savings-optimal threshold ({thr:.2f}): **send a retention offer**.")
    else:
        st.success(f"Decision at the savings-optimal threshold ({thr:.2f}): no offer needed — keep monitoring.")

else:  # Business Impact
    st.title("Business Impact Calculator")
    st.caption("Change the assumptions to see how much a model-driven retention campaign saves. "
               f"Results use the {len(y_test):,} unseen test customers and are scaled to the full base.")
    s = st.columns(4)
    clv = s[0].number_input("Customer lifetime value ($)", 100, 10_000, int(DEFAULT_CLV), step=100)
    offer = s[1].number_input("Retention offer cost ($)", 0, 2_000, int(DEFAULT_OFFER_COST), step=10)
    success = s[2].slider("Offer success rate", 0.05, 0.9, DEFAULT_SUCCESS_RATE, 0.05)
    base_size = s[3].number_input("Customer base size", 1_000, 10_000_000, len(data), step=1_000)
    # Threshold is chosen on out-of-fold TRAIN predictions, then evaluated on the unseen test set.
    best_t, _ = optimal_threshold(oof["y"].to_numpy(), oof["prob"].to_numpy(), clv, offer, success)
    auto = st.toggle("Use the savings-maximising threshold", True)
    thr = best_t if auto else st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)
    r = campaign_impact(y_test, p_test, thr, clv, offer, success)
    allr = campaign_impact(y_test, p_test, 0.0, clv, offer, success)
    scale = base_size / len(y_test)
    m = st.columns(4)
    m[0].metric("Threshold", f"{thr:.2f}")
    m[1].metric("Customers contacted", f"{r.customers_targeted*scale:,.0f}")
    m[2].metric("Customers saved", f"{r.customers_saved*scale:,.0f}")
    m[3].metric("Net savings", f"${r.net_savings*scale:,.0f}",
                delta=f"${(r.net_savings-allr.net_savings)*scale:,.0f} vs contacting everyone")
    m2 = st.columns(4)
    m2[0].metric("Campaign cost", f"${r.campaign_cost*scale:,.0f}")
    m2[1].metric("Revenue protected", f"${r.revenue_saved*scale:,.0f}")
    m2[2].metric("ROI", f"{r.roi:.0%}")
    m2[3].metric("Churners caught", f"{r.churners_caught / max(1, r.churners_caught + r.churners_missed):.0%}")
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curve = pd.DataFrame({"threshold": grid, "net_savings": [
        campaign_impact(y_test, p_test, t, clv, offer, success).net_savings * scale for t in grid]})
    fig = px.line(curve, x="threshold", y="net_savings", title="Net savings vs decision threshold")
    fig.add_vline(x=thr, line_dash="dash", line_color=ORANGE, annotation_text=f"selected {thr:.2f}", annotation_position="top left")
    fig.add_vline(x=0.5, line_dash="dot", line_color="grey", annotation_text="default 0.5", annotation_position="bottom right")
    fig.update_layout(yaxis_title="Net savings ($)", yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")
    with st.expander("How is this calculated?"):
        st.markdown(
            "- Every customer with churn probability ≥ threshold gets the retention offer.\n"
            "- **Revenue protected** = churners caught × success rate × CLV.\n"
            "- **Campaign cost** = customers contacted × offer cost (offers to loyal customers are wasted).\n"
            "- **Net savings** = revenue protected − campaign cost, compared with doing nothing.\n"
            "- The savings-maximising threshold is chosen on out-of-fold *training* predictions, "
            "then measured on unseen test customers and scaled to the base size you enter.")
