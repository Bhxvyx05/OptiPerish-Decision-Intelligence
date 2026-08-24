"""
OptiPerish — Interactive Decision Intelligence Dashboard

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.feature_pipeline import get_splits_and_transformers
from src.model_forecaster import LightGBMForecaster
from src.conformal_engine import SplitConformalPredictor
from src.simulation_engine import StochasticLeadTimeSimulator
from src.inventory_optimizer import PerishableInventoryOptimizer


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="OptiPerish | Decision Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>

    /* ---------- App background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(0, 212, 170, 0.08), transparent 30%),
            radial-gradient(circle at 90% 0%, rgba(99, 102, 241, 0.10), transparent 28%),
            linear-gradient(180deg, #070B14 0%, #0B1020 45%, #0D1324 100%);
        color: #F8FAFC;
    }

    /* ---------- Main content ---------- */
    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #0A1020 0%, #10182B 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    /* ---------- Header ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 32px;
        border-radius: 24px;
        margin-bottom: 22px;
        background:
            linear-gradient(135deg,
                rgba(15, 23, 42, 0.96) 0%,
                rgba(18, 35, 65, 0.94) 45%,
                rgba(12, 51, 62, 0.93) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 55px rgba(0,0,0,0.30);
    }

    .hero:before,
    .hero:after {
        content: "";
        position: absolute;
        border-radius: 999px;
        filter: blur(4px);
        opacity: 0.50;
    }

    .hero:before {
        width: 260px;
        height: 260px;
        right: -70px;
        top: -130px;
        background: radial-gradient(circle, rgba(0, 212, 170, 0.65) 0%, transparent 70%);
    }

    .hero:after {
        width: 220px;
        height: 220px;
        left: -90px;
        bottom: -140px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.60) 0%, transparent 70%);
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        margin-bottom: 11px;
        background: rgba(0, 212, 170, 0.12);
        border: 1px solid rgba(0, 212, 170, 0.35);
        color: #67F6D6;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        font-size: 2.45rem;
        line-height: 1.08;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #F8FAFC;
    }

    .hero-subtitle {
        margin: 11px 0 0;
        color: #CBD5E1;
        font-size: 1rem;
        max-width: 900px;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        margin: 24px 0 12px;
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    .section-caption {
        color: #94A3B8;
        font-size: 0.87rem;
        margin-top: -4px;
        margin-bottom: 14px;
    }

    /* ---------- KPI cards ---------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        margin: 12px 0 22px;
    }

    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 135px;
        padding: 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 45px rgba(0,0,0,0.28);
        border-color: rgba(103, 246, 214, 0.38);
    }

    .kpi-card:before {
        content: "";
        position: absolute;
        width: 100px;
        height: 100px;
        right: -45px;
        top: -45px;
        border-radius: 50%;
        background: var(--glow);
        opacity: 0.20;
        filter: blur(3px);
    }

    .kpi-label {
        color: #94A3B8;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .kpi-value {
        margin-top: 10px;
        color: #F8FAFC;
        font-size: 1.72rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .kpi-sub {
        margin-top: 11px;
        color: #A5B4FC;
        font-size: 0.76rem;
    }

    /* ---------- Info cards ---------- */
    .info-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 35px rgba(0,0,0,0.16);
    }

    .info-title {
        color: #F8FAFC;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .info-text {
        color: #A5B4FC;
        line-height: 1.6;
        font-size: 0.88rem;
    }

    /* ---------- Risk badges ---------- */
    .risk-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    .risk-low {
        color: #86EFAC;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.25);
    }

    .risk-medium {
        color: #FDE68A;
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.25);
    }

    .risk-high {
        color: #FDA4AF;
        background: rgba(244,63,94,0.12);
        border: 1px solid rgba(244,63,94,0.25);
    }

    /* ---------- Streamlit widgets ---------- */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.90) !important;
        border-color: rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: rgba(15, 23, 42, 0.88);
        color: #F8FAFC;
        border-radius: 10px;
    }

    button[kind="primary"] {
        border-radius: 11px !important;
        background: linear-gradient(90deg, #00D4AA 0%, #14B8A6 100%) !important;
        color: #04111A !important;
        border: none !important;
        font-weight: 800 !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        margin-top: 28px;
        padding: 18px 0 4px;
        text-align: center;
        color: #64748B;
        font-size: 0.75rem;
    }

    /* ---------- Responsive ---------- */
    @media (max-width: 1100px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 700px) {
        .hero-title {
            font-size: 1.75rem;
        }
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def metric_card(label: str, value: str, sub: str, glow: str) -> str:
    return f"""
    <div class="kpi-card" style="--glow:{glow};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def risk_level(service_level: float) -> tuple[str, str]:
    stockout_pct = max(0.0, 1.0 - service_level) * 100

    if stockout_pct <= 3:
        return "LOW RISK", "risk-low"
    if stockout_pct <= 8:
        return "MEDIUM RISK", "risk-medium"
    return "HIGH RISK", "risk-high"


def styled_plotly(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.40)",
        font=dict(color="#CBD5E1"),
        margin=dict(l=20, r=20, t=35, b=20),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=13,
            font_color="#F8FAFC",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.08)",
        ),
    )
    return fig


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train_pipeline():
    (
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        y_test,
        transformers,
    ) = get_splits_and_transformers()

    ct = transformers["column_transformer"]
    feature_names = transformers["feature_names"]

    X_train_arr = ct.transform(X_train)
    X_cal_arr = ct.transform(X_cal)
    X_test_arr = ct.transform(X_test)

    lgb = LightGBMForecaster(
        objective="huber",
        random_state=42,
    )

    lgb.fit(
        X_train_arr,
        y_train.to_numpy(),
        feature_names=feature_names,
        num_boost_round=150,
    )

    conformal = SplitConformalPredictor(
        forecaster=lgb,
        X_cal=X_cal_arr,
        y_cal=y_cal.to_numpy(),
    )

    return (
        lgb,
        conformal,
        transformers,
        X_test,
        y_test,
        X_test_arr,
    )


# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Predictive + Prescriptive Analytics</div>
        <h1 class="hero-title">📦 OptiPerish</h1>
        <div class="hero-subtitle">
            Uncertainty-Aware Perishable Inventory Optimization —
            turning demand forecasts into cost-aware ordering decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.spinner("Initializing forecasting model and conformal calibration..."):
    (
        lgb,
        conformal,
        transformers,
        X_test,
        y_test,
        X_test_arr,
    ) = load_and_train_pipeline()


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Scenario Control Center")

store_options = sorted(X_test["store_id"].unique())
item_options = sorted(X_test["item_id"].unique())

sel_store = st.sidebar.selectbox(
    "Store",
    store_options,
    index=0,
)

sel_item = st.sidebar.selectbox(
    "Item SKU",
    item_options,
    index=0,
)

mask = (
    (X_test["store_id"] == sel_store)
    & (X_test["item_id"] == sel_item)
)

item_df = X_test[mask].copy().reset_index(drop=True)
item_arr = X_test_arr[mask.values]
item_y = y_test[mask.values].to_numpy()

if item_df.empty:
    st.error("No data is available for the selected store/SKU.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Stress Testing")

demand_shock_pct = st.sidebar.slider(
    "Demand Shock",
    min_value=-40,
    max_value=60,
    value=0,
    step=5,
    format="%d%%",
)

supplier_delay = st.sidebar.slider(
    "Supplier Delay",
    min_value=0,
    max_value=5,
    value=0,
    step=1,
    format="+%d days",
)

target_service_pct = st.sidebar.slider(
    "Target Service Level",
    min_value=85,
    max_value=99,
    value=95,
    step=1,
    format="%d%%",
)

demand_shock = demand_shock_pct / 100.0
target_service_level = target_service_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Cost Assumptions")

unit_cost = st.sidebar.number_input(
    "Unit Cost (₹)",
    min_value=0.0,
    value=float(item_df["unit_cost"].iloc[0]),
    step=1.0,
)

holding_cost = st.sidebar.number_input(
    "Holding Cost / Day (₹)",
    min_value=0.0,
    value=float(item_df["holding_cost_per_day"].iloc[0]),
    step=0.5,
)

spoilage_cost = st.sidebar.number_input(
    "Spoilage Penalty / Unit (₹)",
    min_value=0.0,
    value=float(item_df["spoilage_cost_per_unit"].iloc[0]),
    step=1.0,
)

stockout_cost = st.sidebar.number_input(
    "Stockout Cost / Unit (₹)",
    min_value=0.0,
    value=float(item_df["stockout_cost_per_unit"].iloc[0]),
    step=1.0,
)

shelf_life = int(item_df["shelf_life_days"].iloc[0])

current_inv = st.sidebar.number_input(
    "Initial On-Hand Inventory",
    min_value=0,
    value=20,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype uses synthetic grocery data and simulated operational assumptions."
)


# ---------------------------------------------------------------------
# FORECAST + UNCERTAINTY
# ---------------------------------------------------------------------
lower, yhat, upper = conformal.predict_interval(
    item_arr,
    coverage=target_service_level,
)

lower = np.maximum(0.0, lower * (1.0 + demand_shock))
yhat = np.maximum(0.0, yhat * (1.0 + demand_shock))
upper = np.maximum(yhat, upper * (1.0 + demand_shock))


# ---------------------------------------------------------------------
# STOCHASTIC LEAD TIME
# ---------------------------------------------------------------------
base_lt_dist = {
    max(1, 2 + supplier_delay): 0.15,
    max(1, 3 + supplier_delay): 0.50,
    max(1, 5 + supplier_delay): 0.25,
    max(1, 7 + supplier_delay): 0.10,
}

max_L = max(base_lt_dist.keys())

if len(lower) >= max_L:
    lower_bounds = lower[:max_L]
    upper_bounds = upper[:max_L]
else:
    lower_bounds = np.pad(
        lower,
        (0, max_L - len(lower)),
        mode="edge",
    )
    upper_bounds = np.pad(
        upper,
        (0, max_L - len(upper)),
        mode="edge",
    )

sim = StochasticLeadTimeSimulator(
    lead_time_probs=base_lt_dist,
    lower_bounds=np.maximum(0.0, lower_bounds),
    upper_bounds=np.maximum(0.0, upper_bounds),
    random_state=42,
)

ddlt_samples, sim_summary = sim.simulate_ddlt(
    num_simulations=5000,
)


# ---------------------------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------------------------
optimizer = PerishableInventoryOptimizer(
    unit_cost=unit_cost,
    holding_cost=holding_cost,
    spoilage_cost=spoilage_cost,
    stockout_cost=stockout_cost,
    shelf_life_days=shelf_life,
    current_inventory=current_inv,
)

opt_result = optimizer.optimize_order_quantity(
    ddlt_samples,
    min_service_level=target_service_level,
)

optimal_q = float(opt_result["optimal_order_quantity"])
inventory_after = float(opt_result["inventory_after_order"])
service_level = float(opt_result["service_level"])
expected_shortage = float(opt_result["expected_shortage_units"])
expected_leftover = float(opt_result["expected_leftover_units"])
expected_cost = float(opt_result["expected_total_cost"])

risk_text, risk_class = risk_level(service_level)

# ---------------------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------------------
st.markdown(
    '<div class="section-title">📊 Decision Snapshot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-grid">

        {metric_card(
            "Optimal Order (Q*)",
            f"{optimal_q:.0f} units",
            "Cost-minimizing replenishment quantity",
            "#00D4AA"
        )}

        {metric_card(
            "Post-Order Inventory",
            f"{inventory_after:.0f} units",
            f"Initial inventory: {current_inv:.0f}",
            "#6366F1"
        )}

        {metric_card(
            "Service Level",
            f"{service_level * 100:.2f}%",
            f"Target: {target_service_pct}%",
            "#38BDF8"
        )}

        {metric_card(
            "Expected Shortage",
            f"{expected_shortage:.2f}",
            "Units at risk during lead time",
            "#F59E0B"
        )}

        {metric_card(
            "Expected Total Cost",
            f"₹{expected_cost:,.0f}",
            "Expected operational loss",
            "#EC4899"
        )}

    </div>
    """,
    unsafe_allow_html=True,
)

risk_col1, risk_col2 = st.columns([1, 3])

with risk_col1:
    st.markdown(
        f'<span class="risk-badge {risk_class}">{risk_text}</span>',
        unsafe_allow_html=True,
    )

with risk_col2:
    st.caption(
        f"Expected leftover inventory: {expected_leftover:.2f} units • "
        f"Simulated DDLT mean: {sim_summary['mean']:.2f} units • "
        f"Lead-time scenarios: {supplier_delay:+d} day shock"
    )


st.markdown("---")


# ---------------------------------------------------------------------
# CHARTS — FORECAST
# ---------------------------------------------------------------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown(
        '<div class="section-title">📈 Demand Forecast + Conformal Uncertainty</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">The shaded band represents the calibrated prediction interval used by the optimizer.</div>',
        unsafe_allow_html=True,
    )

    dates = pd.to_datetime(item_df["date"])

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=dates,
            y=item_y,
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#F8FAFC", width=1.7),
            marker=dict(size=5),
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=dates,
            y=yhat,
            mode="lines",
            name="LightGBM Forecast",
            line=dict(color="#38BDF8", width=2.5, dash="dash"),
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=dates,
            y=upper,
            mode="lines",
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=dates,
            y=lower,
            mode="lines",
            name=f"{target_service_pct}% Prediction Interval",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )

    fig_forecast.add_annotation(
        x=dates.iloc[-1],
        y=float(yhat[-1]),
        text=f"Forecast: {yhat[-1]:.0f}",
        showarrow=True,
        arrowcolor="#00D4AA",
        font=dict(color="#E2E8F0"),
    )

    styled_plotly(fig_forecast, height=430)
    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        config={"displayModeBar": False},
    )


with right:
    st.markdown(
        '<div class="section-title">🎲 Demand During Lead Time</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Monte Carlo simulation combines demand uncertainty with stochastic supplier lead time.</div>',
        unsafe_allow_html=True,
    )

    fig_ddlt = px.histogram(
        ddlt_samples,
        nbins=40,
        color_discrete_sequence=["#14B8A6"],
        labels={"value": "DDLT (Units)", "count": "Simulations"},
    )

    fig_ddlt.add_vline(
        x=current_inv + optimal_q,
        line_width=3,
        line_dash="dash",
        line_color="#FB7185",
        annotation_text="Post-order inventory",
        annotation_position="top",
    )

    fig_ddlt.add_vline(
        x=np.quantile(ddlt_samples, target_service_level),
        line_width=2,
        line_dash="dot",
        line_color="#A78BFA",
        annotation_text=f"{target_service_pct}% demand quantile",
        annotation_position="bottom",
    )

    styled_plotly(fig_ddlt, height=430)
    st.plotly_chart(
        fig_ddlt,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ---------------------------------------------------------------------
# COST CURVE
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-title">📉 Asymmetric Cost Optimization</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">The optimizer balances stockout, spoilage and holding costs instead of relying on a fixed safety buffer.</div>',
    unsafe_allow_html=True,
)

q_grid = np.linspace(
    0,
    float(np.percentile(ddlt_samples, 99)) + max(20, optimal_q * 0.20),
    120,
)

costs = [
    optimizer.evaluate_cost(q, ddlt_samples)
    for q in q_grid
]

fig_cost = go.Figure()

fig_cost.add_trace(
    go.Scatter(
        x=q_grid,
        y=costs,
        mode="lines",
        name="Expected Cost",
        line=dict(
            color="#A78BFA",
            width=3,
        ),
        hovertemplate="Order Qty: %{x:.1f}<br>Expected Cost: ₹%{y:,.2f}<extra></extra>",
    )
)

fig_cost.add_vline(
    x=optimal_q,
    line_width=3,
    line_dash="dot",
    line_color="#22C55E",
    annotation_text=f"Q* = {optimal_q:.0f}",
    annotation_position="top",
)

styled_plotly(fig_cost, height=360)
st.plotly_chart(
    fig_cost,
    use_container_width=True,
    config={"displayModeBar": False},
)


# ---------------------------------------------------------------------
# SCENARIO SUMMARY
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-title">🧪 Current Scenario</div>',
    unsafe_allow_html=True,
)

scenario_data = pd.DataFrame(
    {
        "Parameter": [
            "Store",
            "Item SKU",
            "Demand Shock",
            "Supplier Delay",
            "Target Service Level",
            "Shelf Life",
            "Initial Inventory",
            "Optimal Order Quantity",
            "Expected Cost",
        ],
        "Value": [
            sel_store,
            sel_item,
            f"{demand_shock_pct:+d}%",
            f"+{supplier_delay} days",
            f"{target_service_pct}%",
            f"{shelf_life} days",
            f"{current_inv} units",
            f"{optimal_q:.0f} units",
            f"₹{expected_cost:,.0f}",
        ],
    }
)

st.dataframe(
    scenario_data,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------
# EXPLANATION CARD
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="info-card">
        <div class="info-title">💡 What the engine is doing</div>
        <div class="info-text">
            The system first generates a point demand forecast with LightGBM,
            calibrates uncertainty using conformal prediction, samples uncertain
            supplier lead times through Monte Carlo simulation, and finally chooses
            an order quantity that minimizes expected inventory-related cost under
            the selected service-level target.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="footer">
        OptiPerish • Predictive + Prescriptive Inventory Intelligence • Portfolio Prototype
    </div>
    """,
    unsafe_allow_html=True,
)