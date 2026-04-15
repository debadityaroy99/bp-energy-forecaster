"""
app.py
------
BP Strategic Energy Transition Dashboard
Hybrid LinReg-LSTM forecast + Renewable Deficit analysis.

Answers the core business question:
  "How much clean-energy capacity must BP build to offset
   fossil-fuel depletion by a chosen Net-Zero target year?"

Run from the project root:
    streamlit run app.py
"""

import os, random, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"]        = "42"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

random.seed(42);  np.random.seed(42);  tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "BP Strategic Transition",
    layout     = "wide",
    page_icon  = "🌍",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data",
                         "world-crude-oil-price-vs-oil-consumption.csv")
LOOK_BACK = 12

# ---------------------------------------------------------------------------
# 1. Data pipeline
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load World data, upsample to monthly frequency."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df.rename(columns={
        df.columns[3]: "Price",
        df.columns[4]: "Total_Energy_Demand",   # framed as total demand proxy
        df.columns[5]: "Population",
    }, inplace=True)

    world = (df[(df["Entity"] == "World") & (df["Year"] >= 1965)]
               .copy().reset_index(drop=True))
    world[["Price", "Population"]] = world[["Price", "Population"]].ffill().bfill()

    world["Date"] = pd.to_datetime(world["Year"].astype(str) + "-01-01")
    world.set_index("Date", inplace=True)

    monthly = (world[["Total_Energy_Demand", "Price", "Population"]]
               .resample("MS")
               .interpolate(method="linear")
               .ffill().bfill())
    monthly.reset_index(inplace=True)
    monthly["Time_Index"] = np.arange(len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# 2. Hybrid Forecasting Engine
# ---------------------------------------------------------------------------
@st.cache_resource
def train_hybrid_engine(cache_key: str, _historical: pd.DataFrame):
    """
    Two-stage Hybrid LinReg-LSTM.
    Stage 1: LinearRegression on Time_Index → deterministic trend.
    Stage 2: LSTM on residuals → non-linear correction.
    Returns (lr, lstm_model, scaler, hist_mae, hist_rmse).
    """
    TARGET = "Total_Energy_Demand"

    # Stage 1 — linear trend
    lr = LinearRegression()
    lr.fit(_historical[["Time_Index"]], _historical[TARGET])
    linreg_preds = lr.predict(_historical[["Time_Index"]])
    residuals    = _historical[TARGET].values - linreg_preds

    # Stage 2 — LSTM on residuals
    lstm_input = np.column_stack([
        residuals,
        _historical["Price"].values,
        _historical["Population"].values,
    ])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(lstm_input)

    X, y = [], []
    for i in range(len(scaled) - LOOK_BACK):
        X.append(scaled[i : i + LOOK_BACK])
        y.append(scaled[i + LOOK_BACK, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(32, activation="relu", input_shape=(LOOK_BACK, 3)),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X, y,
        epochs          = 30,
        batch_size      = 32,
        validation_split= 0.1,
        callbacks       = [EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=0)],
        verbose=0,
    )

    # Held-out test metrics (2016+)
    split_mask  = _historical["Date"] > "2015-12-01"
    test_df     = _historical[split_mask].reset_index(drop=True)
    test_scaled = scaler.transform(np.column_stack([
        test_df[TARGET].values - lr.predict(test_df[["Time_Index"]]),
        test_df["Price"].values,
        test_df["Population"].values,
    ]))
    X_te, y_te = [], []
    for i in range(len(test_scaled) - LOOK_BACK):
        X_te.append(test_scaled[i : i + LOOK_BACK])
        y_te.append(test_scaled[i + LOOK_BACK, 0])
    X_te = np.array(X_te)

    def inv_col0(v):
        d = np.zeros((len(v), 3)); d[:, 0] = np.array(v).flatten()
        return scaler.inverse_transform(d)[:, 0]

    test_preds  = inv_col0(model.predict(X_te, verbose=0)) + \
                  lr.predict(test_df[["Time_Index"]])[LOOK_BACK:]
    test_actual = test_df[TARGET].values[LOOK_BACK:]

    hist_mae  = mean_absolute_error(test_actual, test_preds)
    hist_rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
    return lr, model, scaler, hist_mae, hist_rmse


def generate_future_forecast(
    lr, lstm_model, scaler,
    historical: pd.DataFrame,
    future_year: int,
) -> tuple:
    """
    Extend the Hybrid forecast to `future_year` via a rolling LSTM window.
    Returns (future_dates, future_demand).
    """
    TARGET = "Total_Energy_Demand"
    last_date = historical["Date"].max()
    last_ti   = historical["Time_Index"].max()
    last_price = historical["Price"].iloc[-1]
    last_pop   = historical["Population"].iloc[-1]

    future_dates = pd.date_range(
        start = last_date + pd.DateOffset(months=1),
        end   = f"{future_year}-12-01",
        freq  = "MS",
    )
    n_future  = len(future_dates)
    future_ti = np.arange(last_ti + 1, last_ti + 1 + n_future)

    # Build scaled history for the seed window
    hist_resid  = historical[TARGET].values - lr.predict(historical[["Time_Index"]])
    hist_scaled = scaler.transform(np.column_stack([
        hist_resid,
        historical["Price"].values,
        historical["Population"].values,
    ]))
    seed               = hist_scaled[-LOOK_BACK:].copy()
    const_price_sc     = hist_scaled[-1, 1]
    const_pop_sc       = hist_scaled[-1, 2]

    future_resid_sc = []
    for _ in range(n_future):
        pred_sc = lstm_model.predict(seed.reshape(1, LOOK_BACK, 3), verbose=0)[0, 0]
        future_resid_sc.append(pred_sc)
        seed = np.vstack([seed[1:], [pred_sc, const_price_sc, const_pop_sc]])

    def inv_col0(v):
        d = np.zeros((len(v), 3)); d[:, 0] = np.array(v)
        return scaler.inverse_transform(d)[:, 0]

    future_resid_real = inv_col0(future_resid_sc)
    future_trend      = lr.predict(future_ti.reshape(-1, 1))
    future_demand     = future_trend + future_resid_real

    return future_dates, future_demand


def build_phase_out_curve(
    future_dates: pd.DatetimeIndex,
    last_consumption: float,
    net_zero_year: int,
) -> np.ndarray:
    """
    Linear hydrocarbon phase-out: starts at `last_consumption` at the first
    future month, drops linearly to 0 by Jan of `net_zero_year`, then stays 0.
    """
    values = []
    net_zero_ts = pd.Timestamp(f"{net_zero_year}-01-01")
    start_ts    = future_dates[0]
    total_months = max(
        (net_zero_ts.year - start_ts.year) * 12 +
        (net_zero_ts.month - start_ts.month), 1
    )
    for d in future_dates:
        months_elapsed = (d.year - start_ts.year) * 12 + (d.month - start_ts.month)
        frac = min(months_elapsed / total_months, 1.0)
        values.append(max(0.0, last_consumption * (1.0 - frac)))
    return np.array(values)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.title("🌍 BP Strategic Energy Transition Dashboard")
st.markdown(
    "**Core question:** How much renewable capacity must BP build to offset "
    "fossil-fuel depletion and still meet total energy demand by the chosen "
    "Net-Zero target year? — powered by the **Hybrid LinReg-LSTM** champion model."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Scenario Controls")
st.sidebar.markdown("---")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Year)", min_value=2030, max_value=2070, value=2060, step=1,
)
net_zero_year = st.sidebar.slider(
    "Net-Zero Target Year",    min_value=2035, max_value=2070, value=2050, step=1,
)

# Ensure the forecast window covers the Net-Zero year.
if forecast_horizon < net_zero_year:
    forecast_horizon = net_zero_year
    st.sidebar.warning(f"Forecast horizon extended to {net_zero_year} to cover the target year.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Scenario assumptions**\n\n"
    "- Oil price held at last 2023 value\n"
    "- Population growth held flat\n"
    "- Hydrocarbon phase-out is linear\n"
    "- Renewable fills 100 % of the gap"
)

# ── Load data & train ────────────────────────────────────────────────────────
historical = load_data()

with st.spinner("Training Hybrid Engine... (cached after first run)"):
    cache_key = str(historical["Date"].max())
    lr, lstm_model, scaler, hist_mae, hist_rmse = train_hybrid_engine(
        cache_key, historical
    )

# ── Generate forecasts ───────────────────────────────────────────────────────
with st.spinner(f"Generating forecast to {forecast_horizon}..."):
    future_dates, future_demand = generate_future_forecast(
        lr, lstm_model, scaler, historical, forecast_horizon
    )

# Last known consumption (Dec 2023 proxy = last available monthly value).
last_consumption = historical["Total_Energy_Demand"].iloc[-1]

# Hydrocarbon phase-out curve.
phase_out = build_phase_out_curve(future_dates, last_consumption, net_zero_year)

# Renewable deficit: demand minus what hydrocarbons still supply.
renewable_required = np.maximum(0.0, future_demand - phase_out)

# Point estimates at the Net-Zero target year (closest January).
nz_mask = pd.DatetimeIndex(future_dates).year == net_zero_year
nz_demand     = float(future_demand[nz_mask].mean()) if nz_mask.any() else float(future_demand[-1])
nz_renewable  = float(renewable_required[nz_mask].mean()) if nz_mask.any() else float(renewable_required[-1])

# ---------------------------------------------------------------------------
# Row 1 — Strategic KPIs
# ---------------------------------------------------------------------------
st.markdown("### Strategic KPIs")
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    label = "Model Accuracy (RMSE)",
    value = f"{hist_rmse:,.0f} m\u00b3/day",
    delta = f"-{(526_578 - hist_rmse)/526_578*100:.1f}% vs linear baseline",
)
kpi2.metric(
    label = f"Total Demand in {net_zero_year}",
    value = f"{nz_demand/1e6:.2f}M m\u00b3/day",
    delta = f"{(nz_demand - last_consumption)/last_consumption*100:+.1f}% vs today",
)
kpi3.metric(
    label = f"Renewable Capacity Required ({net_zero_year})",
    value = f"{nz_renewable/1e6:.2f}M m\u00b3/day",
    delta = f"{nz_renewable/nz_demand*100:.1f}% of total demand",
    delta_color = "normal",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2 — Net-Zero Strategy Chart
# ---------------------------------------------------------------------------
st.markdown(f"### Energy Transition Roadmap to {net_zero_year}")

st.info(
    "**How to read this chart:** The **green shaded area** is the *Renewable Deficit* — "
    "the volume of clean-energy infrastructure BP must build and operate each year to "
    "compensate for the declining hydrocarbon supply while still meeting total energy demand. "
    "The wider the green band, the greater the investment urgency."
)

fig = go.Figure()

# Trace 1 — Historical actual consumption (blue).
fig.add_trace(go.Scatter(
    x    = historical["Date"],
    y    = historical["Total_Energy_Demand"],
    mode = "lines",
    name = "Historical Oil Consumption",
    line = dict(color="#4C9BE8", width=2),
    hovertemplate = "%{x|%b %Y}<br>%{y:,.0f} m\u00b3/day<extra></extra>",
))

# Trace 2 — Hydrocarbon phase-out (orange solid).
# Must be added BEFORE the demand trace so fill='tonexty' shades the gap.
fig.add_trace(go.Scatter(
    x    = future_dates,
    y    = phase_out,
    mode = "lines",
    name = f"Hydrocarbon Phase-Out (→ 0 by {net_zero_year})",
    line = dict(color="#FF7F0E", width=2.5),
    hovertemplate = "%{x|%b %Y}<br>%{y:,.0f} m\u00b3/day<extra></extra>",
))

# Trace 3 — Renewable deficit fill.
# fill='tonexty' shades from THIS trace UP to the previous trace (phase-out → demand).
# Because future_demand >= phase_out, the fill area represents the renewable gap.
fig.add_trace(go.Scatter(
    x         = future_dates,
    y         = future_demand,
    mode      = "lines",
    name      = "Renewable Capacity Required",
    fill      = "tonexty",
    fillcolor = "rgba(0, 200, 100, 0.25)",
    line      = dict(color="rgba(0,0,0,0)"),   # invisible line; only fill shown
    hovertemplate = "%{x|%b %Y}<br>Gap: %{y:,.0f} m\u00b3/day<extra></extra>",
))

# Trace 4 — Total demand forecast (dashed grey) drawn ON TOP for visibility.
fig.add_trace(go.Scatter(
    x    = future_dates,
    y    = future_demand,
    mode = "lines",
    name = "Total Demand Forecast (Hybrid LSTM)",
    line = dict(color="#AAAAAA", width=2, dash="dash"),
    hovertemplate = "%{x|%b %Y}<br>%{y:,.0f} m\u00b3/day<extra></extra>",
))

# Net-Zero vertical line — add_shape avoids the Plotly 6.x add_vline bug.
nz_date_str = f"{net_zero_year}-01-01"
fig.add_shape(
    type="line",
    x0=nz_date_str, x1=nz_date_str,
    y0=0, y1=1, yref="paper",
    line=dict(color="#FF4B4B", width=2, dash="dash"),
)
fig.add_annotation(
    x=nz_date_str, y=0.97, yref="paper",
    text=f"🎯 Net-Zero Target ({net_zero_year})",
    showarrow=False,
    font=dict(color="#FF4B4B", size=12, family="Arial Black"),
    xanchor="left", xshift=8,
)

# Train/test split marker.
fig.add_shape(
    type="line",
    x0="2016-01-01", x1="2016-01-01",
    y0=0, y1=1, yref="paper",
    line=dict(color="grey", width=1, dash="dot"),
)
fig.add_annotation(
    x="2016-01-01", y=0.97, yref="paper",
    text="Train | Test",
    showarrow=False,
    font=dict(color="grey", size=10),
    xanchor="left", xshift=6,
)

fig.update_layout(
    template      = "plotly_dark",
    paper_bgcolor = "#0E1117",
    plot_bgcolor  = "#0E1117",
    height        = 560,
    margin        = dict(l=60, r=30, t=20, b=60),
    xaxis = dict(
        title     = "Year",
        showgrid  = True, gridcolor="#1f1f1f",
        tickformat= "%Y",
    ),
    yaxis = dict(
        title      = "Energy Volume (m\u00b3/day)",
        showgrid   = True, gridcolor="#1f1f1f",
        tickformat = ",.0f",
    ),
    legend = dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="left", x=0, font=dict(size=11),
    ),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3 — Renewable build-out timeline + annual summary table
# ---------------------------------------------------------------------------
st.markdown("### Renewable Build-Out Timeline")

col_left, col_right = st.columns([3, 1])

with col_left:
    # Annual renewable deficit bar chart.
    deficit_df = pd.DataFrame({
        "Date":     future_dates,
        "Deficit":  renewable_required,
    })
    deficit_df["Year"] = pd.DatetimeIndex(deficit_df["Date"]).year
    annual_deficit = (deficit_df.groupby("Year")["Deficit"]
                                 .mean().reset_index())

    fig_bar = go.Figure(go.Bar(
        x         = annual_deficit["Year"],
        y         = annual_deficit["Deficit"],
        marker_color = [
            "#FF4B4B" if yr == net_zero_year else "#00C864"
            for yr in annual_deficit["Year"]
        ],
        hovertemplate = "%{x}<br>Required: %{y:,.0f} m\u00b3/day<extra></extra>",
        name      = "Renewable Capacity Needed",
    ))
    fig_bar.update_layout(
        template      = "plotly_dark",
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#0E1117",
        height        = 300,
        margin        = dict(l=50, r=20, t=30, b=40),
        title         = dict(
            text="Annual Average Renewable Capacity Required (m\u00b3/day eq.)",
            font=dict(size=12),
        ),
        xaxis = dict(showgrid=False, dtick=5),
        yaxis = dict(showgrid=True, gridcolor="#1f1f1f", tickformat=",.0f"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("**Milestone Summary**")
    # Show every 5th year only for readability.
    milestone_years = [y for y in annual_deficit["Year"] if y % 5 == 0 or y == net_zero_year]
    milestone_df = annual_deficit[annual_deficit["Year"].isin(milestone_years)].copy()
    milestone_df["Renewable (M m\u00b3/d)"] = (
        milestone_df["Deficit"] / 1e6
    ).map("{:.2f}M".format)
    milestone_df = milestone_df[["Year", "Renewable (M m\u00b3/d)"]].reset_index(drop=True)
    st.dataframe(milestone_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Row 4 — Raw data expander
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📂 View Raw Macroeconomic Data (1965–2024)"):
    display_df = historical[["Date", "Total_Energy_Demand", "Price", "Population"]].copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
    display_df = display_df.rename(columns={
        "Total_Energy_Demand": "Oil Consumption (m\u00b3/day)",
        "Price"              : "Oil Price (2024 US$)",
        "Population"         : "World Population",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "BP Strategic Energy Transition Project  |  "
    "Champion Model: Hybrid LinReg-LSTM  |  "
    f"Forecast horizon: {forecast_horizon}  |  "
    "Data: Our World in Data (1965–2024)"
)
