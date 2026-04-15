"""
08_baselines_advanced.py
------------------------
Upgrade all three statistical baselines to use the same monthly, multivariate
data pipeline as the Advanced LSTM (script 07), ensuring a truly fair
apples-to-apples comparison.

Upgrades vs. the original annual baselines:
  Linear Reg  : annual + univariate  ->  monthly + 3 features (Time, Price, Pop)
  Prophet     : annual + no regressors -> monthly + yearly seasonality + 2 regressors
  ARIMA       : annual, univariate    ->  SARIMAX (monthly, 2 exogenous features)

Run from the `scripts/` directory:
    python 08_baselines_advanced.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------------------------
# 0. Paths & hardcoded Advanced LSTM scores for the leaderboard
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
RAW_CSV   = os.path.join(DATA_DIR, "world-crude-oil-price-vs-oil-consumption.csv")

LSTM_ADV_MAE  = 313_602.13
LSTM_ADV_RMSE = 396_690.03

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Consistent date formatter for all test-window plots.
DATE_FMT = mdates.DateFormatter("%Y-%m")

def style_date_axis(ax):
    ax.xaxis.set_major_formatter(DATE_FMT)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

# ---------------------------------------------------------------------------
# 1. Data pipeline  (mirrors 07_model_lstm_advanced.py exactly)
# ---------------------------------------------------------------------------
print("[1/7] Building monthly World pipeline ...")

df_raw = pd.read_csv(RAW_CSV, encoding="latin-1")
df_raw.rename(columns={
    df_raw.columns[3]: "Price",
    df_raw.columns[4]: "Consumption",
    df_raw.columns[5]: "Population",
}, inplace=True)

world = (df_raw[(df_raw["Entity"] == "World") & (df_raw["Year"] >= 1965)]
         .copy()
         .reset_index(drop=True))

# Fill any sparse Price / Population gaps within the World series.
world[["Price", "Population"]] = world[["Price", "Population"]].ffill().bfill()

# Convert to datetime index and upsample to monthly via linear interpolation.
world["Date"] = pd.to_datetime(world["Year"].astype(str) + "-01-01")
world.set_index("Date", inplace=True)

monthly = (world[["Consumption", "Price", "Population"]]
           .resample("MS")
           .interpolate(method="linear")
           .ffill()
           .bfill())

monthly.reset_index(inplace=True)

# Numeric time index for Linear Regression (0, 1, 2 … n-1).
monthly["Time_Index"] = np.arange(len(monthly))

print(f"      Monthly shape : {monthly.shape}")
print(f"      Date range    : {monthly['Date'].min().date()} "
      f"-> {monthly['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 2. Train / Test split
# ---------------------------------------------------------------------------
train = monthly[monthly["Date"] <= "2015-12-01"].copy()
test  = monthly[monthly["Date"] >  "2015-12-01"].copy()

print(f"      Train : {len(train)} months | Test : {len(test)} months\n")

y_test_actual = test["Consumption"].values

# ---------------------------------------------------------------------------
# 3. Model 1: Multiple Linear Regression
#    Features: Time_Index, Price, Population
# ---------------------------------------------------------------------------
print("[2/7] Multiple Linear Regression ...")

LR_FEATURES = ["Time_Index", "Price", "Population"]

X_train_lr = train[LR_FEATURES].values
y_train_lr = train["Consumption"].values
X_test_lr  = test[LR_FEATURES].values

lr_model     = LinearRegression().fit(X_train_lr, y_train_lr)
linreg_preds = lr_model.predict(X_test_lr)

lr_mae  = mean_absolute_error(y_test_actual, linreg_preds)
lr_rmse = rmse(y_test_actual, linreg_preds)
print(f"      MAE={lr_mae:,.0f}  RMSE={lr_rmse:,.0f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train["Date"], train["Consumption"],
        color="steelblue", linewidth=1.5, alpha=0.7, label="Train Data")
ax.plot(test["Date"], y_test_actual,
        color="darkorange", linewidth=2, label="Test Data Actual")
ax.plot(test["Date"], linreg_preds,
        color="green", linewidth=2, linestyle=":",
        label="Multi-LinReg Prediction")
ax.axvline(pd.Timestamp("2015-12-01"), color="grey",
           linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("Multiple Linear Regression — Monthly Multivariate Forecast",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Consumption (m\u00b3/day)")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)
style_date_axis(ax)
ax.text(0.02, 0.97,
        f"MAE  = {lr_mae/1e6:.3f}M\nRMSE = {lr_rmse/1e6:.3f}M",
        transform=ax.transAxes, va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="honeydew", alpha=0.9))
plt.tight_layout()
path = os.path.join(DATA_DIR, "linreg_advanced_forecast.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"      Plot saved -> {path}")

# ---------------------------------------------------------------------------
# 4. Model 2: Prophet with External Regressors
# ---------------------------------------------------------------------------
print("\n[3/7] Prophet with regressors (yearly_seasonality=True) ...")

def make_prophet_df(df):
    """Convert pipeline output to Prophet's required format."""
    return df.rename(columns={"Date": "ds", "Consumption": "y"})[
        ["ds", "y", "Price", "Population"]
    ].copy()

train_prophet = make_prophet_df(train)
test_prophet  = make_prophet_df(test)

prophet_model = Prophet(yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False)

# Regressors MUST be added before .fit() — Prophet needs to allocate
# coefficient slots for them during the Stan model compilation.
prophet_model.add_regressor("Price")
prophet_model.add_regressor("Population")
prophet_model.fit(train_prophet)

# The prediction dataframe must contain the future 'ds', 'Price', and
# 'Population' values. We have actuals for both exogenous features in the
# test set, so we pass them directly (realistic out-of-sample scenario).
forecast      = prophet_model.predict(test_prophet)
prophet_preds = forecast["yhat"].values

p_mae  = mean_absolute_error(y_test_actual, prophet_preds)
p_rmse = rmse(y_test_actual, prophet_preds)
print(f"      MAE={p_mae:,.0f}  RMSE={p_rmse:,.0f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train["Date"], train["Consumption"],
        color="steelblue", linewidth=1.5, alpha=0.7, label="Train Data")
ax.plot(test["Date"], y_test_actual,
        color="darkorange", linewidth=2, label="Test Data Actual")
ax.plot(test["Date"], prophet_preds,
        color="purple", linewidth=2, linestyle="--",
        label="Prophet Prediction")
# Shade the 80 % uncertainty interval from Prophet's built-in output.
ax.fill_between(test["Date"],
                forecast["yhat_lower"].values,
                forecast["yhat_upper"].values,
                color="purple", alpha=0.13,
                label="80% Uncertainty Interval")
ax.axvline(pd.Timestamp("2015-12-01"), color="grey",
           linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("Prophet + Regressors — Monthly Multivariate Forecast",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Consumption (m\u00b3/day)")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)
style_date_axis(ax)
ax.text(0.02, 0.97,
        f"MAE  = {p_mae/1e6:.3f}M\nRMSE = {p_rmse/1e6:.3f}M",
        transform=ax.transAxes, va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", alpha=0.9))
plt.tight_layout()
path = os.path.join(DATA_DIR, "prophet_advanced_forecast.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"      Plot saved -> {path}")

# ---------------------------------------------------------------------------
# 5. Model 3: SARIMAX(1,1,1)
#    Endogenous  : Consumption
#    Exogenous   : Price, Population
# ---------------------------------------------------------------------------
print("\n[4/7] SARIMAX(1,1,1) with exogenous features ...")

EXOG_COLS = ["Price", "Population"]

train_y    = train["Consumption"].values
train_exog = train[EXOG_COLS].values
test_exog  = test[EXOG_COLS].values

sarimax_model = SARIMAX(
    endog  = train_y,
    exog   = train_exog,
    order  = (1, 1, 1),
    # trend='n' disables the deterministic trend constant; the I(1) differencing
    # already handles the long-run upward drift in consumption.
    trend  = "n",
    enforce_stationarity  = False,
    enforce_invertibility = False,
)
sarimax_fit = sarimax_model.fit(disp=False)

# get_forecast() requires the exogenous variables for the test horizon.
sarimax_forecast = sarimax_fit.get_forecast(
    steps = len(test),
    exog  = test_exog,
)
sarimax_preds = sarimax_forecast.predicted_mean
conf_int_raw  = sarimax_forecast.conf_int(alpha=0.20)   # 80 % CI to match Prophet
# conf_int() may return a DataFrame or ndarray depending on statsmodels version.
conf_int = (conf_int_raw if hasattr(conf_int_raw, "iloc")
            else pd.DataFrame(conf_int_raw, columns=["lower", "upper"]))

s_mae  = mean_absolute_error(y_test_actual, sarimax_preds)
s_rmse = rmse(y_test_actual, sarimax_preds)
print(f"      MAE={s_mae:,.0f}  RMSE={s_rmse:,.0f}")
print(f"\n{sarimax_fit.summary()}\n")

# --- Plot ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train["Date"], train["Consumption"],
        color="steelblue", linewidth=1.5, alpha=0.7, label="Train Data")
ax.plot(test["Date"], y_test_actual,
        color="darkorange", linewidth=2, label="Test Data Actual")
ax.plot(test["Date"], sarimax_preds,
        color="red", linewidth=2, linestyle="-.",
        label="SARIMAX(1,1,1) Prediction")
ax.fill_between(test["Date"],
                conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                color="red", alpha=0.10, label="80% Confidence Interval")
ax.axvline(pd.Timestamp("2015-12-01"), color="grey",
           linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("SARIMAX(1,1,1) — Monthly Multivariate Forecast",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Consumption (m\u00b3/day)")
ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)
style_date_axis(ax)
ax.text(0.02, 0.97,
        f"MAE  = {s_mae/1e6:.3f}M\nRMSE = {s_rmse/1e6:.3f}M",
        transform=ax.transAxes, va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe8e8", alpha=0.9))
plt.tight_layout()
path = os.path.join(DATA_DIR, "sarimax_advanced_forecast.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"      Plot saved -> {path}")

# ---------------------------------------------------------------------------
# 6. Leaderboard
# ---------------------------------------------------------------------------
print("\n[5/7] Evaluation leaderboard ...")

models = [
    ("Advanced LSTM",         LSTM_ADV_MAE,  LSTM_ADV_RMSE,  "monthly, multivariate"),
    ("Multi-LinReg (adv.)",   lr_mae,        lr_rmse,         "monthly, multivariate"),
    ("Prophet + regressors",  p_mae,         p_rmse,          "monthly, multivariate"),
    ("SARIMAX(1,1,1)",        s_mae,         s_rmse,          "monthly, multivariate"),
]
models.sort(key=lambda x: x[1])   # sort by MAE ascending

print("\n============ Advanced Baseline Leaderboard (sorted by MAE) ============")
print(f"  {'Rank':<5} {'Model':<26} {'MAE (m3/d)':>13}  {'RMSE (m3/d)':>13}  Notes")
print(f"  {'-'*72}")
for rank, (name, mae_v, rmse_v, note) in enumerate(models, 1):
    flag = " <-- BEST" if rank == 1 else ""
    print(f"  {rank:<5} {name:<26} {mae_v:>13,.0f}  {rmse_v:>13,.0f}  {note}{flag}")
print("=======================================================================\n")

# ---------------------------------------------------------------------------
# 7. Master comparison chart
# ---------------------------------------------------------------------------
print("[6/7] Master advanced baselines comparison chart ...")

fig, ax = plt.subplots(figsize=(14, 7))

# Background zone shading.
ax.axvspan(train["Date"].min(), pd.Timestamp("2015-12-01"),
           alpha=0.05, color="steelblue")
ax.axvspan(pd.Timestamp("2016-01-01"), test["Date"].max(),
           alpha=0.05, color="orange")
ax.axvline(pd.Timestamp("2015-12-01"), color="grey",
           linestyle="--", linewidth=1.1, alpha=0.55)

# Training history for context.
ax.plot(train["Date"], train["Consumption"],
        color="steelblue", linewidth=1.8, alpha=0.55,
        label="Train Data (1965-2015)")

# Ground truth.
ax.plot(test["Date"], y_test_actual,
        color="black", linewidth=3,
        label="Actual Consumption (2016+)")

# The three upgraded baselines.
ax.plot(test["Date"], linreg_preds,
        color="green", linewidth=2, linestyle=":",
        label=f"Multi-LinReg   (MAE={lr_mae/1e6:.3f}M | RMSE={lr_rmse/1e6:.3f}M)")
ax.plot(test["Date"], prophet_preds,
        color="purple", linewidth=2, linestyle="--",
        label=f"Prophet+Reg    (MAE={p_mae/1e6:.3f}M | RMSE={p_rmse/1e6:.3f}M)")
ax.plot(test["Date"], sarimax_preds,
        color="red", linewidth=2.5, linestyle="-.",
        label=f"SARIMAX(1,1,1) (MAE={s_mae/1e6:.3f}M | RMSE={s_rmse/1e6:.3f}M)")

# LSTM reference annotation (score from script 07 — no re-train needed).
ax.annotate(
    f"Adv. LSTM best score\nMAE={LSTM_ADV_MAE/1e6:.3f}M | RMSE={LSTM_ADV_RMSE/1e6:.3f}M",
    xy=(test["Date"].iloc[len(test)//2],
        y_test_actual[len(test)//2] * 0.975),
    fontsize=8.5, color="teal",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0f7f7", alpha=0.85),
)

# Zone labels.
ax.text(0.30, 1.015, "Training Zone (1965-2015)",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="steelblue", alpha=0.8)
ax.text(0.85, 1.015, "Test Zone (2016-2024)",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="darkorange", alpha=0.8)

# Metric table bottom-right.
best_mae_name  = min(models, key=lambda x: x[1])[0]
best_rmse_name = min(models, key=lambda x: x[2])[0]
table = (
    f"{'Model':<24} {'MAE':>9}  {'RMSE':>9}\n"
    f"{'-'*45}\n"
    f"{'Adv. LSTM':<24} {LSTM_ADV_MAE/1e6:>8.3f}M  {LSTM_ADV_RMSE/1e6:>8.3f}M\n"
    f"{'Multi-LinReg':<24} {lr_mae/1e6:>8.3f}M  {lr_rmse/1e6:>8.3f}M\n"
    f"{'Prophet+Reg':<24} {p_mae/1e6:>8.3f}M  {p_rmse/1e6:>8.3f}M\n"
    f"{'SARIMAX(1,1,1)':<24} {s_mae/1e6:>8.3f}M  {s_rmse/1e6:>8.3f}M"
)
ax.text(0.98, 0.04, table,
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="lightgrey", alpha=0.93))

ax.set_title(
    "Advanced Baselines — Monthly Multivariate Forecast vs Actual (2016-2024)",
    fontsize=14, fontweight="bold", pad=18)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Oil Consumption (m\u00b3/day)", fontsize=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.35)
style_date_axis(ax)

plt.tight_layout()
path = os.path.join(DATA_DIR, "advanced_baselines_comparison.png")
plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
print(f"      Plot saved -> {path}")

print("\n[OK] Advanced baselines script complete.")
print(f"     Best MAE  -> {best_mae_name}")
print(f"     Best RMSE -> {best_rmse_name}")
