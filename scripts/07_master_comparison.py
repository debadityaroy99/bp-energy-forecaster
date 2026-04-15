"""
07_master_comparison.py
-----------------------
Final stage: re-train all four models in one self-contained script and produce
a single presentation-quality chart that places every forecast side-by-side
against the ground truth for direct visual comparison.

Run from the `scripts/` directory:
    python 07_master_comparison.py
"""

import os

# Suppress TensorFlow C++ / oneDNN startup noise before any TF import.
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "master_model_comparison.png")

LOOK_BACK  = 3

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[1/5] Loading data ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
y_test   = test_df["Oil_Consumption_m3_day"].values

print(f"      Train : {len(train_df)} rows  "
      f"({train_df['Year'].min()} - {train_df['Year'].max()})")
print(f"      Test  : {len(test_df)} rows  "
      f"({test_df['Year'].min()} - {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 2. Linear Regression
# ---------------------------------------------------------------------------
print("[2/5] Linear Regression ...")

X_train = train_df["Year"].values.reshape(-1, 1)
X_test  = test_df["Year"].values.reshape(-1, 1)
y_train = train_df["Oil_Consumption_m3_day"].values

lr_model     = LinearRegression().fit(X_train, y_train)
linreg_preds = lr_model.predict(X_test)

lr_mae, lr_rmse = (
    mean_absolute_error(y_test, linreg_preds),
    np.sqrt(mean_squared_error(y_test, linreg_preds)),
)
print(f"      MAE={lr_mae:,.0f}  RMSE={lr_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 3. Prophet
# ---------------------------------------------------------------------------
print("[3/5] Prophet ...")

def to_prophet_df(df):
    """Convert raw dataframe to Prophet's required ds/y format."""
    return pd.DataFrame({
        "ds": pd.to_datetime(df["Year"].astype(str) + "-01-01"),
        "y" : df["Oil_Consumption_m3_day"].values,
    })

prophet_model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
prophet_model.fit(to_prophet_df(train_df))
forecast      = prophet_model.predict(to_prophet_df(test_df)[["ds"]])
prophet_preds = forecast["yhat"].values

p_mae, p_rmse = (
    mean_absolute_error(y_test, prophet_preds),
    np.sqrt(mean_squared_error(y_test, prophet_preds)),
)
print(f"      MAE={p_mae:,.0f}  RMSE={p_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 4. ARIMA(1, 1, 1)
# ---------------------------------------------------------------------------
print("[4/5] ARIMA(1,1,1) ...")

train_series = pd.Series(
    data  = train_df["Oil_Consumption_m3_day"].values,
    index = train_df["Year"].values,
)
arima_fit   = ARIMA(train_series, order=(1, 1, 1)).fit()
arima_preds = arima_fit.forecast(steps=len(test_df)).values

a_mae, a_rmse = (
    mean_absolute_error(y_test, arima_preds),
    np.sqrt(mean_squared_error(y_test, arima_preds)),
)
print(f"      MAE={a_mae:,.0f}  RMSE={a_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 5. LSTM
# ---------------------------------------------------------------------------
print("[5/5] LSTM (micro network, verbose=0) ...")

# -- Scale (fit ONLY on train to prevent data leakage) ----------------------
scaler       = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(
    train_df["Oil_Consumption_m3_day"].values.reshape(-1, 1)
)
test_scaled  = scaler.transform(
    test_df["Oil_Consumption_m3_day"].values.reshape(-1, 1)
)

# -- Sliding-window sequence builder ----------------------------------------
def create_dataset(data, look_back=3):
    """Build (X, y) supervised pairs from a 1-D scaled array."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

X_tr, y_tr = create_dataset(train_scaled, LOOK_BACK)
X_te, y_te = create_dataset(test_scaled,  LOOK_BACK)

# Keras LSTM requires 3-D input: [samples, time_steps, features].
X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
X_te = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)

# -- Build & train ----------------------------------------------------------
lstm_model = Sequential([
    LSTM(50, activation="relu", input_shape=(LOOK_BACK, 1)),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(
    X_tr, y_tr,
    epochs=100, batch_size=4,
    callbacks=[EarlyStopping(monitor="loss", patience=15,
                             restore_best_weights=True, verbose=0)],
    verbose=0,   # silent — metrics are printed in the summary table below
)

# -- Predict & inverse-scale ------------------------------------------------
lstm_preds_scaled = lstm_model.predict(X_te, verbose=0)
lstm_preds        = scaler.inverse_transform(lstm_preds_scaled).flatten()

# y_te is scaled; inverse-transform for fair metric calculation.
y_te_real = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

l_mae, l_rmse = (
    mean_absolute_error(y_te_real, lstm_preds),
    np.sqrt(mean_squared_error(y_te_real, lstm_preds)),
)

# LSTM covers only test years 4-9 (2019-2024) due to the look_back window.
lstm_years = test_df["Year"].values[LOOK_BACK:]

print(f"      MAE={l_mae:,.0f}  RMSE={l_rmse:,.0f}  "
      f"(evaluated on {len(lstm_years)} points: "
      f"{lstm_years[0]}-{lstm_years[-1]})")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
print("\n============== Final Model Comparison ==============")
print(f"  {'Model':<22} {'MAE (m3/day)':>14}  {'RMSE (m3/day)':>14}")
print(f"  {'-'*52}")
print(f"  {'Linear Regression':<22} {lr_mae:>14,.0f}  {lr_rmse:>14,.0f}")
print(f"  {'Prophet':<22} {p_mae:>14,.0f}  {p_rmse:>14,.0f}")
print(f"  {'ARIMA(1,1,1)':<22} {a_mae:>14,.0f}  {a_rmse:>14,.0f}")
print(f"  {'LSTM (6-pt window)':<22} {l_mae:>14,.0f}  {l_rmse:>14,.0f}")
print("=====================================================\n")

mae_winner  = min(
    {"Linear Reg": lr_mae,  "Prophet": p_mae,  "ARIMA": a_mae},
    key=lambda k: {"Linear Reg": lr_mae, "Prophet": p_mae, "ARIMA": a_mae}[k]
)
rmse_winner = min(
    {"Linear Reg": lr_rmse, "Prophet": p_rmse, "ARIMA": a_rmse},
    key=lambda k: {"Linear Reg": lr_rmse, "Prophet": p_rmse, "ARIMA": a_rmse}[k]
)
print(f"  Best MAE  (excl. LSTM*) -> {mae_winner}")
print(f"  Best RMSE (excl. LSTM*) -> {rmse_winner}")
print("  * LSTM evaluated on 6 points; others on 9. Not directly comparable.\n")

# ---------------------------------------------------------------------------
# 7. Master comparison chart
# ---------------------------------------------------------------------------
print("Generating master comparison chart ...")

fig, ax = plt.subplots(figsize=(14, 8))

# --- Background zone shading ------------------------------------------------
ax.axvspan(
    train_df["Year"].min(), 2015,
    alpha=0.05, color="steelblue", zorder=0
)
ax.axvspan(
    2015, test_df["Year"].max() + 0.5,
    alpha=0.05, color="orange", zorder=0
)
ax.axvline(x=2015.5, color="grey", linestyle="--", linewidth=1, alpha=0.55, zorder=1)

# Zone labels pinned near the top of the axes using axis-fraction coordinates.
y_top = ax.get_ylim()[1]
ax.text(0.37, 1.01, "Training Zone (1965-2015)",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="steelblue", alpha=0.8)
ax.text(0.88, 1.01, "Test Zone (2016-2024)",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="darkorange", alpha=0.8)

# --- Training history (context only) ----------------------------------------
ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2, alpha=0.6,
    label="Train Data (1965-2015)"
)

# --- Ground truth (test window) ---------------------------------------------
ax.plot(
    test_df["Year"], y_test,
    color="black", linewidth=3,
    label="Actual Consumption (2016+)"
)

# --- Model forecasts --------------------------------------------------------
ax.plot(
    test_df["Year"], linreg_preds,
    color="green", linewidth=2, linestyle=":",
    label="Linear Regression (Straight Average)"
)
ax.plot(
    test_df["Year"], prophet_preds,
    color="purple", linewidth=2, linestyle="--",
    label="Prophet (Momentum)"
)
ax.plot(
    test_df["Year"], arima_preds,
    color="red", linewidth=2.5, linestyle="-.",
    label="ARIMA (Conservative/Winner)"
)

# LSTM only covers 2019-2024 (look_back gap of 3 years).
ax.plot(
    lstm_years, lstm_preds,
    color="teal", linewidth=2, linestyle="-", marker="o", markersize=5,
    label="LSTM (Data Starved)"
)

# Annotate the LSTM warm-up gap so readers understand why it starts at 2019.
ax.annotate(
    f"LSTM warm-up\n(look_back={LOOK_BACK})",
    xy=(test_df["Year"].iloc[0] + 0.3, lstm_preds[0]),
    xytext=(30, -40), textcoords="offset points",
    fontsize=8, color="teal",
    arrowprops=dict(arrowstyle="->", color="teal", lw=1.1),
)

# --- Metric table embedded bottom-right ------------------------------------
table_text = (
    f"{'Model':<20} {'MAE':>9}  {'RMSE':>9}\n"
    f"{'─'*41}\n"
    f"{'Linear Regression':<20} {lr_mae/1e6:>8.3f}M  {lr_rmse/1e6:>8.3f}M\n"
    f"{'Prophet':<20} {p_mae/1e6:>8.3f}M  {p_rmse/1e6:>8.3f}M\n"
    f"{'ARIMA(1,1,1)':<20} {a_mae/1e6:>8.3f}M  {a_rmse/1e6:>8.3f}M\n"
    f"{'LSTM (6-pt)*':<20} {l_mae/1e6:>8.3f}M  {l_rmse/1e6:>8.3f}M\n"
    f"\n* evaluated on 2019-2024 only"
)
ax.text(
    0.98, 0.04, table_text,
    transform=ax.transAxes,
    fontsize=8, verticalalignment="bottom", horizontalalignment="right",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
              edgecolor="lightgrey", alpha=0.92)
)

# --- Labels, legend, formatting --------------------------------------------
ax.set_title(
    "World Oil Consumption — All Four Models vs Actual (2016-2024)",
    fontsize=15, fontweight="bold", pad=18
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Oil Consumption (m\u00b3/day)", fontsize=12)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(train_df["Year"].min() - 1, test_df["Year"].max() + 1)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n      Plot saved -> {PLOT_PATH}")
print("\n[OK] Master comparison script complete.")
