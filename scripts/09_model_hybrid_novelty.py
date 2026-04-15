"""
09_model_hybrid_novelty.py
--------------------------
Hybrid LinReg-LSTM Ensemble — architectural novelty for the final-year project.

The core idea (residual decomposition):
  A linear regression captures the long-run deterministic trend in consumption.
  What it cannot capture is the structured, non-linear deviation from that
  trend — the residuals.  These residuals still contain learnable patterns
  (price shocks, demand plateaus, pandemic dips) that a sequence model can
  exploit.

  Stage 1:  LinReg(Time)  ->  trend prediction T(t)
  Stage 2:  LSTM([r, Price, Pop])  ->  residual prediction R(t)
  Output :  Hybrid(t) = T(t) + R(t)

This architecture is well-established in the time-series literature under the
name "error-correction ensemble" and has been cited in energy forecasting papers
(e.g., Hu et al., 2020; Li et al., 2022).

Run from the `scripts/` directory:
    python 09_model_hybrid_novelty.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

# Fix all random seeds for reproducible, reportable scores.
import random
import numpy as np
random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
import tensorflow as tf
tf.random.set_seed(42)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 0. Constants
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
RAW_CSV   = os.path.join(DATA_DIR, "world-crude-oil-price-vs-oil-consumption.csv")
PLOT_PATH = os.path.join(DATA_DIR, "hybrid_novelty_forecast.png")

LOOK_BACK     = 12
SPLIT_DATE    = "2015-12-01"
ADV_LSTM_MAE  = 313_602.13
ADV_LSTM_RMSE = 396_690.03

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ---------------------------------------------------------------------------
# 1. Data pipeline  (identical preprocessing to scripts 07 & 08)
# ---------------------------------------------------------------------------
print("[1/6] Building monthly World pipeline ...")

df_raw = pd.read_csv(RAW_CSV, encoding="latin-1")
df_raw.rename(columns={
    df_raw.columns[3]: "Price",
    df_raw.columns[4]: "Consumption",
    df_raw.columns[5]: "Population",
}, inplace=True)

world = (df_raw[(df_raw["Entity"] == "World") & (df_raw["Year"] >= 1965)]
         .copy().reset_index(drop=True))
world[["Price", "Population"]] = world[["Price", "Population"]].ffill().bfill()

world["Date"] = pd.to_datetime(world["Year"].astype(str) + "-01-01")
world.set_index("Date", inplace=True)

monthly = (world[["Consumption", "Price", "Population"]]
           .resample("MS")
           .interpolate(method="linear")
           .ffill().bfill())
monthly.reset_index(inplace=True)

# Numeric time index for the linear stage (0, 1, 2 … N-1).
monthly["Time_Index"] = np.arange(len(monthly))

train = monthly[monthly["Date"] <= SPLIT_DATE].copy().reset_index(drop=True)
test  = monthly[monthly["Date"] >  SPLIT_DATE].copy().reset_index(drop=True)

print(f"      Train : {len(train)} months | Test : {len(test)} months")

# ---------------------------------------------------------------------------
# 2. Stage 1 — Linear Regression on Time_Index  (trend extractor)
# ---------------------------------------------------------------------------
print("[2/6] Stage 1: fitting linear trend ...")

lr = LinearRegression()
lr.fit(train[["Time_Index"]], train["Consumption"])

train_linreg_preds = lr.predict(train[["Time_Index"]])
test_linreg_preds  = lr.predict(test[["Time_Index"]])

# Report how much variance the linear trend already explains.
ss_res  = np.sum((train["Consumption"].values - train_linreg_preds) ** 2)
ss_tot  = np.sum((train["Consumption"].values - train["Consumption"].mean()) ** 2)
r2_train = 1 - ss_res / ss_tot
print(f"      Linear trend R² (train) : {r2_train:.4f}")

# ---------------------------------------------------------------------------
# 3. Residual extraction
# ---------------------------------------------------------------------------
print("[3/6] Extracting residuals ...")

train_residuals = train["Consumption"].values - train_linreg_preds
test_residuals  = test["Consumption"].values  - test_linreg_preds

print(f"      Train residual range : [{train_residuals.min():+,.0f}, "
      f"{train_residuals.max():+,.0f}]  m3/day")
print(f"      Test  residual range : [{test_residuals.min():+,.0f}, "
      f"{test_residuals.max():+,.0f}]  m3/day")

# ---------------------------------------------------------------------------
# 4. Stage 2 — LSTM on residuals  (non-linear correction learner)
# ---------------------------------------------------------------------------
print("[4/6] Stage 2: LSTM on residuals ...")

# Build arrays for scaling: [residual, Price, Population].
# The scaler must be fit ONLY on train data to prevent data leakage.
train_lstm_input = np.column_stack([
    train_residuals,
    train["Price"].values,
    train["Population"].values,
])
test_lstm_input = np.column_stack([
    test_residuals,
    test["Price"].values,
    test["Population"].values,
])

scaler       = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_lstm_input)   # (612, 3)
test_scaled  = scaler.transform(test_lstm_input)         # (97,  3)

# -- Sliding-window sequence builder ----------------------------------------
def create_sequences(data, look_back=12):
    """
    Build (X, y) supervised pairs.
    X[i] = data[i : i+look_back]        shape (look_back, 3)
    y[i] = data[i+look_back, 0]         next-step residual (column 0)
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])        # (12, 3)
        y.append(data[i + look_back, 0])          # scalar
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOK_BACK)
X_test,  y_test  = create_sequences(test_scaled,  LOOK_BACK)

print(f"      X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"      X_test  : {X_test.shape}    y_test  : {y_test.shape}")

# -- Build model ------------------------------------------------------------
lstm_model = Sequential([
    # 32 units — intentionally lightweight; the linear stage has already
    # removed the dominant trend, so the LSTM only needs to learn low-amplitude
    # non-linear structure.
    LSTM(32, activation="relu",
         input_shape=(LOOK_BACK, 3),
         return_sequences=False),
    Dropout(0.2),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=10,
    restore_best_weights=True, verbose=1,
)

print(f"\n      Training residual-correction LSTM ...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)
best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"      Best epoch : {best_epoch}  |  "
      f"val_loss : {min(history.history['val_loss']):.6f}")

# ---------------------------------------------------------------------------
# 5. Stage 3 — Hybrid output synthesis
# ---------------------------------------------------------------------------
print("\n[5/6] Stage 3: synthesising hybrid forecast ...")

# Predict scaled residuals for the test sequences.
pred_residuals_scaled = lstm_model.predict(X_test, verbose=0)  # (85, 1)

# Inverse-transform only the residual column (index 0).
# Scaler was fit on 3 features, so we use the dummy-array trick.
def inv_scale_col0(scaler, values):
    dummy = np.zeros((len(values), 3))
    dummy[:, 0] = values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

lstm_residual_preds = inv_scale_col0(scaler, pred_residuals_scaled)

# Align the linear predictions: the first LOOK_BACK test months are used
# purely as the seed window for the LSTM, so the hybrid forecast starts
# at test index 12 (Jan 2017) and covers 85 months through Jan 2024.
hybrid_linreg_base = test_linreg_preds[LOOK_BACK:]        # (85,)
final_hybrid_preds = hybrid_linreg_base + lstm_residual_preds  # (85,)

# Corresponding ground-truth slice and dates.
y_actual      = test["Consumption"].values[LOOK_BACK:]    # (85,)
forecast_dates = test["Date"].values[LOOK_BACK:]          # (85,)

# Actual residuals for the bottom-panel plot.
actual_residuals_plot = test_residuals[LOOK_BACK:]        # (85,)

# ---------------------------------------------------------------------------
# 6. Evaluation & leaderboard
# ---------------------------------------------------------------------------
h_mae  = mean_absolute_error(y_actual, final_hybrid_preds)
h_rmse = rmse(y_actual, final_hybrid_preds)

print("\n========== Hybrid LinReg-LSTM Leaderboard ==========")
print(f"  {'Model':<28} {'MAE (m3/day)':>14}  {'RMSE (m3/day)':>14}")
print(f"  {'-'*58}")

scores = [
    ("Hybrid LinReg-LSTM",  h_mae,        h_rmse),
    ("Advanced LSTM (scr07)", ADV_LSTM_MAE, ADV_LSTM_RMSE),
]
scores.sort(key=lambda x: x[1])

for rank, (name, mae_v, rmse_v) in enumerate(scores, 1):
    crown = " <-- BEST MAE" if rank == 1 else ""
    print(f"  {rank}. {name:<26} {mae_v:>14,.0f}  {rmse_v:>14,.0f}{crown}")

print("=====================================================\n")

if h_mae < ADV_LSTM_MAE and h_rmse < ADV_LSTM_RMSE:
    print("  [PASS] Hybrid beats Advanced LSTM on BOTH metrics.")
elif h_mae < ADV_LSTM_MAE or h_rmse < ADV_LSTM_RMSE:
    print("  [PARTIAL] Hybrid beats Advanced LSTM on one metric.")
else:
    print("  [INFO] Hybrid did not outscore Advanced LSTM. "
          "Consider LOOK_BACK tuning or a deeper LSTM correction stage.")

# ---------------------------------------------------------------------------
# 7. Visualisation — 2-panel figure
# ---------------------------------------------------------------------------
print("[6/6] Generating 2-panel plot ...")

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(14, 10),
    gridspec_kw={"height_ratios": [3, 2]},
)
date_fmt   = mdates.DateFormatter("%Y-%m")
year_loc   = mdates.YearLocator()

def style_ax(ax):
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(year_loc)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.grid(True, linestyle="--", alpha=0.38)

# ── Top panel: full consumption picture ─────────────────────────────────────
# Training history for context.
ax_top.plot(train["Date"], train["Consumption"],
            color="steelblue", linewidth=1.6, alpha=0.55,
            label="Train Data (1965-2015)")

# Linear trend line extended across both windows so the decomposition is
# visually obvious — the LSTM corrects the gap between this line and reality.
all_dates  = pd.concat([train["Date"], test["Date"]], ignore_index=True)
all_ti     = pd.concat([train[["Time_Index"]], test[["Time_Index"]]], ignore_index=True)
all_trend  = lr.predict(all_ti)
ax_top.plot(all_dates, all_trend,
            color="grey", linewidth=1.4, linestyle=":",
            alpha=0.7, label="LinReg Base Trend (Stage 1)")

# Ground truth for the test window.
ax_top.plot(test["Date"], test["Consumption"],
            color="darkorange", linewidth=2.5,
            label="Actual Consumption (2016+)")

# Hybrid forecast.
ax_top.plot(forecast_dates, final_hybrid_preds,
            color="teal", linewidth=2.2, linestyle="--",
            label="Hybrid LinReg-LSTM Forecast")

# Fill between actual and hybrid to make error bands obvious.
ax_top.fill_between(forecast_dates, y_actual, final_hybrid_preds,
                    alpha=0.12, color="teal")

# Warm-up region shading (months used as the LSTM seed window, not predicted).
ax_top.axvspan(test["Date"].iloc[0], test["Date"].iloc[LOOK_BACK - 1],
               alpha=0.10, color="grey",
               label=f"LSTM warm-up ({LOOK_BACK} months)")
ax_top.axvline(pd.Timestamp(SPLIT_DATE), color="grey",
               linestyle="--", linewidth=1.1, alpha=0.55)

ax_top.set_title(
    "Hybrid LinReg-LSTM Ensemble — Trend + Residual Correction\n"
    f"Stage 1: Linear Trend  |  Stage 2: LSTM Residual (look_back={LOOK_BACK}mo)",
    fontsize=13, fontweight="bold",
)
ax_top.set_xlabel("Date", fontsize=11)
ax_top.set_ylabel("Oil Consumption (m\u00b3/day)", fontsize=11)
ax_top.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
style_ax(ax_top)

metric_text = (
    f"Hybrid   MAE  = {h_mae/1e6:.3f}M\n"
    f"Hybrid   RMSE = {h_rmse/1e6:.3f}M\n"
    f"Adv LSTM MAE  = {ADV_LSTM_MAE/1e6:.3f}M\n"
    f"Adv LSTM RMSE = {ADV_LSTM_RMSE/1e6:.3f}M"
)
ax_top.text(0.02, 0.97, metric_text,
            transform=ax_top.transAxes, fontsize=8.5,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#e0f7f7", alpha=0.92))

# ── Bottom panel: residual decomposition ────────────────────────────────────
ax_bot.plot(forecast_dates, actual_residuals_plot,
            color="darkorange", linewidth=2,
            label="Actual Residuals (Consumption - LinReg Trend)")
ax_bot.plot(forecast_dates, lstm_residual_preds,
            color="teal", linewidth=2, linestyle="--",
            label="LSTM Predicted Residuals (Stage 2 output)")
ax_bot.fill_between(forecast_dates,
                    actual_residuals_plot, lstm_residual_preds,
                    alpha=0.12, color="purple")
ax_bot.axhline(0, color="grey", linewidth=1, linestyle=":", alpha=0.6)

ax_bot.set_title("Residual Decomposition — What the LSTM Learns to Correct",
                 fontsize=12, fontweight="bold")
ax_bot.set_xlabel("Date", fontsize=11)
ax_bot.set_ylabel("Residual (m\u00b3/day)", fontsize=11)
ax_bot.legend(fontsize=9.5)
style_ax(ax_bot)

plt.tight_layout(h_pad=2.5)
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] Hybrid novelty script complete.")
