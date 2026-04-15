"""
04_model_arima.py
-----------------
Stage 4 of the forecasting pipeline: train an ARIMA(1,1,1) model and benchmark
it against both the Linear Regression baseline and the Prophet model.

ARIMA(p, d, q) recap:
  p=1  AutoRegressive term  -- the forecast uses 1 lag of the series itself.
  d=1  Differencing term    -- first-difference the series once to remove the
                               linear trend and achieve stationarity.
  q=1  Moving Average term  -- 1 lag of the forecast error is fed back in.

For a slowly-trending energy series with no strong seasonality, ARIMA(1,1,1) is
a well-established starting point. It often outperforms a global linear fit
because the AR term lets the model "remember" the recent level of the series.

Run from the `scripts/` directory:
    python 04_model_arima.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress the routine convergence / optimization notes that statsmodels emits.
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")

TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "arima_forecast.png")

# Baseline scores recorded from 02_model_linreg.py.
BASELINE_MAE  = 402_485.43
BASELINE_RMSE = 526_577.50

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
print("[1/4] Loading train and test splits ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"      Train : {train_df.shape[0]} rows  "
      f"({train_df['Year'].min()} - {train_df['Year'].max()})")
print(f"      Test  : {test_df.shape[0]} rows  "
      f"({test_df['Year'].min()} - {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 3. Prepare the training series
# ---------------------------------------------------------------------------
# ARIMA operates on a univariate time series — just the target values.
# Setting the integer Year as the index gives statsmodels a clean time axis
# and makes any diagnostic plots produced by the library readable.
train_series = pd.Series(
    data  = train_df["Oil_Consumption_m3_day"].values,
    index = train_df["Year"].values,
    name  = "Oil_Consumption_m3_day"
)

# ---------------------------------------------------------------------------
# 4. Train the ARIMA(1, 1, 1) model
# ---------------------------------------------------------------------------
print("[2/4] Training ARIMA(1, 1, 1) model ...")

arima_model = ARIMA(train_series, order=(1, 1, 1))
model_fit   = arima_model.fit()

# Print the full summary — this is important for a project report as it shows
# coefficient significance, AIC/BIC, and residual diagnostics at a glance.
print("\n--- ARIMA Model Summary ---")
print(model_fit.summary())
print("---------------------------\n")

# ---------------------------------------------------------------------------
# 5. Forecast the test horizon
# ---------------------------------------------------------------------------
print("[3/4] Generating forecast ...")

# forecast() steps forward from the end of the training series.
# It returns a plain numpy array of length `steps`.
forecast_values = model_fit.forecast(steps=len(test_df))
y_test          = test_df["Oil_Consumption_m3_day"].values

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
mae  = mean_absolute_error(y_test, forecast_values)
rmse = np.sqrt(mean_squared_error(y_test, forecast_values))

print("--- ARIMA(1,1,1) Model Evaluation ---")
print(f"  MAE  : {mae:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_MAE:>15,.2f}  |  "
      f"delta: {BASELINE_MAE - mae:+,.2f})")
print(f"  RMSE : {rmse:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_RMSE:>15,.2f}  |  "
      f"delta: {BASELINE_RMSE - rmse:+,.2f})")
print("--------------------------------------\n")

if mae < BASELINE_MAE and rmse < BASELINE_RMSE:
    print("  [PASS] ARIMA beats the Linear Regression baseline on both metrics.")
elif mae < BASELINE_MAE or rmse < BASELINE_RMSE:
    print("  [PARTIAL] ARIMA beats the baseline on one metric. "
          "Consider tuning the (p, d, q) order.")
else:
    print("  [INFO] ARIMA did not improve over baseline. "
          "Try order=(2,1,2) or check for non-stationarity with an ADF test.")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
print("[4/4] Generating forecast plot ...")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2,
    label="Train Data (1965-2015)"
)
ax.plot(
    test_df["Year"], y_test,
    color="darkorange", linewidth=2,
    label="Test Data Actual (2016+)"
)
ax.plot(
    test_df["Year"], forecast_values,
    color="red", linewidth=2, linestyle="--",
    label="ARIMA (1,1,1) Prediction"
)

# Annotate each forecast point with its value in millions for quick reading.
for year, pred in zip(test_df["Year"], forecast_values):
    ax.annotate(
        f"{pred/1e6:.2f}M",
        xy=(year, pred),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=7, color="red"
    )

# Vertical split-boundary marker.
ax.axvline(x=2015, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(2015.1, ax.get_ylim()[0], " Split (2015)",
        color="grey", fontsize=8, va="bottom")

ax.set_title("ARIMA Model Forecast vs Actual", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Oil Consumption (m3/day)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

# Metric comparison box embedded in the plot for report use.
metric_text = (
    f"ARIMA    MAE  = {mae/1e6:.3f}M\n"
    f"ARIMA    RMSE = {rmse/1e6:.3f}M\n"
    f"Baseline MAE  = {BASELINE_MAE/1e6:.3f}M\n"
    f"Baseline RMSE = {BASELINE_RMSE/1e6:.3f}M"
)
ax.text(
    0.02, 0.97, metric_text,
    transform=ax.transAxes,
    fontsize=8.5, verticalalignment="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffe8e8", alpha=0.85)
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] ARIMA script complete.")
