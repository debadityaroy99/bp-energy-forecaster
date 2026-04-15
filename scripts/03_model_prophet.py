"""
03_model_prophet.py
-------------------
Stage 3 of the forecasting pipeline: train a Facebook Prophet model and compare
its performance against the Linear Regression baseline from stage 2.

Prophet is a decomposable time-series model (trend + seasonality + holidays).
For annual data like this, only the trend component is active, but Prophet's
piecewise-linear trend detection still tends to outperform a single global
linear fit when the growth rate has changed over time.

Run from the `scripts/` directory:
    python 03_model_prophet.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress the verbose Stan/cmdstanpy compilation messages during fit.
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")

TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "prophet_forecast.png")

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
print("[1/5] Loading train and test splits ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"      Train : {train_df.shape[0]} rows  "
      f"({train_df['Year'].min()} - {train_df['Year'].max()})")
print(f"      Test  : {test_df.shape[0]} rows  "
      f"({test_df['Year'].min()} - {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 3. Prophet data formatting
# ---------------------------------------------------------------------------
# Prophet is strict: input MUST have exactly two columns named 'ds' and 'y'.
#   ds  -- datestamp (datetime64).  We represent each year as Jan 1 of that year.
#   y   -- the numeric target value.
print("[2/5] Formatting data for Prophet (ds / y columns) ...")

def to_prophet_df(df):
    """Convert a raw Year/Oil_Consumption_m3_day dataframe to Prophet format."""
    prophet_df = pd.DataFrame()
    # pd.to_datetime understands plain year integers when we cast to string first.
    prophet_df["ds"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
    prophet_df["y"]  = df["Oil_Consumption_m3_day"].values
    return prophet_df

train_prophet = to_prophet_df(train_df)
test_prophet  = to_prophet_df(test_df)

print(f"      train_prophet dtypes:\n{train_prophet.dtypes}")

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
print("[3/5] Training Prophet model ...")

# yearly_seasonality=False because the data is already at annual granularity —
# there is no sub-year pattern to detect. Disabling it keeps the model honest.
# weekly_seasonality and daily_seasonality are False for the same reason.
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
model.fit(train_prophet)
print("      Model fitted successfully.")

# ---------------------------------------------------------------------------
# 5. Predict on the test period
# ---------------------------------------------------------------------------
print("[4/5] Generating predictions ...")

# Build a future dataframe from the test set's actual dates so we predict
# exactly the years we want to evaluate — not an auto-extended horizon.
future_df = test_prophet[["ds"]].copy()
forecast  = model.predict(future_df)

# 'yhat' is Prophet's point-estimate column; yhat_lower / yhat_upper are the
# 80 % uncertainty interval but we only need the point estimate here.
y_pred = forecast["yhat"].values
y_test = test_prophet["y"].values

# ---------------------------------------------------------------------------
# 6. Evaluate and compare against baseline
# ---------------------------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Baseline scores from 02_model_linreg.py for direct comparison.
BASELINE_MAE  = 402_485.43
BASELINE_RMSE = 526_577.50

print("\n--- Prophet Model Evaluation ---")
print(f"  MAE  : {mae:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_MAE:>15,.2f}  |  "
      f"delta: {BASELINE_MAE - mae:+,.2f})")
print(f"  RMSE : {rmse:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_RMSE:>15,.2f}  |  "
      f"delta: {BASELINE_RMSE - rmse:+,.2f})")
print("--------------------------------\n")

if mae < BASELINE_MAE and rmse < BASELINE_RMSE:
    print("  [PASS] Prophet beats the Linear Regression baseline on both metrics.")
else:
    print("  [INFO] Prophet did not improve over baseline on all metrics. "
          "Consider tuning changepoint_prior_scale.")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
print("[5/5] Generating forecast plot ...")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2,
    label="Train Data (1965-2015)"
)
ax.plot(
    test_df["Year"], test_df["Oil_Consumption_m3_day"],
    color="darkorange", linewidth=2,
    label="Test Data Actual (2016+)"
)
ax.plot(
    test_df["Year"], y_pred,
    color="purple", linewidth=2, linestyle="--",
    label="Prophet Prediction"
)

# Shade the uncertainty interval to show where Prophet thinks values could fall.
ax.fill_between(
    test_df["Year"],
    forecast["yhat_lower"].values,
    forecast["yhat_upper"].values,
    color="purple", alpha=0.12,
    label="Prophet 80% Uncertainty Interval"
)

# Vertical split-boundary marker.
ax.axvline(x=2015, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(2015.1, ax.get_ylim()[0], " Split (2015)",
        color="grey", fontsize=8, va="bottom")

ax.set_title("Prophet Model Forecast vs Actual", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Oil Consumption (m3/day)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Embed metric comparison box so the plot is self-documenting.
metric_text = (
    f"Prophet  MAE  = {mae/1e6:.3f}M\n"
    f"Prophet  RMSE = {rmse/1e6:.3f}M\n"
    f"Baseline MAE  = {BASELINE_MAE/1e6:.3f}M\n"
    f"Baseline RMSE = {BASELINE_RMSE/1e6:.3f}M"
)
ax.text(
    0.02, 0.97, metric_text,
    transform=ax.transAxes,
    fontsize=8.5, verticalalignment="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", alpha=0.85)
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] Prophet script complete.")
