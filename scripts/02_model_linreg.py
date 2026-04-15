"""
02_model_linreg.py
------------------
Stage 2 of the forecasting pipeline: train a Linear Regression baseline,
evaluate it on the held-out test set, and produce a forecast plot.

Linear Regression is intentionally used as a *baseline*. Its performance
sets the minimum bar that every subsequent model (ARIMA, LSTM, etc.) must beat.

Run from the `scripts/` directory:
    python 02_model_linreg.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")

TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "linreg_forecast.png")

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
# 3. Prepare features and target
# ---------------------------------------------------------------------------
print("[2/4] Preparing features ...")

# scikit-learn expects X to be a 2-D array even for a single feature,
# so reshape from (n,) to (n, 1).
X_train = train_df["Year"].values.reshape(-1, 1)
y_train = train_df["Oil_Consumption_m3_day"].values

X_test  = test_df["Year"].values.reshape(-1, 1)
y_test  = test_df["Oil_Consumption_m3_day"].values

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
print("[3/4] Training Linear Regression model ...")

model = LinearRegression()
model.fit(X_train, y_train)

print(f"      Slope     (coefficient) : {model.coef_[0]:,.2f} m3/day per year")
print(f"      Intercept               : {model.intercept_:,.2f}")

# ---------------------------------------------------------------------------
# 5. Predict and evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Baseline Model Evaluation (Linear Regression) ---")
print(f"  MAE  (Mean Absolute Error)       : {mae:>15,.2f} m3/day")
print(f"  RMSE (Root Mean Squared Error)   : {rmse:>15,.2f} m3/day")
print("------------------------------------------------------\n")

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
print("[4/4] Generating forecast plot ...")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    train_df["Year"], y_train,
    color="steelblue", linewidth=2,
    label="Train Data (1965-2015)"
)
ax.plot(
    test_df["Year"], y_test,
    color="darkorange", linewidth=2,
    label="Test Data Actual (2016+)"
)
ax.plot(
    test_df["Year"], y_pred,
    color="green", linewidth=2, linestyle="--",
    label="Linear Regression Prediction"
)

# Annotate each predicted point so the gap from actuals is immediately visible.
for year, actual, pred in zip(test_df["Year"], y_test, y_pred):
    ax.annotate(
        f"{pred/1e6:.2f}M",
        xy=(year, pred),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=7, color="green"
    )

# Draw a vertical line at the train/test boundary for clarity.
ax.axvline(x=2015, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(2015.1, ax.get_ylim()[0], " Split (2015)",
        color="grey", fontsize=8, va="bottom")

ax.set_title("Linear Regression Baseline Forecast", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Oil Consumption (m3/day)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

# Add metric box in the top-left corner so results are visible in the image.
metric_text = f"MAE  = {mae/1e6:.3f}M\nRMSE = {rmse/1e6:.3f}M"
ax.text(
    0.02, 0.97, metric_text,
    transform=ax.transAxes,
    fontsize=9, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8)
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] Linear Regression script complete.")
