"""
05_model_comparison.py
----------------------
Stage 5 (Final): Re-train all three models in a single script and produce one
master comparison chart suitable for a project presentation or report.

This script is intentionally self-contained — it does not import results from
earlier scripts. Re-training from scratch guarantees reproducibility: the plot
always reflects exactly what the code produces, not a cached state.

Run from the `scripts/` directory:
    python 05_model_comparison.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "final_model_comparison.png")

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
print("[1/5] Loading data ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

y_test = test_df["Oil_Consumption_m3_day"].values

print(f"      Train : {train_df.shape[0]} rows  "
      f"({train_df['Year'].min()} - {train_df['Year'].max()})")
print(f"      Test  : {test_df.shape[0]} rows  "
      f"({test_df['Year'].min()} - {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 3a. Linear Regression
# ---------------------------------------------------------------------------
print("[2/5] Fitting Linear Regression ...")

X_train    = train_df["Year"].values.reshape(-1, 1)
X_test     = test_df["Year"].values.reshape(-1, 1)
y_train    = train_df["Oil_Consumption_m3_day"].values

lr_model   = LinearRegression()
lr_model.fit(X_train, y_train)
linreg_preds = lr_model.predict(X_test)

# ---------------------------------------------------------------------------
# 3b. Prophet
# ---------------------------------------------------------------------------
print("[3/5] Fitting Prophet ...")

def to_prophet_df(df):
    """Convert Year / Oil_Consumption_m3_day columns to Prophet's ds / y format."""
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

future_df    = to_prophet_df(test_df)[["ds"]]
forecast     = prophet_model.predict(future_df)
prophet_preds = forecast["yhat"].values

# ---------------------------------------------------------------------------
# 3c. ARIMA(1, 1, 1)
# ---------------------------------------------------------------------------
print("[4/5] Fitting ARIMA(1,1,1) ...")

train_series = pd.Series(
    data  = train_df["Oil_Consumption_m3_day"].values,
    index = train_df["Year"].values,
    name  = "Oil_Consumption_m3_day",
)
arima_fit  = ARIMA(train_series, order=(1, 1, 1)).fit()
arima_preds = arima_fit.forecast(steps=len(test_df)).values

# ---------------------------------------------------------------------------
# 4. Compute metrics for all three models
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

lr_mae,     lr_rmse     = metrics(y_test, linreg_preds)
prophet_mae, prophet_rmse = metrics(y_test, prophet_preds)
arima_mae,  arima_rmse  = metrics(y_test, arima_preds)

print("\n========= Final Model Comparison =========")
print(f"  {'Model':<22} {'MAE':>16}  {'RMSE':>16}")
print(f"  {'-'*56}")
print(f"  {'Linear Regression':<22} {lr_mae:>16,.0f}  {lr_rmse:>16,.0f}")
print(f"  {'Prophet':<22} {prophet_mae:>16,.0f}  {prophet_rmse:>16,.0f}")
print(f"  {'ARIMA(1,1,1)':<22} {arima_mae:>16,.0f}  {arima_rmse:>16,.0f}")
print("==========================================\n")

# Programmatically determine the winner for each metric so labels stay correct
# even if someone re-runs with different data or tuned hyperparameters.
mae_scores  = {"Linear Reg": lr_mae,  "Prophet": prophet_mae,  "ARIMA": arima_mae}
rmse_scores = {"Linear Reg": lr_rmse, "Prophet": prophet_rmse, "ARIMA": arima_rmse}
mae_winner  = min(mae_scores,  key=mae_scores.get)
rmse_winner = min(rmse_scores, key=rmse_scores.get)
print(f"  Best MAE  -> {mae_winner}  ({mae_scores[mae_winner]:,.0f} m3/day)")
print(f"  Best RMSE -> {rmse_winner}  ({rmse_scores[rmse_winner]:,.0f} m3/day)\n")

# ---------------------------------------------------------------------------
# 5. Master comparison chart
# ---------------------------------------------------------------------------
print("[5/5] Generating master comparison chart ...")

fig, ax = plt.subplots(figsize=(14, 7))

# --- Shaded training zone ---------------------------------------------------
# A soft background fill is cleaner and more readable than a single vertical
# line when distinguishing two distinctly different phases of the data.
ax.axvspan(
    train_df["Year"].min(), 2015,
    alpha=0.06, color="steelblue", label="_nolegend_"
)
ax.axvspan(
    2016, test_df["Year"].max(),
    alpha=0.06, color="orange", label="_nolegend_"
)

# Crisp boundary line so the exact split year is unambiguous.
ax.axvline(x=2015.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.6)

# Zone labels just below the top of the axes.
ax.text(1988, ax.get_ylim()[1] * 0.995, "Training Zone",
        ha="center", va="top", fontsize=9, color="steelblue", alpha=0.7)
ax.text(2020, ax.get_ylim()[1] * 0.995, "Test Zone",
        ha="center", va="top", fontsize=9, color="darkorange", alpha=0.7)

# --- Data lines -------------------------------------------------------------
ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2.5,
    label="Train Data (1965-2015)"
)
ax.plot(
    test_df["Year"], y_test,
    color="black", linewidth=3,
    label="Actual Consumption (2016+)"
)

# --- Model prediction lines -------------------------------------------------
ax.plot(
    test_df["Year"], linreg_preds,
    color="green", linewidth=2, linestyle=":",
    label=f"Linear Reg  (MAE={lr_mae/1e6:.3f}M | RMSE={lr_rmse/1e6:.3f}M)"
)
ax.plot(
    test_df["Year"], prophet_preds,
    color="purple", linewidth=2, linestyle="--",
    label=f"Prophet     (MAE={prophet_mae/1e6:.3f}M | RMSE={prophet_rmse/1e6:.3f}M)"
)
ax.plot(
    test_df["Year"], arima_preds,
    color="red", linewidth=2, linestyle="-.",
    label=f"ARIMA(1,1,1)(MAE={arima_mae/1e6:.3f}M | RMSE={arima_rmse/1e6:.3f}M)"
)

# --- Winner callout annotations --------------------------------------------
# Star the best MAE and best RMSE points so the reader can see the gap
# between predictions and actual values for the winning models.
best_mae_preds  = {"Linear Reg": linreg_preds,
                   "Prophet":    prophet_preds,
                   "ARIMA":      arima_preds}[mae_winner]
best_rmse_preds = {"Linear Reg": linreg_preds,
                   "Prophet":    prophet_preds,
                   "ARIMA":      arima_preds}[rmse_winner]

# Annotate the midpoint of the test window for clarity.
mid_idx = len(test_df) // 2
ax.annotate(
    f"Best MAE\n({mae_winner})",
    xy=(test_df["Year"].iloc[mid_idx], best_mae_preds[mid_idx]),
    xytext=(20, 30), textcoords="offset points",
    fontsize=8, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
)
if rmse_winner != mae_winner:
    ax.annotate(
        f"Best RMSE\n({rmse_winner})",
        xy=(test_df["Year"].iloc[mid_idx], best_rmse_preds[mid_idx]),
        xytext=(20, -45), textcoords="offset points",
        fontsize=8, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
    )

# --- Labels, legend, formatting ---------------------------------------------
ax.set_title(
    "World Oil Consumption Forecast — All Models vs Actual (2016-2024)",
    fontsize=15, fontweight="bold", pad=14
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Oil Consumption (m3/day)", fontsize=12)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_xlim(train_df["Year"].min() - 1, test_df["Year"].max() + 1)

# Metric summary table embedded as a text box bottom-right.
summary = (
    f"{'Model':<14} {'MAE':>10}  {'RMSE':>10}\n"
    f"{'-'*37}\n"
    f"{'Linear Reg':<14} {lr_mae/1e6:>9.3f}M  {lr_rmse/1e6:>9.3f}M\n"
    f"{'Prophet':<14} {prophet_mae/1e6:>9.3f}M  {prophet_rmse/1e6:>9.3f}M\n"
    f"{'ARIMA(1,1,1)':<14} {arima_mae/1e6:>9.3f}M  {arima_rmse/1e6:>9.3f}M"
)
ax.text(
    0.98, 0.04, summary,
    transform=ax.transAxes,
    fontsize=8.5, verticalalignment="bottom",
    horizontalalignment="right", family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9,
              edgecolor="lightgrey")
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] Final model comparison script complete.")
print(f"     Best MAE  winner : {mae_winner}  ({mae_scores[mae_winner]:,.0f} m3/day)")
print(f"     Best RMSE winner : {rmse_winner}  ({rmse_scores[rmse_winner]:,.0f} m3/day)")
