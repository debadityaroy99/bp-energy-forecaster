"""
07_model_lstm_advanced.py
--------------------------
Advanced multivariate, panel-data LSTM with monthly upsampling.

Why this solves the data-starvation problem from script 06:
  - Annual data gives the LSTM only 48 training sequences for one country.
  - Monthly upsampling of 21 entities multiplies the training corpus to
    ~21 * 51 years * 12 months = ~12,852 sequences — enough for a deep
    network to learn meaningful temporal patterns.
  - Adding Price and Population as co-variates turns the model multivariate:
    the LSTM can learn that a price spike precedes a consumption dip, etc.

Run from the `scripts/` directory:
    python 07_model_lstm_advanced.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 0. Paths & hyper-parameters
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
RAW_CSV     = os.path.join(DATA_DIR, "world-crude-oil-price-vs-oil-consumption.csv")
PLOT_PATH   = os.path.join(DATA_DIR, "lstm_advanced_forecast.png")

LOOK_BACK   = 12      # 12-month sliding window
SPLIT_YEAR  = 2015
N_ENTITIES  = 20      # top-N by mean consumption (World is already #1)
EPOCHS      = 100
BATCH_SIZE  = 64

BASELINE_MAE  = 402_485.43
BASELINE_RMSE = 526_577.50

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("[1/8] Loading raw CSV ...")
df_raw = pd.read_csv(RAW_CSV, encoding="latin-1")
print(f"      Raw shape: {df_raw.shape}")

# ---------------------------------------------------------------------------
# 2. Rename columns positionally (avoids encoding issues with '³' character)
# ---------------------------------------------------------------------------
print("[2/8] Cleaning & renaming columns ...")

df_raw.rename(columns={
    df_raw.columns[3]: "Price",        # Oil price (constant 2024 US$)
    df_raw.columns[4]: "Consumption",  # Oil consumption - m³/day
    df_raw.columns[5]: "Population",   # Population (historical)
}, inplace=True)

# Keep only years with meaningful data and rows where Consumption exists.
df = df_raw[df_raw["Year"] >= 1965].copy()
df.dropna(subset=["Consumption"], inplace=True)

# Fill sparse Price & Population values within each entity's time series.
df = (df.groupby("Entity", group_keys=False)
        .apply(lambda g: g.ffill().bfill()))

# Drop any remaining NaN rows (entities with no price data at all).
df.dropna(subset=["Price", "Population"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"      After cleaning: {df.shape} | unique entities: {df['Entity'].nunique()}")

# ---------------------------------------------------------------------------
# 3. Panel data — top-20 entities by mean Consumption + World (guaranteed)
# ---------------------------------------------------------------------------
print("[3/8] Selecting panel entities ...")

top_entities = (df.groupby("Entity")["Consumption"]
                  .mean()
                  .sort_values(ascending=False)
                  .head(N_ENTITIES)
                  .index.tolist())

# Guarantee 'World' is present regardless of its rank.
if "World" not in top_entities:
    top_entities.append("World")

df = df[df["Entity"].isin(top_entities)].copy()
print(f"      Panel entities ({len(top_entities)}): {', '.join(sorted(top_entities))}")
print(f"      Panel shape: {df.shape}")

# ---------------------------------------------------------------------------
# 4. Monthly upsampling via linear interpolation
# ---------------------------------------------------------------------------
print("[4/8] Upsampling to monthly frequency ...")

# Convert Year integer to Jan-1 datetime so resample() has a time axis.
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
df.set_index("Date", inplace=True)

monthly_groups = []
for entity, group in df.groupby("Entity"):
    # Sort by time, keep only numeric columns that make sense to interpolate.
    g = group[["Consumption", "Price", "Population"]].sort_index()

    # resample to Month-Start; interpolate fills the 11 new months linearly.
    g_monthly = g.resample("MS").interpolate(method="linear")

    # Forward-fill / backward-fill any edge NaN that interpolation can't fix.
    g_monthly = g_monthly.ffill().bfill()
    g_monthly["Entity"] = entity
    monthly_groups.append(g_monthly)

df_monthly = pd.concat(monthly_groups)
df_monthly.reset_index(inplace=True)           # brings Date back as a column
df_monthly.rename(columns={"index": "Date"}, errors="ignore", inplace=True)

# Extract Year and Month for the train/test split logic.
df_monthly["Year"]  = df_monthly["Date"].dt.year
df_monthly["Month"] = df_monthly["Date"].dt.month

df_monthly.dropna(subset=["Consumption", "Price", "Population"], inplace=True)
print(f"      Monthly shape: {df_monthly.shape}")
print(f"      Date range: {df_monthly['Date'].min().date()} "
      f"to {df_monthly['Date'].max().date()}")

# ---------------------------------------------------------------------------
# 5. Scale the 3 features to [0, 1]
# ---------------------------------------------------------------------------
print("[5/8] Scaling features ...")

FEATURE_COLS = ["Consumption", "Price", "Population"]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the full panel so every entity shares the same scale, allowing the
# network to compare absolute levels across countries.
df_monthly[FEATURE_COLS] = scaler.fit_transform(df_monthly[FEATURE_COLS])

# ---------------------------------------------------------------------------
# 6. Sequence generation — never mix entities
# ---------------------------------------------------------------------------
print("[6/8] Creating sequences (look_back={} months) ...".format(LOOK_BACK))

def create_sequences(data_df, look_back=12, split_year=2015):
    """
    Build sliding-window (X, y) pairs grouped by Entity.

    Rules:
      - Training : sequences whose TARGET month falls in year <= split_year,
                   drawn from ALL panel entities.
      - Testing  : sequences whose TARGET month falls in year >  split_year,
                   drawn ONLY from 'World' (our evaluation target).

    Returns four numpy arrays: X_train, y_train, X_test, y_test.
    """
    X_tr, y_tr = [], []
    X_te, y_te = [], []

    for entity, group in data_df.groupby("Entity"):
        group = group.sort_values("Date").reset_index(drop=True)
        features = group[FEATURE_COLS].values   # (T, 3)  — already scaled
        years    = group["Year"].values          # (T,)

        for i in range(len(features) - look_back):
            X_seq       = features[i : i + look_back]   # (look_back, 3)
            y_val       = features[i + look_back, 0]     # next Consumption
            target_year = years[i + look_back]

            if target_year <= split_year:
                X_tr.append(X_seq)
                y_tr.append(y_val)
            elif entity == "World":
                # Only evaluate on the World series.
                X_te.append(X_seq)
                y_te.append(y_val)

    return (np.array(X_tr), np.array(y_tr),
            np.array(X_te), np.array(y_te))

X_train, y_train, X_test, y_test = create_sequences(
    df_monthly, look_back=LOOK_BACK, split_year=SPLIT_YEAR
)

# Keras LSTM input must be 3-D: [samples, time_steps, features].
# create_sequences already returns (n, look_back, 3) — no reshape needed.
print(f"      X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"      X_test : {X_test.shape}    y_test : {y_test.shape}")

# ---------------------------------------------------------------------------
# 7. Build the advanced LSTM
# ---------------------------------------------------------------------------
print("[7/8] Building Advanced LSTM model ...")

model = Sequential([
    # 64 units — larger than the baseline (50) to capture cross-entity patterns
    # in the multivariate input.  return_sequences=False: we only need the
    # hidden state at the final time step for a one-step-ahead forecast.
    LSTM(64, activation="relu",
         input_shape=(LOOK_BACK, len(FEATURE_COLS)),
         return_sequences=False),

    # Dropout regularises the network by randomly zeroing 20 % of activations
    # each forward pass during training, reducing overfitting on any one entity.
    Dropout(0.2),

    Dense(1),
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ---------------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------------
print(f"\n[8/8] Training (epochs={EPOCHS}, batch_size={BATCH_SIZE}) ...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val   = min(history.history["val_loss"])
print(f"\n      Best epoch: {best_epoch}  |  best val_loss: {best_val:.6f}")

# ---------------------------------------------------------------------------
# Predict & inverse-transform
# ---------------------------------------------------------------------------
print("\nPredicting on World test set ...")

preds_scaled = model.predict(X_test, verbose=0)   # shape (n, 1)

def inverse_scale_consumption(scaled_values):
    """
    Inverse-transform only the Consumption column (index 0 in the scaler).
    The scaler was fit on 3 features, so we build a dummy (n, 3) array,
    fill column 0 with the values we care about, and discard the rest.
    """
    dummy = np.zeros((len(scaled_values), len(FEATURE_COLS)))
    dummy[:, 0] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

lstm_preds = inverse_scale_consumption(preds_scaled)
y_test_real = inverse_scale_consumption(y_test)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
mae  = mean_absolute_error(y_test_real, lstm_preds)
rmse = np.sqrt(mean_squared_error(y_test_real, lstm_preds))

print("\n--- Advanced LSTM Evaluation (World, 2016-2024) ---")
print(f"  MAE  : {mae:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_MAE:>15,.2f}  |  delta: {BASELINE_MAE - mae:+,.2f})")
print(f"  RMSE : {rmse:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_RMSE:>15,.2f}  |  delta: {BASELINE_RMSE - rmse:+,.2f})")
print("---------------------------------------------------\n")

if mae < BASELINE_MAE and rmse < BASELINE_RMSE:
    print("  [PASS] Advanced LSTM beats the Linear Regression baseline on BOTH metrics.")
elif mae < BASELINE_MAE or rmse < BASELINE_RMSE:
    print("  [PARTIAL] Advanced LSTM beats the baseline on one metric.")
else:
    print("  [INFO] Advanced LSTM did not beat the baseline. "
          "Consider tuning LOOK_BACK, LSTM units, or adding a second LSTM layer.")

# ---------------------------------------------------------------------------
# Build a monthly date range for the X-axis of the test plot
# ---------------------------------------------------------------------------
world_test_dates = (df_monthly[
    (df_monthly["Entity"] == "World") &
    (df_monthly["Year"]   >  SPLIT_YEAR)
]["Date"].sort_values().reset_index(drop=True))

# Trim to match the number of test sequences (look_back offset removes the
# first LOOK_BACK months which are used only as seed windows).
plot_dates = world_test_dates.iloc[LOOK_BACK:].reset_index(drop=True)

# Guard: truncate arrays to the shortest dimension if lengths diverge.
n = min(len(plot_dates), len(y_test_real), len(lstm_preds))
plot_dates, y_test_real, lstm_preds = (
    plot_dates[:n], y_test_real[:n], lstm_preds[:n]
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("Generating forecast plot ...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]

ax.plot(plot_dates, y_test_real,
        color="darkorange", linewidth=2,
        label="Actual World Consumption (2016+)")
ax.plot(plot_dates, lstm_preds,
        color="teal", linewidth=2, linestyle="--",
        label=f"Advanced LSTM Prediction (look_back={LOOK_BACK} months)")

ax.fill_between(plot_dates, y_test_real, lstm_preds,
                alpha=0.12, color="teal", label="Prediction error band")

ax.set_title("Advanced Multivariate LSTM Forecast vs Actual\n"
             "(21-entity panel, monthly granularity, 3 features)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Oil Consumption (m\u00b3/day)", fontsize=11)
ax.legend(fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.4)

metric_text = (
    f"Advanced LSTM  MAE  = {mae/1e6:.3f}M\n"
    f"Advanced LSTM  RMSE = {rmse/1e6:.3f}M\n"
    f"LinReg baseline MAE  = {BASELINE_MAE/1e6:.3f}M\n"
    f"LinReg baseline RMSE = {BASELINE_RMSE/1e6:.3f}M"
)
ax.text(0.02, 0.97, metric_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#e0f7f7", alpha=0.9))

# --- Training loss curve (bottom panel) ------------------------------------
ax2 = axes[1]
ax2.plot(history.history["loss"],     color="steelblue",  label="Train loss")
ax2.plot(history.history["val_loss"], color="darkorange", label="Val loss")
ax2.axvline(x=best_epoch - 1, color="grey",
            linestyle="--", linewidth=1, alpha=0.7)
ax2.set_title("Training & Validation Loss (MSE on scaled data)",
              fontsize=11)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("MSE Loss", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout(h_pad=2.5)
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] Advanced LSTM script complete.")
