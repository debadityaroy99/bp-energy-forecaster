"""
01_data_prep.py
---------------
Stage 1 of the forecasting pipeline: load raw CSV data, clean and filter it,
perform a sequential train/test split, generate a diagnostic plot, and export
the processed splits for downstream modelling.

Run from the `scripts/` directory:
    python 01_data_prep.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../scripts/
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
RAW_CSV   = os.path.join(DATA_DIR, "world-crude-oil-price-vs-oil-consumption.csv")

CLEANED_CSV       = os.path.join(DATA_DIR, "cleaned_energy_data.csv")
TRAIN_CSV         = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV          = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH         = os.path.join(DATA_DIR, "train_test_split.png")

SPLIT_YEAR = 2015

# ---------------------------------------------------------------------------
# 2. Load raw data
# ---------------------------------------------------------------------------
print("[1/5] Loading raw CSV ...")
# Use latin-1 encoding to safely handle special characters (e.g. ³) in headers.
df_raw = pd.read_csv(RAW_CSV, encoding="latin-1")
print(f"      Raw shape: {df_raw.shape}")

# ---------------------------------------------------------------------------
# 3. Clean & Filter
# ---------------------------------------------------------------------------
print("[2/5] Cleaning and filtering ...")

# Keep only global aggregates and post-1964 observations.
df = df_raw[
    (df_raw["Entity"] == "World") &
    (df_raw["Year"]   >= 1965)
].copy()

# Column at index 4 carries an encoding-unsafe name ('Oil consumption - m³/day').
# Rename it positionally so the script is robust to any encoding environment.
consumption_col = df.columns[4]
df.rename(columns={consumption_col: "Oil_Consumption_m3_day"}, inplace=True)

# Retain only the columns needed for forecasting.
df = df[["Year", "Oil_Consumption_m3_day"]]

# Remove rows where the target variable is missing.
df.dropna(inplace=True)

# Ensure Year is stored as an integer for clean axis labels.
df["Year"] = df["Year"].astype(int)

df.reset_index(drop=True, inplace=True)
print(f"      Cleaned shape: {df.shape}")

# ---------------------------------------------------------------------------
# 4. Sequential train / test split
# ---------------------------------------------------------------------------
print(f"[3/5] Splitting at Year = {SPLIT_YEAR} ...")

train_df = df[df["Year"] <= SPLIT_YEAR].reset_index(drop=True)
test_df  = df[df["Year"] >  SPLIT_YEAR].reset_index(drop=True)

print(f"      Training set : {train_df.shape[0]} rows  "
      f"({train_df['Year'].min()} – {train_df['Year'].max()})")
print(f"      Testing set  : {test_df.shape[0]}  rows  "
      f"({test_df['Year'].min()} – {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 5. Diagnostic plot
# ---------------------------------------------------------------------------
print("[4/5] Generating train/test split plot ...")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2, label="Training Data (<= 2015)"
)
ax.plot(
    test_df["Year"], test_df["Oil_Consumption_m3_day"],
    color="darkorange", linewidth=2, label="Testing Data (> 2015)"
)

# Mark the split boundary with a subtle dashed line.
ax.axvline(x=SPLIT_YEAR, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.text(SPLIT_YEAR + 0.2, ax.get_ylim()[0], f" Split ({SPLIT_YEAR})",
        color="grey", fontsize=9, va="bottom")

ax.set_title("World Oil Consumption — Train / Test Split", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Oil Consumption (m³/day)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------
print("[5/5] Exporting artefacts ...")

os.makedirs(DATA_DIR, exist_ok=True)

# Save the plot before any plt.show() call to guarantee the file is written.
fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"      Plot saved  -> {PLOT_PATH}")

df.to_csv(CLEANED_CSV, index=False)
print(f"      Master CSV  -> {CLEANED_CSV}")

train_df.to_csv(TRAIN_CSV, index=False)
print(f"      Train CSV   -> {TRAIN_CSV}")

test_df.to_csv(TEST_CSV, index=False)
print(f"      Test CSV    -> {TEST_CSV}")

print("\n[OK] Data preparation complete. All artefacts saved to ../data/")
