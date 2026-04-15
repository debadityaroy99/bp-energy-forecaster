# BP Strategic Energy Transition Dashboard

A final-year B.Tech project forecasting **global oil consumption** and modelling the **renewable energy capacity required** to achieve a net-zero transition by a user-selected target year.

---

## Live Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   ├── world-crude-oil-price-vs-oil-consumption.csv   # raw dataset (not in repo — see below)
│   ├── cleaned_energy_data.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   └── *.png                       # all generated forecast charts
└── scripts/
    ├── 01_data_prep.py             # data cleaning & train/test split
    ├── 02_model_linreg.py          # Linear Regression baseline
    ├── 03_model_prophet.py         # Facebook Prophet baseline
    ├── 04_model_arima.py           # ARIMA(1,1,1) baseline
    ├── 05_model_comparison.py      # Annual 3-model comparison chart
    ├── 06_model_lstm.py            # Basic LSTM (annual, univariate)
    ├── 07_model_lstm_advanced.py   # Advanced LSTM (monthly, multivariate)
    ├── 07_master_comparison.py     # 4-model master comparison chart
    ├── 08_baselines_advanced.py    # SARIMAX + advanced monthly baselines
    └── 09_model_hybrid_novelty.py  # Champion: Hybrid LinReg-LSTM ensemble
```

---

## Champion Model — Hybrid LinReg-LSTM

The novel architecture that achieved the best results across all models tested:

| Stage | Component | Role |
|-------|-----------|------|
| 1 | Linear Regression on `Time_Index` | Captures deterministic long-run trend (R² = 0.9485) |
| 2 | LSTM (32 units) on residuals | Learns non-linear corrections to the trend |
| Output | `Trend + LSTM Residual` | Final hybrid forecast |

**Why it works:** By pre-removing the dominant linear trend, the LSTM's learning target shrinks from millions to hundreds-of-thousands of m³/day — a far simpler signal that converges in as few as 5 epochs.

---

## Full Model Leaderboard

| Rank | Model | MAE (m³/day) | RMSE (m³/day) | Notes |
|------|-------|-------------|--------------|-------|
| 1 | **Hybrid LinReg-LSTM** | **~171,647** | **~307,016** | Champion — novel architecture |
| 2 | Advanced LSTM | 313,602 | 396,690 | Monthly, multivariate |
| 3 | Multi-LinReg (advanced) | 513,458 | 586,749 | Monthly, 3 features |
| 4 | SARIMAX(1,1,1) | 502,395 | 693,094 | Monthly, 2 exogenous |
| 5 | Linear Regression | 402,486 | 526,578 | Annual baseline |
| 6 | ARIMA(1,1,1) | 469,645 | 522,920 | Annual baseline |
| 7 | Prophet | 422,644 | 545,539 | Annual baseline |
| 8 | Basic LSTM | 613,700 | 895,389 | Data-starved (annual) |

---

## Dataset

**Source:** [Our World in Data — Oil Consumption vs. Price](https://ourworldindata.org/grapher/oil-consumption-vs-price)

The raw CSV (`world-crude-oil-price-vs-oil-consumption.csv`) is excluded from this repository due to file size. Download it from the link above and place it in the `data/` folder before running.

---

## Installation

```bash
pip install -r requirements.txt
```

Then run the data pipeline first:

```bash
cd scripts
python 01_data_prep.py
```

Then launch the dashboard:

```bash
cd ..
streamlit run app.py
```

---

## Dashboard Features

- **Hybrid LinReg-LSTM** forecast to any user-selected year (2030–2070)
- **Hydrocarbon Phase-Out** curve: linear ramp from current consumption → 0 at Net-Zero year
- **Renewable Deficit** visualisation: green shaded area showing the exact clean-energy capacity BP must build
- Interactive **Plotly** charts with dark corporate theme
- Sidebar sliders for forecast horizon and Net-Zero target year
- Annual milestone summary table

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow/Keras | 2.21.0 | LSTM model |
| scikit-learn | 1.6.1 | LinearRegression, MinMaxScaler, metrics |
| statsmodels | 0.14.6 | SARIMAX |
| Prophet | 1.3.0 | Facebook Prophet |
| Streamlit | 1.56.0 | Dashboard UI |
| Plotly | 6.7.0 | Interactive charts |
| pandas | 2.2.3 | Data pipeline |
| numpy | 2.2.3 | Numerical operations |
| matplotlib | 3.10.1 | Static script charts |
