from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path("/home/apalo/projects/TMF-Market-Regimes-Risk-Portfolio")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures" / "chapter2"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT = RAW_DATA_DIR / "ibex_prices.csv"
PROCESSED_OUTPUT = PROCESSED_DATA_DIR / "dataset_cap3_master.csv"

# =========================
# PARAMETERS
# =========================
TICKER = "^IBEX"
START_DATE = "2002-01-01"
SHORT_WINDOW = 20
MEDIUM_WINDOW = 60
VAR_TAIL_PROB = 0.05

# =========================
# HELPERS
# =========================
def downside_volatility(series: pd.Series) -> float:
    negative_returns = series[series < 0]
    if len(negative_returns) == 0:
        return np.nan
    return negative_returns.std(ddof=1)


def historical_var(series: np.ndarray, alpha: float = VAR_TAIL_PROB) -> float:
    return np.quantile(series, alpha)


def historical_cvar(series: np.ndarray, alpha: float = VAR_TAIL_PROB) -> float:
    var_threshold = np.quantile(series, alpha)
    tail_losses = series[series <= var_threshold]
    return tail_losses.mean() if len(tail_losses) > 0 else np.nan


# =========================
# STEP 1: DOWNLOAD IBEX
# =========================
print("Downloading IBEX data...")

ibex = yf.download(
    TICKER,
    start=START_DATE,
    progress=False,
    auto_adjust=True
)

if ibex.empty:
    raise ValueError("Downloaded dataset is empty.")

ibex = ibex.reset_index()

if "Date" not in ibex.columns:
    raise ValueError("Column 'Date' not found in downloaded dataset.")

if "Close" not in ibex.columns:
    raise ValueError("Column 'Close' not found in downloaded dataset.")

ibex = ibex[["Date", "Close"]].copy()
ibex.columns = ["date", "ibex_close"]

ibex["date"] = pd.to_datetime(ibex["date"])
ibex = ibex.sort_values("date").reset_index(drop=True)

# daily log returns
ibex["ret_1d"] = np.log(ibex["ibex_close"] / ibex["ibex_close"].shift(1))
ibex = ibex.dropna(subset=["ret_1d"]).reset_index(drop=True)

ibex.to_csv(RAW_OUTPUT, index=False)

print(f"Raw data exported to: {RAW_OUTPUT}")
print(f"Raw shape: {ibex.shape}")

# =========================
# STEP 2: FEATURE ENGINEERING
# =========================
print("Computing risk features...")

df = ibex.copy()

# performance features
df["ret_20d"] = df["ret_1d"].rolling(
    window=SHORT_WINDOW,
    min_periods=SHORT_WINDOW
).sum()

df["ret_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).sum()

# volatility features
df["vol_20d"] = df["ret_1d"].rolling(
    window=SHORT_WINDOW,
    min_periods=SHORT_WINDOW
).std(ddof=1)

df["vol_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).std(ddof=1)

# downside risk
df["downside_vol_20d"] = df["ret_1d"].rolling(
    window=SHORT_WINDOW,
    min_periods=SHORT_WINDOW
).apply(downside_volatility, raw=False)

df["downside_vol_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).apply(downside_volatility, raw=False)

# tail risk
df["var_95_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).apply(lambda x: historical_var(x), raw=True)

df["cvar_95_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).apply(lambda x: historical_cvar(x), raw=True)

# drawdown
running_max = df["ibex_close"].cummax()
df["drawdown"] = df["ibex_close"] / running_max - 1.0

# skewness
df["skew_60d"] = df["ret_1d"].rolling(
    window=MEDIUM_WINDOW,
    min_periods=MEDIUM_WINDOW
).skew()

# risk-adjusted return
df["ret_risk_20d"] = df["ret_20d"] / df["vol_20d"]
df["ret_risk_60d"] = df["ret_60d"] / df["vol_60d"]

feature_columns = [
    "date",
    "ibex_close",
    "ret_1d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_60d",
    "downside_vol_20d",
    "downside_vol_60d",
    "var_95_60d",
    "cvar_95_60d",
    "drawdown",
    "skew_60d",
    "ret_risk_20d",
    "ret_risk_60d",
]

master_df = df[feature_columns].copy()
master_df = master_df.dropna(subset=feature_columns).reset_index(drop=True)
master_df = master_df.sort_values("date").reset_index(drop=True)

master_df.to_csv(PROCESSED_OUTPUT, index=False)

print(f"Processed data exported to: {PROCESSED_OUTPUT}")
print(f"Processed shape: {master_df.shape}")
print(f"Date coverage: {master_df['date'].min()} to {master_df['date'].max()}")

# =========================
# STEP 3: GENERATE FIGURES
# =========================
print("Generating figures...")

master_df["date"] = pd.to_datetime(master_df["date"])

# 1. Daily returns
plt.figure(figsize=(12, 5))
plt.plot(master_df["date"], master_df["ret_1d"], linewidth=0.8)
plt.title("Daily log returns of the IBEX 35")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "daily_returns_ibex35.png", dpi=300)
plt.close()

# 2. Volatility vs downside volatility
plt.figure(figsize=(12, 5))
plt.plot(master_df["date"], master_df["vol_60d"], label="Volatility (60d)", linewidth=1)
plt.plot(master_df["date"], master_df["downside_vol_60d"], label="Downside volatility (60d)", linewidth=1)
plt.title("Volatility and downside volatility of the IBEX 35 (60-day window)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "volatility_vs_downside_60d.png", dpi=300)
plt.close()

# 3. Drawdown
plt.figure(figsize=(12, 5))
plt.plot(master_df["date"], master_df["drawdown"], linewidth=1)
plt.title("Market drawdown of the IBEX 35")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.ylim(master_df["drawdown"].min() * 1.05, 0.05)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "drawdown_ibex35.png", dpi=300)
plt.close()

# 4. VaR vs CVaR
plt.figure(figsize=(12, 5))
plt.plot(master_df["date"], master_df["var_95_60d"], label="VaR 95% (60d)", linewidth=1)
plt.plot(master_df["date"], master_df["cvar_95_60d"], label="CVaR 95% (60d)", linewidth=1)
plt.title("VaR and CVaR of the IBEX 35 (95%, 60-day window)")
plt.xlabel("Date")
plt.ylabel("Risk measure")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "var_cvar_60d.png", dpi=300)
plt.close()

# 5. Skewness
plt.figure(figsize=(10, 4))
plt.plot(master_df["date"], master_df["skew_60d"])
plt.title("Rolling skewness (60d)")
plt.xlabel("Date")
plt.ylabel("Skewness")
plt.axhline(0, linestyle="--")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "skew_60d_series.png", dpi=300)
plt.close()

# 6. Risk-adjusted return
plt.figure(figsize=(8, 5))
plt.plot(master_df["date"], master_df["ret_risk_60d"])
plt.title("Risk-adjusted return (60-day window) of the IBEX 35")
plt.xlabel("Date")
plt.ylabel("Return / Volatility (60d)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ret_risk_60d_series.png", dpi=300)
plt.close()

print("Figures generated successfully.")
print("Daily update completed successfully.")