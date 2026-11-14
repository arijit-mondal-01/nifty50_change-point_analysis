"""
Nifty 50 Change-Point Pipeline — Steps 0 to 5.

Data acquisition, stationarity testing, ARMA order selection, final fit.

Produces the three objects the Step 6-9 script expects:
    data       : DataFrame with DatetimeIndex and a "log_return" column
    best_order : (p, d, q) chosen by AIC
    final_fit  : fitted ARIMA result on the full sample

Run standalone:      python src/steps0to5.py
Or import:           from steps0to5 import data, best_order, final_fit
"""
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
# Always run from the src folder, whichever way the script is launched
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─── Configuration ───────────────────────────────────────────────────────
TICKER = "^NSEI"          # Nifty 50 index on Yahoo Finance
START_DATE = "2007-09-17"  # first date with reliable ^NSEI history
END_DATE = "2026-08-27"    # PIN THIS. Do not use date.today().

# Pinning the end date is not fussiness — it is what makes your numbers
# reproducible. If the end date drifts every time you run the script, your
# breakpoints and penalty selection drift with it, and a figure in your
# report stops matching the code that produced it.

DATA_DIR = "data/raw"
CACHE_PATH = os.path.join(DATA_DIR, f"nifty50_{START_DATE}_to_{END_DATE}.parquet")

MAX_P = 5   # ARMA order search grid
MAX_Q = 5


# ═══════════════════════════════════════════════════════════════════════
# STEP 0 — DOWNLOAD AND CACHE
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """
    Download once, cache to parquet, reuse thereafter.

    Re-downloading on every run is usually the slowest single operation in
    the whole notebook, and it silently couples your results to whenever
    you happened to run the code. Cache it.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print(f"Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
    raw = yf.download(TICKER, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError("Download returned no rows — check ticker and dates.")

    # yfinance returns a MultiIndex column layout for some versions/queries
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.to_parquet(CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return raw


df = load_data()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — LOG RETURNS AND CLEANING
# ═══════════════════════════════════════════════════════════════════════

data = pd.DataFrame(index=df.index)
data["close"] = df["Close"]

# Log returns rather than simple returns: they are additive across time,
# closer to symmetric, and are what the ARMA/Gaussian machinery downstream
# assumes. r_t = ln(P_t) - ln(P_{t-1})
data["log_return"] = np.log(data["close"]).diff()

# Drop the first NaN from differencing, and any exchange-holiday gaps that
# slipped through as zero-volume rows.
data = data.dropna(subset=["log_return"])
data = data[data["log_return"] != 0]  # remove non-trading artefacts

print(f"\nObservations: {len(data)}")
print(f"Date range   : {data.index[0].date()} to {data.index[-1].date()}")
print(f"Mean return  : {data['log_return'].mean():.6f}")
print(f"Std dev      : {data['log_return'].std():.6f}")
print(f"Annualised σ : {data['log_return'].std() * np.sqrt(252):.4f}")
print(f"Skewness     : {data['log_return'].skew():.4f}")
print(f"Kurtosis     : {data['log_return'].kurtosis():.4f}")

# Excess kurtosis will be large (typically 8-15 for Indian equities). Note it:
# it is the reason your simulation study should include t-distributed
# innovations, not just Gaussian ones.

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(data.index, data["close"], lw=0.8)
axes[0].set_ylabel("Nifty 50 close")
axes[0].set_title("Nifty 50 — price level and log returns")
axes[1].plot(data.index, data["log_return"], lw=0.5, alpha=0.8)
axes[1].set_ylabel("Log return")
plt.tight_layout()
plt.savefig("results/fig1_price_and_returns.png", dpi=150)
plt.show()

# Volatility clustering should be visible by eye in the lower panel — calm
# stretches broken by bursts. That is the phenomenon the change-point
# pipeline is trying to formalise, and it is worth saying so in the report.


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — STATIONARITY TESTING
# ═══════════════════════════════════════════════════════════════════════

def stationarity_report(series, name):
    """
    ADF and KPSS together.

    ADF null  : unit root present (non-stationary)
    KPSS null : series is stationary

    They have opposite nulls, so running both guards against the weak power
    of either alone. Agreement is what you want to report; disagreement is
    a genuine finding worth a sentence.
    """
    print(f"\n─── Stationarity: {name} ───")

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag="AIC")
    print(f"ADF  statistic {adf_stat:.4f}, p = {adf_p:.6f}")
    print(f"     critical values: {dict((k, round(v, 3)) for k, v in adf_crit.items())}")
    print(f"     -> {'stationary' if adf_p < 0.05 else 'NON-stationary'} at 5%")

    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")
    print(f"KPSS statistic {kpss_stat:.4f}, p = {kpss_p:.4f}")
    print(f"     -> {'stationary' if kpss_p > 0.05 else 'NON-stationary'} at 5%")

    return adf_p, kpss_p


stationarity_report(data["close"].dropna(), "price level")
stationarity_report(data["log_return"], "log returns")

# Expected outcome: prices non-stationary, log returns stationary. That is
# the justification for modelling returns rather than prices, and for
# setting d = 0 in the ARIMA order below.


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — ACF / PACF INSPECTION
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
plot_acf(data["log_return"], lags=40, ax=axes[0, 0])
axes[0, 0].set_title("ACF — log returns")
plot_pacf(data["log_return"], lags=40, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title("PACF — log returns")

# The squared-return plots matter more than the raw ones for this project.
plot_acf(data["log_return"] ** 2, lags=40, ax=axes[1, 0])
axes[1, 0].set_title("ACF — SQUARED returns (volatility persistence)")
plot_pacf(data["log_return"] ** 2, lags=40, ax=axes[1, 1], method="ywm")
axes[1, 1].set_title("PACF — squared returns")
plt.tight_layout()
plt.savefig("results/fig2_acf_pacf.png", dpi=150)
plt.show()

# Read these two rows against each other and put the comparison in your
# report. The top row will show almost nothing — daily returns are close to
# serially uncorrelated. The bottom row will show strong, slowly decaying
# autocorrelation. That contrast is the empirical basis for the decision in
# Step 6 to detect on squared residuals rather than on the residuals
# themselves: the structure in this series lives in the variance.

acf_ret = acf(data["log_return"], nlags=10)
acf_sq = acf(data["log_return"] ** 2, nlags=10)
print("\nACF of returns   (lags 1-5):", np.round(acf_ret[1:6], 4))
print("ACF of squared   (lags 1-5):", np.round(acf_sq[1:6], 4))


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — ARMA ORDER SELECTION BY AIC GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════

def select_order(series, max_p=MAX_P, max_q=MAX_Q, d=0):
    """
    Exhaustive (p, q) grid scored by AIC and BIC.

    Failed fits are recorded rather than silently skipped, so you can see
    whether a region of the grid was simply unreachable.
    """
    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(series, order=(p, d, q)).fit()
            rows.append({"p": p, "d": d, "q": q,
                         "aic": res.aic, "bic": res.bic, "converged": True})
        except Exception as e:
            rows.append({"p": p, "d": d, "q": q,
                         "aic": np.nan, "bic": np.nan,
                         "converged": False, "error": str(e)[:60]})
    return pd.DataFrame(rows)


print("\nRunning ARMA order grid search ...")
order_df = select_order(data["log_return"])

n_failed = (~order_df["converged"]).sum()
print(f"Grid complete: {len(order_df)} candidates, {n_failed} failed to converge")

valid = order_df.dropna(subset=["aic"])
best_aic_row = valid.loc[valid["aic"].idxmin()]
best_bic_row = valid.loc[valid["bic"].idxmin()]

print("\nTop 5 by AIC:")
print(valid.nsmallest(5, "aic")[["p", "d", "q", "aic", "bic"]].to_string(index=False))

best_order = (int(best_bic_row["p"]), int(best_bic_row["d"]), int(best_bic_row["q"]))
bic_order = (int(best_bic_row["p"]), int(best_bic_row["d"]), int(best_bic_row["q"]))

print(f"\nBest by AIC: {best_order}")
print(f"Best by BIC: {bic_order}")

# AIC and BIC frequently disagree here, BIC preferring the smaller model
# because its complexity penalty grows with sample size. Report both and
# state which you used and why. For this pipeline the ARMA is a nuisance
# filter rather than the object of interest, so the parsimonious BIC choice
# is easy to defend — but say so explicitly rather than leaving it implicit.

order_df.to_csv("results/arma_order_search.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — FIT FINAL MODEL AND CHECK RESIDUALS
# ═══════════════════════════════════════════════════════════════════════

final_fit = ARIMA(data["log_return"], order=best_order).fit()
print("\n", final_fit.summary())

resid = final_fit.resid.dropna()

# Ljung-Box on residuals: null is no remaining autocorrelation. Large
# p-values mean the ARMA has done its job on the conditional mean.
lb_resid = acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)
print("\nLjung-Box on residuals (want large p — no leftover autocorrelation):")
print(lb_resid)

# Ljung-Box on SQUARED residuals: small p-values mean strong ARCH effects
# remain. Expect them, and do not treat it as a failure of the model —
# it is the signal your change-point analysis is built to detect.
lb_sq = acorr_ljungbox(resid ** 2, lags=[5, 10, 20], return_df=True)
print("\nLjung-Box on SQUARED residuals (small p = volatility clustering remains):")
print(lb_sq)

# This pair of tables is the cleanest single piece of evidence in the whole
# report for why you detect breaks on squared residuals. Quote it directly.

# Proportion of return variance the ARMA actually explains
r_squared = 1 - resid.var() / data["log_return"].var()
print(f"\nARMA R-squared on returns: {r_squared:.5f}")

# This will be very small — well under 1%. Say so plainly in the write-up.
# It is the honest reason the "residuals vs raw returns" distinction in the
# original guide is weaker than it sounds, and it motivates the shift to
# variance-based detection.

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
axes[0, 0].plot(resid.index, resid.values, lw=0.5)
axes[0, 0].set_title("ARMA residuals")
axes[0, 1].hist(resid.values, bins=100, density=True, alpha=0.7)
axes[0, 1].set_title("Residual distribution")
plot_acf(resid, lags=40, ax=axes[1, 0])
axes[1, 0].set_title("ACF — residuals")
plot_acf(resid ** 2, lags=40, ax=axes[1, 1])
axes[1, 1].set_title("ACF — squared residuals")
plt.tight_layout()
plt.savefig("results/fig3_residual_diagnostics.png", dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════
# HANDOFF
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Steps 0-5 complete. Available for Steps 6-9:")
    print(f"  data       : {len(data)} rows, columns {list(data.columns)}")
    print(f"  best_order : {best_order}")
    print(f"  final_fit  : ARIMA{best_order}, AIC {final_fit.aic:.2f}")
    print("=" * 60)
