# Structural Change in Nifty 50 Volatility

Detecting calm and crisis volatility regimes in the Nifty 50 index (2007–2026) using change-point analysis.

**Author:** Arijit Mondal
**Supervisor:** Dr. Buddhananda Banerjee, IIT Kharagpur

## Research question
Has Nifty 50 volatility changed structurally? If so, when, and by how much?

## Data
- Nifty 50 daily closing prices from Yahoo Finance (`^NSEI`)
- 18 Sep 2007 – 26 Aug 2026, 4,642 trading days
- Converted to log returns: r_t = ln P_t − ln P_{t−1}

## Method
1. **Stationarity:** ADF and KPSS tests. Prices are non-stationary; log returns are stationary.
2. **Autocorrelation:** returns show no autocorrelation; squared returns are strongly persistent, so breaks are sought in the variance.
3. **Mean model:** 35 ARMA models compared by AIC and BIC. BIC selects MA(1), θ = 0.037; it explains 0.14% of return variance.
4. **Diagnostics:** Ljung-Box tests on residuals and squared residuals.
5. **Change-point detection:** penalised least-squares segmentation (`ruptures.KernelCPD`, linear kernel) on standardised squared residuals, minimum segment length 40 days.
6. **Penalty:** 3·ln(n) ≈ 25.3, with a sweep from 2 to 80 to check robustness.

## Results
Five breaks, identical for every penalty from 20 to 80:

| Regime | Trading days | Annual volatility | Event |
|---|---|---|---|
| Sep 2007 – Oct 2008 | 259 | 36.1% | Crisis building |
| Oct 2008 – Dec 2008 | 40 | 71.2% | Lehman collapse |
| Dec 2008 – Jul 2009 | 148 | 41.5% | Crisis easing |
| Jul 2009 – Feb 2020 | 2,591 | 15.7% | Long calm |
| Feb 2020 – May 2020 | 41 | 68.5% | COVID crash |
| May 2020 – Aug 2026 | 1,563 | 14.6% | Post-COVID calm |

Crisis regimes are 4–5 times more volatile than calm ones, and last about 40 trading days.

## Repository structure
- `src/steps0to5.py`: data download, returns, stationarity tests, ACF, ARMA selection and diagnostics
- `src/final_results.py`: change-point detection, penalty sweep, regime table and Figure 4
- `src/results/`: figures and regime table
- `requirements.txt`: Python packages

## How to run
pip install -r requirements.txt
cd src
python final_results.py

## Tools
Python 3: pandas, numpy, statsmodels, ruptures, matplotlib, yfinance

## Limitations and future work
Simulation study with planted breaks, comparison with GARCH, bootstrap confidence intervals for break dates, and extension to other indices.
