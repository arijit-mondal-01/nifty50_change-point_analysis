"""
Final, simplified change-point analysis for Nifty 50.
Detects shifts in volatility (squared ARMA residuals) with PELT-style
segmentation, then reports each regime's volatility and saves a figure.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

from steps0to5 import data, final_fit   # reuses Steps 0-5 (data is cached)

MIN_SIZE = 40   # shortest regime allowed, in trading days

# 1. Signal: squared residuals = daily volatility proxy, standardised
resid = final_fit.resid.dropna()
dates = resid.index
sig = resid.values ** 2
sig = (sig - sig.mean()) / sig.std()

# 2. Detector: finds dates where the AVERAGE level of volatility shifts
#    ("linear" kernel = classic least-squares change-point, fast)
algo = rpt.KernelCPD(kernel="linear", min_size=MIN_SIZE).fit(sig.reshape(-1, 1))

# 3. Penalty sweep: higher penalty = fewer breaks
print("\nPenalty sweep")
for pen in [2, 5, 10, 20, 30, 50, 80]:
    bk = algo.predict(pen=pen)[:-1]
    print(f"pen={pen:>3}: {len(bk):>2} breaks ->",
          [str(dates[b].date()) for b in bk])

# 4. Chosen penalty: a BIC-style rule, 3 x log(n). Adjust after the sweep.
PENALTY = 3 * np.log(len(sig))
breaks = algo.predict(pen=PENALTY)[:-1]
print(f"\nChosen penalty {PENALTY:.1f}: {len(breaks)} breaks")

# 5. Results table: volatility in each regime
bounds = [0] + breaks + [len(resid)]
rows = []
for a, b in zip(bounds[:-1], bounds[1:]):
    seg = data["log_return"].loc[dates[a]:dates[b - 1]]
    rows.append({
        "start": dates[a].date(),
        "end": dates[b - 1].date(),
        "days": len(seg),
        "annual_vol_pct": round(seg.std() * np.sqrt(252) * 100, 1),
    })
table = pd.DataFrame(rows)
print("\nVolatility regimes\n", table.to_string(index=False))
table.to_csv("results/volatility_regimes.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(data.index, data["log_return"], lw=0.4, alpha=0.5, color="steelblue")

# Each label gets its own height so neighbouring labels never overlap
heights = [0.16, 0.13, 0.10, 0.16, 0.13, 0.16]

for i, r in enumerate(rows):
    start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
    colour = "red" if r["annual_vol_pct"] > 30 else "green"
    ax.axvspan(start, end, color=colour, alpha=0.12)

    mid = start + (end - start) / 2
    ax.annotate(
        f'{r["annual_vol_pct"]}%',
        xy=(mid, 0.06),                 # arrow points into the band
        xytext=(mid, heights[i]),       # label sits at its own height
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, lw=0.8),
        arrowprops=dict(arrowstyle="-", color=colour, lw=0.8),
    )

ax.set_ylim(-0.16, 0.19)                # leaves room for the labels at the top
ax.set_ylabel("Daily log return")
ax.set_title("Nifty 50 — annualised volatility by regime (red = crisis, green = calm)")
plt.tight_layout()
plt.savefig("results/fig4_volatility_regimes.png", dpi=150)
plt.show()