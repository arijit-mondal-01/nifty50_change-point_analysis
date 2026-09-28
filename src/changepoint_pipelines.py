"""
Nifty 50 Change-Point Pipeline — Steps 6 to 9, corrected and sped up.

Drop-in replacement for Steps 6-9 of the advanced guide.
Assumes Steps 0-5 already ran and left these in memory:
    data       : DataFrame with a DatetimeIndex and a "log_return" column
    best_order : tuple (p, d, q) chosen by AIC grid search
    final_fit  : fitted ARIMA result on the full sample

What changed vs the original guide, and why, is commented inline at each step.
"""

# ═══════════════════════════════════════════════════════════════════════
# 0. SETUP
# ═══════════════════════════════════════════════════════════════════════

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ARIMA fits on short segments throw hundreds of convergence warnings.
# We suppress them for readability but COUNT the actual failures later,
# rather than hiding them behind a bare `except: continue` as the guide did.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

SEED = 42
rng = np.random.default_rng(SEED)

# One place to change the segment-length constraint. The original guide used
# min_size=30 for PELT but distance=60 for the Bayesian peak finder, which
# meant the two methods were operating under different constraints — that
# quietly biased the agreement test in Step 6c. Now they share one value.
MIN_SIZE = 40          # minimum trading days between change points
PENALTY = 10           # initial penalty; Step 9 tunes this properly
N_PERMUTATIONS = 1000  # for the agreement null distribution


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def cps_to_positions(bkps, n):
    """
    Normalise ruptures output to ONE convention: each change point is the
    positional index of the FIRST observation of the new regime.

    ruptures' .predict() returns segment END indices, with the final entry
    equal to n. So dropping the last entry gives exactly the "first obs of
    new regime" positions. The guide's `cp - 1` gave the LAST obs of the OLD
    regime instead — a different convention from the one it used for the
    Bayesian dates, creating a systematic 1-2 day offset between the two
    methods and inflating the measured disagreement for no real reason.
    """
    return [int(b) for b in bkps if 0 < int(b) < n]


def standardise(x):
    """
    The rbf/kernel cost is scale-dependent: the same penalty value means
    completely different things on a series with sd 0.01 vs sd 1.0. Log
    returns have sd ~0.01, so an unstandardised penalty of 10 is effectively
    enormous. Standardising makes penalty values interpretable and portable
    between Step 6 and Step 9 — the guide applied a penalty tuned on raw
    returns to residuals, which are on a different scale.
    """
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def detect(signal_1d, penalty, min_size=MIN_SIZE, cached_algo=None):
    """
    Kernel change-point detection.

    SPEED: rpt.KernelCPD(kernel="rbf") solves the same segmentation problem
    as rpt.Pelt(model="rbf") but runs in compiled C and caches the Gram
    matrix internally. Sweeping ten penalty values costs barely more than
    one — which is what makes Step 9 go from minutes to seconds. Pass the
    fitted algo back in via `cached_algo` to reuse it across penalties.
    """
    sig = standardise(signal_1d).reshape(-1, 1)
    algo = cached_algo if cached_algo is not None else \
        rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(sig)
    bkps = algo.predict(pen=penalty)
    return cps_to_positions(bkps, len(sig)), algo


def gaussian_loglik(x, mu, sigma):
    """
    Mean per-observation Gaussian log-likelihood.

    This replaces MSE as the scoring rule throughout. Reason: an ARMA
    forecast of daily returns reverts to the unconditional mean within 2-3
    steps, so comparing two forecasts by MSE is really just comparing two
    sample means against 20 days of noise — the result flips sign if you
    move the horizon by one day. Log-likelihood scores the VARIANCE too,
    and volatility regimes in equity returns are strongly persistent, so
    this test actually has discriminating power.
    """
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    return float(np.mean(-0.5 * np.log(2 * np.pi * sigma ** 2)
                         - 0.5 * ((x - mu) / sigma) ** 2))


# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — DETECTION ON ARMA RESIDUALS (two-pass, variance-targeted)
# ═══════════════════════════════════════════════════════════════════════

# FIX 1: index alignment.
# The guide used data.index[-len(residuals):], which silently misaligns if
# d > 0 in the ARIMA order. resid carries its own correct index — use it.
residuals = final_fit.resid
residuals = residuals.dropna()
residual_dates = residuals.index
resid_values = residuals.values

# FIX 2: what we actually detect on.
# Daily equity returns have near-zero autocorrelation, so an ARMA explains
# well under 1% of variance and the residuals are numerically almost
# identical to the raw returns — "residuals vs raw returns" is close to a
# distinction without a difference. What genuinely shifts regime in return
# series is the CONDITIONAL VARIANCE. Detecting on squared residuals targets
# that directly, and is honest about what is being claimed.
variance_proxy = resid_values ** 2

pelt_pos, algo_cached = detect(variance_proxy, PENALTY)
pelt_dates = residual_dates[pelt_pos]
print("Change points (variance proxy, pass 1):", list(pelt_dates))


# FIX 3: the circularity.
# The guide fits ARMA on the full sample ASSUMING no break, then hunts for
# breaks in its residuals — so the parameters are averaged across regimes.
# This loop refits ARMA within each detected segment and re-detects on the
# new residuals. It converges in 2-3 iterations and removes the objection.

def segment_residuals(series, cut_positions, order):
    """Refit ARMA separately within each segment; return stitched residuals."""
    out = np.empty(len(series))
    out[:] = np.nan
    bounds = [0] + list(cut_positions) + [len(series)]
    n_failed = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = series.iloc[a:b]
        if len(seg) < max(MIN_SIZE, 30):
            out[a:b] = seg.values - seg.values.mean()  # fallback: demean only
            continue
        try:
            r = ARIMA(seg, order=order).fit()
            out[a:b] = r.resid.values
        except Exception:
            n_failed += 1
            out[a:b] = seg.values - seg.values.mean()
    return out, n_failed


log_ret = data["log_return"].loc[residual_dates]  # align to residual window
cuts = pelt_pos

for iteration in range(3):
    seg_resid, n_failed = segment_residuals(log_ret, cuts, best_order)
    new_cuts, _ = detect(seg_resid ** 2, PENALTY)
    print(f"  pass {iteration + 2}: {len(new_cuts)} change points "
          f"({n_failed} segment fits failed)")
    if new_cuts == cuts:
        print("  converged.")
        break
    cuts = new_cuts

pelt_pos = cuts
pelt_dates = residual_dates[pelt_pos]
final_resid = seg_resid  # regime-aware residuals, used from here on
print("Final change points:", list(pelt_dates))


# ─── 6b. Offline Bayesian ────────────────────────────────────────────────
# This is the compute bottleneck: O(n^2) in time AND memory. truncate=-40
# is doing a lot of work — keep it. At ~6000+ observations consider
# subsampling to weekly, or dropping this method and relying on the
# permutation test in 6c for validation instead.

import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial
from scipy.signal import find_peaks

bayes_input = standardise(final_resid ** 2)

Q, P, Pcp = offcd.offline_changepoint_detection(
    bayes_input,
    partial(offcd.const_prior, l=(len(bayes_input) + 1)),
    offcd.gaussian_obs_log_likelihood,
    truncate=-40,
)
change_prob = np.exp(Pcp).sum(0)

# FIX 4: the arbitrary threshold.
# height=0.1 in the guide is unjustifiable in a write-up. Plot the curve,
# then set the threshold from the curve's own distribution so you can
# defend it. 95th percentile is a defensible, statable rule.
plt.figure(figsize=(12, 3))
plt.plot(residual_dates[1:len(change_prob) + 1], change_prob)
plt.title("Bayesian change-point probability — inspect before thresholding")
plt.tight_layout()
plt.show()

THRESHOLD = float(np.quantile(change_prob, 0.95))
print(f"Bayesian threshold (95th pct of probability curve): {THRESHOLD:.4f}")

# distance now matches MIN_SIZE so both methods face the same constraint
peaks, _ = find_peaks(change_prob, height=THRESHOLD, distance=MIN_SIZE)

# +1 converts "probability index i" to "first observation of new regime",
# matching the PELT convention set in cps_to_positions().
bayes_pos = [int(p) + 1 for p in peaks]
bayes_dates = residual_dates[bayes_pos]
print("Bayesian change points:", list(bayes_dates))


# ─── 6c. Agreement between methods, WITH a null baseline ─────────────────

def mean_nearest_gap(set_a, set_b):
    """Mean distance from each point in A to its nearest neighbour in B,
    measured in TRADING DAYS (positional index), not calendar days.
    The guide used (d1 - d2).days but then interpreted the result as
    trading days — off by a factor of ~1.45 across weekends and holidays."""
    if len(set_a) == 0 or len(set_b) == 0:
        return np.nan
    b = np.asarray(set_b)
    return float(np.mean([np.min(np.abs(b - a)) for a in set_a]))


observed_gap = mean_nearest_gap(pelt_pos, bayes_pos)

# THE KEY ADDITION: "the methods agree" is meaningless without knowing what
# chance agreement looks like. Draw random breakpoint sets of the same size
# on the same series and measure the gap distribution under the null.
n_obs = len(final_resid)
null_gaps = []
for _ in range(N_PERMUTATIONS):
    fake_a = np.sort(rng.choice(np.arange(MIN_SIZE, n_obs - MIN_SIZE),
                                size=len(pelt_pos), replace=False))
    fake_b = np.sort(rng.choice(np.arange(MIN_SIZE, n_obs - MIN_SIZE),
                                size=len(bayes_pos), replace=False))
    null_gaps.append(mean_nearest_gap(fake_a, fake_b))
null_gaps = np.array(null_gaps)

p_value = float(np.mean(null_gaps <= observed_gap))
print(f"\nObserved mean nearest gap : {observed_gap:.1f} trading days")
print(f"Random baseline (median)  : {np.median(null_gaps):.1f} trading days")
print(f"Permutation p-value       : {p_value:.4f}")

plt.figure(figsize=(8, 4))
plt.hist(null_gaps, bins=40, alpha=0.7, label="random baseline")
plt.axvline(observed_gap, color="red", lw=2, label="observed")
plt.xlabel("Mean nearest gap (trading days)")
plt.legend()
plt.title("Cross-method agreement vs chance")
plt.tight_layout()
plt.show()

# Per-breakpoint table, now in trading days with a guard for empty sets
overlap_rows = []
for pos, date in zip(pelt_pos, pelt_dates):
    if len(bayes_pos) == 0:
        overlap_rows.append({"date": date, "closest_bayesian": None,
                             "gap_trading_days": None})
        continue
    j = int(np.argmin(np.abs(np.array(bayes_pos) - pos)))
    overlap_rows.append({
        "date": date,
        "closest_bayesian": bayes_dates[j],
        "gap_trading_days": abs(bayes_pos[j] - pos),
    })
overlap_df = pd.DataFrame(overlap_rows)
print("\n", overlap_df)


# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — OUT-OF-SAMPLE VALIDATION (rewritten)
# ═══════════════════════════════════════════════════════════════════════
#
# Three bugs in the guide's version, plus one design flaw:
#
#   (a) MISALIGNMENT — forecast_global started AT the break but was scored
#       against actuals starting 30 days AFTER it.
#   (b) UNFAIR ADVANTAGE — the local model saw 30 days the global model
#       didn't, so any "improvement" could be pure recency rather than a
#       structural break. Fixed by giving BOTH models the same information
#       set (expanding window), isolating the break itself.
#   (c) 30 OBSERVATIONS is far too few to fit an ARMA. Raised to 200.
#   (d) DESIGN — MSE on mean forecasts is near-noise (see gaussian_loglik).
#       Scored on predictive log-likelihood instead.

LOCAL_WINDOW = 200   # was 30 in the guide — too short to fit anything
HORIZON = 60         # evaluation window after the break


def validate_break(series, break_pos, horizon=HORIZON, window=LOCAL_WINDOW):
    """
    Does a regime-specific model predict the post-break period better than
    a model that ignores the break? Both models see identical data.
    """
    pre = series.iloc[:break_pos]
    post = series.iloc[break_pos:]
    if len(pre) < window or len(post) < window + horizon:
        return None

    local_train = post.iloc[:window]                  # post-break only
    global_train = series.iloc[:break_pos + window]   # SAME end point, more history
    actual = post.iloc[window:window + horizon]

    # Global model: ignores the break, pools everything up to the same date
    mu_g, sd_g = global_train.mean(), global_train.std()
    # Local model: uses only the new regime
    mu_l, sd_l = local_train.mean(), local_train.std()

    ll_global = gaussian_loglik(actual.values, mu_g, sd_g)
    ll_local = gaussian_loglik(actual.values, mu_l, sd_l)

    return {
        "break_pos": break_pos,
        "date": series.index[break_pos],
        "loglik_global": ll_global,
        "loglik_local": ll_local,
        "loglik_gain": ll_local - ll_global,   # >0 means the break matters
        "sd_before": float(sd_g),
        "sd_after": float(sd_l),
    }


# FIX: validate the UNION of both methods' candidates.
# The guide only validated PELT dates, silently discarding every
# Bayesian-only candidate before it could be tested.
candidate_pos = sorted(set(pelt_pos) | set(bayes_pos))

validation = [r for r in (validate_break(log_ret, p) for p in candidate_pos) if r]
validation_df = pd.DataFrame(validation)
print("\nStep 7 — forecast validation:\n", validation_df)


# ═══════════════════════════════════════════════════════════════════════
# STEP 8 — COMBINED REPORT TABLE
# ═══════════════════════════════════════════════════════════════════════
#
# FIX: key the validation results by position in a dict rather than
# scanning with `next(r for r in ... if r["breakpoint"] == cp_date)`.
# Timestamp equality is fragile after any resampling or timezone handling,
# and a silent miss there just produced None in the guide's table.

val_by_pos = {r["break_pos"]: r for r in validation}

report_rows = []
bounds = [0] + candidate_pos + [len(log_ret)]
for i, pos in enumerate(candidate_pos):
    v = val_by_pos.get(pos)
    seg = log_ret.iloc[bounds[i + 1]:bounds[i + 2]]   # segment AFTER this break
    j = int(np.argmin(np.abs(np.array(bayes_pos) - pos))) if bayes_pos else None
    report_rows.append({
        "date": log_ret.index[pos],
        "found_by_pelt": pos in pelt_pos,
        "found_by_bayesian": pos in bayes_pos,
        "gap_trading_days": abs(bayes_pos[j] - pos) if j is not None else None,
        # segment statistics — a break table without these is hard to
        # interpret economically
        "segment_n": len(seg),
        "segment_mean": seg.mean(),
        "segment_sd": seg.std(),
        "annualised_vol": seg.std() * np.sqrt(252),
        "loglik_gain": v["loglik_gain"] if v else None,
        "forecast_validated": (v["loglik_gain"] > 0) if v else None,
        # so no row looks more credible than the evidence supports
        "agreement_p_value": p_value,
    })

final_report_df = pd.DataFrame(report_rows)
print("\nStep 8 — final report:\n", final_report_df)
final_report_df.to_csv("nifty50_changepoint_final_report.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# STEP 9 — PENALTY SELECTION (leakage-free, comparable, log-likelihood)
# ═══════════════════════════════════════════════════════════════════════
#
# Four problems with the guide's version:
#
#   1. LEAKAGE — it ran PELT on the FULL series including the holdout days,
#      then held out the last 20 days of each segment. The breakpoints had
#      already seen the evaluation data, which systematically favours more
#      breakpoints. Fixed with rolling-origin: at each origin, detection
#      only ever sees data strictly before it.
#
#   2. NOT COMPARABLE — min_segment_size=60 dropped short segments, so
#      different penalties were scored on different numbers of segments
#      covering different days. Fixed by holding the EVALUATION SET fixed
#      across all penalties and varying only the segmentation.
#
#   3. FLAT-FORECAST MSE — near-independent of penalty, so idxmin() picked
#      essentially at random. Scored on predictive log-likelihood instead.
#
#   4. SIGNAL MISMATCH — it tuned on data["log_return"] but detected on
#      residuals. Same signal is used throughout here.

BLOCK = 40  # days predicted at each evaluation origin

# Fixed evaluation origins: identical for every penalty, so the comparison
# is like-for-like. Spread across the back 40% of the sample.
origins = list(range(int(0.60 * n_obs), n_obs - BLOCK, BLOCK))
print(f"\nStep 9 — {len(origins)} evaluation origins, block size {BLOCK}")

# SPEED: fit KernelCPD ONCE per origin, then sweep all penalties against the
# cached Gram matrix. This is the single biggest saving in the script.
cached_algos = {}
for o in origins:
    sig = standardise(final_resid[:o] ** 2).reshape(-1, 1)
    cached_algos[o] = rpt.KernelCPD(kernel="rbf", min_size=MIN_SIZE).fit(sig)

penalty_candidates = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
results = []

for pen in penalty_candidates:
    block_scores, n_cps_seen, n_failures = [], [], 0
    for o in origins:
        try:
            bkps = cached_algos[o].predict(pen=pen)
            cps = cps_to_positions(bkps, o)
        except Exception:
            n_failures += 1
            continue
        n_cps_seen.append(len(cps))

        # Use only the CURRENT regime (data since the last change point) to
        # parameterise the forecast — that is the whole point of detecting
        # breaks. Larger penalty -> fewer breaks -> longer, staler window.
        start = cps[-1] if cps else 0
        train = log_ret.iloc[start:o]
        if len(train) < MIN_SIZE:
            n_failures += 1
            continue
        actual = log_ret.iloc[o:o + BLOCK]
        block_scores.append(gaussian_loglik(actual.values,
                                            train.mean(), train.std()))

    results.append({
        "penalty": pen,
        "mean_loglik": np.mean(block_scores) if block_scores else np.nan,
        "mean_n_changepoints": np.mean(n_cps_seen) if n_cps_seen else np.nan,
        "n_blocks_scored": len(block_scores),
        # REPORTED, not swallowed by a bare except — a penalty whose
        # segments mostly failed to fit would otherwise look artificially clean
        "n_failures": n_failures,
    })
    print(f"  penalty {pen:>3}: mean loglik = {results[-1]['mean_loglik']:.4f}, "
          f"avg {results[-1]['mean_n_changepoints']:.1f} change points, "
          f"{n_failures} failures")

penalty_df = pd.DataFrame(results)

# ─── 9a. Pick the best penalty ───────────────────────────────────────────
# Drop NaNs first — idxmin/idxmax misbehaves otherwise, which the guide's
# version did not guard against. HIGHER log-likelihood is better.
valid = penalty_df.dropna(subset=["mean_loglik"])
best_row = valid.loc[valid["mean_loglik"].idxmax()]
BEST_PENALTY = float(best_row["penalty"])
print(f"\nBest penalty by rolling-origin CV: {BEST_PENALTY}"
      f" (mean loglik {best_row['mean_loglik']:.4f},"
      f" ~{best_row['mean_n_changepoints']:.1f} change points)")

# ─── 9b. Trade-off plot ──────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(penalty_df["penalty"], penalty_df["mean_loglik"],
         color="red", marker="o")
ax1.set_xlabel("Penalty value")
ax1.set_ylabel("Mean predictive log-likelihood (higher is better)", color="red")
ax1.axvline(BEST_PENALTY, color="grey", ls="--", alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(penalty_df["penalty"], penalty_df["mean_n_changepoints"],
         color="blue", marker="s")
ax2.set_ylabel("Number of change points", color="blue")
plt.title("Penalty vs out-of-sample predictive fit and model complexity")
plt.tight_layout()
plt.show()

print("\nRe-run Steps 6-8 with PENALTY = BEST_PENALTY for final results.")
