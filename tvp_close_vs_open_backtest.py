"""
Close-to-close vs. same-day open-to-close vs. close-to-close-but-skip-
pre-gap-days, on the real holdout window, using the ALREADY-LOCKED
3-regressor model (no new fitting -- same hardcoded hyperparameters as
every other live script).

Three strategies, same underlying forecast signal (direction + Kelly
fraction from the model's close-to-close prediction), different
REALIZED return used for P&L:

  A. CLOSE_TO_CLOSE, unrestricted: position * gold's close-to-close
     return, every day. This is the strategy every backtest number in
     the README is actually reporting.

  B. CLOSE_TO_CLOSE, skip pre-gap days: same as A, but position forced
     to 0 on any day where the NEXT trading day is more than 1 calendar
     day away (a weekend and/or holiday gap -- data-driven detection,
     not a hardcoded "skip Fridays" rule, so it also catches holiday
     Mondays/Thursdays automatically).

  C. SAME_DAY_OPEN_TO_CLOSE: same signal, but P&L computed against
     TODAY's own open-to-close return instead of close-to-close. Never
     holds overnight at all, so there's no pre-gap day to skip -- every
     day is already immune to weekend/holiday gap risk by construction.

The real question: does skipping pre-gap days (B) actually fix the
problem close-to-close (A) has, or does meaningful risk still leak
through some other way? And does going same-day (C) sacrifice real edge
that close-to-close was capturing from overnight moves, or is that edge
mostly illusory once you account for how it got there?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_calibration import forecasts_and_variance

DATA_PATH = "gold_macro_data.csv"
REGRESSORS = ["real_rate_diff", "usd_logret", "gvz_logret"]
SPLIT_DATE = "2022-07-06"  # same fixed holdout boundary as every other comparison

TRAIN_STD = {"real_rate_diff": 0.050393, "usd_logret": 0.003527, "gvz_logret": 0.054245}
SIGMA_OBS = 0.00796
SIGMA_BETA = {"real_rate_diff": 0.000548, "usd_logret": 0.000481, "gvz_logret": 0.001496}

BUCKET_EDGES = [-np.inf, 0.131, 0.306, 0.507, 0.839, np.inf]
BUCKET_HIT_RATES = [0.5782, 0.6014, 0.6735, 0.7646, 0.8041]
KELLY_SCALE = 0.5


def bucket_and_kelly(z):
    abs_z = abs(z)
    for b in range(len(BUCKET_HIT_RATES)):
        if BUCKET_EDGES[b] <= abs_z < BUCKET_EDGES[b + 1]:
            p = BUCKET_HIT_RATES[b]
            return max(2 * p - 1, 0.0) * KELLY_SCALE
    p = BUCKET_HIT_RATES[-1]
    return max(2 * p - 1, 0.0) * KELLY_SCALE


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    return ((equity - running_max) / running_max).min()


def sharpe_like(daily_returns):
    if daily_returns.std() == 0:
        return np.nan
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    first_valid = df["gvz_logret"].first_valid_index()
    df = df[df.index >= first_valid]
    split = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))

    # --- pull gold's Open for the whole history (never archived before) ---
    print("Pulling gold Open prices (GC=F) -- not in the archive, needed fresh...")
    raw = yf.download("GC=F", start=df.index[0], end=df.index[-1] + pd.Timedelta(days=2),
                       progress=False, auto_adjust=True)
    gold_open = raw["Open"]
    if hasattr(gold_open, "columns"):
        gold_open = gold_open.iloc[:, 0]
    gold_close = raw["Close"]
    if hasattr(gold_close, "columns"):
        gold_close = gold_close.iloc[:, 0]

    open_to_close_logret = np.log(gold_close / gold_open)
    open_to_close_logret = open_to_close_logret.reindex(df.index)
    missing = open_to_close_logret.isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} days missing open-to-close data after "
              f"reindexing -- those days will be excluded from strategy C.")

    # --- data-driven pre-gap detection: does the NEXT trading day skip >1 calendar day? ---
    date_diffs = df.index.to_series().diff().shift(-1).dt.days
    pre_gap = date_diffs > 1
    print(f"\n{pre_gap.sum()} pre-gap days identified out of {len(df)} total "
          f"(weekends and/or holidays ahead).\n")

    # --- get forecasts from the already-locked model, no new fitting ---
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    exog_full = df[REGRESSORS] / pd.Series(TRAIN_STD)
    sm_model = TVPRegression(df["gold_logret"], exog_full)
    sm_model.exog_names = REGRESSORS
    params = np.array([SIGMA_OBS**2] + [SIGMA_BETA[r]**2 for r in REGRESSORS])

    forecasts, var = forecasts_and_variance(sm_model, params)
    z = np.where((var > 0) & ~np.isnan(forecasts), forecasts / np.sqrt(np.where(var > 0, var, 1)), np.nan)

    test_slice = slice(split, None)
    test_valid = ~np.isnan(forecasts[test_slice]) & ~np.isnan(z[test_slice])
    test_forecast = forecasts[test_slice]
    test_z = z[test_slice]
    test_pre_gap = pre_gap.values[test_slice]
    test_close_ret = np.expm1(df["gold_logret"].values[test_slice])
    test_open_close_ret = np.expm1(open_to_close_logret.values[test_slice])
    test_open_close_valid = ~np.isnan(test_open_close_ret)

    kelly_frac = np.array([bucket_and_kelly(zz) if not np.isnan(zz) else 0.0 for zz in test_z])
    long_signal = test_valid & (test_forecast > 0)  # long-only: only size up on UP calls

    # --- Strategy A: close-to-close, unrestricted ---
    pos_a = np.where(long_signal, kelly_frac, 0.0)
    ret_a = pos_a * test_close_ret

    # --- Strategy B: close-to-close, skip pre-gap days ---
    pos_b = np.where(long_signal & ~test_pre_gap, kelly_frac, 0.0)
    ret_b = pos_b * test_close_ret

    # --- Strategy C: same-day open-to-close (never holds overnight, no pre-gap risk at all) ---
    pos_c = np.where(long_signal & test_open_close_valid, kelly_frac, 0.0)
    ret_c = pos_c * np.nan_to_num(test_open_close_ret)

    results = {}
    for name, ret, pos in [("A: CLOSE_TO_CLOSE (unrestricted)", ret_a, pos_a),
                            ("B: CLOSE_TO_CLOSE (skip pre-gap)", ret_b, pos_b),
                            ("C: SAME_DAY_OPEN_TO_CLOSE", ret_c, pos_c)]:
        equity = np.cumprod(1 + ret)
        total_ret = equity[-1] - 1
        dd = max_drawdown(equity)
        sr = sharpe_like(pd.Series(ret))
        days_in_market = (pos != 0).mean()

        # how much of the losses came specifically from pre-gap days?
        pre_gap_losses = ret[test_pre_gap & (ret < 0)].sum()
        total_losses = ret[ret < 0].sum()
        pre_gap_loss_share = pre_gap_losses / total_losses if total_losses != 0 else 0.0

        results[name] = equity
        print(f"--- {name} ---")
        print(f"  Total return: {total_ret:.2%}  |  Max DD: {dd:.2%}  |  "
              f"Sharpe-like: {sr:.2f}  |  Days in market: {days_in_market:.1%}")
        print(f"  Share of total losses from pre-gap days: {pre_gap_loss_share:.1%}\n")

    print("What to look for: does B's pre-gap-loss-share drop to ~0% (confirms the")
    print("skip rule actually works)? Does B's total return/Sharpe hold up close to")
    print("A's (skipping pre-gap days didn't cost much) or drop a lot (it was")
    print("carrying real edge)? Does C's Sharpe beat or lag A/B (same-day-only")
    print("misses overnight edge, or dodges gap risk for free)?")

    fig, ax = plt.subplots(figsize=(11, 5))
    test_dates = df.index[split:]
    for name, equity in results.items():
        ax.plot(test_dates, equity, label=name)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Equity (starting capital = 1.0)")
    ax.set_title("Close-to-close vs. open-to-close vs. pre-gap-skip")
    ax.legend()
    plt.tight_layout()
    plt.savefig("close_vs_open_backtest.png", dpi=150)
    print("\nSaved close_vs_open_backtest.png")


if __name__ == "__main__":
    main()