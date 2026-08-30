"""
Resimulates position sizing under 4 variants, using the SAME already-
fitted hyperparameters and calibration buckets from tonight's runs (no
new MCMC needed -- one fast Kalman filter pass, then just resimulate the
sizing rule different ways on the same forecasts):

  1. Half-Kelly,    long/short  (the original tvp_position_sizing.py result)
  2. Quarter-Kelly, long/short
  3. Half-Kelly,    long-only   (matches a standard cash-account Roth IRA
                                  -- no shorting allowed, sit out "down"
                                  days instead of betting against them)
  4. Quarter-Kelly, long-only

Long-only means: on days the model forecasts DOWN, position = 0 (sit
out) rather than a short position. This directly matches the real
constraint of a standard brokerage Roth IRA, which can't short.

All four use the exact same forecasts and confidence buckets -- only the
Kelly scale and the long-only constraint change -- so differences
between rows are cleanly attributable to those two design choices, not
to noise from re-fitting anything.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_calibration import forecasts_and_variance

DATA_PATH = "gold_macro_data.csv"
REGRESSORS = ["real_rate_diff", "usd_logret"]
TRAIN_FRAC = 0.8

# --- real fitted numbers from tvp_holdout_backtest.py ---
TRAIN_STD = {"real_rate_diff": 0.050198, "usd_logret": 0.003392}
SIGMA_OBS = 0.009572
SIGMA_BETA = {"real_rate_diff": 0.000197, "usd_logret": 0.000622}

# --- real calibration bucket edges + train hit rates (tvp_calibration.py) ---
BUCKET_EDGES = [-np.inf, 0.099, 0.238, 0.402, 0.660, np.inf]
BUCKET_HIT_RATES = [0.5870, 0.5819, 0.6678, 0.7305, 0.8179]


def bucket_for(z):
    abs_z = abs(z)
    for b in range(len(BUCKET_HIT_RATES)):
        if BUCKET_EDGES[b] <= abs_z < BUCKET_EDGES[b + 1]:
            return b
    return len(BUCKET_HIT_RATES) - 1


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    return ((equity - running_max) / running_max).min()


def sharpe_like(daily_returns):
    if daily_returns.std() == 0:
        return np.nan
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    split = int(len(df) * TRAIN_FRAC)

    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    exog_full = df[REGRESSORS] / pd.Series(TRAIN_STD)

    sm_model = TVPRegression(df["gold_logret"], exog_full)
    sm_model.exog_names = REGRESSORS
    params = np.array([SIGMA_OBS**2, SIGMA_BETA["real_rate_diff"]**2,
                        SIGMA_BETA["usd_logret"]**2])

    forecasts, var = forecasts_and_variance(sm_model, params)
    gold_simple_ret = np.expm1(df["gold_logret"].values)

    valid = ~np.isnan(forecasts) & ~np.isnan(var) & (var > 0)
    z = np.where(valid, forecasts / np.sqrt(np.where(var > 0, var, 1)), np.nan)
    bucket = np.array([bucket_for(zz) if not np.isnan(zz) else -1 for zz in z])
    hit_rate = np.array([BUCKET_HIT_RATES[b] if b >= 0 else 0.5 for b in bucket])

    test_slice = slice(split, None)
    test_valid = valid[test_slice]
    test_sign = np.sign(forecasts[test_slice])
    test_ret = gold_simple_ret[test_slice]
    test_hit_rate = hit_rate[test_slice]

    variants = {
        "Half-Kelly, long/short":    (0.5,  False),
        "Quarter-Kelly, long/short": (0.25, False),
        "Half-Kelly, long-only":     (0.5,  True),
        "Quarter-Kelly, long-only":  (0.25, True),
    }

    print(f"--- Sizing variant comparison on holdout ({test_valid.sum()} valid obs) ---")
    print(f"{'Variant':<30} {'Total ret':>10} {'Max DD':>10} {'Sharpe-like':>12} {'Days in market':>15}")

    equity_curves = {}
    for name, (kelly_scale, long_only) in variants.items():
        f_full = np.maximum(2 * test_hit_rate - 1, 0.0)
        f = f_full * kelly_scale

        if long_only:
            position = np.where(test_valid & (test_sign > 0), f, 0.0)
        else:
            position = np.where(test_valid, f * test_sign, 0.0)

        daily_ret = position * test_ret
        equity = np.cumprod(1 + daily_ret)
        equity_curves[name] = equity

        total_ret = equity[-1] - 1
        dd = max_drawdown(equity)
        sr = sharpe_like(pd.Series(daily_ret))
        days_in_market = (position != 0).mean()

        print(f"{name:<30} {total_ret:>10.2%} {dd:>10.2%} {sr:>12.2f} {days_in_market:>14.1%}")

    print("\nRows 1 vs 2 isolate the Kelly-fraction effect (half vs quarter, both long/short).")
    print("Rows 1 vs 3 isolate the long-only effect (both half-Kelly).")
    print("Row 4 is the realistic Roth IRA version: quarter-Kelly AND no shorting.")

    test_dates = df.index[split:]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, equity in equity_curves.items():
        ax.plot(test_dates, equity, label=name)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Equity (starting capital = 1.0)")
    ax.set_title("Sizing variant comparison: Kelly fraction x long-only constraint")
    ax.legend()
    plt.tight_layout()
    plt.savefig("sizing_variants.png", dpi=150)
    print("\nSaved sizing_variants.png")


if __name__ == "__main__":
    main()