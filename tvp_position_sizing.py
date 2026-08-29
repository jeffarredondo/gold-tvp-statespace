"""
Position sizing on top of the locked 2-regressor Bayes-shrinkage model.

Uses the same confidence buckets as tvp_calibration.py (|z| = forecast /
predicted std, bucketed using TRAIN-only edges). Each bucket's TRAIN hit
rate p becomes that bucket's Kelly fraction: full Kelly f* = 2p - 1,
we use HALF Kelly (f*/2) since p is an estimate with real sampling
uncertainty, not a known true probability -- overbetting on an
overestimated edge compounds catastrophically in a way underbetting
never does, which is why half-Kelly (or smaller) is the standard
practitioner default rather than full Kelly.

Simplification being made explicitly: this treats each day as an
even-money bet on direction (f* = 2p-1), not the fuller Kelly formula
that also accounts for the actual size of wins vs losses. Good enough
for "does confidence-weighted sizing help" as an academic question; a
production version would want the continuous-outcome Kelly variant.

Three strategies compared on the TEST (holdout) window only:
  1. Confidence-weighted half-Kelly: position size = bucket's half-Kelly
     fraction, varies day to day with how confident the model is.
  2. Flat sizing, same average exposure: constant position size across
     all days, set equal to strategy 1's average size on the test
     window. Same total risk budget, no use of confidence -- this is
     the fair baseline. If strategy 1 doesn't beat this, confidence-
     weighting isn't earning its keep, it's just "smaller bets."
  3. Buy-and-hold gold: for context/reference, not a fair comparison
     (no directional signal at all, always long).

If a bucket's train hit rate is <= 0.5, that bucket's Kelly fraction is
clipped to 0 (no edge to size against) rather than left negative --
we're not flipping the trade against the model's own forecast sign,
just declining to size that day if the model's confidence in this slice
of history didn't actually pay off.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import fit_shrinkage_model
from tvp_calibration import forecasts_and_variance

DATA_PATH = "gold_macro_data.csv"
TRAIN_FRAC = 0.8
REGRESSORS = ["real_rate_diff", "usd_logret"]
N_BUCKETS = 5
KELLY_SCALE = 0.5  # half-Kelly

DRAWS = 1500
TUNE = 1500
CHAINS = 4


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return dd.min()


def sharpe_like(daily_returns):
    if daily_returns.std() == 0:
        return np.nan
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    split = int(len(df) * TRAIN_FRAC)

    endog_full = df["gold_logret"]
    exog_raw_full = df[REGRESSORS]
    exog_std = exog_raw_full.iloc[:split].std()
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    print("Fitting Bayes shrinkage model on train window...")
    _, trace = fit_shrinkage_model(endog_train, exog_train,
                                    draws=DRAWS, tune=TUNE, chains=CHAINS)

    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    sm_model_full = TVPRegression(endog_full, exog_full)
    sm_model_full.exog_names = REGRESSORS

    forecasts, var = forecasts_and_variance(sm_model_full, bayes_params)
    gold_logret = endog_full.values
    gold_simple_ret = np.expm1(gold_logret)  # for correct compounding

    valid = ~np.isnan(forecasts) & ~np.isnan(var) & (var > 0)
    z = np.full_like(forecasts, np.nan)
    z[valid] = forecasts[valid] / np.sqrt(var[valid])
    correct = (np.sign(forecasts) == np.sign(gold_logret))

    # --- bucket edges + per-bucket Kelly fraction, from TRAIN only ---
    train_abs_z = np.abs(z[:split])
    train_abs_z = train_abs_z[~np.isnan(train_abs_z)]
    bucket_edges = np.quantile(train_abs_z, np.linspace(0, 1, N_BUCKETS + 1))
    bucket_edges[0], bucket_edges[-1] = -np.inf, np.inf

    abs_z = np.abs(z)
    bucket_idx = np.digitize(abs_z, bucket_edges[1:-1])

    bucket_kelly = np.zeros(N_BUCKETS)
    print(f"\n--- Bucket Kelly fractions (from TRAIN hit rate, half-Kelly) ---")
    for b in range(N_BUCKETS):
        train_mask = (bucket_idx == b) & valid & (np.arange(len(df)) < split)
        p = correct[train_mask].mean() if train_mask.sum() > 0 else 0.5
        f_full = max(2 * p - 1, 0.0)
        f_half = f_full * KELLY_SCALE
        bucket_kelly[b] = f_half
        print(f"  bucket {b}: train hit rate = {p:.4f}, half-Kelly fraction = {f_half:.4f}")

    # --- apply sizing on the TEST window ---
    test_slice = slice(split, None)
    test_valid = valid[test_slice]
    test_bucket = bucket_idx[test_slice]
    test_sign = np.sign(forecasts[test_slice])
    test_ret = gold_simple_ret[test_slice]

    size_conf = np.where(test_valid, bucket_kelly[np.clip(test_bucket, 0, N_BUCKETS - 1)], 0.0)
    position_conf = size_conf * test_sign

    flat_size = size_conf[test_valid].mean()  # same average exposure
    position_flat = np.where(test_valid, flat_size * test_sign, 0.0)

    position_bh = np.ones_like(test_ret)  # buy-and-hold

    strategies = {
        "Confidence-weighted half-Kelly": position_conf,
        f"Flat sizing (avg={flat_size:.3f})": position_flat,
        "Buy-and-hold": position_bh,
    }

    print(f"\n--- Backtest results on holdout window ({test_valid.sum()} valid obs) ---")
    print(f"{'Strategy':<35} {'Total ret':>10} {'Max DD':>10} {'Sharpe-like':>12}")

    equity_curves = {}
    for name, position in strategies.items():
        daily_strategy_ret = position * test_ret
        equity = np.cumprod(1 + daily_strategy_ret)
        equity_curves[name] = equity
        total_ret = equity[-1] - 1
        dd = max_drawdown(equity)
        sr = sharpe_like(pd.Series(daily_strategy_ret))
        print(f"{name:<35} {total_ret:>10.2%} {dd:>10.2%} {sr:>12.2f}")

    print("\nNote: 'Sharpe-like' is mean/std * sqrt(252), no risk-free rate")
    print("subtracted -- a rough annualized reward-to-volatility ratio, not a")
    print("formal Sharpe ratio. Max DD is the worst peak-to-trough equity drop.")
    print("\nThe meaningful comparison is row 1 vs row 2 (same average exposure --")
    print("isolates whether confidence-weighting helps). Row 3 (buy-and-hold) is")
    print("context only, not a fair comparison (no signal, no risk management).")

    # --- plot equity curves ---
    test_dates = df.index[split:]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, equity in equity_curves.items():
        ax.plot(test_dates, equity, label=name)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Equity (starting capital = 1.0)")
    ax.set_title("Holdout equity curves: sizing strategy comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("position_sizing_backtest.png", dpi=150)
    print("\nSaved position_sizing_backtest.png")


if __name__ == "__main__":
    main()