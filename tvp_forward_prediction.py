"""
A genuine forward prediction using the LOCKED 3-regressor model (real
rate, USD, and GVZ -- gold's own options-implied volatility index),
built from hyperparameters already fit tonight -- no new MCMC needed,
just one fast Kalman filter pass over the latest data.

GVZ added 2026-09-01 after clearing the same bar every other candidate
(VIX, 2Y Treasury diff, 3mo-10Y curve slope) was held to: a real,
non-noise directional accuracy gain (McNemar p=0.0009) AND a nonzero
sigma_beta role (in fact the largest of the three, ~3x either of the
others) -- confirmed stable under a stricter test that restricts BOTH
models to GVZ's actual 2008-06-04 inception, ruling out the missing-data
confound as the explanation.

Hardcoded below are the real posterior means from tvp_holdout_backtest.py
and the real calibration bucket edges/hit rates from tvp_calibration.py.
If you rerun those scripts later and get different numbers (more data,
different train window, etc.), update the constants at the top.

IMPORTANT HONEST LIMITATION, carried over from the very first message of
this whole project: this model's inputs are SAME-DAY changes. To
forecast forward at all, you need a view on what those will do next --
there's no way around this. Assuming "no change" (the default, honest
baseline) makes the forecast collapse to exactly zero, because 0 * beta
= 0 regardless of beta. So every forward prediction here is explicitly
CONDITIONAL on a stated scenario, not an unconditional oracle call. Two
scenarios are run:

  1. FLAT baseline: none of the three regressors move. Included
     specifically to make the "no view -> no signal" limitation visible,
     not to hide it.
  2. TREND PERSISTS: the last 20 trading days' average daily change in
     each regressor continues. A real, if modest and naive, forecasting
     assumption -- simple persistence, not a market-implied forward path.

Output is a NEXT TRADING DAY call (the model's actual native horizon),
plus a rough "if this holds across ~21 trading days" monthly
extrapolation -- explicitly labeled as a much cruder extension, since
compounding a single day's edge over a month assumes both the beta and
the scenario stay constant that whole time, which is a real stretch.
"""

import numpy as np
import pandas as pd

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
REGRESSORS = ["real_rate_diff", "usd_logret", "gvz_logret"]
HYPOTHETICAL_DOLLARS = 1000
TRADING_DAYS_PER_MONTH = 21

# --- real fitted numbers from tonight's completed runs (tvp_holdout_backtest.py) ---
TRAIN_STD = {"real_rate_diff": 0.050393, "usd_logret": 0.003527, "gvz_logret": 0.054245}
SIGMA_OBS = 0.00796
SIGMA_BETA = {"real_rate_diff": 0.000548, "usd_logret": 0.000481, "gvz_logret": 0.001496}

# --- real calibration bucket edges + train hit rates (tvp_calibration.py) ---
BUCKET_EDGES = [-np.inf, 0.131, 0.306, 0.507, 0.839, np.inf]
BUCKET_HIT_RATES = [0.5782, 0.6014, 0.6735, 0.7646, 0.8041]
KELLY_SCALE = 0.5


def bucket_and_kelly(z):
    abs_z = abs(z)
    for b in range(len(BUCKET_HIT_RATES)):
        if BUCKET_EDGES[b] <= abs_z < BUCKET_EDGES[b + 1]:
            p = BUCKET_HIT_RATES[b]
            f_half = max(2 * p - 1, 0.0) * KELLY_SCALE
            return b, p, f_half
    return len(BUCKET_HIT_RATES) - 1, BUCKET_HIT_RATES[-1], \
        max(2 * BUCKET_HIT_RATES[-1] - 1, 0.0) * KELLY_SCALE


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    exog_full = df[REGRESSORS] / pd.Series(TRAIN_STD)

    sm_model = TVPRegression(df["gold_logret"], exog_full)
    sm_model.exog_names = REGRESSORS

    params = np.array([SIGMA_OBS**2] + [SIGMA_BETA[r]**2 for r in REGRESSORS])
    res = sm_model.filter(params)

    # predicted state going INTO the next (unobserved) day, in standardized units
    beta_std = res.predicted_state[:, -1]
    beta_cov_std = res.predicted_state_cov[:, :, -1]

    # convert beta back to original units for readability
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    beta_orig = beta_std / std_vec

    as_of_date = df.index[-1].date()
    last_gold_price = df["gold"].iloc[-1]
    print(f"As of {as_of_date} (last date in gold_macro_data.csv -- rerun")
    print("fetch_gold_data.py first if you want this current-day):")
    print(f"  Last gold price: ${last_gold_price:,.2f}")
    for r, b in zip(REGRESSORS, beta_orig):
        print(f"  Current beta ({r}): {b:.4f}")

    # --- scenario 1: flat baseline ---
    recent = df[REGRESSORS].iloc[-20:]
    scenarios = {
        "FLAT (no change)": {r: 0.0 for r in REGRESSORS},
        "TREND PERSISTS (last 20-day avg)": {r: recent[r].mean() for r in REGRESSORS},
    }

    print(f"\n{'='*70}\nNext trading day prediction, ${HYPOTHETICAL_DOLLARS} hypothetical stake\n{'='*70}")

    for name, scenario in scenarios.items():
        x = np.array([scenario[r] for r in REGRESSORS])
        forecast_logret = beta_orig @ x

        x_std = x / std_vec
        pred_var = x_std @ beta_cov_std @ x_std + params[0]
        z = forecast_logret / np.sqrt(pred_var) if pred_var > 0 else 0.0

        bucket, p, f_half = bucket_and_kelly(z)
        direction = "UP" if forecast_logret > 0 else ("DOWN" if forecast_logret < 0 else "FLAT")

        if forecast_logret == 0:
            f_half = 0.0  # no directional edge -> no stake, not a default "long"
        stake = HYPOTHETICAL_DOLLARS * f_half

        print(f"\n--- Scenario: {name} ---")
        assumed_str = ", ".join(f"{r}: {scenario[r]:+.6f}" for r in REGRESSORS)
        print(f"  Assumed {assumed_str}")
        print(f"  Forecast gold log-return: {forecast_logret:+.5f}  -> direction: {direction}")
        print(f"  |z| = {abs(z):.3f}  -> confidence bucket {bucket} "
              f"(historical train hit rate {p:.1%})")
        print(f"  Half-Kelly position size: {f_half:.1%} of capital")
        stake_direction = "long" if forecast_logret > 0 else ("short" if forecast_logret < 0 else "n/a")
        print(f"  On ${HYPOTHETICAL_DOLLARS}: hypothetical stake = ${stake:.2f} "
              f"{stake_direction}")

        # rough monthly extrapolation, clearly caveated
        monthly_logret = forecast_logret * TRADING_DAYS_PER_MONTH
        implied_price = last_gold_price * np.exp(monthly_logret)
        print(f"  [Rough ~{TRADING_DAYS_PER_MONTH}-trading-day extrapolation, NOT a real")
        print(f"   forecast -- assumes beta AND the scenario both hold constant all month]")
        print(f"   implied cumulative move: {monthly_logret:+.4f} log-return "
              f"-> ${implied_price:,.2f} if realized exactly")

    print("\nThe FLAT scenario above should show ~0 forecast and bucket 0 (least confident)")
    print("-- that's the model correctly telling you it has nothing to say without a view.")
    print("The TREND PERSISTS scenario is the more real (if still simple) prediction.")


if __name__ == "__main__":
    main()