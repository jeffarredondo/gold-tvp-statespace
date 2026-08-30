"""
DTWEXBGS (the Fed's broad USD index) has a structural ~1-week publication
lag on FRED -- it's an H.10 statistical release, genuinely slower to post
than DFII10 or gold, not a pipeline bug. That means gold_macro_data.csv
is always going to be missing the most recent several days of real
usd_logret, which blocks the model's STATE from advancing all the way to
"yesterday" -- not just the display, the actual Kalman filter beta.

This script bridges that gap for a LIVE forecast only:
  1. Loads the archived (lagged but 100% real) dataset for history.
  2. Pulls DFII10 directly from FRED for the missing days (it's usually
     only 1-2 days behind, often already caught up).
  3. Pulls a FAST-updating dollar proxy (ICE Dollar Index futures,
     DX-Y.NYB) via yfinance for the same gap window, standing in for the
     not-yet-published DTWEXBGS values.
  4. Runs one more Kalman filter step through the gap days (frozen
     hyperparameters, no refitting) to get the model's CURRENT beta
     state, not the one-week-stale state the archive alone would give.
  5. Forecasts the actual next trading day off that current state.

IMPORTANT: none of the proxy data is written back into gold_macro_data.csv.
It's a stand-in for a live forecast only -- DX-Y.NYB is correlated with
DTWEXBGS but is NOT the same series the model was fit on (different
constituent weights, different base), so treat gap-day betas as a
reasonable approximation, not exact. Once FRED actually publishes the
real DTWEXBGS values for these dates, rerun fetch_gold_data.py normally
and this bridge becomes unnecessary for that window.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "<Key>")
REGRESSORS = ["real_rate_diff", "usd_logret"]
HYPOTHETICAL_DOLLARS = 1000

TRAIN_STD = {"real_rate_diff": 0.050198, "usd_logret": 0.003392}
SIGMA_OBS = 0.009572
SIGMA_BETA = {"real_rate_diff": 0.000197, "usd_logret": 0.000622}

BUCKET_EDGES = [-np.inf, 0.099, 0.238, 0.402, 0.660, np.inf]
BUCKET_HIT_RATES = [0.5870, 0.5819, 0.6678, 0.7305, 0.8179]
KELLY_SCALE = 0.5


def bucket_and_kelly(z):
    abs_z = abs(z)
    for b in range(len(BUCKET_HIT_RATES)):
        if BUCKET_EDGES[b] <= abs_z < BUCKET_EDGES[b + 1]:
            p = BUCKET_HIT_RATES[b]
            return b, p, max(2 * p - 1, 0.0) * KELLY_SCALE
    p = BUCKET_HIT_RATES[-1]
    return len(BUCKET_HIT_RATES) - 1, p, max(2 * p - 1, 0.0) * KELLY_SCALE


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    last_archived_date = df.index[-1]
    print(f"Archived data (100% real) runs through {last_archived_date.date()}")

    # --- pull the gap: real_rate direct from FRED, usd proxy from yfinance ---
    if FRED_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("Set FRED_API_KEY as an env var or paste it into the script.")
    fred = Fred(api_key=FRED_API_KEY)
    gap_start = last_archived_date - pd.Timedelta(days=5)  # buffer for diffing + holidays

    real_rate_gap = fred.get_series("DFII10", observation_start=gap_start)
    usd_proxy_gap = yf.download("DX-Y.NYB", start=gap_start, progress=False,
                                 auto_adjust=True)["Close"]
    if hasattr(usd_proxy_gap, "columns"):
        usd_proxy_gap = usd_proxy_gap.iloc[:, 0]

    gap_df = pd.DataFrame({"real_rate": real_rate_gap}).join(
        pd.DataFrame({"usd_proxy": usd_proxy_gap}), how="outer"
    ).ffill().dropna()

    # diff BEFORE filtering to post-archive dates -- otherwise the first
    # retained day has nothing left to diff against, pandas correctly
    # returns NaN, and a blanket fillna(0.0) would silently turn a real,
    # unknown change into a fake "nothing happened" -- exactly the kind
    # of silent-zero bug this project has spent all day hunting for
    gap_df["real_rate_diff"] = gap_df["real_rate"].diff()
    gap_df["usd_logret"] = np.log(gap_df["usd_proxy"] / gap_df["usd_proxy"].shift(1))

    gap_df = gap_df[gap_df.index > last_archived_date]

    if gap_df.empty:
        print("No new data beyond the archive -- FRED/proxy haven't advanced. "
              "Nothing to bridge; run tvp_forward_prediction.py as-is.")
        return

    gap_df = gap_df.dropna(subset=["real_rate_diff", "usd_logret"])
    if gap_df.empty:
        print("Gap data didn't have enough history before it to compute real "
              "diffs -- try increasing the lookback window (gap_start).")
        return

    print(f"\nBridging {len(gap_df)} day(s) beyond the archive using a live "
          f"dollar proxy (DX-Y.NYB, NOT the real DTWEXBGS):")
    print(gap_df[["real_rate_diff", "usd_logret"]])

    # --- advance the Kalman filter through history + the bridge days ---
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    full_exog = pd.concat([df[REGRESSORS], gap_df[REGRESSORS]]) / pd.Series(TRAIN_STD)
    full_endog = pd.concat([df["gold_logret"], pd.Series(np.nan, index=gap_df.index)])

    sm_model = TVPRegression(full_endog.fillna(0), full_exog)  # dummy endog for bridge rows
    sm_model.exog_names = REGRESSORS
    params = np.array([SIGMA_OBS**2, SIGMA_BETA["real_rate_diff"]**2,
                        SIGMA_BETA["usd_logret"]**2])
    res = sm_model.filter(params)

    beta_std = res.predicted_state[:, -1]
    beta_cov_std = res.predicted_state_cov[:, :, -1]
    beta_orig = beta_std / std_vec

    print(f"\nCurrent beta (bridged through {gap_df.index[-1].date()}):")
    print(f"  real_rate_diff: {beta_orig[0]:.4f}")
    print(f"  usd_logret:     {beta_orig[1]:.4f}")

    # --- forecast the next trading day: trend-persists using bridged recent data ---
    recent = pd.concat([df[REGRESSORS].iloc[-15:], gap_df[REGRESSORS]])
    scenario = {r: recent[r].mean() for r in REGRESSORS}
    x = np.array([scenario[r] for r in REGRESSORS])
    forecast_logret = beta_orig @ x

    x_std = x / std_vec
    pred_var = x_std @ beta_cov_std @ x_std + params[0]
    z = forecast_logret / np.sqrt(pred_var) if pred_var > 0 else 0.0
    bucket, p, f_half = bucket_and_kelly(z)
    direction = "UP" if forecast_logret > 0 else ("DOWN" if forecast_logret < 0 else "FLAT")
    stake = HYPOTHETICAL_DOLLARS * f_half if forecast_logret != 0 else 0.0

    print(f"\n{'='*60}\nBridged forecast for the next trading day\n{'='*60}")
    print(f"Scenario (trend persists, using bridged recent data):")
    print(f"  real_rate_diff: {scenario['real_rate_diff']:+.5f}, "
          f"usd_logret: {scenario['usd_logret']:+.6f}")
    print(f"Forecast gold log-return: {forecast_logret:+.5f} -> direction: {direction}")
    print(f"|z| = {abs(z):.3f} -> confidence bucket {bucket} (train hit rate {p:.1%})")
    print(f"Half-Kelly stake on ${HYPOTHETICAL_DOLLARS}: ${stake:.2f} "
          f"{'long' if forecast_logret > 0 else ('short' if forecast_logret < 0 else 'n/a')}")
    print("\nCaveat: the dollar figures for the bridge days are a PROXY (DX-Y.NYB),")
    print("not the real DTWEXBGS the model was fit on. Treat this as a reasonable")
    print("approximation, not an exact match to what a fully-updated FRED pull")
    print("would eventually show once DTWEXBGS catches up.")


if __name__ == "__main__":
    main()