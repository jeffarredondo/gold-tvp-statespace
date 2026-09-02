"""
DTWEXBGS (the Fed's broad USD index) has a structural ~1-week publication
lag on FRED -- it's an H.10 statistical release, genuinely slower to post
than DFII10 or gold, not a pipeline bug. That means gold_macro_data.csv
is always going to be missing the most recent several days of real
usd_logret, which blocks the model's STATE from advancing all the way to
"yesterday" -- not just the display, the actual Kalman filter beta.

Since fetch_gold_data.py trims the ENTIRE archived dataframe to whatever
date the slowest CORE series (usually usd_index) has reached, gvz_logret
also gets cut back to that same date even though real, fresher GVZ data
already exists on Yahoo Finance -- GVZ itself has no publication lag, it
just gets caught in the trim alongside everything else.

This script bridges the gap for a LIVE forecast only:
  1. Loads the archived (lagged but 100% real) dataset for history.
  2. Pulls DFII10 directly from FRED for the missing days (usually only
     1-2 days behind, often already caught up).
  3. Pulls a FAST-updating dollar proxy (ICE Dollar Index futures,
     DX-Y.NYB) via yfinance, standing in for the not-yet-published
     DTWEXBGS values.
  4. Pulls fresh ^GVZ directly via yfinance for the same gap window --
     this is the REAL series, not a proxy, just fetched past where the
     archive's trim cut it off.
  5. Runs one more Kalman filter step through the gap days (frozen
     hyperparameters, no refitting) to get the model's CURRENT beta
     state, not the stale one the archive alone would give.
  6. Forecasts the actual next trading day off that current state.

IMPORTANT: none of this gap data is written back into gold_macro_data.csv.
It's a stand-in for a live forecast only. DX-Y.NYB is correlated with
DTWEXBGS but NOT the same series the model was fit on (different
constituent weights, different base) -- treat real_rate/usd_logret
gap-day betas as a reasonable approximation, not exact. GVZ's own gap
values ARE the real series (no proxy substitution needed there), just
fetched fresh rather than pulled from the trimmed archive.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "KEYHERE")
REGRESSORS = ["real_rate_diff", "usd_logret", "gvz_logret"]
HYPOTHETICAL_DOLLARS = 1000

# --- real fitted numbers from tvp_holdout_backtest.py (3-regressor, GVZ-restricted) ---
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
            return b, p, max(2 * p - 1, 0.0) * KELLY_SCALE
    p = BUCKET_HIT_RATES[-1]
    return len(BUCKET_HIT_RATES) - 1, p, max(2 * p - 1, 0.0) * KELLY_SCALE


def _get_close(ticker, start_date):
    raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)["Close"]
    if hasattr(raw, "columns"):
        raw = raw.iloc[:, 0]
    return raw


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    last_archived_date = df.index[-1]
    print(f"Archived data (100% real) runs through {last_archived_date.date()}")

    if FRED_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("Set FRED_API_KEY as an env var or paste it into the script.")
    fred = Fred(api_key=FRED_API_KEY)
    gap_start = last_archived_date - pd.Timedelta(days=5)  # buffer for diffing + holidays

    real_rate_gap = fred.get_series("DFII10", observation_start=gap_start)
    last_real_rate_date = real_rate_gap.index.max()  # capture BEFORE any ffill hides this

    usd_proxy_gap = _get_close("DX-Y.NYB", gap_start)
    # nominal 10Y yield (CBOE ^TNX, quoted as yield*10) as a proxy for
    # DFII10's day-of change when the real series hasn't posted yet.
    # Justified because nominal_yield = real_yield + breakeven inflation,
    # and breakeven inflation moves much less day-to-day than the nominal
    # yield does -- so a day's nominal move is a reasonable stand-in for
    # that same day's real-yield move, NOT an exact match.
    nominal_10y_gap = _get_close("^TNX", gap_start) / 10.0
    # ^GVZ itself, fetched fresh -- this IS the real regressor, not a
    # proxy, it's just past where the archive's trim cut it off
    gvz_gap = _get_close("^GVZ", gap_start)

    gap_df = pd.DataFrame({"real_rate": real_rate_gap}).join(
        pd.DataFrame({"usd_proxy": usd_proxy_gap}), how="outer"
    ).join(pd.DataFrame({"nominal_10y": nominal_10y_gap}), how="outer"
    ).join(pd.DataFrame({"gvz": gvz_gap}), how="outer")
    gap_df["usd_proxy"] = gap_df["usd_proxy"].ffill()
    gap_df["nominal_10y"] = gap_df["nominal_10y"].ffill()
    gap_df["gvz"] = gap_df["gvz"].ffill()
    # NOTE: deliberately NOT ffilling 'real_rate' -- we want its trailing
    # NaNs to survive so we know exactly which days are genuinely missing
    # and substitute the nominal proxy for THOSE days specifically.

    # diff BEFORE filtering to post-archive dates -- otherwise the first
    # retained day has nothing left to diff against, pandas correctly
    # returns NaN, and a blanket fillna(0.0) would silently turn a real,
    # unknown change into a fake "nothing happened"
    gap_df["real_rate_diff"] = gap_df["real_rate"].diff()
    gap_df["usd_logret"] = np.log(gap_df["usd_proxy"] / gap_df["usd_proxy"].shift(1))
    gap_df["nominal_10y_diff"] = gap_df["nominal_10y"].diff()
    gap_df["gvz_logret"] = np.log(gap_df["gvz"] / gap_df["gvz"].shift(1))

    gap_df = gap_df[gap_df.index > last_archived_date]

    if gap_df.empty:
        print("No new data beyond the archive -- FRED/proxy haven't advanced. "
              "Nothing to bridge; run tvp_forward_prediction.py as-is.")
        return

    # for days beyond real_rate's last published date, substitute the
    # nominal-yield proxy's diff instead of leaving a fake/missing value
    stale_days = gap_df.index[gap_df.index > last_real_rate_date]
    if len(stale_days) > 0:
        print(f"\n*** real_rate (DFII10) hasn't published beyond "
              f"{last_real_rate_date.date()} yet. ***")
        print(f"*** Substituting the nominal 10Y yield's day-of change "
              f"(^TNX) for {[d.date() for d in stale_days]}. ***")
        print("*** This assumes breakeven inflation held roughly steady that "
              "day -- a reasonable approximation, not the real TIPS move. ***\n")
        gap_df.loc[stale_days, "real_rate_diff"] = gap_df.loc[stale_days, "nominal_10y_diff"]

    gap_df = gap_df.dropna(subset=REGRESSORS)
    if gap_df.empty:
        print("Gap data didn't have enough history before it to compute real "
              "diffs -- try increasing the lookback window (gap_start).")
        return

    print(f"\nBridging {len(gap_df)} day(s) beyond the archive:")
    print(gap_df[REGRESSORS])

    # --- advance the Kalman filter through history + the bridge days ---
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    full_exog = pd.concat([df[REGRESSORS], gap_df[REGRESSORS]]) / pd.Series(TRAIN_STD)
    full_endog = pd.concat([df["gold_logret"], pd.Series(np.nan, index=gap_df.index)])

    sm_model = TVPRegression(full_endog.fillna(0), full_exog)  # dummy endog for bridge rows
    sm_model.exog_names = REGRESSORS
    params = np.array([SIGMA_OBS**2] + [SIGMA_BETA[r]**2 for r in REGRESSORS])
    res = sm_model.filter(params)

    beta_std = res.predicted_state[:, -1]
    beta_cov_std = res.predicted_state_cov[:, :, -1]
    beta_orig = beta_std / std_vec

    print(f"\nCurrent beta (bridged through {gap_df.index[-1].date()}):")
    for r, b in zip(REGRESSORS, beta_orig):
        print(f"  {r}: {b:.4f}")

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
    print("Scenario (trend persists, using bridged recent data):")
    assumed_str = ", ".join(f"{r}: {scenario[r]:+.6f}" for r in REGRESSORS)
    print(f"  {assumed_str}")
    print(f"Forecast gold log-return: {forecast_logret:+.5f} -> direction: {direction}")
    print(f"|z| = {abs(z):.3f} -> confidence bucket {bucket} (train hit rate {p:.1%})")
    print(f"Half-Kelly stake on ${HYPOTHETICAL_DOLLARS}: ${stake:.2f} "
          f"{'long' if forecast_logret > 0 else ('short' if forecast_logret < 0 else 'n/a')}")
    print("\nCaveat: real_rate/usd_logret gap-day figures may include a proxy")
    print("(^TNX for real_rate, DX-Y.NYB for the dollar) if the real FRED series")
    print("hasn't posted yet -- see the warnings above. gvz_logret is the real")
    print("series throughout, just fetched fresh rather than from the trimmed archive.")


if __name__ == "__main__":
    main()