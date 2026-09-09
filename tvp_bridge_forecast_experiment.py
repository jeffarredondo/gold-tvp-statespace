"""
Compares two ways of filling in "what will today's inputs be" when
forecasting TODAY's close, using the exact same beta both times.

Both modes share ONE Kalman filter state, computed using ONLY data
through YESTERDAY's completed close (bridging the archive's FRED lag
with the usual ^TNX/DX-Y.NYB proxies, same as tvp_bridge_forecast.py).
Neither mode ever adds an extra Kalman step for "today" -- doing that
would silently shift the forecast target from TODAY to TOMORROW, which
was a real bug in an earlier version of this script: adding today as an
extra completed row didn't just add information, it changed which day
was being forecast, so the two modes were never actually predicting the
same target day.

The only thing that differs between the two modes is the SCENARIO used
to forecast today, given that shared beta:

  EOD_TREND (last full close, trend-persists): today's
      real_rate_diff/usd_logret/gvz_logret assumed equal to their
      trailing 15-day average. No new information about today at all --
      a pure extrapolation from the last completed close.

  SINCE_LAST_CLOSE (overnight/premarket move): today's
      real_rate_diff/usd_logret computed from the ACTUAL move already
      visible in ^TNX and DX-Y.NYB between yesterday's close and right
      now, used directly instead of extrapolated. gvz_logret still uses
      the trailing average in both modes -- GVZ is computed from GLD
      options, which aren't open pre-market, so there is no real
      overnight GVZ reading to use.

This tests one specific, real question: does the dollar/rate market's
overnight move carry genuine information about today's close beyond
what a simple trend-persists guess already captures? It does NOT test
whether gold's own past return predicts gold's own future return (gold
was never an input to this model), and it does NOT need a new archived
regressor -- it's purely about which SCENARIO INPUT is more accurate
for a day that hasn't closed yet.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "KEY_NOT_SET")
REGRESSORS = ["real_rate_diff", "usd_logret", "gvz_logret"]
HYPOTHETICAL_DOLLARS = 1000

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


def bridge_to_yesterday(df, last_archived_date, fred):
    """
    Bridges the archive up through the LAST GENUINELY CLOSED trading day,
    determined by real Eastern time against a market-close cutoff -- not
    just literal calendar "today" regardless of whether that session has
    actually settled. Past ~5pm ET, today itself counts as closed and
    gets included; before that, today is still open and gets excluded.
    This matters: at 6pm ET the same calendar day, "last close" should
    already mean TODAY, not yesterday -- treating it as still-yesterday
    made an earlier version of this function one day more stale than it
    needed to be, and wrongly implied EOD_TREND would shift overnight
    when nothing new had actually closed in between.

    Returns (gap_df through the last closed day, dict of the NEXT
    (still-open) day's overnight-observed move or None if none yet).
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    # Both DX-Y.NYB and ^TNX settle their DAILY bar around 2:59-3:00pm ET
    # specifically (confirmed via ICE's own contract documentation for
    # DX-Y.NYB, and consistently shown across multiple quote sources for
    # ^TNX) -- electronic trading continues into the evening for both,
    # but that later activity feeds the NEXT day's bar, not a correction
    # to today's already-settled one.
    market_closed_today = now_et.hour >= 16
    today = pd.Timestamp(now_et.date())

    gap_start = last_archived_date - pd.Timedelta(days=6)

    real_rate_gap = fred.get_series("DFII10", observation_start=gap_start)
    last_real_rate_date = real_rate_gap.index.max()

    usd_proxy_gap = _get_close("DX-Y.NYB", gap_start)
    nominal_10y_gap = _get_close("^TNX", gap_start) / 10.0
    last_usd_proxy_ts = usd_proxy_gap.index.max()  # capture BEFORE ffill hides this
    last_nominal_10y_ts = nominal_10y_gap.index.max()
    gvz_gap = _get_close("^GVZ", gap_start)
    # real gold closes for the bridge window -- GC=F trades continuously,
    # so this is genuine data, not a proxy, just trimmed out of the
    # archive by the dollar's lag the same way GVZ was. Using this
    # instead of a fake 0 return matters: filling with 0 doesn't mean
    # "unknown" to a Kalman filter, it means "gold definitely didn't
    # move that day," and the filter's UPDATE step would genuinely
    # correct beta based on that fiction -- an actively wrong input, not
    # just an unused one.
    gold_gap = _get_close("GC=F", gap_start)

    merged = pd.DataFrame({"real_rate": real_rate_gap}).join(
        pd.DataFrame({"usd_proxy": usd_proxy_gap}), how="outer"
    ).join(pd.DataFrame({"nominal_10y": nominal_10y_gap}), how="outer"
    ).join(pd.DataFrame({"gvz": gvz_gap}), how="outer"
    ).join(pd.DataFrame({"gold": gold_gap}), how="outer")
    merged["usd_proxy"] = merged["usd_proxy"].ffill()
    merged["nominal_10y"] = merged["nominal_10y"].ffill()
    merged["gvz"] = merged["gvz"].ffill()
    merged["gold"] = merged["gold"].ffill()

    # NOTE: plain .diff()/.shift() here would be positionally wrong --
    # the outer join creates rows for dates where SOME series traded but
    # others didn't (e.g. currency futures trade on a US bond-market
    # holiday), leaving a real NaN gap in between. .diff() subtracts the
    # IMMEDIATELY PRECEDING ROW, not the last real value, so a single
    # NaN day silently poisons the very next real day's diff into NaN
    # too, even though that next day's own value is perfectly valid.
    # Fix: drop each series' own NaNs first, diff against its own last
    # real observation (however many calendar days back that is), then
    # reindex back to the full table -- correctly leaves NaN only on
    # days that series itself has no value, not on the day after a gap.
    def _safe_diff(s):
        return s.dropna().diff().reindex(s.index)

    def _safe_logret(s):
        return np.log(s.dropna() / s.dropna().shift(1)).reindex(s.index)

    merged["real_rate_diff"] = _safe_diff(merged["real_rate"])
    merged["usd_logret"] = _safe_logret(merged["usd_proxy"])
    merged["nominal_10y_diff"] = _safe_diff(merged["nominal_10y"])
    merged["gvz_logret"] = _safe_logret(merged["gvz"])

    # Derive "last closed day" from the ACTUAL data returned, not from
    # calendar arithmetic. Naive "today minus 1 day" breaks on Mondays
    # (lands on Sunday, never a trading day) and on any market holiday --
    # asking the real data what its most recent date is sidesteps both
    # automatically, since weekends/holidays simply have no rows to find.
    if market_closed_today:
        candidates = merged.index[merged.index <= today]
    else:
        candidates = merged.index[merged.index < today]
    last_closed_date = candidates.max() if len(candidates) > 0 else today - pd.Timedelta(days=1)
    yesterday_cutoff = last_closed_date + pd.Timedelta(days=1)
    merged["gold_logret"] = _safe_logret(merged["gold"])

    # bridge = strictly completed days only (through yesterday)
    # diagnostic: show each raw series (pre-ffill, pre-dropna) for every
    # date after the archive, so a missing day is directly visible
    # instead of silently dropped -- confirms exactly which of the 5
    # source pulls (real_rate, usd_proxy, nominal_10y, gvz, gold) is
    # actually missing a date, rather than guessing after the fact
    diag_window = merged[merged.index > last_archived_date][
        ["real_rate", "usd_proxy", "nominal_10y", "gvz", "gold"]]
    print("--- Raw pulled values, pre-ffill/pre-dropna (for diagnosing gaps) ---")
    print(diag_window)
    print()

    gap_df = merged[(merged.index > last_archived_date) & (merged.index < yesterday_cutoff)]
    stale_days = gap_df.index[gap_df.index > last_real_rate_date] if len(gap_df) else []
    if len(stale_days) > 0:
        print(f"real_rate proxied via ^TNX for: {[d.date() for d in stale_days]}")
        gap_df.loc[stale_days, "real_rate_diff"] = gap_df.loc[stale_days, "nominal_10y_diff"]
    gap_df = gap_df.dropna(subset=REGRESSORS + ["gold_logret"])

    # the NEXT still-open day's move so far, if any row exists for it yet
    # (this is TOMORROW's date if today's session has already closed)
    next_open_day = last_closed_date + pd.Timedelta(days=1)
    today_rows = merged[merged.index >= next_open_day]
    observed_today = None
    if len(today_rows) == 0:
        print(f"(no data yet for {next_open_day.date()} -- most recent real "
              f"timestamps: usd_proxy={last_usd_proxy_ts}, "
              f"nominal_10y={last_nominal_10y_ts})")
    if len(today_rows) > 0:
        row = today_rows.iloc[0]
        # for real_rate: use the nominal proxy's overnight move directly
        # (DFII10 itself never has intraday/premarket data -- it's a
        # once-daily government release, not a traded instrument)
        rr = row["nominal_10y_diff"] if pd.notna(row["nominal_10y_diff"]) else None
        usd = row["usd_logret"] if pd.notna(row["usd_logret"]) else None
        if rr is not None and usd is not None:
            observed_today = {"real_rate_diff": rr, "usd_logret": usd}

    return gap_df, observed_today


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    last_archived_date = df.index[-1]
    print(f"Archived data (100% real) runs through {last_archived_date.date()}")

    if FRED_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("Set FRED_API_KEY as an env var or paste it into the script.")
    fred = Fred(api_key=FRED_API_KEY)

    gap_df, observed_today = bridge_to_yesterday(df, last_archived_date, fred)

    if not gap_df.empty:
        print(f"\nBridged through yesterday ({gap_df.index[-1].date()}):")
        print(gap_df[REGRESSORS])

    # --- ONE shared beta, computed through yesterday only ---
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    full_exog = pd.concat([df[REGRESSORS], gap_df[REGRESSORS]]) / pd.Series(TRAIN_STD)
    # use REAL gold returns for the bridge days -- these are already-
    # completed sessions with genuine, known gold_logret, not something
    # to fake as 0. A fake 0 would tell the Kalman filter's update step
    # "gold definitely didn't move," actively biasing beta rather than
    # just leaving it uninformed.
    full_endog = pd.concat([df["gold_logret"], gap_df["gold_logret"]])

    sm_model = TVPRegression(full_endog, full_exog)
    sm_model.exog_names = REGRESSORS
    params = np.array([SIGMA_OBS**2] + [SIGMA_BETA[r]**2 for r in REGRESSORS])
    res = sm_model.filter(params)

    beta_std = res.predicted_state[:, -1]
    beta_cov_std = res.predicted_state_cov[:, :, -1]
    beta_orig = beta_std / std_vec

    print("\nShared beta (as of yesterday's close, used by BOTH scenarios below):")
    for r, b in zip(REGRESSORS, beta_orig):
        print(f"  {r}: {b:.4f}")

    recent_window = pd.concat([df[REGRESSORS].iloc[-15:], gap_df[REGRESSORS]])
    trend_avg = {r: recent_window[r].mean() for r in REGRESSORS}

    def forecast_from_scenario(x_dict, label):
        x = np.array([x_dict[r] for r in REGRESSORS])
        forecast_logret = beta_orig @ x
        x_std = x / std_vec
        pred_var = x_std @ beta_cov_std @ x_std + params[0]
        z = forecast_logret / np.sqrt(pred_var) if pred_var > 0 else 0.0
        bucket, p, f_half = bucket_and_kelly(z)
        direction = "UP" if forecast_logret > 0 else ("DOWN" if forecast_logret < 0 else "FLAT")
        stake = HYPOTHETICAL_DOLLARS * f_half if forecast_logret != 0 else 0.0

        print(f"\n--- {label} ---")
        assumed_str = ", ".join(f"{r}: {x_dict[r]:+.6f}" for r in REGRESSORS)
        print(f"  Scenario: {assumed_str}")
        print(f"  Forecast gold log-return: {forecast_logret:+.5f} -> direction: {direction}")
        print(f"  |z| = {abs(z):.3f} -> confidence bucket {bucket} (train hit rate {p:.1%})")
        print(f"  Half-Kelly stake on ${HYPOTHETICAL_DOLLARS}: ${stake:.2f} "
              f"{'long' if forecast_logret > 0 else ('short' if forecast_logret < 0 else 'n/a')}")
        return {"direction": direction, "bucket": bucket, "stake": stake}

    guess = forecast_from_scenario(trend_avg, "EOD_TREND (last full close, trend-persists extrapolation -- no new info)")

    if observed_today is None:
        print("\n--- SINCE_LAST_CLOSE (overnight/premarket) ---")
        print("  No overnight move detected yet for today (^TNX/DX-Y.NYB haven't")
        print("  moved from yesterday's close, or no data yet) -- nothing to compare.")
        observed = None
    else:
        observed_scenario = dict(trend_avg)  # gvz stays at trend-average (no overnight GVZ)
        observed_scenario["real_rate_diff"] = observed_today["real_rate_diff"]
        observed_scenario["usd_logret"] = observed_today["usd_logret"]
        observed = forecast_from_scenario(
            observed_scenario, "SINCE_LAST_CLOSE (today's actual dollar/rate move since yesterday's close)")

    print(f"\n{'='*60}\nComparison -- both forecast the SAME target day (today)\n{'='*60}")
    if observed:
        agree = guess["direction"] == observed["direction"]
        print(f"EOD_TREND:        {guess['direction']}, bucket {guess['bucket']}, ${guess['stake']:.2f}")
        print(f"SINCE_LAST_CLOSE: {observed['direction']}, bucket {observed['bucket']}, ${observed['stake']:.2f}")
        print(f"\n{'AGREE' if agree else 'DISAGREE'} on direction.")
        if not agree:
            print("Disagreement day -- log which one turns out right once today closes.")


if __name__ == "__main__":
    main()