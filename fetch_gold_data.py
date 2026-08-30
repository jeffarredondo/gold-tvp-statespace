"""
Data pipeline: pull gold, real rates, and USD index.

Setup:
    pip install fredapi yfinance pandas numpy
    Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
    export FRED_API_KEY=your_key_here      (or pass it in directly below)

Series pulled:
    DFII10          - 10Y TIPS real yield, from FRED (daily, market days)
    DTWEXBGS        - Trade-weighted USD index, broad, from FRED (daily)
    GC=F            - COMEX gold futures continuous contract, from yfinance

Note: FRED discontinued GOLDPMGBD228NLBM/GOLDAMGBD228NLBM (the LBMA fix
series) -- licensing lapse, not a typo. Gold now comes from yfinance
instead. GC=F (futures) is used rather than GLD (the ETF) to avoid
expense-ratio drag distorting the return series; swap GOLD_TICKER to
"GLD" below if you'd rather track the ETF specifically.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# --- Config ---
FRED_API_KEY = os.environ.get("FRED_API_KEY", "<KEY>")
START_DATE = "2003-01-01"  # DFII10 doesn't really start clean until ~2003
GOLD_TICKER = "GC=F"  # COMEX gold futures; use "GLD" for the ETF instead

FRED_SERIES = {
    "real_rate": "DFII10",
    "usd_index": "DTWEXBGS",
    "breakeven_5y": "T5YIE",
    "breakeven_10y": "T10YIE",
}


def pull_fred_data(api_key: str, start_date: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)

    data = {}
    for label, series_id in FRED_SERIES.items():
        s = fred.get_series(series_id, observation_start=start_date)
        s.name = label
        data[label] = s
        print(f"Pulled {series_id} -> '{label}': {len(s)} obs, "
              f"{s.index.min().date()} to {s.index.max().date()}")

    df = pd.concat(data.values(), axis=1)
    return df


def pull_gold_data(ticker: str, start_date: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    gold = raw["Close"].copy()
    gold.columns = ["gold"]
    gold.index.name = None
    print(f"Pulled {ticker} -> 'gold': {len(gold)} obs, "
          f"{gold.index.min().date()} to {gold.index.max().date()}")
    return gold


CORE_SERIES = ["gold", "real_rate", "usd_index"]  # required for the locked model
OPTIONAL_SERIES = ["breakeven_5y", "breakeven_10y"]  # nice-to-have, allowed to lag


def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRED series don't all publish on the same days (holidays, data lags,
    gold fix licensing gaps). Forward-fill small gaps, then drop any rows
    still missing data at the start/end where a series hasn't begun yet.

    Only CORE_SERIES (what the locked 2-regressor model actually needs)
    determine the trim cutoff -- the breakeven series are optional extras
    the model doesn't use for its main prediction, and FRED sometimes
    publishes them on a noticeably different schedule than the core
    series. Forcing everything to share one cutoff meant a lagging
    breakeven pull could hold back real_rate/usd_index/gold data that
    was actually already current -- exactly the kind of thing that
    silently stales out a "next day" prediction for no good reason.
    """
    df = df.sort_index()

    # cutoff based on CORE series only
    core_last_dates = df[CORE_SERIES].apply(lambda col: col.last_valid_index())
    cutoff = core_last_dates.min()
    if cutoff < df.index.max():
        n_trimmed = (df.index > cutoff).sum()
        print(f"Trimming {n_trimmed} trailing row(s) after {cutoff.date()} "
              f"-- '{core_last_dates.idxmin()}' hadn't published yet.")
        df = df.loc[:cutoff]

    df = df.ffill(limit=5)  # tolerate short gaps, don't paper over long ones

    # only require CORE_SERIES to be non-null; optional series (breakevens)
    # can be NaN at the tail without dropping otherwise-good core rows
    df = df.dropna(subset=CORE_SERIES)
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per the project guardrails: model in returns/diffs, not levels.
    - Gold and USD index: log returns (they're strictly positive levels)
    - Real rate and breakevens: simple diffs (all can dip negative --
      real rates during easy-money periods, breakevens during deflation
      scares like 2008/2020 -- so log returns are undefined for them)
    """
    out = df.copy()
    out["gold_logret"] = np.log(df["gold"] / df["gold"].shift(1))
    out["usd_logret"] = np.log(df["usd_index"] / df["usd_index"].shift(1))
    out["real_rate_diff"] = df["real_rate"].diff()
    out["breakeven_5y_diff"] = df["breakeven_5y"].diff()
    out["breakeven_10y_diff"] = df["breakeven_10y"].diff()
    # only require the CORE columns to be non-null; breakeven diffs are
    # optional extras and shouldn't drop otherwise-good trailing rows
    core_derived = ["gold_logret", "usd_logret", "real_rate_diff"]
    return out.dropna(subset=core_derived)


def main():
    if FRED_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "Set FRED_API_KEY as an env var or paste it into the script."
        )

    fred_raw = pull_fred_data(FRED_API_KEY, START_DATE)
    gold_raw = pull_gold_data(GOLD_TICKER, START_DATE)
    raw = fred_raw.join(gold_raw, how="outer")
    clean = clean_and_align(raw)
    modeling_df = add_log_returns(clean)

    print("\n--- Sample of modeling dataframe ---")
    print(modeling_df.tail())

    out_path = "gold_macro_data.csv"
    modeling_df.to_csv(out_path)
    print(f"\nSaved {len(modeling_df)} rows to {out_path}")

    return modeling_df


if __name__ == "__main__":
    main()