import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# --- Config ---
FRED_API_KEY = os.environ.get("FRED_API_KEY", "<YOUR_API_KEY>")
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


def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRED series don't all publish on the same days (holidays, data lags,
    gold fix licensing gaps). Forward-fill small gaps, then drop any rows
    still missing data at the start/end where a series hasn't begun yet.

    Also trims the tail to the last date every source series actually had
    a real (non-NaN, pre-ffill) observation. FRED series like DTWEXBGS
    often lag a few business days behind gold/rates; without this trim,
    ffill(limit=5) quietly repeats the last known value for those days,
    producing fake zero returns right at the end of the series -- exactly
    the kind of stale-tail artifact that's easy to miss and misread as
    "the dollar went flat" when it's really just a publication lag.
    """
    df = df.sort_index()

    # last real observation date per column, before any filling
    last_real_dates = df.apply(lambda col: col.last_valid_index())
    cutoff = last_real_dates.min()
    if cutoff < df.index.max():
        n_trimmed = (df.index > cutoff).sum()
        print(f"Trimming {n_trimmed} trailing row(s) after {cutoff.date()} "
              f"-- '{last_real_dates.idxmin()}' hadn't published yet.")
        df = df.loc[:cutoff]

    df = df.ffill(limit=5)  # tolerate short gaps, don't paper over long ones
    df = df.dropna()
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
    return out.dropna()


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