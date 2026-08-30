"""
Grades recent gold direction for the paper-trading log, using REAL gold
prices pulled fresh from yfinance -- deliberately independent of
fetch_gold_data.py / gold_macro_data.csv.

Why separate: grading "was the forecast right" only needs the actual
direction gold moved, day to day. It does NOT need real_rate_diff or
usd_logret at all -- those are only needed to make tomorrow's NEW
forecast, not to check whether yesterday's forecast was correct. FRED's
lag on DFII10/DTWEXBGS is irrelevant to grading, since gold's own price
already updates in near real time.

Deliberately does NOT write anything back into gold_macro_data.csv. That
file should stay 100% real, unmixed data -- injecting an assumed
"dollar held flat" placeholder into the modeling dataset would quietly
contaminate it with synthetic values that look identical to real ones in
every later script (calibration, position sizing, the amplification
chart, any future refit). Keep grading and modeling data completely
separate; this script only prints, never writes to the shared CSV.
"""

import yfinance as yf
import numpy as np
import pandas as pd

GOLD_TICKER = "GC=F"
LOOKBACK_DAYS = 10


def main():
    raw = yf.download(GOLD_TICKER, period=f"{LOOKBACK_DAYS}d", progress=False,
                       auto_adjust=True)
    if raw.empty:
        print(f"No data returned for {GOLD_TICKER} -- check your internet "
              f"connection or try again in a moment.")
        return
    closes = raw["Close"].iloc[:, 0] if hasattr(raw["Close"], "columns") else raw["Close"]

    print(f"--- Real gold ({GOLD_TICKER}) closes, last {LOOKBACK_DAYS} trading days ---\n")
    print(f"{'Date':<12} {'Close':>10} {'Log return':>12} {'Direction':>10}")

    prev_close = None
    for date, close in closes.items():
        if prev_close is not None:
            logret = np.log(close / prev_close)
            direction = "UP" if logret > 0 else ("DOWN" if logret < 0 else "FLAT")
            print(f"{date.date()!s:<12} {close:>10.2f} {logret:>+12.5f} {direction:>10}")
        else:
            print(f"{date.date()!s:<12} {close:>10.2f} {'--':>12} {'--':>10}")
        prev_close = close

    print("\nCompare the Direction column above against your tracker's")
    print("forecast_direction for the matching date to fill in the hit column.")
    print("This does NOT touch gold_macro_data.csv -- for a new forecast, you")
    print("still need fetch_gold_data.py to catch up with fresh FRED data.")


if __name__ == "__main__":
    main()