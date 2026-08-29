"""
The headline finding of this project, made into one clean derived series:
gold's beta on USD log returns isn't just the mechanical -1 you'd expect
from pure currency-denomination effects (gold priced in dollars means a
1% dollar move should mechanically produce a ~1% opposite move in the
USD gold price, full stop, with zero economic story behind it -- this is
sometimes called the numeraire effect).

amplification(t) = -beta_usd(t)   [positive number; 1.0 = purely
                                    mechanical, no real story beyond
                                    currency translation]

Values well above 1 mean gold is responding to dollar moves as a
genuine economic signal (risk appetite, global liquidity, Fed-policy
proxy) on top of the mechanical conversion -- not just doing currency
arithmetic. The interesting claim isn't a fixed multiplier -- it's that
this multiplier ITSELF moves over time, which is the whole reason a
time-varying model was worth building instead of a static regression.

Uses the real, already-fitted hyperparameters from the locked 2-regressor
holdout run (tvp_holdout_backtest.py) -- one fast Kalman smoother pass
over the full ~20-year history, no new MCMC run needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
REGRESSORS = ["real_rate_diff", "usd_logret"]

# --- real fitted numbers from tvp_holdout_backtest.py ---
TRAIN_STD = {"real_rate_diff": 0.050198, "usd_logret": 0.003392}
SIGMA_OBS = 0.009572
SIGMA_BETA = {"real_rate_diff": 0.000197, "usd_logret": 0.000622}

MECHANICAL_BENCHMARK = 1.0  # pure currency-denomination pass-through


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    std_vec = np.array([TRAIN_STD[r] for r in REGRESSORS])
    exog_full = df[REGRESSORS] / pd.Series(TRAIN_STD)

    sm_model = TVPRegression(df["gold_logret"], exog_full)
    sm_model.exog_names = REGRESSORS

    params = np.array([SIGMA_OBS**2, SIGMA_BETA["real_rate_diff"]**2,
                        SIGMA_BETA["usd_logret"]**2])
    res = sm_model.smooth(params)

    beta_usd_std = res.smoothed_state[1]
    beta_usd = beta_usd_std / std_vec[1]  # back to original units

    amplification = -beta_usd / MECHANICAL_BENCHMARK

    current = amplification[-1]
    hist_mean = amplification.mean()
    hist_max = amplification.max()
    hist_min = amplification.min()
    pct_rank = (amplification < current).mean()

    print(f"Current (as of {df.index[-1].date()}) amplification factor: {current:.2f}x")
    print(f"20-year history: mean={hist_mean:.2f}x, min={hist_min:.2f}x, max={hist_max:.2f}x")
    print(f"Current value is at the {pct_rank:.1%} percentile of the full history")
    print(f"(1.0x = purely mechanical currency pass-through, no story beyond that)")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, amplification, color="tab:red", label="amplification(t)")
    ax.axhline(MECHANICAL_BENCHMARK, color="gray", linewidth=1, linestyle="--",
               label="mechanical benchmark (1.0x)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Amplification factor (-beta_usd)")
    ax.set_title("Gold's dollar sensitivity: how far above pure currency mechanics?")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dollar_amplification.png", dpi=150)
    print("\nSaved dollar_amplification.png")


if __name__ == "__main__":
    main()