"""
Fits the FULL (4-regressor) Bayes shrinkage model once, saves smoothed
beta paths for all four regressors, and runs the same delta-correlation
diagnostic used earlier on real_rate/usd -- this time specifically on
breakeven_5y vs breakeven_10y, since those two are the obvious
collinearity risk (same underlying inflation view, different duration).

Same logic as before: if the two betas' period-over-period CHANGES are
strongly negatively correlated, that's the Kalman filter trading
"wiggle room" between two collinear regressors rather than each beta
tracking its own genuine, independently identified signal. A large
sigma_beta split (10Y wanting ~2x the movement of 5Y, per the last run)
could be real differential signal, or could be this artifact -- the
correlation check is what tells them apart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import fit_shrinkage_model

DATA_PATH = "gold_macro_data.csv"
TRAIN_FRAC = 0.8
REGRESSORS = ["real_rate_diff", "usd_logret", "breakeven_5y_diff", "breakeven_10y_diff"]

DRAWS = 1500
TUNE = 1500
CHAINS = 4


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    split = int(len(df) * TRAIN_FRAC)

    endog_full = df["gold_logret"]
    exog_raw_full = df[REGRESSORS]
    exog_std = exog_raw_full.iloc[:split].std()
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    print("Fitting FULL (4-regressor) Bayes shrinkage model on train window...")
    sm_model_train = TVPRegression(endog_train, exog_train)
    sm_model_train.exog_names = REGRESSORS
    _, trace = fit_shrinkage_model(endog_train, exog_train,
                                    draws=DRAWS, tune=TUNE, chains=CHAINS)

    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    print("\nPosterior means:")
    print(f"  sigma_obs: {post_mean_sigma_obs:.6g}")
    for name, val in zip(REGRESSORS, post_mean_sigma_beta):
        print(f"  sigma_beta[{name}]: {val:.6g}")

    # smooth over the FULL series with these frozen hyperparameters
    sm_model_full = TVPRegression(endog_full, exog_full)
    sm_model_full.exog_names = REGRESSORS
    res = sm_model_full.smooth(bayes_params)

    # convert back to original (unstandardized) units for interpretation
    betas = pd.DataFrame(
        {name: res.smoothed_state[i] / exog_std[name]
         for i, name in enumerate(REGRESSORS)},
        index=df.index,
    )
    betas.to_csv("tvp_betas_full_4reg.csv")
    print(f"\nSaved smoothed beta paths to tvp_betas_full_4reg.csv ({len(betas)} rows)")

    # --- diagnostic: breakeven_5y vs breakeven_10y delta correlation ---
    d5 = betas["breakeven_5y_diff"].diff()
    d10 = betas["breakeven_10y_diff"].diff()

    std5, std10 = d5.std(), d10.std()
    corr = d5.corr(d10)

    print("\n--- Delta diagnostic: breakeven_5y vs breakeven_10y ---")
    print(f"  std(delta beta_5y):  {std5:.6g}")
    print(f"  std(delta beta_10y): {std10:.6g}")
    print(f"  ratio (10y / 5y):    {std10/std5:.2f}x")
    print(f"  corr(delta beta_5y, delta beta_10y): {corr:.3f}")
    print("  (strong negative correlation => variance-trading artifact between")
    print("   two collinear regressors, not independent genuine signal)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(d5, d10, s=8, alpha=0.4, color="tab:purple")
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[0].axvline(0, color="gray", linewidth=0.6)
    axes[0].set_xlabel("Δ beta_breakeven_5y")
    axes[0].set_ylabel("Δ beta_breakeven_10y")
    axes[0].set_title(f"corr = {corr:.3f}")

    axes[1].plot(betas.index, betas["breakeven_5y_diff"], label="beta_5y", alpha=0.8)
    axes[1].plot(betas.index, betas["breakeven_10y_diff"], label="beta_10y", alpha=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.6, linestyle="--")
    axes[1].legend()
    axes[1].set_title("Beta paths (original units)")

    plt.tight_layout()
    plt.savefig("breakeven_beta_diagnostic.png", dpi=150)
    print("\nSaved breakeven_beta_diagnostic.png")


if __name__ == "__main__":
    main()