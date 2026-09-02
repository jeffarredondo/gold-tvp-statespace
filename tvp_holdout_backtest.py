"""
Honest holdout backtest: fit hyperparameters (MLE point estimate, and the
Bayes shrinkage-prior posterior) using only the first 80% of the data in
time, then freeze those hyperparameters and run the Kalman filter forward
through the ENTIRE series -- states keep updating normally at each t using
only past y's (that's just how filtering works), but the variance/scale
hyperparameters themselves never see the last 20% during fitting.

Directional accuracy is then scored separately on the train window and
the held-out test window. This is the test that can actually detect
overfitting: an unconstrained beta that's fitting noise looks fine (or
even great) in-sample, by definition -- the cost only shows up when it's
asked to predict noise it hasn't seen yet. The previous full-sample
comparison (tvp_bayes_shrinkage.py) couldn't detect that at all, which is
exactly why MLE and Bayes came back statistically identical there.

Bumped default draws/tune/chains up since the M4 Mac mini ran the smaller
config in ~5 minutes -- there's headroom to chase better rhat/ESS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import (
    fit_shrinkage_model,
    filtered_forecasts,
    directional_accuracy,
)

DATA_PATH = "gold_macro_data.csv"
REGRESSORS = ["real_rate_diff", "usd_logret", "gvz_logret"]
SPLIT_DATE = "2022-07-06"  # same fixed date used throughout every comparison tonight

# beefed up now that a 300/300/4-chain run only took ~5 min on the M4
DRAWS = 1500
TUNE = 1500
CHAINS = 4


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Restrict to gvz_logret's actual first observation -- otherwise the
    # early years get treated as fully-missing observations by statsmodels
    # (any NaN in the design row skips the WHOLE row, not just that
    # column), giving real_rate/usd_logret worse informational footing
    # too, not just gvz. Confirmed via testing this doesn't meaningfully
    # change results (tvp_new_regressor_test.py, restricted vs
    # unrestricted GVZ comparison came back nearly identical) -- doing it
    # here too for full consistency with how GVZ was validated.
    first_valid = df["gvz_logret"].first_valid_index()
    original_len = len(df)
    df = df[df.index >= first_valid]
    print(f"Restricting to {first_valid.date()} onward (gvz_logret's first "
          f"observation) -- dropped {original_len - len(df)} early rows.\n")

    endog_full = df["gold_logret"]
    exog_raw_full = df[REGRESSORS]

    split = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
    split_date = df.index[split]
    print(f"Train: {df.index[0].date()} to {df.index[split-1].date()} "
          f"({split} obs)")
    print(f"Test:  {split_date.date()} to {df.index[-1].date()} "
          f"({len(df)-split} obs)\n")

    # standardize using TRAIN stats only, apply the same fixed scale to
    # the whole series -- avoids leaking test-period variance into the
    # very numbers we're using to keep the two betas comparable
    exog_std = exog_raw_full.iloc[:split].std()
    print("Regressor std devs (train-only, used to standardize):")
    print(exog_std, "\n")
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    # --- fit hyperparameters on TRAIN ONLY ---
    print("Fitting MLE on train window...")
    sm_model_train = TVPRegression(endog_train, exog_train)
    sm_model_train.exog_names = list(exog_train.columns)
    mle_res = sm_model_train.fit(disp=False, maxiter=500)
    mle_params = mle_res.params
    print(f"MLE params (sigma2_obs, then sigma2_beta per regressor in order {REGRESSORS}):")
    print(mle_params, "\n")

    print("Fitting Bayes shrinkage model on train window "
          f"({CHAINS} chains x {TUNE} tune + {DRAWS} draws -- this is the long part)...")
    _, trace = fit_shrinkage_model(endog_train, exog_train,
                                    draws=DRAWS, tune=TUNE, chains=CHAINS)

    import arviz as az
    summary = az.summary(trace, var_names=["sigma_obs", "tau", "sigma_beta"])
    print("\n--- Posterior summary (train window only, standardized units) ---")
    print(summary)

    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    # --- freeze those hyperparameters, filter forward through the FULL series ---
    sm_model_full = TVPRegression(endog_full, exog_full)
    sm_model_full.exog_names = list(exog_full.columns)

    mle_forecasts_full = filtered_forecasts(sm_model_full, mle_params)
    bayes_forecasts_full = filtered_forecasts(sm_model_full, bayes_params)
    actuals_full = endog_full.values

    def split_acc(forecasts):
        train_acc = directional_accuracy(forecasts[:split], actuals_full[:split])
        test_acc = directional_accuracy(forecasts[split:], actuals_full[split:])
        return train_acc, test_acc

    mle_train_acc, mle_test_acc = split_acc(mle_forecasts_full)
    bayes_train_acc, bayes_test_acc = split_acc(bayes_forecasts_full)

    print("\n--- Directional accuracy: train (in-sample) vs test (holdout) ---")
    print(f"{'Model':<20} {'Train acc':>12} {'Test acc':>12} {'Gap':>10}")
    print(f"{'MLE':<20} {mle_train_acc:>12.4f} {mle_test_acc:>12.4f} "
          f"{mle_train_acc - mle_test_acc:>10.4f}")
    print(f"{'Bayes (shrinkage)':<20} {bayes_train_acc:>12.4f} {bayes_test_acc:>12.4f} "
          f"{bayes_train_acc - bayes_test_acc:>10.4f}")
    print("\nWhat to look for: if shrinkage is doing its job, its train-test gap")
    print("(overfitting) should be smaller than MLE's, and/or its test accuracy")
    print("should be higher than MLE's test accuracy. Train accuracy alone tells")
    print("you nothing about generalization -- only the test column and the gap do.")

    # --- plot: forecasts on the holdout window only, both models ---
    test_dates = df.index[split:]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test_dates, np.sign(mle_forecasts_full[split:]), label="MLE sign", alpha=0.6)
    ax.plot(test_dates, np.sign(bayes_forecasts_full[split:]), label="Bayes sign", alpha=0.6)
    ax.plot(test_dates, np.sign(actuals_full[split:]), label="Actual sign", color="black", linewidth=0.8)
    ax.set_title("Holdout window: forecast sign vs actual sign")
    ax.legend()
    plt.tight_layout()
    plt.savefig("holdout_backtest.png", dpi=150)
    print("\nSaved holdout_backtest.png")


if __name__ == "__main__":
    main()