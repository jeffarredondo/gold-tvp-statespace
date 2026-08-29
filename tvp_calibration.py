"""
Calibration check for the locked 2-regressor Bayes-shrinkage TVP model
(real_rate_diff, usd_logret).

The Kalman filter gives more than a point forecast -- at each t it also
gives the predicted variance of that forecast, Var(y_t | t-1), built from
the predicted state covariance P_t and that day's regressor values:

    Var(y_t | t-1) = z_t' P_t z_t + sigma2_obs

z_t = forecast_t / sqrt(Var(y_t | t-1)) is then "how many standard
deviations from zero is this forecast" -- a built-in confidence measure,
no extra modeling required. If the model is well-calibrated, forecasts
with larger |z| (more confident) should have a HIGHER realized hit rate
than forecasts with small |z| (model shrugging). If hit rate is flat
across |z| buckets, the model doesn't actually know when it doesn't know
-- which would undercut any position-sizing scheme built on top of it,
since sizing bigger on "confident" calls only makes sense if confidence
is actually informative.

Same train/test discipline as before: hyperparameters fit on the first
80% only, bucket boundaries also defined on train-only z-scores (so the
test set can't leak into bucket definitions), then hit rate is reported
separately for train and test within each bucket.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import fit_shrinkage_model

DATA_PATH = "gold_macro_data.csv"
TRAIN_FRAC = 0.8
REGRESSORS = ["real_rate_diff", "usd_logret"]  # locked: 2-regressor model
N_BUCKETS = 5

DRAWS = 1500
TUNE = 1500
CHAINS = 4


def forecasts_and_variance(sm_model, params):
    """Returns (point forecast, predicted variance) at each t, using only
    information available before t (predicted, not filtered/smoothed)."""
    res = sm_model.filter(params)
    predicted_state = res.predicted_state[:, :-1]      # state predicted before seeing y_t
    predicted_cov = res.predicted_state_cov[:, :, :-1]  # P_t, same alignment
    exog = sm_model.exog                                # (nobs, k)
    sigma2_obs = params[0]

    forecasts = np.einsum("ij,ji->i", exog, predicted_state)

    # Var(y_t|t-1) = z_t' P_t z_t + sigma2_obs, done per-t via einsum
    var = np.einsum("ti,ijt,tj->t", exog, predicted_cov, exog) + sigma2_obs
    return forecasts, var


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    split = int(len(df) * TRAIN_FRAC)

    endog_full = df["gold_logret"]
    exog_raw_full = df[REGRESSORS]
    exog_std = exog_raw_full.iloc[:split].std()
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    print("Fitting Bayes shrinkage model on train window...")
    _, trace = fit_shrinkage_model(endog_train, exog_train,
                                    draws=DRAWS, tune=TUNE, chains=CHAINS)

    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    sm_model_full = TVPRegression(endog_full, exog_full)
    sm_model_full.exog_names = REGRESSORS

    forecasts, var = forecasts_and_variance(sm_model_full, bayes_params)
    actuals = endog_full.values

    valid = ~np.isnan(forecasts) & ~np.isnan(var) & (var > 0)
    z = np.full_like(forecasts, np.nan)
    z[valid] = forecasts[valid] / np.sqrt(var[valid])

    correct = (np.sign(forecasts) == np.sign(actuals))

    # bucket boundaries from TRAIN |z| only, so test can't leak into them
    train_abs_z = np.abs(z[:split])
    train_abs_z = train_abs_z[~np.isnan(train_abs_z)]
    bucket_edges = np.quantile(train_abs_z, np.linspace(0, 1, N_BUCKETS + 1))
    bucket_edges[0], bucket_edges[-1] = -np.inf, np.inf

    abs_z = np.abs(z)
    bucket_idx = np.digitize(abs_z, bucket_edges[1:-1])

    print(f"\n--- Calibration: hit rate by |z| confidence bucket ({N_BUCKETS} buckets) ---")
    print(f"{'Bucket':<20} {'|z| range':<22} {'n (train)':>10} {'hit% train':>11} "
          f"{'n (test)':>9} {'hit% test':>10}")

    bucket_summary = []
    for b in range(N_BUCKETS):
        in_bucket = (bucket_idx == b) & valid
        train_mask = in_bucket & (np.arange(len(df)) < split)
        test_mask = in_bucket & (np.arange(len(df)) >= split)

        n_train, n_test = train_mask.sum(), test_mask.sum()
        hit_train = correct[train_mask].mean() if n_train > 0 else np.nan
        hit_test = correct[test_mask].mean() if n_test > 0 else np.nan

        lo, hi = bucket_edges[b], bucket_edges[b + 1]
        lo_str = f"{lo:.3f}" if np.isfinite(lo) else "0"
        hi_str = f"{hi:.3f}" if np.isfinite(hi) else "inf"
        print(f"{'bucket ' + str(b):<20} {f'[{lo_str}, {hi_str})':<22} "
              f"{n_train:>10} {hit_train:>11.4f} {n_test:>9} {hit_test:>10.4f}")

        bucket_summary.append((b, hit_train, hit_test))

    print("\nWhat to look for: hit% should generally RISE from bucket 0 (least")
    print("confident) to the last bucket (most confident), in BOTH columns.")
    print("A flat or non-monotonic pattern means the model's confidence isn't")
    print("actually informative -- it doesn't know when it doesn't know.")

    # plot: hit rate by bucket, train vs test
    buckets, hit_train_list, hit_test_list = zip(*bucket_summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(buckets, hit_train_list, marker="o", label="train")
    ax.plot(buckets, hit_test_list, marker="o", label="test (holdout)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="coin flip")
    ax.set_xlabel("Confidence bucket (0 = least confident, "
                  f"{N_BUCKETS-1} = most confident)")
    ax.set_ylabel("Directional hit rate")
    ax.set_title("Calibration: does model confidence predict accuracy?")
    ax.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=150)
    print("\nSaved calibration_curve.png")


if __name__ == "__main__":
    main()