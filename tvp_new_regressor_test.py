"""
Tests ONE new candidate regressor at a time against the locked
2-regressor baseline (real_rate_diff, usd_logret) -- same discipline as
the original breakeven test: train/test holdout, McNemar's test for
whether any accuracy gain is real, and the new regressor's own
sigma_beta posterior to check whether it's actually earning a nonzero
role or getting shrunk toward "doesn't matter."

Change NEW_REGRESSOR below and rerun to test the next candidate. Testing
one at a time (rather than all three at once) is deliberate -- it's the
only way to attribute any accuracy change to the RIGHT variable, and
avoids walking straight into the same collinearity trap the 5Y/10Y
breakevens fell into by throwing correlated regressors in together
without checking first.

Candidates, in the order worth testing:
  1. "vix_logret"               -- pure risk-appetite/fear proxy
  2. "treasury_2y_diff"         -- near-term Fed-expectation shifts
  3. "curve_slope_3mo10y_diff"  -- Estrella-Mishkin recession spread (10Y-3mo)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import fit_shrinkage_model, filtered_forecasts

DATA_PATH = "gold_macro_data.csv"

BASE_REGRESSORS = ["real_rate_diff", "usd_logret"]
NEW_REGRESSOR = "gvz_logret"  # <-- change this to test a different candidate
FULL_REGRESSORS = BASE_REGRESSORS + [NEW_REGRESSOR]

SPLIT_DATE = "2022-07-06"  # same fixed date as every other holdout comparison tonight

DRAWS = 1500
TUNE = 1500
CHAINS = 4


def run_regressor_set(df, split, regressor_cols, label):
    print(f"\n{'='*60}\nRunning: {label} -> {regressor_cols}\n{'='*60}")

    endog_full = df["gold_logret"]
    exog_raw_full = df[regressor_cols]

    exog_std = exog_raw_full.iloc[:split].std()
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    sm_model_train = TVPRegression(endog_train, exog_train)
    sm_model_train.exog_names = regressor_cols
    _, trace = fit_shrinkage_model(endog_train, exog_train,
                                    draws=DRAWS, tune=TUNE, chains=CHAINS)

    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    sm_model_full = TVPRegression(endog_full, exog_full)
    sm_model_full.exog_names = regressor_cols
    bayes_forecasts_full = filtered_forecasts(sm_model_full, bayes_params)

    return {
        "label": label,
        "regressors": regressor_cols,
        "forecasts": bayes_forecasts_full,
        "sigma_beta_posterior": post_mean_sigma_beta,
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Restrict BOTH models to start from the new regressor's actual first
    # valid observation -- otherwise BASE gets real data the whole way
    # through while FULL treats the entire early window as missing (any
    # NaN in the design row makes statsmodels skip the WHOLE observation,
    # not just that one column), giving the two models genuinely
    # different informational footing for however long the gap lasts.
    # Confirmed via testing: statsmodels correctly contributes exactly
    # 0.0 log-likelihood for missing rows rather than corrupting the fit
    # with NaN, but that's still less real information than BASE gets in
    # the same window -- restricting both to equal footing removes the
    # question entirely instead of arguing it's probably fine.
    first_valid = df[NEW_REGRESSOR].first_valid_index()
    original_len = len(df)
    df = df[df.index >= first_valid]
    print(f"Restricting to {first_valid.date()} onward "
          f"({NEW_REGRESSOR}'s actual first observation) -- "
          f"dropped {original_len - len(df)} early rows so BASE and FULL "
          f"are compared on equal informational footing throughout.")

    split = df.index.searchsorted(pd.Timestamp(SPLIT_DATE))
    print(f"Train: {df.index[0].date()} to {df.index[split-1].date()} "
          f"({split} obs)")
    print(f"Test:  {df.index[split].date()} to {df.index[-1].date()} "
          f"({len(df)-split} obs)\n")

    actuals = df["gold_logret"].values

    result_base = run_regressor_set(df, split, BASE_REGRESSORS, "BASE (2 regressors)")
    result_full = run_regressor_set(df, split, FULL_REGRESSORS,
                                     f"FULL (BASE + {NEW_REGRESSOR})")

    def acc(forecasts, start, end):
        f = forecasts[start:end]
        a = actuals[start:end]
        valid = ~np.isnan(f)
        return np.mean(np.sign(f[valid]) == np.sign(a[valid]))

    print(f"\n{'='*60}\nHoldout comparison\n{'='*60}")
    print(f"{'Model':<35} {'Train acc':>12} {'Test acc':>12}")
    for res in (result_base, result_full):
        train_acc = acc(res["forecasts"], 0, split)
        test_acc = acc(res["forecasts"], split, None)
        print(f"{res['label']:<35} {train_acc:>12.4f} {test_acc:>12.4f}")

    # --- McNemar's test on the holdout window ---
    base_test_forecasts = result_base["forecasts"][split:]
    full_test_forecasts = result_full["forecasts"][split:]
    test_actuals = actuals[split:]

    valid = ~np.isnan(base_test_forecasts) & ~np.isnan(full_test_forecasts)
    base_correct = np.sign(base_test_forecasts[valid]) == np.sign(test_actuals[valid])
    full_correct = np.sign(full_test_forecasts[valid]) == np.sign(test_actuals[valid])

    both_correct = np.sum(base_correct & full_correct)
    base_only = np.sum(base_correct & ~full_correct)
    full_only = np.sum(~base_correct & full_correct)
    neither = np.sum(~base_correct & ~full_correct)
    table = [[both_correct, base_only], [full_only, neither]]

    result = mcnemar(table, exact=False, correction=True)
    print(f"\n--- McNemar's test: does adding {NEW_REGRESSOR} help? ---")
    print(f"Contingency table [ [both_correct, base_only], [full_only, neither] ]:")
    print(f"  {table}")
    print(f"  statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")
    print(f"  base_only = {base_only} (BASE right, FULL wrong)")
    print(f"  full_only = {full_only} (FULL right, BASE wrong)")
    if result.pvalue < 0.05:
        winner = "FULL" if full_only > base_only else "BASE"
        print(f"  -> significant at p<0.05: {winner} is genuinely different, not noise.")
    else:
        print(f"  -> not significant at p<0.05: can't distinguish the two on this holdout.")

    # --- the actual "did it earn its spot" check ---
    print(f"\n--- sigma_beta posteriors (standardized units) ---")
    print(f"  BASE: {dict(zip(BASE_REGRESSORS, result_base['sigma_beta_posterior']))}")
    print(f"  FULL: {dict(zip(FULL_REGRESSORS, result_full['sigma_beta_posterior']))}")
    new_reg_sigma = result_full["sigma_beta_posterior"][-1]
    base_reg_sigmas = result_full["sigma_beta_posterior"][:-1]
    print(f"\n  {NEW_REGRESSOR}'s sigma_beta: {new_reg_sigma:.6g}")
    print(f"  existing regressors' sigma_beta range: "
          f"{base_reg_sigmas.min():.6g} to {base_reg_sigmas.max():.6g}")
    if new_reg_sigma < base_reg_sigmas.min() * 0.1:
        print(f"  -> {NEW_REGRESSOR} is being shrunk toward ~0 relative to the "
              f"others -- the model isn't finding it needs to move.")
    else:
        print(f"  -> {NEW_REGRESSOR}'s sigma_beta is comparable to the existing "
              f"regressors' -- it's carrying a genuinely nonzero role.")
    print("\nVerdict needs BOTH signals to include this regressor: a real,")
    print("non-noise accuracy gain (McNemar) AND a nonzero sigma_beta role.")
    print("Either alone isn't enough -- in-sample movement without out-of-")
    print("sample benefit, or the reverse, are both reasons to leave it out.")


if __name__ == "__main__":
    main()