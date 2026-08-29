"""
Runs the train/test holdout backtest twice -- once with the original
2-regressor model (real_rate_diff, usd_logret), once with the 4-regressor
version adding breakeven_5y_diff and breakeven_10y_diff -- then tests
whether the bigger model's holdout directional accuracy is actually
distinguishable from the smaller one, rather than just eyeballing two
numbers next to each other.

Test used: McNemar's test on the paired correct/incorrect sequences from
both models over the SAME holdout window. This directly matches our
metric (binary right/wrong calls) and is simple to justify, but assumes
the disagreements are independent across time -- shaky for time series,
since today's forecast error is probably correlated with yesterday's.
If this comes back significant and you want a second opinion, the proper
econometric tool for nested-model forecast comparison is the Diebold-
Mariano test with the Clark-West correction (Clark & West 2007), which
handles that serial correlation via a Newey-West-style variance estimate
and corrects for the fact that a nested (bigger) model's forecast errors
are mechanically noisier even when the extra variables carry zero true
signal. Not implemented here -- flagging it as the natural next step if
this first pass looks interesting.

Both models' hyperparameters are still fit train-only and filtered
forward through the full series, exactly as in tvp_holdout_backtest.py --
this script just runs that same procedure twice, with two different
regressor sets, and adds the significance test at the end.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from tvp_gold_model import TVPRegression
from tvp_bayes_shrinkage import fit_shrinkage_model, filtered_forecasts

DATA_PATH = "gold_macro_data.csv"
TRAIN_FRAC = 0.8

BASE_REGRESSORS = ["real_rate_diff", "usd_logret"]
FULL_REGRESSORS = BASE_REGRESSORS + ["breakeven_5y_diff", "breakeven_10y_diff"]

DRAWS = 1500
TUNE = 1500
CHAINS = 4


def run_regressor_set(df, split, regressor_cols, label):
    print(f"\n{'='*60}\nRunning regressor set: {label} -> {regressor_cols}\n{'='*60}")

    endog_full = df["gold_logret"]
    exog_raw_full = df[regressor_cols]

    exog_std = exog_raw_full.iloc[:split].std()
    exog_full = exog_raw_full / exog_std

    endog_train = endog_full.iloc[:split]
    exog_train = exog_full.iloc[:split]

    print(f"[{label}] Fitting Bayes shrinkage model on train window...")
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
    split = int(len(df) * TRAIN_FRAC)
    actuals = df["gold_logret"].values

    result_base = run_regressor_set(df, split, BASE_REGRESSORS, "BASE (2 regressors)")
    result_full = run_regressor_set(df, split, FULL_REGRESSORS, "FULL (4 regressors, +breakevens)")

    def acc(forecasts, start, end):
        f = forecasts[start:end]
        a = actuals[start:end]
        valid = ~np.isnan(f)
        return np.mean(np.sign(f[valid]) == np.sign(a[valid]))

    print(f"\n{'='*60}\nHoldout comparison\n{'='*60}")
    print(f"{'Model':<30} {'Train acc':>12} {'Test acc':>12}")
    for res in (result_base, result_full):
        train_acc = acc(res["forecasts"], 0, split)
        test_acc = acc(res["forecasts"], split, None)
        print(f"{res['label']:<30} {train_acc:>12.4f} {test_acc:>12.4f}")

    # --- McNemar's test on the holdout window only ---
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
    print(f"\n--- McNemar's test: does FULL disagree with BASE more than chance? ---")
    print(f"Contingency table [ [both_correct, base_only], [full_only, neither] ]:")
    print(f"  {table}")
    print(f"  statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")
    print(f"\n  base_only = {base_only} (times BASE was right, FULL was wrong)")
    print(f"  full_only = {full_only} (times FULL was right, BASE was wrong)")
    if result.pvalue < 0.05:
        winner = "FULL" if full_only > base_only else "BASE"
        print(f"  -> significant at p<0.05: {winner} is genuinely different, not noise.")
    else:
        print(f"  -> not significant at p<0.05: can't distinguish the two on this holdout.")

    print(f"\nsigma_beta posterior means:")
    print(f"  BASE: {dict(zip(BASE_REGRESSORS, result_base['sigma_beta_posterior']))}")
    print(f"  FULL: {dict(zip(FULL_REGRESSORS, result_full['sigma_beta_posterior']))}")


if __name__ == "__main__":
    main()