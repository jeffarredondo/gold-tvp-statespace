"""
Full-Bayes version of the TVP gold model, with a shrinkage prior on the
beta innovation variances instead of hand-tuning them.

The problem this solves: MLE (tvp_gold_model.py) fit sigma2.beta_usd too
large, letting that coefficient re-fit to near-daily noise instead of
tracking genuine slow drift (confirmed via diagnose_tvp_betas.py -- the
period-over-period beta_usd changes were ~150x larger than beta_real_rate's,
with no correlation between the two, ruling out a shared-variance-budget
explanation and pointing to plain overfitting of that one variance).

The fix, following the TVP shrinkage literature (Bitto & Frühwirth-
Schnatter 2019, "Achieving shrinkage in a time-varying parameter model
framework", and the R package shrinkTVP that implements it): put a
horseshoe-style shrinkage prior directly on each beta's innovation std
dev, sigma_beta_i = tau * lambda_i, where tau is a shared global scale
and each lambda_i is a coefficient-specific local scale. If a beta's true
process variance is ~0, the local scale shrinks it there; if a beta
genuinely needs to move, its local scale can escape the shrinkage even
if the global scale is small. Nothing about which beta gets shrunk is
decided by hand -- it falls out of the posterior.

Note: their machinery mostly exists to handle Gibbs-sampling the full
latent state history, which needs a non-centered parameterization to
avoid degenerate funnel geometry. We don't need that here: our Kalman
filter already integrates the states out analytically (that's what
TVPRegression.loglike computes), so we only need to put the shrinkage
prior on the 2-3 variance hyperparameters and sample those directly with
PyMC -- a much smaller problem than sampling the full state history.

This wraps the statsmodels TVPRegression's loglike/score as a PyTensor Op
so PyMC/NUTS can call it as a black-box likelihood. This is slow relative
to a native PyMC model (each gradient still needs to run the full Kalman
filter), so budget real time for this to run -- expect somewhere in the
10-40 minute range for the full ~5400-row dataset with a modest number of
chains/draws. Kick it off and walk away.
"""

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from tvp_gold_model import TVPRegression

DATA_PATH = "gold_macro_data.csv"
DRAWS = 300
TUNE = 300
CHAINS = 4


# --- wrap the statsmodels Kalman filter as a black-box PyTensor Op ---

class KalmanScore(pt.Op):
    """Gradient of the Kalman filter loglike, via statsmodels' own score()
    (complex-step differentiation under the hood -- more accurate and no
    slower than hand-rolled finite differences)."""
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, sm_model):
        self.sm_model = sm_model

    def perform(self, node, inputs, outputs):
        (params,) = inputs
        try:
            sc = self.sm_model.score(np.asarray(params, dtype=np.float64))
        except Exception:
            sc = np.zeros_like(params)
        outputs[0][0] = np.asarray(sc, dtype=np.float64)


class KalmanLogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, sm_model):
        self.sm_model = sm_model
        self.score_op = KalmanScore(sm_model)

    def perform(self, node, inputs, outputs):
        (params,) = inputs
        try:
            ll = self.sm_model.loglike(params)
            if not np.isfinite(ll):
                ll = -1e10
        except Exception:
            ll = -1e10
        outputs[0][0] = np.array(ll, dtype=np.float64)

    def grad(self, inputs, output_grads):
        (params,) = inputs
        g = self.score_op(params)
        return [output_grads[0] * g]


def fit_shrinkage_model(endog, exog, draws=DRAWS, tune=TUNE, chains=CHAINS):
    """Fit the horseshoe-shrinkage TVP model. Returns the PyMC trace."""
    sm_model = TVPRegression(endog, exog)
    sm_model.exog_names = list(exog.columns)
    loglike_op = KalmanLogLike(sm_model)

    with pm.Model() as model:
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=np.std(endog))

        # horseshoe: shared global scale + per-coefficient local scale
        tau = pm.HalfCauchy("tau", beta=1.0)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=exog.shape[1])
        sigma_beta = pm.Deterministic("sigma_beta", tau * lam)

        params = pt.concatenate([pt.stack([sigma_obs**2]), sigma_beta**2])
        pm.Potential("loglike", loglike_op(params))

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                           cores=min(chains, 4), random_seed=42)

    return sm_model, trace


def filtered_forecasts(sm_model, params):
    """
    One-step-ahead forecasts using the Kalman filter's *predicted* state
    at each t (i.e. only using data up to t-1) -- this is the honest
    forecast series for a directional-accuracy backtest. Smoothed states
    use the whole sample and would leak future information into the
    backtest.
    """
    res = sm_model.filter(params)
    predicted_state = res.predicted_state[:, :-1]  # state predicted before seeing y_t
    forecasts = np.einsum("ij,ji->i", sm_model.exog, predicted_state)
    return forecasts


def directional_accuracy(forecasts, actuals):
    valid = ~np.isnan(forecasts)
    return np.mean(np.sign(forecasts[valid]) == np.sign(actuals[valid]))


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    endog = df["gold_logret"]

    # Standardize regressors to unit variance before fitting. Without this,
    # the shrinkage prior compares innovation variances across regressors
    # on different natural scales (real_rate_diff has ~15x the std of
    # usd_logret in this dataset), which alone would push sigma_beta_usd
    # much larger than sigma_beta_real_rate regardless of whether the true
    # sensitivity is actually noisier -- confounding scale with signal.
    exog_raw = df[["real_rate_diff", "usd_logret"]]
    exog_std = exog_raw.std()
    exog = exog_raw / exog_std
    print("Regressor std devs (used to standardize):")
    print(exog_std)
    print()

    print("Fitting shrinkage-prior Bayesian TVP model...")
    sm_model, trace = fit_shrinkage_model(endog, exog)

    summary = az.summary(trace, var_names=["sigma_obs", "tau", "sigma_beta"])
    print("\n--- Posterior summary (sigma_beta in STANDARDIZED regressor units) ---")
    print(summary)
    print("\nsigma_beta[0] = real_rate_diff, sigma_beta[1] = usd_logret")
    print("Compare these two directly -- same units now, so a large gap")
    print("here reflects genuine signal difference, not scale mismatch.")

    # posterior mean hyperparameters -> one filter pass for beta paths + forecasts
    post_mean_sigma_obs = trace.posterior["sigma_obs"].mean().item()
    post_mean_sigma_beta = trace.posterior["sigma_beta"].mean(dim=["chain", "draw"]).values
    bayes_params = np.concatenate([[post_mean_sigma_obs**2], post_mean_sigma_beta**2])

    res_bayes = sm_model.smooth(bayes_params)
    # convert betas back to original (unstandardized) units for interpretation:
    # y = beta_std * (x / std_x) = (beta_std / std_x) * x, so beta_orig = beta_std / std_x
    betas_bayes = pd.DataFrame(
        {"beta_real_rate": res_bayes.smoothed_state[0] / exog_std["real_rate_diff"],
         "beta_usd": res_bayes.smoothed_state[1] / exog_std["usd_logret"]},
        index=df.index,
    )
    betas_bayes.to_csv("tvp_betas_bayes.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(betas_bayes.index, betas_bayes["beta_real_rate"], color="tab:blue")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("Bayes-shrinkage beta: real-rate sensitivity")
    axes[1].plot(betas_bayes.index, betas_bayes["beta_usd"], color="tab:orange")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Bayes-shrinkage beta: USD sensitivity")
    plt.tight_layout()
    plt.savefig("tvp_betas_bayes.png", dpi=150)
    print("\nSaved shrinkage beta paths to tvp_betas_bayes.csv / .png")

    # --- backtest: MLE point-estimate vs Bayes-shrinkage, directional accuracy ---
    print("\nFitting plain MLE model for comparison...")
    mle_res = sm_model.fit(disp=False, maxiter=500)
    mle_forecasts = filtered_forecasts(sm_model, mle_res.params)
    bayes_forecasts = filtered_forecasts(sm_model, bayes_params)
    actuals = endog.values

    acc_mle = directional_accuracy(mle_forecasts, actuals)
    acc_bayes = directional_accuracy(bayes_forecasts, actuals)

    print("\n--- Directional accuracy (one-step-ahead, full sample) ---")
    print(f"  MLE (unconstrained):        {acc_mle:.4f}")
    print(f"  Bayes (shrinkage prior):    {acc_bayes:.4f}")
    print("\nNote: both hyperparameter fits used the full sample, so this")
    print("compares whether shrinking the noisy beta helps the FORECASTS")
    print("(which only use past states), not a fully out-of-sample test of")
    print("the hyperparameters themselves. A true walk-forward refit of the")
    print("variances periodically (e.g. every ~6-12 months, expanding window)")
    print("would be the next step if this looks promising.")


if __name__ == "__main__":
    main()