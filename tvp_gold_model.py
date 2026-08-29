"""
Time-varying parameter (TVP) regression for gold, via a custom Kalman
filter state space model.

    gold_logret_t = beta1_t * real_rate_diff_t + beta2_t * usd_logret_t + eps_t
    beta1_t = beta1_{t-1} + eta1_t
    beta2_t = beta2_{t-1} + eta2_t

Each beta is its own latent state following a random walk, with its own
freely estimated transition variance (sigma2.beta1, sigma2.beta2). MLE
fits those variances from the data -- if a beta doesn't actually need to
move, the fitted variance collapses toward zero and it behaves like a
fixed OLS coefficient. If it does move, the variance is large enough to
let the smoothed state track it.

Why not statsmodels' built-in UnobservedComponents(mle_regression=False)?
Tested it directly against a synthetic series with a real regime break
in beta, and it completely failed to track it -- it has no estimated
process variance on the regression states, so it's actually just
recursive OLS (converges to the static in-sample average), not a real
TVP model. Confirmed via source inspection: no state_cov parameter is
ever assigned to those states. Hence the custom MLEModel below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEModel

DATA_PATH = "gold_macro_data.csv"


class TVPRegression(MLEModel):
    """Time-varying parameter regression via Kalman filter."""

    def __init__(self, endog, exog):
        exog = np.asarray(exog)
        k_states = exog.shape[1]

        super().__init__(endog, k_states=k_states, k_posdef=k_states,
                          initialization="diffuse")

        self.k_exog = k_states
        self.exog = exog

        self.ssm["transition"] = np.eye(k_states)
        self.ssm["selection"] = np.eye(k_states)

        self.ssm["design"] = np.zeros((1, k_states, self.nobs))
        for t in range(self.nobs):
            self.ssm["design", 0, :, t] = exog[t]

    @property
    def param_names(self):
        return ["sigma2.obs"] + [f"sigma2.beta_{n}" for n in self.exog_names]

    @property
    def start_params(self):
        return [np.var(self.endog)] + [1e-4] * self.k_exog

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return constrained ** 0.5

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self.ssm["obs_cov", 0, 0] = params[0]
        self.ssm["state_cov"] = np.diag(params[1:])


def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    endog = df["gold_logret"]
    exog = df[["real_rate_diff", "usd_logret"]]

    model = TVPRegression(endog, exog)
    model.exog_names = list(exog.columns)  # used in param_names
    res = model.fit(disp=True, maxiter=500)

    print("\n--- Fitted variances ---")
    for name, val in zip(model.param_names, res.params):
        print(f"  {name}: {val:.6g}")

    beta_real_rate = res.smoothed_state[0]
    beta_usd = res.smoothed_state[1]

    betas = pd.DataFrame(
        {"beta_real_rate": beta_real_rate, "beta_usd": beta_usd},
        index=df.index,
    )
    betas.to_csv("tvp_betas.csv")
    print(f"\nSaved smoothed beta paths to tvp_betas.csv ({len(betas)} rows)")

    # quick plot so you can eyeball whether the betas actually move
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(betas.index, betas["beta_real_rate"], color="tab:blue")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("Time-varying beta: gold return sensitivity to real-rate changes")

    axes[1].plot(betas.index, betas["beta_usd"], color="tab:orange")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Time-varying beta: gold return sensitivity to USD index returns")

    plt.tight_layout()
    plt.savefig("tvp_betas.png", dpi=150)
    print("Saved plot to tvp_betas.png")

    return res, betas


if __name__ == "__main__":
    main()