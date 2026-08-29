# gold-tvp-statespace

A time-varying-parameter (TVP) Bayesian state space model for gold's daily
directional sensitivity to real interest rates and the US dollar — built
as a weekend deep-dive into Kalman filtering, shrinkage priors, and honest
backtesting. **This is an educational/exploratory project, not a trading
system or investment advice.**

## What this is

Gold's relationship to real rates and the dollar isn't fixed — it visibly
shifts across macro regimes (QE era vs. hiking cycle vs. now). Instead of
a single OLS coefficient averaged over 20 years, this project models both
sensitivities as **time-varying parameters**: latent states that follow a
random walk through a Kalman filter, estimated via a custom
`statsmodels.tsa.statespace.mlemodel.MLEModel` subclass, with the random
walk variances themselves fit via full Bayesian inference (PyMC) using a
horseshoe shrinkage prior — so the model decides *for itself* whether a
given coefficient should genuinely move over time or behave like an
ordinary fixed regression coefficient.

Underneath the Bayesian machinery, don't lose sight of what this is: still
just linear regression (`gold_return = β₁·real_rate_change +
β₂·usd_change + ε`), just with the coefficients allowed to wiggle instead
of being pinned to one number for two decades.

## The model

```
gold_logret_t = beta1_t * real_rate_diff_t + beta2_t * usd_logret_t + eps_t
beta1_t = beta1_{t-1} + eta1_t          (random walk, eta1 ~ N(0, sigma2_beta1))
beta2_t = beta2_{t-1} + eta2_t          (random walk, eta2 ~ N(0, sigma2_beta2))
```

- **real_rate_diff**: daily change in the 10Y TIPS real yield (FRED: `DFII10`)
- **usd_logret**: daily log return of the trade-weighted USD index (FRED: `DTWEXBGS`)
- **gold_logret**: daily log return of gold (via `yfinance`, `GC=F` continuous futures)

Both regressors are standardized (unit variance, using train-only stats)
before fitting — they live on very different natural scales, and skipping
this step was an early, real mistake that made one coefficient's noise
look ~65x larger than the other's when the true gap (after fixing it) was
closer to ~4x. See `diagnose_tvp_betas.py` / `tvp_breakeven_diagnostic.py`
for the diagnostic tooling that caught this.

## Why not a simpler tool?

`statsmodels.tsa.statespace.structural.UnobservedComponents` with
`mle_regression=False` looks like it gives time-varying regression
coefficients, but it doesn't — it has no estimated process variance on
the regression states, so it's actually **recursive OLS** (converges to
the static in-sample average, can't track a real regime break). Confirmed
directly with a synthetic test where the true coefficient shifted
regimes halfway through the series and the built-in tool completely
missed it. That's why this project uses a hand-built `MLEModel` instead —
see the docstring in `tvp_gold_model.py` for the full test.

## Key findings

- **The shrinkage prior works.** On synthetic data with one genuinely
  time-varying coefficient and one genuinely fixed one, the horseshoe
  prior correctly separated them without any manual tuning.
- **A units confound nearly produced a false "this coefficient is all
  noise" conclusion.** Real rate changes have ~15x the standard deviation
  of USD log returns in this dataset; standardizing first was essential
  before comparing the two coefficients' innovation variances.
- **Adding TIPS breakevens (5Y/10Y) as extra regressors was tested and
  rejected.** A 4-regressor model showed a small directional accuracy
  improvement on the holdout (0.6852 → 0.6955), but McNemar's test found
  it statistically indistinguishable from chance (p=0.35), and the two
  breakeven durations showed the classic collinear-regressor symptom
  (unstable variance split between two highly correlated inputs). **The
  2-regressor model (real rate + USD only) is what's locked in.**
- **The model's confidence is genuinely informative.** Bucketing forecasts
  by |z| = forecast / Kalman-predicted-std and checking realized hit rate
  per bucket shows a clean, monotonic climb from ~57% (least confident)
  to ~87% (most confident) on held-out data — confirmed well outside
  sampling noise (~10 standard errors from a coin flip in the top
  bucket). This is what makes confidence-based position sizing meaningful
  rather than cosmetic.
- **Confidence-weighted half-Kelly sizing beat flat sizing at the same
  average exposure** on the holdout window (202.68% vs. 110.78% total
  return, similar max drawdown ~1.6%). The backtest's Sharpe-like ratio
  (~6) is implausibly high by real-world standards and should be read as
  "this specific holdout window was unusually favorable," not "this beats
  professional funds."
- **Gold's dollar sensitivity is currently far above what pure currency
  mechanics would predict.** If gold's USD price moved purely because
  it's dollar-denominated (the "numeraire effect"), `beta_usd` should sit
  near -1 (a 1% dollar move -> ~1% opposite gold move, no story beyond
  currency translation). As of the last data pull (2026-08-21), the
  fitted `beta_usd` is **-4.19** — about **4.2x** the mechanical
  benchmark, and sitting at the **99.1st percentile** of its own 20-year
  history (mean 1.57x, range -1.33x to 5.03x over the full sample; the
  negative low end means the relationship has, at least once
  historically, actually flipped -- gold moving *with* the dollar rather
  than against it, plausibly a shared flight-to-safety episode). See
  `tvp_dollar_amplification.py`.

  **This is a point-in-time snapshot, not a permanent structural fact.**
  The entire reason this project models `beta_usd` as time-varying rather
  than fixed is that this exact multiplier moves — sometimes it's near
  1x (gold behaving like it's "just" doing currency arithmetic),
  sometimes far above it. Rerun `tvp_dollar_amplification.py` for a
  current reading before treating this number as still true; the
  percentile-rank framing ("currently near a 20-year extreme") is more
  honest than quoting -4.19x as a fixed law of how gold behaves.

  ![Gold's dollar-sensitivity amplification factor over time](dollar_amplification.png)

## Honest limitations

- One signal pair, one asset. Real systematic strategies lean on
  *breadth* — many weak, partially independent signals — more than on
  any single signal being individually strong. A genuinely more complete
  version of this would be hierarchical, pooling partial evidence across
  many macro factors/asset classes, not just two.
- One train/test split, one historical regime mix (2006-2022 train,
  2022-2026 test). That test window looks like an easy trending period
  for directional calls — no guarantee this generalizes to a choppier
  regime.
- No transaction costs, no taxes, no capacity constraints modeled. Fine
  for a backtest exploring whether the *methodology* holds up; not fine
  as a real strategy P&L.
- The forward-prediction script (`tvp_forward_prediction.py`) requires an
  explicit scenario for future real-rate/dollar moves — there's no way
  around this, since the model's own inputs are same-day changes. A
  "no view" (flat) scenario correctly forecasts exactly zero.

## Pipeline (run in this order)

| Script | What it does | Runtime |
|---|---|---|
| `fetch_gold_data.py` | Pulls DFII10, DTWEXBGS (FRED) and gold futures (yfinance), aligns and cleans, computes log returns/diffs | seconds |
| `tvp_gold_model.py` | Defines the custom `TVPRegression` `MLEModel`; MLE point-estimate baseline fit + diagnostic plot | seconds |
| `diagnose_tvp_betas.py` | Delta/correlation diagnostics on MLE beta paths (catches noise-chasing, variance-trading) | seconds |
| `tvp_bayes_shrinkage.py` | Library: horseshoe-shrinkage Bayes model (PyMC), reusable fit/forecast functions | ~15-20 min |
| `tvp_holdout_backtest.py` | 80/20 train/test split, MLE vs. Bayes directional accuracy on genuinely held-out data | ~20-25 min |
| `tvp_regressor_comparison.py` | 2-regressor vs. 4-regressor (with breakevens) comparison + McNemar's test | ~40-50 min |
| `tvp_breakeven_diagnostic.py` | Delta-correlation check on the two breakeven betas specifically | ~18 min |
| `tvp_calibration.py` | Confidence-bucket calibration curve (does model confidence predict accuracy?) | ~18 min |
| `tvp_position_sizing.py` | Half-Kelly position sizing backtest vs. flat sizing vs. buy-and-hold | ~18 min |
| `tvp_dollar_amplification.py` | Gold's USD-sensitivity vs. the mechanical -1 currency-pass-through benchmark, full history + current percentile rank | seconds |
| `tvp_forward_prediction.py` | Scenario-conditional next-day (and rough monthly) prediction using the locked model | seconds |

All the PyMC-fitting scripts wrap a black-box Kalman filter likelihood
from `tvp_gold_model.py`/`tvp_bayes_shrinkage.py` as a PyTensor `Op`
(using `statsmodels`' own `score()` for the gradient), which is why
they're slow relative to a native PyMC model — every NUTS step still
runs a full Kalman filter pass. Ran comfortably on an M4 Mac mini;
budget accordingly on slower hardware.

## Sample prediction (point-in-time, not a standing forecast)

Output from `tvp_forward_prediction.py`, using data through 2026-08-21 and
the "trend persists" scenario (last 20-day average real-rate/dollar
moves continuing):

- **Direction: UP**, forecast next-day log-return +0.465%
- **Confidence bucket 3** of 5 (historical hit rate in this bucket: 73.0%)
- **Half-Kelly stake: 23.1% of capital** — on a hypothetical $1,000,
  that's **$230.50 long**

The FLAT ("no view") scenario correctly returns exactly $0 staked — the
model has nothing to say without an assumption about where real rates
and the dollar are headed, and it says so rather than guessing.

This is a live number, not a fixed result: rerun `fetch_gold_data.py`
then `tvp_forward_prediction.py` for a current call. Like the dollar-
amplification finding above, treat this as "what the model says today,"
not "what the model will always say."

## Requirements

```
pip install fredapi yfinance pandas numpy statsmodels matplotlib pymc arviz
```

Needs a free FRED API key (`FRED_API_KEY` env var) — see
`fetch_gold_data.py` for details.