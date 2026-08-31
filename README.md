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
- **Long-only has a real, measurable cost, not just a theoretical one.**
  Removing shorting entirely (to match a standard cash-account Roth IRA,
  which can't short) dropped the Sharpe-like ratio from 6.04 to 4.98 at
  the same Kelly fraction, with max drawdown barely changing (-1.63% ->
  -1.55%). In this specific window, avoiding shorts mostly gave up
  upside rather than avoiding downside -- which doesn't undercut the
  structural case against shorting (a short's loss is unbounded, a
  long's is capped at 100%), it just means this particular backtest
  window never hit the scenario where that structural risk bites.
- **Manually excluding low-confidence buckets (0-1) from a long-only
  strategy made things WORSE, not better.** Half-Kelly long-only with
  all buckets included returned 89.01%; restricting to bucket>=2 only
  dropped that to 80.54%, with Sharpe falling too (4.98 -> 4.66) and
  almost no drawdown improvement (-1.55% -> -1.49%). The reason: Kelly
  sizing already discounts weak-edge buckets correctly (bucket 0's 8.7%
  stake reflects its thin 58.7% edge) -- manually zeroing them out on
  top of that doesn't remove risk that wasn't already handled, it just
  throws away real, if modest, positive-expectancy bets. See
  `tvp_sizing_variants.py`.
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

## Live forecasting: two FRED series lag, and that's not a bug

`DTWEXBGS` (the broad USD index) has a **structural ~1-week publication
lag** on FRED -- it's a Fed H.10 statistical release, genuinely slower to
post than the other series, not a pipeline bug. `DFII10` (the real
yield) also lags by ~1 business day on any given day, for the mundane
reason that FRED hasn't finished its daily update yet.

For historical backtesting this doesn't matter -- `fetch_gold_data.py`
correctly trims to whatever's actually published (only using
`gold`/`real_rate`/`usd_index` to determine the cutoff; the optional
breakeven columns are allowed to lag independently without holding back
data that's already current). But it means the *archived* dataset is
never quite caught up to "yesterday," which blocks a genuinely live
"what does the model say about tomorrow" call.

`tvp_bridge_forecast.py` solves this for live use only, never touching
the archive:
- Pulls `DFII10` directly from FRED for whatever's missing.
- If today's `DFII10` genuinely hasn't posted, substitutes that day's
  move in the **nominal** 10Y yield (`^TNX`) instead of assuming no
  change -- justified because breakeven inflation expectations move
  much less day-to-day than nominal yields do, so a nominal move is a
  reasonable (not exact) stand-in for the same day's real-yield move.
- Substitutes `DX-Y.NYB` (ICE Dollar Index futures) for the
  not-yet-published `DTWEXBGS` days.
- Advances the Kalman filter's state through the bridged days (frozen
  hyperparameters, no refitting), then forecasts the next trading day
  off the *current* state instead of a stale one.

None of this proxy data is ever written into `gold_macro_data.csv`.
Mixing assumed/proxy values into what's supposed to be pure real data
would quietly contaminate every downstream script that trusts that file
as ground truth -- a mistake worth avoiding even when the assumption
seems harmless (an early version of this script silently zero-filled a
missing `real_rate_diff` this way; the fix was substituting the `^TNX`
proxy explicitly rather than pretending nothing happened that day).

**Daily live workflow:**
```
python fetch_gold_data.py          # refresh with whatever's real
python tvp_bridge_forecast.py      # bridge the gap, forecast off current state
```
Check the printed warnings for which series (if any) needed bridging
before trusting the forecast.

## Forward paper-trading log

Started 2026-08-31. A backtest assumes perfect knowledge of the day's
actual macro inputs; a live forecast only has whatever's actually
knowable (including proxy-bridged data) at the moment a call gets made.
Those are different questions, and only the second one tells you whether
this survives contact with real-world data constraints -- not just a
clean historical dataset.

**Rule, decided and validated, not just a preference:** size every
bucket per the Kelly table, no manual exclusion of low-confidence
buckets. Confirmed as the better rule against the naive "skip 0-1"
instinct (see the sizing-variants finding above) -- Kelly's own math
already discounts weak edges correctly.

**Standing constraint:** long-only, no shorting, regardless of what the
model outputs. A short's structural risk (unbounded loss) doesn't
appear in one backtest's numbers by chance; that's a property of the
instrument, not something a good historical window can vouch for.

**Log columns:**
```
date,scenario,forecast_direction,confidence_bucket,kelly_stake_pct,prior_close,next_close,actual_direction,hit,daily_return,cumulative_return,running_hit_rate
```

**What to watch for over the next few months isn't any single day's
outcome -- it's whether the bucket ordering holds.** If bucket 4 keeps
meaningfully beating bucket 0 over many logged days, the calibration
survived contact with a new period. If the ordering goes flat or
scrambles, that's real evidence the calibration decayed -- a legitimate
trigger to refit, separate from any fixed calendar schedule (see below).


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
| `tvp_sizing_variants.py` | Half vs. quarter Kelly, long-only vs. long/short, bucket>=2-only vs. all buckets, plus a buy-and-hold benchmark | seconds |
| `tvp_dollar_amplification.py` | Gold's USD-sensitivity vs. the mechanical -1 currency-pass-through benchmark, full history + current percentile rank | seconds |
| `tvp_forward_prediction.py` | Scenario-conditional next-day (and rough monthly) prediction using the locked model, off the archived (lagged) dataset | seconds |
| `tvp_bridge_forecast.py` | Same as above, but bridges the FRED publication lag with fast-updating proxies first -- see "Live forecasting" below | seconds |
| `tvp_grade_latest.py` | Pulls real, fresh gold prices (yfinance) to grade recent forecast calls without waiting on FRED | seconds |

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
then `tvp_forward_prediction.py` (or `tvp_bridge_forecast.py` for the
freshest possible inputs) for a current call. Like the dollar-
amplification finding above, treat this as "what the model said on
2026-08-21," not "what the model will always say" — check the paper-
trading log above for what it's actually said since.

## Requirements

```
pip install fredapi yfinance pandas numpy statsmodels matplotlib pymc arviz
```

Needs a free FRED API key (`FRED_API_KEY` env var) — see
`fetch_gold_data.py` for details.