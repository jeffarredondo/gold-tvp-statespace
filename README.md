# gold-tvp-statespace

A time-varying-parameter (TVP) Bayesian state space model for gold's daily
directional sensitivity to real interest rates, the US dollar, and gold's
own options-implied volatility — built as a weekend deep-dive into Kalman
filtering, shrinkage priors, and honest backtesting that turned into an
ongoing forward-testing project. **This is an educational/exploratory
project, not a trading system or investment advice.**

## What this is

Gold's relationship to real rates and the dollar isn't fixed — it visibly
shifts across macro regimes (QE era vs. hiking cycle vs. now). Instead of
a single OLS coefficient averaged over 20 years, this project models
sensitivities as **time-varying parameters**: latent states that follow a
random walk through a Kalman filter, estimated via a custom
`statsmodels.tsa.statespace.mlemodel.MLEModel` subclass, with the random
walk variances themselves fit via full Bayesian inference (PyMC) using a
horseshoe shrinkage prior — so the model decides *for itself* whether a
given coefficient should genuinely move over time or behave like an
ordinary fixed regression coefficient.

Underneath the Bayesian machinery, don't lose sight of what this is: still
just linear regression (`gold_return = β₁·real_rate_change +
β₂·usd_change + β₃·gvz_change + ε`), just with the coefficients allowed
to wiggle instead of being pinned to one number for two decades.

## The model

```
gold_logret_t = b1_t*real_rate_diff_t + b2_t*usd_logret_t + b3_t*gvz_logret_t + eps_t
b1_t = b1_{t-1} + eta1_t   (random walk, eta1 ~ N(0, sigma2_beta1))
b2_t = b2_{t-1} + eta2_t   (random walk, eta2 ~ N(0, sigma2_beta2))
b3_t = b3_{t-1} + eta3_t   (random walk, eta3 ~ N(0, sigma2_beta3))
```

- **real_rate_diff**: daily change in the 10Y TIPS real yield (FRED: `DFII10`)
- **usd_logret**: daily log return of the trade-weighted USD index (FRED: `DTWEXBGS`)
- **gvz_logret**: daily log return of the CBOE Gold Volatility Index (`^GVZ`
  via yfinance) — gold's own options-implied forward-looking volatility,
  added 2026-09-01 after clearing a real bar (see Key Findings below)
- **gold_logret**: daily log return of gold (via `yfinance`, `GC=F` continuous futures)

Both regressors are standardized (unit variance, using train-only stats)
before fitting — they live on very different natural scales, and skipping
this step was an early, real mistake that made one coefficient's noise
look ~65x larger than the other's when the true gap (after fixing it) was
closer to ~4x. See `diagnose_tvp_betas.py` / `tvp_breakeven_diagnostic.py`
for the diagnostic tooling that caught this.

`^GVZ` only exists from 2008-06-04 onward. Every script that fits the
3-regressor model restricts training to that date forward on ALL
regressors, not just GVZ — `statsmodels` treats any `NaN` in the design
matrix as the WHOLE observation missing, so without this restriction,
real_rate and usd_logret would also get worse informational footing
during 2006-2008 in the GVZ-inclusive model than in a model without it.
Confirmed via direct testing that this restriction doesn't meaningfully
change results — it just removes a legitimate objection before trusting
them.

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
  (unstable variance split between two highly correlated inputs).
- **Four more candidates were tested one at a time, same bar every time:
  a real McNemar-significant accuracy gain AND a nonzero sigma_beta
  role, neither alone sufficient** (`tvp_new_regressor_test.py`):

  | Candidate | McNemar p-value | sigma_beta role | Verdict |
  |---|---|---|---|
  | VIX (`vix_logret`) | 1.000 | comparable, but... | **Rejected** |
  | 2Y Treasury (`treasury_2y_diff`) | 0.525 | comparable, but... | **Rejected** |
  | 10Y-3mo curve slope | 0.918 | comparable, but... | **Rejected** |
  | **Gold Vol Index (`gvz_logret`)** | **0.0009** | **largest of all 3 betas** | **Accepted** |

  For the first three, sigma_beta alone would have said "include it" —
  McNemar is what actually distinguished real signal from in-sample
  noise that doesn't generalize. GVZ is the one candidate where both
  signals agreed, and it held up nearly identically (p=0.0009, same
  effect size) when re-tested with both models restricted to GVZ's
  actual 2008-06-04 inception, ruling out the missing-data confound as
  the explanation. **The model is now 3-regressor: real rate, USD, GVZ.**
- **The model's confidence is genuinely informative, and got MORE
  informative with GVZ added.** Bucketing forecasts by |z| = forecast /
  Kalman-predicted-std and checking realized hit rate per bucket: the
  2-regressor model climbed from ~57% to ~87% test-set hit rate across 5
  buckets; the 3-regressor model climbs from ~66% to ~89%, with the
  biggest gains at the top end (bucket 3: 71% → 81%, bucket 4: 87% →
  89%) — the buckets that actually drive most of the sizing.
- **Confidence-weighted half-Kelly sizing beat flat sizing at the same
  average exposure**, both before and after adding GVZ. 2-regressor
  holdout: 202.68% vs. 110.78% total return (long/short). 3-regressor,
  **long-only** (the version that actually matters, see below): 124.40%
  vs. flat sizing, Sharpe-like 5.84. The backtest's Sharpe-like ratios
  (~5-7 depending on variant) are implausibly high by real-world
  standards and should be read as "this specific holdout window was
  unusually favorable," not "this beats professional funds."
- **Long-only has a real, measurable cost, not just a theoretical one.**
  Removing shorting (to match a standard cash-account Roth IRA, which
  can't short) dropped the Sharpe-like ratio from 7.43 to 5.84 in the
  3-regressor model at the same Kelly fraction. Doesn't undercut the
  structural case against shorting (a short's loss is unbounded, a
  long's is capped at 100%) — it just means this backtest window never
  hit the scenario where that structural risk bites.
- **Manually excluding low-confidence buckets (0-1) from a long-only
  strategy made things WORSE, not better — confirmed twice, on two
  different models.** 2-regressor: 89.01% (all buckets) vs. 80.54%
  (bucket≥2 only). 3-regressor: 124.40% vs. 113.66%, same pattern. Kelly
  sizing already discounts weak-edge buckets correctly (a thin 58% edge
  gets an appropriately thin ~8-10% stake) — manually zeroing them out
  on top of that doesn't remove risk that wasn't already handled, it
  just throws away real, if modest, positive-expectancy bets. See
  `tvp_sizing_variants.py`.
- **A recurring bug pattern worth naming explicitly: hardcoded regressor
  names left over from the 2-regressor era.** Twice during the GVZ
  rollout, a script crashed or would have silently used stale values
  because a `params` array or beta printout referenced
  `SIGMA_BETA["real_rate_diff"]`/`["usd_logret"]` by name instead of
  looping over the current `REGRESSORS` list. Fixed by building these
  generically (`[SIGMA_BETA[r] for r in REGRESSORS]`) everywhere —
  worth checking for this specific pattern any time a new regressor
  gets added to any of these scripts.
- **Gold's dollar sensitivity was far above what pure currency mechanics
  would predict, in the 2-regressor model.** If gold's USD price moved
  purely because it's dollar-denominated (the "numeraire effect"),
  `beta_usd` should sit near -1 (a 1% dollar move -> ~1% opposite gold
  move, no story beyond currency translation). As of 2026-08-21, the
  fitted `beta_usd` was **-4.19** — about 4.2x the mechanical benchmark,
  at the 99.1st percentile of its own 20-year history (range -1.33x to
  5.03x; the negative low end means the relationship has, at least once,
  actually flipped — gold moving *with* the dollar, plausibly a shared
  flight-to-safety episode). See `tvp_dollar_amplification.py`.

  **This is a point-in-time snapshot, not a permanent structural fact,**
  and it's specifically a 2-regressor-model number — `tvp_dollar_amplification.py`
  hasn't been rerun against the 3-regressor model yet. With GVZ now
  absorbing some of what was previously attributed to `beta_usd`, the
  amplification reading will very likely be smaller once recomputed
  (GVZ and the dollar plausibly share some variance during risk-off
  episodes). Rerunning this script under the 3-regressor model is the
  natural next follow-up, not done here.

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
- Pulls fresh `^GVZ` directly for the gap window. This ISN'T a proxy —
  GVZ has no FRED-style publication lag at all, it just gets caught in
  the archive's trim alongside the dollar (the trim cutoff is driven by
  the slowest CORE series, currently `usd_index`), so real, current GVZ
  data exists on Yahoo Finance even when the archive hasn't caught up.
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

## Honest limitations

- Three signals, one asset. Real systematic strategies lean on
  *breadth* — many weak, partially independent signals — more than on
  a handful being individually strong. GVZ going from "candidate" to
  "included" is a step in that direction, not the end of it; a
  genuinely more complete version of this would be hierarchical,
  pooling partial evidence across many macro/vol factors, not three.
- One train/test split, one historical regime mix (2008-2022 train,
  2022-2026 test, for the 3-regressor model). That test window looks
  like an easy trending period for directional calls — no guarantee
  this generalizes to a choppier regime.
- No transaction costs, no taxes, no capacity constraints modeled in the
  backtests. A cash-account Roth IRA sidesteps the tax question entirely
  (no gains reporting on any holding period, ever), which is a genuinely
  good fit for a daily-rebalanced strategy — but the backtests
  themselves still don't model costs, so treat them as testing the
  *methodology*, not a realized P&L.
- The forward-prediction scripts require an explicit scenario for
  future real-rate/dollar/GVZ moves — there's no way around this, since
  the model's own inputs are same-day changes. A "no view" (flat)
  scenario correctly forecasts exactly zero.
- **When to refit is deliberately NOT on a fixed calendar.** The beta
  *state* updates continuously, by construction, every time new data
  comes in — that's the entire point of a Kalman filter, and it needs
  no manual "update." What's frozen is the *hyperparameters* (the
  hard-coded `sigma_beta`/`sigma_obs` values and calibration buckets
  scattered through the live-forecast scripts) — fit once, on data
  through July 2022. Refitting too often costs real validation data (you
  consume holdout every time you fold more history into training) for a
  process that, based on the two extreme-amplification periods found so
  far (2008, 2026 — eighteen years apart), looks like it moves on a
  multi-year cadence anyway. The honest trigger is the paper-trading
  log's bucket ordering breaking down, with an annual check-in as a
  floor, not a fixed quarterly/annual refit regardless of evidence.

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
| `tvp_new_regressor_test.py` | Tests ONE new candidate regressor at a time against the locked baseline: McNemar's test + sigma_beta check. Change `NEW_REGRESSOR` and rerun for each candidate | ~40-50 min |
| `tvp_dollar_amplification.py` | Gold's USD-sensitivity vs. the mechanical -1 currency-pass-through benchmark, full history + current percentile rank (2-regressor model, not yet updated for 3) | seconds |
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

Output from `tvp_bridge_forecast.py`, 2026-09-01, 3-regressor model,
"trend persists" scenario (bridging through the most recent real/proxy
data at the time):

- **Direction: DOWN**, forecast next-day log-return -0.065%
- **Confidence bucket 0** of 5 (historical hit rate in this bucket: 57.8%)
- **Half-Kelly stake: 7.8% of capital** — on a hypothetical $1,000,
  that's **$78.20 short**

Under the standing long-only rule, a short call at bucket 0 means **no
position taken** — this is a real example of the paper log's most common
entry so far: a genuine forecast that the sizing/direction rules say to
skip, not a strong conviction call. The FLAT ("no view") scenario still
correctly returns exactly $0 staked in every run.

This is a live number, not a fixed result: rerun `fetch_gold_data.py`
then `tvp_bridge_forecast.py` for a current call. Like the dollar-
amplification finding above, treat this as "what the model said on
2026-09-01," not "what the model will always say" — check the paper-
trading log for what it's actually said since.

## Requirements

```
pip install fredapi yfinance pandas numpy statsmodels matplotlib pymc arviz
```

Needs a free FRED API key (`FRED_API_KEY` env var) — see
`fetch_gold_data.py` for details.