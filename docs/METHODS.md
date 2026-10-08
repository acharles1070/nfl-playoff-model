# The statistics behind the model

This is a plain-language tour of every statistical and mathematical method in
the project: what it is, why it is here, the core equation, where it lives in
the code, and what it actually bought me. The last section is a scorecard of
everything I tried, including the ideas that did **not** work.

A theme you will see repeatedly: with ~4,600 games (and only ~200 playoff
games) **sophistication is cheap and signal is expensive.** Each technique was
admitted only if it improved predictions on seasons it had never seen.

---

## 1. How I grade a prediction (the part that keeps me honest)

**Proper scoring rule: log loss.** If you say 70% and the home team wins you
lose `-ln(0.70) = 0.357`; if it loses you lose `-ln(0.30) = 1.204`. Average it
over games. Lower is better. Log loss is a *proper* scoring rule: the only way
to minimise it in the long run is to report your true beliefs, so it cannot be
gamed by being bold or timid. Accuracy ("did the favourite win?") throws away
almost everything and is only a diagnostic here. A coin flip scores 0.693.
`src/models/compare.py`

**Walk-forward validation.** Each test season is predicted by a model that has
only seen *earlier* seasons, exactly as it would have been used in real time.
Random cross-validation would let 2024 help predict 2019.
`src/models/walk_forward.py`

**Dev window / holdout discipline.** Ideas are *selected* on a dev window
(test seasons 2002-2016, ~4,000 games, after extending history back to 1999 on
2026-10-06; before that it was 2011-2016, ~1,600 games). The 2017-2025 holdout
is for confirmation only. The extra history does not make the final model more
accurate (holdout 0.6308 vs 0.6304), but it more than doubles the data available
for choosing between ideas, which is where statistical power was short. Nothing I
report as the headline result was used to choose anything. This is what
protects against the classic trap of reporting numbers from the data you tuned
on (the v1 notebook's 0.61 was that trap, plus look-ahead leakage).

**Paired bootstrap.** With ~2,500 games, is a 0.004 improvement real? I
resample games with replacement 5,000 times, recompute the *difference*
between two models on the same games each time, and read off a 95% interval.
"Paired" matters: both models face identical games, so most of the noise
cancels. If the interval contains 0 I say "not significant". `compare.py`

**Leakage tests (perturbation).** A pregame feature for game G must depend
only on games before G. I *prove* it by rewriting one game's box score and
asserting that no feature at or before that game moves. I confirmed the test
can fail by injecting a deliberate leak. `tests/test_no_leakage.py`

**The market as a yardstick.** The de-vigged closing moneyline: convert each
American price to an implied probability, then rescale the two sides to sum to
1 (removing the bookmaker's margin). It is used only for comparison, never as
an input. `src/models/market.py`

**The encompassing test.** (Caution when reading its output: with two predictors that are
0.89 correlated, the *individual* coefficients are unstable. An earlier version of this
project read the market's coefficient of 1.23 as evidence the market was under-confident;
that was an artifact. Regressing outcomes on the market alone gives a slope of **1.03**
over 5,359 games, i.e. the market is calibrated.) The sharpest question: does my model contain
*anything* the market lacks? Regress outcomes on both log-odds:
`logit P(win) = a + b_market * logit(p_market) + b_model * logit(p_model)`.
If `b_model` is 0 once the market is included, I add nothing.
`src/models/encompassing.py`

---

## 2. Team strength: an online Kalman filter

**Problem.** A team's true strength is invisible. I only see noisy games
(single-game EPA/play has a standard deviation of ~0.18, while the true
spread between teams is only ~0.09). And strength changes: injuries, trades,
a new season.

**Idea.** Keep, for every team, a *belief* (a mean and a variance) about its
offensive and defensive strength, and update that belief after each game in
proportion to how surprising the result was *and* how unsure I was. This is
a Kalman filter: the same maths that tracks a missile or your phone's GPS
position, here tracking a football team.

**The model.** All in EPA-per-play units.

```
home offence EPA/play = mu + O_home + sum_k(share_k * Q_k) + Dv_away + noise
away offence EPA/play = mu + O_away + sum_k(share_k * Q_k) + Dv_home + noise
```

`O` is offence, `Dv` is how much a defence *allows* (lower is better), `Q` is a
quarterback (section 3), `mu` the league mean.

**The update.** For one observation with innovation `z = observed - predicted`
and total uncertainty `S = Var(O) + Var(Dv) + Var(QBs) + noise`:

```
gain for each component  K = Var(component) / S
new belief               mean += K * z
                         Var  *= (1 - K)
```

A team I know well (small variance) barely moves. A team I barely know moves
a lot. Game noise shrinks as plays increase (`noise = obs_var * 62.7 / plays`).

**Between games and seasons.** Variance drifts up a little each game
(`drift_var`: teams change). Across a season break every mean is pulled toward
league average (`carryover`) and variance jumps (`season_var`). That carryover
is the model's *prior*: last year's strength is where I start, and this
year's games gradually overwrite it. Offence, defence and QBs get their own
carryover (year-over-year correlations of ~0.40 offence vs ~0.28 defence, so
defence regresses harder in the raw data).

**Why this is "Bayesian".** The mean/variance pair is exactly a Gaussian
posterior, and the update is Bayes' rule. The season carryover is shrinkage
toward a prior mean (the same idea as ridge regression or empirical Bayes).

**What it bought.** Log loss 0.6541 (season-to-date EPA) to 0.6341 on the
frozen holdout, closing ~43% of the gap to the market. Tuning surface was
almost perfectly flat (every setting within 0.0005), so there is little
overfitting risk from its five knobs. `src/features/ratings.py`

**Timing guarantee.** Teams play at most once a week, so updating a whole
week at once is identical to updating game by game, and every rating used to
predict a game is computed from strictly earlier weeks.

---

## 3. Quarterbacks: credit assignment by uncertainty

**Problem.** A team's offence is partly its QB. When a QB changes teams, or a
backup plays, a team-only model is blind.

**Idea.** Give every QB his *own* latent rating `Q` that follows him between
teams. An offence observation becomes `O_team + sum(dropback_share * Q_qb)`.
Mathematically this is a Kalman filter with a larger state vector. Credit for
a good game is split between the team and the QB **in proportion to their
variances**: the one I know less about absorbs more of the surprise. The
QB's weight is his share of dropbacks, so a 5-dropback cameo barely counts.

The QB who took the first snap is the pregame starter (known at kickoff); his
rating *before* the game is used. A QB's rating carries over seasons at 0.85
(more persistent than team context).

**Sanity check that it learned real football.** Without being told who anyone
is, it ranks 2024 as Lamar Jackson, Josh Allen, Jalen Hurts, Patrick Mahomes
at the top and Drew Lock, Will Levis, Bryce Young at the bottom.

**What it bought.** Holdout 0.6341 to 0.6304 overall. Where it should matter
(a team starts a *different* QB than last game, n=550) it improves by 0.013,
significantly (95% CI [-0.025, -0.002]). **It did not help the postseason**
(+0.011, not significant, n=111): an open question, not assumed.
`src/features/ratings.py`, `src/features/qb.py`

---

## 4. From ratings to a win probability

**Adopted:** logistic regression on two features, the difference in
offensive strength (incl. the starting QB) and the difference in defensive
strength. The intercept *is* the home-field advantage (features are
antisymmetric). Standardised inputs, light L2 regularisation.

**Tested and rejected** (each against the pre-set rule: choose on the tuning
window, report the holdout):

| Idea | Why it should help | Result |
|---|---|---|
| Regress on **point margin**, then `P = Phi(mu/sigma)` | Continuous outcome, ~2x the information per game | -0.0008, not significant |
| **Heteroscedastic** margin model, `sigma_i^2 = s0 + s1 * rating_variance_i` (posterior-predictive: wider when ratings are uncertain) | Uses the filter's own uncertainty | +0.0062 worse on tune, ~0 on holdout |
| A second Kalman filter on **point margin** | Captures kicking/ST/defensive scores EPA ignores | -0.0002, not significant |
| **Rest, bye, neutral site, division, dome** | Known real effects | +0.0007 to +0.0011 (worse) |
| **Garbage-time filter** (only plays with win prob 10-90%) | Decided-game EPA is noisy | +0.002 to +0.003 (worse) |
| **Postseason recalibration** | Suspected underconfidence | Slope 0.67 [0.22, 1.15]: not underconfident |

The lesson: the game model is limited by the **information in the features**,
not by the link function. Fancier maths on the same inputs did not help.

---

## 4b. Is the market itself improvable? (market-input models)

Separate from the fundamentals model, I tested whether the market's own
probabilities can be improved (`src/models/market_models.py`):

- *De-vig method* (multiplicative / additive / power / Shin): log loss differs by
  ~0.0002, so the benchmark is not an artifact of how the margin is removed.
- *Favorite-longshot bias*: the slope of outcomes on logit(price) is **1.03**
  (5,359 games, 2006-2025); bins track the price. No usable bias.
- *Calibrated market* (walk-forward slope/intercept): -0.0001 vs raw (holdout). *Stacked
  market + my model*: +0.0002. Neither beats the closing line.

Conclusion: the NFL closing moneyline is efficient in this sample; models that take it
as input can match it but not beat it.

---

## 5. Injuries: measured, not guessed

The v1 notebook used hand-picked weights ("QB1 out = -0.85"). Here everything
is measured from nflverse data (`src/data/injuries.py`, `src/features/injuries.py`).

**Importance.** A player's importance is his mean snap share over his last 4
games *strictly before* this week, via an as-of join (for each injury row,
look up the most recent rolling value from an earlier week). Wednesday's
report can never see Sunday's snaps.

**Availability.** `P(out | status, position group)`. "Out" means out (0.1% of
Out players took a snap) and "Doubtful" means out (0.7%), but "Questionable"
players play ~65% of the time, and that **drifted** (roughly 0.4 out in
2013-2015 to about 0.33 from 2017 on, after the league dropped "Probable"). So each season uses
rates measured on the *preceding four seasons only*, with small position
groups shrunk toward the overall rate by a beta-binomial prior (strength 50).

**Burden.** `lost[team, week, group] = sum(importance * P(out))`: the expected
number of starter-equivalents missing. Missing *report* is treated as unknown
(imputed), never as "zero injuries".

**Two tests.** (a) Put the home-minus-away burdens straight into the logistic.
(b) A **two-stage estimator** that is statistically more efficient: stage 1
ridge-regresses each team-game's continuous EPA *residual* (actual minus what
ratings predicted) on its own and the opponent's burden (~9,000 observations
instead of ~4,000 win/loss outcomes); stage 2 shifts the ratings and feeds
them to the logistic.

**Result.** Sensible effects (a lost defensive starter is worth about -0.012
EPA/play, ~0.75 points), but the holdout gains (-0.0014, -0.0018) are **not
significant**, so the champion is unchanged. Injuries run as a *challenger*
model in the live ledger.

**Data quirks found on the way:** the snap-count feed has no 2012 rows despite
its documentation, and 2023's playoff rounds have no injury reports.

---

## 6. The bracket simulator

**Goal.** From team strengths to the probability each team wins the
conference and the Super Bowl, round by round.

**Naive approach and its flaw.** Give each game a probability and simulate
brackets. That treats a team's strength as *known*. In truth a team's
strength is uncertain, and that same uncertainty applies to *all* of its
games, so the rounds are correlated.

**Latent strength shocks.** Draw, once per simulation, a shock `eps_t` for each
team on the probit scale (`Var = tau_t^2`, sized from the Kalman posterior).
Given the shocks, game i-vs-j is won by i with

```
p(eps) = Phi( (m_ij + eps_i - eps_j) / s_ij ),     s_ij^2 = 1 - tau_i^2 - tau_j^2
```

where `m_ij = Phi^-1(p_ij)` is the probit index of the fitted model's
single-game probability. **Marginal preservation:** averaging `p(eps)` over
the shocks returns exactly `p_ij`, so every single-game probability stays
calibrated while multi-round outcomes get the correct dependence. (Verified
numerically in `tests/test_bracket.py`.)

**Exact enumeration (Rao-Blackwellisation).** Conditional on the shocks, the
bracket has a finite number of outcomes. Instead of sampling game results I
*enumerate all of them* with the NFL's reseeding rules (1 plays the lowest
surviving seed, etc.), weight each by its probability, and collapse branches
by surviving-team set. The only Monte Carlo noise left is in the shocks. Less
noise for the same computation is the entire point of Rao-Blackwellisation.

**Quasi-Monte Carlo + antithetic variates.** The shocks come from scrambled
**Sobol sequences** (low-discrepancy points that fill space more evenly than
random ones) paired with their negatives (antithetic draws). With 16,384
draws a title probability has a Monte Carlo standard error of about 0.0015.

**Verification.** The exact solver is checked against an *independently
written* brute-force sampler (60,000 full brackets, both the 12-team and
14-team formats), plus checks that probabilities form a proper distribution,
that the shocks have the right variance, and that marginals are preserved.

**Rebuilding historical seeds.** The schedule has no seeds, so they are
reconstructed from structure (byes, mirror pairings 2v7/3v6/4v5) and
**validated by replaying the real results through reseeding** (zero
mismatches across all 17 seasons). Records cannot break every tie (2015:
Denver over New England on head-to-head), so among orderings that reproduce
reality I search for the one most consistent with regular-season records in
every block. This reproduces your hand-entered 2025 bracket exactly.

**Scoring against the market.** For each season and each round I simulate
from the actual survivors and compare with BetMGM's de-vigged futures price at
the same moment, using `-ln P(actual winner)`. 15 seasons, walk-forward.
`src/simulation/`

**Honest result.** The simulator is worse than the futures market in 9 of 10
market-round cells on the 2017-2025 holdout (e.g. prior to the Wild Card,
Super Bowl log score 2.51 vs 2.25; a uniform guess is 2.64). With 9 seasons
this is not statistically resolved, and single seasons dominate (Kansas City
2023 alone accounts for about half of the average gap), but the direction is
consistent.

---

## 6b. The season simulator: title odds from any point in the season

The bracket simulator starts from a known field. The season simulator
(`src/simulation/season.py`) starts from *today*: current records, the
remaining schedule, and ratings as of now.

For each of ~65,000 simulated seasons (one draw of the per-team latent shocks
from section 6, shared by every game that team plays all year):

1. play every remaining regular-season game with `p = Phi((m + eps_i - eps_j) / s)`;
2. resolve divisions and seeds (division winners 1-4, then wild cards) from
   the final records. Ties are broken **at random**, which is the main
   approximation: the real NFL tiebreakers (head-to-head, division record,
   common games, strength of victory...) are far more elaborate;
3. play the playoffs with NFL reseeding and a neutral Super Bowl.

Everything is vectorized across simulations. It is checked three ways: it
agrees with the exact bracket solver on a fixed field in both the 12- and
14-team formats, probabilities obey the identities (title odds sum to 1,
exactly 14 playoff spots, 8 division winners), and single-game marginals are
preserved. The shared shocks matter more here than in the bracket: over 17
games plus playoffs, uncertainty about a team's *true* strength is the dominant
source of spread in its outcomes.

**Backtest.** For each of 15 seasons and weeks 1, 5, 9, 13, 17: rebuild the
league as it stood, simulate, and compare the title odds with the market's
price for the same moment (BetMGM via Covers, weekly prices). Seasons, not
snapshots, are the resampling unit for confidence intervals, because every
snapshot in a season shares one champion.

**Results (15 seasons x weeks 1/5/9/13/17, 2,400 team-snapshots).**
- *Regular-season outcomes are well calibrated.* Predicted P(make playoffs)
  averages 0.400 against an actual 0.400, and every reliability bin tracks at
  every point in the season (e.g. prior to week 5: predicted 0.497 vs actual
  0.490). The schedule simulation, division/seed logic and random tiebreaks are
  sound.
- *Title odds trail the market in every slice* (pooled about +0.25 to +0.34
  log score; 2011-2016 significant, 2017-2025 borderline), and the gap widens
  as the season goes on, i.e. the shortfall is in the playoff stage.
- *Is the title distribution too sharp?* Flattening (p^alpha) helps at the start
  of the playoffs (alpha ~0.5-0.6 chosen on 2011-16 also helps 2017-25) but only
  mildly across all weekly snapshots (best alpha ~0.8). The evidence rests on 15
  champions and one fitted parameter (chi-square ~3.4, p ~0.07): suggestive, not
  established, so it is NOT applied. Scaling the shared shock variance (kappa)
  does not flatten anything, and shrinking single playoff games (beta ~0.9)
  changes playoff log loss by only ~0.0005. The raw title probabilities are
  logged for every team, so flattening can be tested later on forward data.

**Live.** `python -m src.simulation.title_odds --log` writes each team's odds
into `ledger/title_odds.csv`, hash-chained like the prediction ledger, to be
scored against the champion and the market when the season ends.

---

## 7. The forward-test ledger

The only truly out-of-sample evidence is predictions made *before* games.

- **Append-only hash chain.** Each row's SHA-256 covers its own fields plus
  the previous row's hash. Editing, deleting or reordering any row breaks
  every hash after it; recomputing one row's hash still breaks the next row.
  (All tested in `tests/test_ledger.py`.)
- **Provenance.** Each row records the git commit and a hash of the model's
  parameters, refuses to log from a working tree with uncommitted code, and
  never logs a game whose kickoff has passed.
- **Market at logging time.** The comparison line is the price when I
  predicted, not the closing line.
- **Champion / challenger.** The frozen champion and the injury challenger are
  logged side by side; the forward test settles questions the history is too
  small to answer. `src/live/`

---

## 8. Scorecard: everything I tried

All 37 comparisons are in [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) with Holm and
Benjamini-Hochberg corrections: 4 are nominally significant (about 1.9 expected by chance),
and **only the cross-season ratings survive**. The paired per-game log-loss SD is 0.100, so
the 2,494-game holdout can only detect improvements of about 0.0056; confirming 0.002 would
take ~69 seasons.

Holdout 2017-2025 unless noted; log loss, lower is better; "gain" is versus
the model before it; "adopted" requires a paired-bootstrap interval that
excludes 0 where the sample allows, or strong structural justification.

| Idea | Outcome | Verdict |
|---|---|---|
| Cross-season Kalman ratings | 0.6541 to 0.6341 (the only result surviving Holm and BH) | **adopted** |
| Quarterback layer | 0.6341 to 0.6304; QB-change games -0.013 (p=0.025, does NOT survive correction for 34 tests) | **adopted on a structural prior, not on significance** |
| Margin / heteroscedastic game models | within +-0.001 | rejected |
| Margin-based Kalman rating | -0.0002 (n.s.) | rejected |
| Rest / bye / neutral / division / dome | worse | rejected |
| Garbage-time filter | +0.002-0.003 (worse) | rejected |
| Postseason recalibration | slope 0.67, not underconfident | rejected |
| Injury burden (direct) | -0.0014 (n.s.) | **challenger** |
| Injury burden (two-stage) | -0.0018 (n.s.) | **challenger** |
| Re-tuning the filter on 4,000 dev games | same carryover values; -0.0005 (n.s.) | no change |
| Pass/rush split filters | dev +0.0010, holdout 0.0000 | rejected |
| Injured-reserve burden | +0.0011 (worse; already in ratings) | rejected |
| Game-day inactive burden | -0.0018 [-0.0034,-0.0003] pooled 2014-25; not multiplicity-robust; closes ~10% of the market gap | kickoff-time candidate |
| Bracket simulator | validated, but behind the market on 9 seasons | **kept (structurally right)** |
| Season simulator | playoff-qualification calibrated (0.400 vs 0.400); title odds trail market | **kept; forward-tested** |
| Flatten title odds (p^alpha) | alpha 0.5-0.6: -0.10 both eras; ~0.8 over all snapshots; 15 champions, p~0.07 | not applied; test forward |
| Scale shared shock variance (kappa) | no effect | rejected |
| Shrink playoff-game logits (beta) | best ~0.9, -0.0005 | rejected |

**Where I stand.** Holdout log loss 0.6304 against the market's 0.6074: a gap
of 0.023 (95% CI [0.016, 0.030]). I have recovered about half of the original
gap. **The encompassing test says the model adds no detectable information
beyond the closing market** (model coefficient -0.18, CI [-0.42, +0.06],
correlation with the market 0.89). Beating a liquid closing market is
genuinely hard; the realistic value of this system is (a) an independent,
market-free forecast of team strength and bracket odds, (b) a rigorous,
reproducible way to find out whether any new idea helps, and (c) the forward
ledger that will tell me the truth over time.

**Limits worth remembering.** ~111 holdout playoff games and 9 holdout
bracket seasons cannot resolve small effects. I ran several experiments on
the same holdout, so a borderline "significant" result should be treated with
suspicion (multiple comparisons); my pre-set adoption rules guard against
that but do not remove it. The Kalman filter treats ratings as independent
(a diagonal covariance approximation), which is an approximation.
