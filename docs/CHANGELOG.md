# Changelog and errata

Model definitions change only by creating a NEW model id (see CLAUDE.md). Bug fixes that move
outputs are recorded here with their measured effect.

## 2026-10-07: neutral-site probabilities fixed (title simulator `season_sim_v1` -> `season_sim_v2`)

**Bug.** `build_table` obtained neutral-site game logits by subtracting the logistic model's
intercept from its home-game logit. The model standardizes its inputs first, so in raw-feature
space the home-field constant is `intercept - sum(coef * mean / scale)` (0.2448 on the 2025
fit, not 0.2599). Every neutral-site logit was therefore shifted by about -0.015 (about 0.4
percentage points at 50%) against whichever team was listed first, so neutral-site odds were
not symmetric. Neutral sites are only the Super Bowl in the simulators.

**Unaffected.** Single-game probabilities (`ratings_qb_logit_v1`, the forward-test champion),
all walk-forward model comparisons, and home-game probabilities in the simulators.

**Effect (bracket backtest, holdout 2017-2025).** Log scores moved by 0.001 to 0.003; the
simulator still trails the futures market in 9 of 10 cells; the Super Bowl-round cell is
-0.0186 (was -0.0185). Conclusions are unchanged.

**Fix.** `GameModel.home_logit_constant`; regression tests in `tests/test_game_model.py`
(symmetry holds with non-zero feature means; verified to fail on the old formula).
The title ledger now logs `season_sim_v2`; the week-5 rows already logged under `season_sim_v1`
stay in the ledger as logged, and `src.simulation.title_score` scores each model id separately.

## 2026-10-07: protocol claims corrected; E01 replication; leakage tests extended

**Protocol wording.** Earlier text called the 2017-2025 holdout "sealed" and said ideas were chosen on the
2002-2016 development window. The registry shows otherwise: every comparison is scored on 2017-2025 (31),
2014-2025 (6) or 2023-2025 (1), and the Kalman model, QB layer and about 20 early experiments were built on
2026-09-30, before the dev/holdout protocol (2026-10-06) existed. Hyperparameters were tuned on 2011-2016
only. The site, paper, README and drafts now say this, and the multiple-testing correction is applied to
those looks.

**E01 replication (pre-specified 2026-10-07).** On test seasons 2003-2009, which played no part in the
design or tuning, Kalman ratings beat season-to-date EPA by -0.0076 [-0.0158, +0.0004] over 1,869 games:
same sign, about half the holdout size, and the interval just touches zero, so the pre-specified criterion
(interval excluding zero) is NOT met. Dev window 2002-2016: -0.0146 [-0.0204, -0.0089]; holdout under the
2000-onward harness: -0.0175 (E01 in the registry, 2010 training start, is -0.0200). `src.models.replication`.

**Training-start definitions.** The live champion and the simulators train from 2010; the experiment harness
(since 2026-10-06), betting simulation and replication train from 2000. Scorecard numbers use the saved
2010-start predictions (they match the pinned live champion); on the holdout the champion's log loss is
0.6304 (2010 start) vs 0.6308 (2000 start).

**Leakage tests.** Added perturbation tests for the schedule-context, matchup and opponent-adjusted
matchup builders (`tests/test_no_leakage_joins.py`; verified to catch an injected leak). Every pregame
feature builder now has one; the `*_build.py` files are thin wrappers over tested functions.

## 2026-10-08: live bracket, bracket replay, and two leaks caught while building them

**New.** `src/simulation/live_bracket.py` (projected or real bracket for any season, phase-aware),
`src/simulation/bracket_replay.py` (week-by-week replay of a finished season scored against the real bracket), a standalone
site in `bracket/`, a replay chapter on the explainer, a paper section, and a workflow that rebuilds the bracket daily.
`simulate_season(seed_probs=True)` adds p_seed_1..7 and leaves the default output (and so the title ledger) unchanged;
`python -m src.pipeline --stages ...` runs only the stages the bracket needs.

**Two leaks found by sanity-checking the first replay (the eventual champion looked like the clear favorite):**
1. In a finished season every game is flagged played, so "are regular-season games left?" was always false and every weekly
   snapshot used a dummy final week that included all games, playoffs included. Now the test is whether schedule rows exist
   from that week on, and the dummy week respects the as-of week.
2. For the same reason, pre-playoff ratings (and the hypothetical later-round matchups in the pre-kickoff bracket) included
   playoff results. Fixed by the same change.
Guards: `tests/test_live_bracket.py` (pre-playoff ratings identical with or without the playoffs in the schedule; replay inputs
cannot see later rounds; field reconstruction matches the validated one for 2020-2025). The corrected replay agrees with the
independent multi-season backtest to three decimals at every comparable checkpoint. The live site is unaffected (nothing
exists in the future of a live season) and equals the logged `season_sim_v2` odds exactly.

**Replay result (2025 season, January 2026 playoffs), descriptive only.** Projected field: the model had more of the 14 eventual
playoff teams than a naive standings baseline in 2 of 19 projections, fewer in 11, level in 6. Games: model 9/13 winners, closing
line 10/13, higher seed 6/13; log loss 0.577 vs 0.557. 2025 was a hard year for qualification forecasts (0.73 log loss at week 1
vs 0.63 averaged over 15 seasons).
