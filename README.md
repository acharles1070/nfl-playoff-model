# NFL playoff model

A leakage-free, yearly-refreshable pipeline that predicts NFL games and
playoff outcomes, scored honestly against the closing betting market.

**Goal:** make the model better every season, and always know by how much.
The market is a yardstick, never an input.

## Quick start

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.pipeline            # rebuild all tables (~25 s, cached data)
.venv/bin/python -m src.models.benchmark    # every feature set vs the market
.venv/bin/python -m src.models.ratings_experiment   # tune on 2011-16, score 2017-25
.venv/bin/python -m pytest                  # leakage + integrity tests

# live season
.venv/bin/python -m src.live.run --commit   # log next week's pre-game predictions
.venv/bin/python -m src.live.score          # grade the ledger vs the market as games finish
.venv/bin/python -m src.simulation.report --season 2025 --round WC   # bracket odds vs market
.venv/bin/python -m src.simulation.title_odds --log --commit          # current Super Bowl odds + ledger
.venv/bin/python -m src.simulation.title_score                        # score the title ledger (after the season)
```

**Every statistical method, explained in plain language: [`docs/METHODS.md`](docs/METHODS.md).**

Data is downloaded from [nflverse](https://github.com/nflverse) via
`nflreadpy` and cached per season in `data/raw/pbp/` (gitignored).

## How it works

```
nflverse PBP + schedules
   -> team-game EPA                      src/features/epa.py
   -> pregame snapshots (season to date) src/features/snapshots.py
   -> opponent-adjusted EPA              src/features/opponent_adjusted.py
   -> QB-game table (who played, dropbacks) src/features/qb.py
   -> Kalman team + QB ratings           src/features/ratings.py
   -> injury burden (snap share x P(out)) src/features/injuries.py
   -> matchup rows (home vs away diffs)  src/features/*matchups.py, ratings.attach_*
   -> walk-forward logistic model        src/models/walk_forward.py
   -> scored vs closing market           src/models/compare.py, market.py
   -> bracket simulator (exact + QMC)    src/simulation/
   -> hash-chained forward ledger        src/live/
```

Ratings model: each team has an offense and a defense strength (EPA/play) and
each quarterback has his own strength that travels with him between teams.
A game updates all of them by the Kalman gain; between seasons ratings regress
toward average and uncertainty resets upward, so last year is a prior that
this year's games overwrite. The pregame QB is the one who took the first snap.

## Rules that keep the numbers honest

1. **No leakage.** A feature for game G may use only games before G.
   `tests/test_no_leakage.py` rewrites a game's box score and asserts nothing
   at or before it moves. Every new feature family needs such a test.
2. **Postseason = `game_type` in {WC, DIV, CON, SB}.** Never `week >= 18`.
3. **Walk-forward only.** Each test season is predicted from earlier seasons.
4. **One common population** when comparing models.
5. **Log loss is the primary metric**; Brier, AUC, calibration are secondary.
6. **Tune on an early window, report on later seasons** (tune 2011-2016, report 2017-2025). The holdout
   was examined for every candidate idea, so results are multiple-testing corrected and the main
   survivor is replicated on 2003-2009 (see `docs/EXPERIMENT_REGISTRY.md`).
7. **Report uncertainty.** `compare.py` gives paired-bootstrap intervals; with
   ~110 holdout postseason games, small gaps are noise.
8. **The market is comparison only.** A market-input model, if built, is a
   separate model and is never the main one.

## Where things stand

Holdout 2017-2025, parameters frozen before scoring (2,494 games):

| model | log loss | gap to market (95% CI) |
|---|---|---|
| home field only | 0.6899 | 0.083 |
| season-to-date EPA (v0 baseline) | 0.6541 | 0.047 [0.037, 0.057] |
| Kalman ratings, no QB | 0.6341 | 0.027 [0.019, 0.035] |
| **Kalman ratings + QB layer** | **0.6304** | **0.023 [0.016, 0.030]** |
| closing market (de-vigged moneyline) | 0.6074 | - |

QB layer vs no-QB (paired bootstrap): games where a team changes starting QB
(550): -0.013 log loss (p=0.025); regular season: -0.004; all games: -0.004. These do
NOT survive correction for the 34 experiments run (see `docs/EXPERIMENT_REGISTRY.md`):
the QB layer is kept because the mechanism is sound, not because it is proven; **postseason (111 games): +0.011, not significant
but pointing the wrong way. Open question, do not assume it helps playoffs.**

**Does the model know anything the market does not?** Encompassing test
(2017-2025, n=2,494): model coefficient **-0.18 [-0.42, +0.06]**, correlation
with the market 0.89. No detectable incremental information. See METHODS.md.

Bracket simulator vs BetMGM futures (15 seasons, walk-forward): worse than the
market in 9 of 10 market-round cells on the 2017-2025 holdout (Super Bowl log
score prior to the Wild Card 2.51 vs 2.25; uniform = 2.64). Not statistically
resolved at 9 seasons, but consistent. Seeds are reconstructed and validated
against all 17 real brackets.

### Tried and rejected (all tested on the frozen holdout)
margin-regression and heteroscedastic game models, a point-margin Kalman
rating, rest/bye/neutral/division/dome context, garbage-time filtering,
postseason recalibration. Injury burden (direct and two-stage) gains -0.0014 /
-0.0018, not significant: run as a forward-tested **challenger**.

### Could it make money? (`docs/TRADING_SIMULATION.md`)
No. Betting the model's disagreements with the closing moneyline lost 9.0% per bet over
2,054 holdout bets (95% CI [-14.5%, -3.2%]), no better than blindly betting favorites (-2.8%; the 6-point gap to the model's picks is not significant);
quarter-Kelly staking is ruined; Deflated Sharpe 0.00. The market is calibrated (slope 1.03)
and the model's information is a subset of the market's.

### Experiment registry (`docs/EXPERIMENT_REGISTRY.md`)
37 out-of-sample comparisons, Holm and Benjamini-Hochberg corrected: only the cross-season
rating filter survives. The 2,494-game holdout can only detect improvements above ~0.0056
log loss; confirming a 0.002 gain would take ~69 NFL seasons.

### Season simulator and title odds
Simulates the remaining schedule, divisions, seeds and playoffs from today.
Playoff-qualification probabilities are well calibrated (2,400 team-snapshots,
predicted 0.400 vs actual 0.400). Title odds trail the market's weekly prices
(+0.25 to +0.34 log score pooled); weekly odds are logged to
`ledger/title_odds.csv` for a forward test.

### Live forward test
`ledger/predictions.csv` holds pre-game predictions, hash-chained and
committed. Champion `ratings_qb_logit_v1` is frozen; challenger
`ratings_qb_injury_logit_v1` should be logged after Friday injury reports.

## Roadmap

- [x] History back to 2009, cached loader, one-command pipeline
- [x] Market benchmark + bootstrap comparison
- [x] Cross-season team ratings
- [x] Quarterback layer (QB ratings that travel between teams)
- [x] Historical playoff futures odds collected (comparison target for the simulator)
- [x] Automated injuries (measured availability, snap-share burden); challenger model
- [ ] Manual override file for game-week news (QB status, weather forecast)
- [x] Rest / bye / neutral tested: no gain (weather, travel not yet tested)
- [x] Bracket simulator scored against historical futures
- [x] Season simulator (random tiebreaks) + weekly title-odds ledger
- [ ] Real NFL tiebreakers in the season simulator
- [x] Append-only pre-game prediction ledger (live 2026 season)
- [x] Trading simulation and experiment registry (multiple testing, power)
- [x] Weekly logging scheduler (macOS LaunchAgent)
- [ ] Yearly retrain loop and model changelog
- [ ] Public website, Quarto paper, repo publishing prep

## Layout

```
config.yaml        seasons, paths, postseason game types
src/data/          nflverse ingestion (schedules, play-by-play cache), futures odds
src/features/      everything computed from data, pregame-safe
src/models/        walk-forward, market benchmark, experiments
src/simulation/    seeding, bracket simulator, futures backtest
src/live/          prediction ledger, weekly logging, scoring
docs/METHODS.md    the statistics, explained
tests/             leakage and integrity tests
notebooks/archive/ the original v1 notebook (reference only)
```

## Website

An interactive, beginner-friendly walkthrough lives in `site/` (plain HTML/CSS/JS, no build step).

```
python -m src.report.site_data              # rebuild site/data/*.json from the pipeline outputs
python -m http.server 8000 --directory site # preview at http://localhost:8000
```

`.github/workflows/pages.yml` deploys it to GitHub Pages (Settings -> Pages -> Source: GitHub
Actions). Set `REPO_URL` in `site/assets/app.js` to link the repository from the page.
Betting prices on the site come from nflverse's schedule file, never from the Covers futures scrape.

## Paper

`paper/paper.qmd` is a Quarto source for the write-up (PDF and HTML). Every number comes from the
pipeline through `paper/_variables.yml`; figures and tables are generated, not drawn.

```
python -m src.report.site_data      # data the site and paper share
python -m src.report.paper_assets   # figures, tables, variables
quarto render paper                 # HTML + PDF; needs Quarto (https://quarto.org), the PDF uses its bundled Typst (no LaTeX)
```

`tests/test_paper.py` checks (without Quarto) that every variable, cross-reference, include,
figure and citation resolves.

## Live bracket

`bracket/` is a second, standalone site: the playoff bracket for the **2026 season (played in January 2027)**, rebuilt every
morning. Until the playoff schedule exists it is a projection (each team's chance at every seed, the lineup that best matches
those odds, the bracket the favorites would produce, title odds); once the Wild Card matchups are set it shows the real field,
finished scores, and odds re-simulated from the teams still alive.

```
python -m src.simulation.live_bracket    # writes bracket/data/bracket.json
python -m src.simulation.bracket_replay  # replay of the January 2026 playoffs -> site/data/bracket_replay.json
python -m http.server 8000 --directory bracket
```

`.github/workflows/pages.yml` deploys both sites together (explainer at `/`, bracket at `/bracket/`) on every push and every
morning from September to February: it downloads fresh nflverse data, rebuilds the ratings, runs the 65,536-season simulation
and publishes. If the refresh fails it deploys the committed snapshot instead. Regular-season numbers equal the title odds in
`ledger/title_odds.csv`. **Each August set `seasons.live` and `seasons.end` in `config.yaml` to the new season.**

The replay rewinds to every week of the 2025 season and scores the projections against what happened: the projected field
versus a standings baseline, the bracket picked before kickoff versus a higher-seed rule, and each game versus the closing
line. It covers 13 games, so it describes one bracket; it is not evidence of skill (`docs/CHANGELOG.md`).
