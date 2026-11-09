# Opponent forecasting

A planning layer on top of the rating model: for any team, how strong will each remaining opponent be
*when the game is played*, how far will that forecast move, what would move it, and whom might the team meet in
the playoffs. The write-up, with the evidence, is `paper/opponent_forecasting.qmd`.

It uses public data only (nflverse), no betting-market input, and the pinned forward-test champion
`ratings_qb_logit_v1` unchanged. It is independent work, not affiliated with or endorsed by any NFL club.

## Run it

```bash
python -m src.pipeline                       # refresh the data tables (live season included)
python -m src.opponents.report --team WAS    # any team abbreviation; --season and --week replay a past date
```

This writes a JSON file, a Markdown brief and a CSV of the games to `outputs/opponents/`. A worked example is
checked in at [`docs/examples/WAS_2026_w10.md`](examples/WAS_2026_w10.md) (the format matters, the numbers change weekly).

The report is built from games before the as-of week only, where the as-of week is the week after the last one
with a result. A game from an earlier week that has not been played yet (Monday night) stays on the schedule.

## What is in a report

| Section | Meaning |
|:--|:--|
| Where the team stands | Overall rating in points per game against an average team (EPA per play x 62.7), league rank, offense / defense / quarterback split |
| Remaining schedule | Per game: win probability, an 80% *revision band*, the opponent's rating and rank, its projected quarterback, and the probability if either quarterback is replaced |
| Schedule strength | Mean rating of the remaining opponents and its league rank |
| Outlook | Expected final wins and an 80% range, playoff / division / seed / title odds (the logged title simulator, same seed rule) |
| Playoff path | For each round, the chance of playing it, and given that, the most likely opponents and the team's win chance against each |

## How to read the numbers

* **Win %** is pulled toward 50-50 for games further away: the game model was fitted on ratings current at
  kickoff and is overconfident on ratings frozen weeks earlier. The shrinkage `logit / (1 + a h)` has one parameter,
  fitted on 2011-2016 and frozen (`HORIZON_SHRINK_A` in `src/opponents/forecast.py`); it is experiment E39.
* **80% band** is where that probability will probably stand when the game arrives, from how far forecasts moved in
  past seasons at that distance (`docs/opponent_revision_sd.csv`). It is not a confidence interval for the true
  probability. When a starting quarterback changes, forecasts move much more, so the JSON also carries the wider band
  that applies in that case.
* **If QB out** replaces that team's starter with a first-start replacement (the mean rating of quarterbacks in
  their first three career starts, from earlier seasons only) and changes nothing else.
* Rest, travel, weather and injuries beyond the starting quarterback are not in the model.

## Studies behind it

| Command | Question | Tracked output |
|:--|:--|:--|
| `python -m src.opponents.horizon` | How does skill decay with weeks between forecast and game? Fit the shrinkage and the revision bands on 2011-2016, score 2017-2025 | `docs/opponent_horizon.csv`, `docs/opponent_revision_sd.csv`, `docs/opponent_shrink.json` |
| `python -m src.opponents.paths_backtest` | Whom does a team really meet in each playoff round, versus the forecast and versus the standings? | `docs/opponent_paths.csv` |

Both take a few minutes. The per-forecast detail goes to `outputs/` (not tracked).

## Limits

The opponent-path study is descriptive (about 350 team-rounds). Tiebreaking in the simulator is random, which
limits matchup accuracy even at the end of the regular season. The holdout (2017-2025) is not sealed. Nothing here
has been compared with any club's internal models, and none of it is betting advice.
