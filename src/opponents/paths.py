"""Playoff paths: who a team is likely to face, round by round.

The season simulator (src/simulation/season.py) already plays the rest of the season and the whole bracket with NFL
reseeding, so every simulated season contains a full set of playoff matchups. `detail=` makes it count them. From
those counts:

    P(meet)      probability that teams i and j play each other in round R (hosting either way)
    P(reach)     probability that team i plays in round R at all  (sum over opponents)
    P(opp | i)   who i faces in R, given that it gets there       (P(meet) / P(reach))
    P(win | meet) how often i wins that game, in the simulations where it happens

Every simulated season shares one set of latent strength shocks across a team's games, so an opponent that is
strong in one round is strong in all of them (see src/simulation/bracket.py). The numbers use the same
65,536-simulation, seed = season * 100 + week rule as the logged title odds, so they tie out to ledger/title_odds.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.backtest import build_table
from src.simulation.live_bracket import Inputs, fit_model, latest_snapshot
from src.simulation.season import SeasonState, simulate_season
from src.teams import TEAM_NAMES

ROUNDS = ("WC", "DIV", "CON", "SB")


def n_seeds_for(season: int) -> int:
    """The playoff field grew from 6 to 7 teams per conference in 2020."""
    return 7 if season >= 2020 else 6


def state_before(games: pd.DataFrame, season: int, week: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Wins so far and the regular-season games still to play, as of before `week`.

    Same as src.simulation.season_backtest.league_state, except that a game from an EARLIER week that has not been
    played yet (a Monday night game when the next report is already due) stays in the schedule instead of vanishing.
    For a finished season every game is played, so the two agree.
    """
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    done = reg.loc[reg["week"].lt(week) & reg["is_played"]]
    wins: dict[str, float] = {}
    for g in done.itertuples(index=False):
        h = 1.0 if g.home_score > g.away_score else 0.5 if g.home_score == g.away_score else 0.0
        wins[g.home_team] = wins.get(g.home_team, 0.0) + h
        wins[g.away_team] = wins.get(g.away_team, 0.0) + (1.0 - h)
    wins = {t: wins.get(t, 0.0) for t in set(reg["home_team"]) | set(reg["away_team"])}
    rest = reg.loc[reg["week"].ge(week) | ~reg["is_played"]]
    remaining = pd.DataFrame({"home": rest["home_team"].to_numpy(), "away": rest["away_team"].to_numpy(),
                              "neutral": rest["location"].astype(str).str.lower().eq("neutral").to_numpy()})
    return wins, remaining


def season_detail(inp: Inputs, season: int, week: int, n_sims: int = 65536):
    """Simulate the rest of `season` as of before `week`, keeping who-plays-whom.

    Returns (summary table, detail dict, fitted game model, latest rating snapshot). Same inputs and seed rule as
    live_bracket.season_odds, so the summary matches the logged title odds when n_sims and week agree (and no earlier
    game is still unplayed).
    """
    model = fit_model(inp, season, week)
    latest = latest_snapshot(inp, season, week)
    wins, remaining = state_before(inp.games, season, week)
    table = build_table(latest, model)
    detail: dict = {}
    sim = simulate_season(SeasonState(wins, remaining, n_seeds_for(season)), table, n_sims=n_sims, seed=season * 100 + week,
                          seed_probs=True, detail=detail)
    return sim, detail, model, latest


def meeting_matrices(detail: dict, rnd: str) -> tuple[np.ndarray, np.ndarray]:
    """(P(meet), P(row team beats column team | meet)); both n x n over detail['teams']."""
    hosted, host_won = detail["hosted"][rnd], detail["host_won"][rnd]
    meet_count = hosted + hosted.T
    i_wins = host_won + (hosted.T - host_won.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_win = np.where(meet_count > 0, i_wins / meet_count, np.nan)
    return meet_count / detail["n_sims"], p_win


def reach_probability(detail: dict, rnd: str) -> np.ndarray:
    p_meet, _ = meeting_matrices(detail, rnd)
    return p_meet.sum(axis=1)


def opponents(detail: dict, team: str, rnd: str, top: int | None = None) -> pd.DataFrame:
    """Opponent table for one team and round, most likely first."""
    teams = detail["teams"]
    i = teams.index(team)
    p_meet, p_win = meeting_matrices(detail, rnd)
    reach = float(p_meet[i].sum())
    rows = pd.DataFrame({
        "opponent": teams, "name": [TEAM_NAMES[t] for t in teams], "p_meet": p_meet[i],
        "p_given_reach": p_meet[i] / reach if reach > 0 else np.nan, "p_win_if_meet": p_win[i],
    })
    rows = rows.loc[rows["p_meet"] > 0].sort_values("p_meet", ascending=False)
    return rows.head(top) if top else rows


def wins_distribution(detail: dict, team: str) -> pd.DataFrame:
    """Probability of finishing with each win total (ties count half a win)."""
    i = detail["teams"].index(team)
    counts = detail["wins_x2"][i]
    return pd.DataFrame({"wins": np.arange(len(counts)) / 2.0, "p": counts / detail["n_sims"]}).query("p > 0").reset_index(drop=True)
