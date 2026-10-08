"""Score the season simulator against weekly Super Bowl futures, 2011-2025.

For each season S and week w in WEEKS: rebuild the league as of "prior to week w"
(ratings from games before w only, last starters held, game model fit on seasons
< S), simulate the rest of the season and the playoffs, and compare the
simulated title odds with the market's de-vigged price for the same moment.
Also records each team's simulated P(make playoffs) for a calibration check.

Usage:
    python -m src.simulation.season_backtest
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import configured_path
from src.data.schedules import last_complete_season
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.simulation.backtest import FIRST_MODEL_SEASON, GameModel, build_table
from src.simulation.season import SeasonState, simulate_season


WEEKS = (1, 5, 9, 13, 17)
N_SIMS = 32768


def league_state(games: pd.DataFrame, season: int, week: int):
    """Wins to date and the remaining regular-season schedule, as of prior to `week`."""
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    done = reg.loc[reg["week"].lt(week) & reg["is_played"]]

    wins: dict[str, float] = {}
    for g in done.itertuples(index=False):
        h = 1.0 if g.home_score > g.away_score else 0.5 if g.home_score == g.away_score else 0.0
        wins[g.home_team] = wins.get(g.home_team, 0.0) + h
        wins[g.away_team] = wins.get(g.away_team, 0.0) + (1.0 - h)

    teams = set(reg["home_team"]) | set(reg["away_team"])
    wins = {t: wins.get(t, 0.0) for t in teams}

    rest = reg.loc[reg["week"].ge(week)]
    remaining = pd.DataFrame(
        {
            "home": rest["home_team"].to_numpy(),
            "away": rest["away_team"].to_numpy(),
            "neutral": rest["location"].astype(str).str.lower().eq("neutral").to_numpy(),
        }
    )
    return wins, remaining, rest


def snapshot_at(team_game, qb_game, games, season, week, params):
    """Rating state for every team prior to `week` (future games carry projections)."""
    before = games.loc[
        (games["season"].lt(season)) | (games["season"].eq(season) & games["week"].lt(week))
    ]
    before = before.loc[before["is_played"]]
    future = games.loc[games["season"].eq(season) & games["game_type"].eq("REG") & games["week"].ge(week)]

    schedule = pd.concat([before, future], ignore_index=True)
    played_ids = set(before["game_id"])

    ratings = build_team_ratings(
        team_game.loc[team_game["game_id"].isin(played_ids)],
        schedule,
        params,
        qb_game.loc[qb_game["game_id"].isin(played_ids)],
    )

    nxt = ratings.loc[ratings["game_id"].isin(set(future["game_id"]))]
    nxt = nxt.sort_values(["week", "game_id"]).groupby("team").head(1)
    return nxt


def main() -> None:
    processed = configured_path("processed")
    outputs = configured_path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(processed / "games.parquet")
    team_game = pd.read_parquet(processed / "team_game_epa.parquet")
    qb_game = pd.read_parquet(processed / "qb_game.parquet")
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    weekly = pd.read_parquet(processed / "futures_weekly_sb.parquet")

    params = RatingParams()
    played = games.loc[games["is_played"]]
    full_ratings = build_team_ratings(team_game, played, params, qb_game)

    # who actually made the playoffs / won the title
    post = games.loc[games["is_postseason"] & games["is_played"]]
    made = {s: set(d["home_team"]) | set(d["away_team"]) for s, d in post.groupby("season")}
    sb = post.loc[post["game_type"].eq("SB")].copy()
    sb["winner"] = np.where(sb["home_score"] > sb["away_score"], sb["home_team"], sb["away_team"])
    champion = sb.set_index("season")["winner"].to_dict()

    title_rows, playoff_rows = [], []
    started = time.time()

    for season in range(2011, last_complete_season() + 1):
        train = attach_ratings_to_matchups(
            matchups.loc[matchups["season"].between(FIRST_MODEL_SEASON, season - 1)], full_ratings
        )
        model = GameModel(train)

        for week in WEEKS:
            wins, remaining, _ = league_state(games, season, week)
            snap = snapshot_at(team_game, qb_game, games, season, week, params)
            table = build_table(snap, model)

            sim = simulate_season(
                SeasonState(wins, remaining, 7 if season >= 2020 else 6),
                table, n_sims=N_SIMS, seed=season * 100 + week,
            ).set_index("team")

            for team, r in sim.iterrows():
                playoff_rows.append(
                    {"season": season, "week": week, "team": team,
                     "p_playoffs": r["p_playoffs"], "made": float(team in made[season]),
                     "p_win_sb": r["p_win_sb"]}
                )

            mkt = weekly.loc[weekly["season"].eq(season) & weekly["week_prior"].eq(week)].set_index("team")
            common = mkt.index.intersection(sim.index)
            champ = champion[season]

            if champ not in common:
                continue

            p_model = sim.loc[common, "p_win_sb"]
            p_model = p_model / p_model.sum()
            p_market = mkt.loc[common, "p_market"] / mkt.loc[common, "p_market"].sum()
            won = pd.Series(0.0, index=common)
            won[champ] = 1.0

            title_rows.append(
                {"season": season, "week": week, "teams": len(common), "champion": champ,
                 "p_model": float(p_model[champ]), "p_market": float(p_market[champ]),
                 "brier_model": float(((p_model - won) ** 2).sum()),
                 "brier_market": float(((p_market - won) ** 2).sum())}
            )

        print(f"season {season} done ({time.time() - started:.0f}s)", flush=True)

    title = pd.DataFrame(title_rows)
    title["logscore_model"] = -np.log(title["p_model"])
    title["logscore_market"] = -np.log(title["p_market"])
    title.to_csv(outputs / "season_backtest_title.csv", index=False)
    pd.DataFrame(playoff_rows).to_csv(outputs / "season_backtest_playoffs.csv", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
