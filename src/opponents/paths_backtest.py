"""Who will we play? Playoff-opponent forecasts scored against what happened.

    python -m src.opponents.paths_backtest

For each completed season and several as-of points, simulate the rest of the season with only the games before that
week (src/opponents/paths.py), then look at the real postseason: for every team that played in a round, how much
probability did the forecast put on the opponent it really drew? Compared with the simplest planner, the standings
as they stood that week (the field if the season ended today, higher seed advances).

Conditional on reaching the round: a team that lost earlier never appears in later rounds, so the question is only
"given you are here, who is across from you". There are about 13 playoff games a year, so this is descriptive.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import configured_path, project_path
from src.data.schedules import last_complete_season
from src.opponents.paths import ROUNDS, meeting_matrices, n_seeds_for, season_detail
from src.simulation.bracket import reseed_pairs, wild_card_pairs
from src.simulation.bracket_replay import standings_field
from src.simulation.live_bracket import load_inputs

FIRST_SEASON = 2011
DEV_LAST = 2016
N_SIMS = 16384
AS_OF = ("6", "10", "14", "end")        # weeks before which the forecast is made; "end" = after the last regular-season week


def chalk_matchups(field: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """The planner's bracket: higher seed wins everything. Returns round -> {team: opponent}."""
    out: dict[str, dict[str, str]] = {r: {} for r in ROUNDS}
    champs = []
    for seeds in field.values():
        seed_of = {t: i + 1 for i, t in enumerate(seeds)}
        byes = 1 if len(seeds) == 7 else 2
        alive = set(seeds[:byes])
        for h, a in wild_card_pairs(seeds):
            out["WC"][h], out["WC"][a] = a, h
            alive.add(h)
        pairs = reseed_pairs(frozenset(alive), seed_of)
        finalists = []
        for h, a in pairs:
            out["DIV"][h], out["DIV"][a] = a, h
            finalists.append(min((h, a), key=seed_of.get))
        out["CON"][finalists[0]], out["CON"][finalists[1]] = finalists[1], finalists[0]
        champs.append(min(finalists, key=seed_of.get))
    if len(champs) == 2:
        out["SB"][champs[0]], out["SB"][champs[1]] = champs[1], champs[0]
    return out


def season_scores(inp, season: int) -> list[dict]:
    games = inp.games
    post = games.loc[games["season"].eq(season) & games["is_postseason"] & games["is_played"]]
    reg = games.loc[games["season"].eq(season) & games["game_type"].eq("REG")]
    last_week = int(reg["week"].max())
    rows = []
    for tag in AS_OF:
        week = last_week + 1 if tag == "end" else int(tag)
        _, detail, _, _ = season_detail(inp, season, week, N_SIMS)
        teams = detail["teams"]
        idx = {t: i for i, t in enumerate(teams)}
        field = standings_field(games, season, week)
        field = {c: s[: n_seeds_for(season)] for c, s in field.items()}
        chalk = chalk_matchups(field)
        for rnd in ROUNDS:
            p_meet, _ = meeting_matrices(detail, rnd)
            reach = p_meet.sum(axis=1)
            for g in post.loc[post["game_type"].eq(rnd)].itertuples(index=False):
                for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                    i, j = idx[team], idx[opp]
                    q = p_meet[i] / reach[i] if reach[i] > 0 else np.zeros(len(teams))
                    order = np.argsort(-q)
                    rows.append({
                        "season": season, "as_of": tag, "week": week, "round": rnd, "team": team, "opponent": opp,
                        "p_actual": float(q[j]), "p_reach": float(reach[i]), "rank": int(np.flatnonzero(order == j)[0]) + 1,
                        "model_top1": bool(order[0] == j), "model_top3": bool(j in order[:3]),
                        "standings_top1": chalk[rnd].get(team) == opp,
                    })
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.assign(window=np.where(rows["season"].le(DEV_LAST), "dev", "holdout"))
    out = []
    for window in ("dev", "holdout", "all"):
        w = rows if window == "all" else rows.loc[rows["window"].eq(window)]
        for tag in AS_OF:
            for rnd in (*ROUNDS, "ALL"):
                g = w.loc[w["as_of"].eq(tag)] if rnd == "ALL" else w.loc[w["as_of"].eq(tag) & w["round"].eq(rnd)]
                if g.empty:
                    continue
                out.append({"window": window, "as_of": tag, "round": rnd, "n_team_rounds": len(g), "n_seasons": int(g["season"].nunique()),
                            "model_top1": float(g["model_top1"].mean()), "model_top3": float(g["model_top3"].mean()),
                            "model_mean_p": float(g["p_actual"].mean()), "standings_top1": float(g["standings_top1"].mean())})
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.time()
    inp = load_inputs()
    print(f"playoff-opponent backtest: seasons {FIRST_SEASON}-{last_complete_season()}, as-of {AS_OF}, {N_SIMS:,} sims")
    rows = []
    for s in range(FIRST_SEASON, last_complete_season() + 1):
        t = time.time()
        rows += season_scores(inp, s)
        print(f"  {s} ({time.time() - t:.0f}s)", flush=True)
    rows = pd.DataFrame(rows)
    summary = summarize(rows)
    rows.to_parquet(configured_path("outputs") / "opponent_paths_games.parquet", index=False)
    summary.round(5).to_csv(project_path("docs", "opponent_paths.csv"), index=False)
    show = summary.loc[summary["window"].isin(["all"]) & summary["round"].eq("ALL")]
    print(show.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
