"""How much persistent team-strength uncertainty does the simulator need?

The shock variance tau^2 comes from the filter's posterior only; strength also
drifts (injuries, rest, lineup changes) between the snapshot and the playoffs.
Scaling tau^2 by kappa changes ONLY how correlated a team's games are, never a
single game's win probability (s^2 = 1 - tau_i^2 - tau_j^2 keeps marginals exact).

kappa is chosen on 2011-2016 and scored on 2017-2025.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import configured_path
from src.features.ratings import RatingParams, build_team_ratings
from src.simulation.backtest import run_season


KAPPAS = (1, 2, 3, 4, 6, 8)


def main() -> None:
    P = configured_path("processed")
    games = pd.read_parquet(P / "games.parquet")
    played = games.loc[games["is_played"]]
    ratings = build_team_ratings(
        pd.read_parquet(P / "team_game_epa.parquet"), played, RatingParams(), pd.read_parquet(P / "qb_game.parquet")
    )
    matchups = pd.read_parquet(P / "historical_matchups_oa.parquet")
    futures = pd.read_parquet(P / "futures_odds.parquet")

    rows = []
    for kappa in KAPPAS:
        for season in range(2011, 2026):
            for r in run_season(season, games, ratings, matchups, futures, tau_scale=kappa):
                rows.append({**r, "kappa": kappa})
        print(f"kappa={kappa} done", flush=True)

    d = pd.DataFrame(rows)
    d["ls_model"] = -np.log(d["p_model_winner"])
    d["ls_market"] = -np.log(d["p_market_winner"])

    sb = d[d["market"].eq("sb")]
    print("\nSuper Bowl market, mean -ln P(champion). Lower is better.")
    print("kappa | " + " | ".join(f"{r} dev / hold" for r in ("WC", "DIV", "CON")) + " | all rounds dev / hold   (market in last row)")
    for kappa, g in sb.groupby("kappa"):
        cells = []
        for rnd in ("WC", "DIV", "CON"):
            h = g[g["round"].eq(rnd)]
            cells.append(f"{h[h.season <= 2016].ls_model.mean():.3f} / {h[h.season >= 2017].ls_model.mean():.3f}")
        cells.append(f"{g[g.season <= 2016].ls_model.mean():.3f} / {g[g.season >= 2017].ls_model.mean():.3f}")
        print(f"{kappa:>5} | " + " | ".join(cells))
    g = sb[sb.kappa.eq(1)]
    cells = []
    for rnd in ("WC", "DIV", "CON"):
        h = g[g["round"].eq(rnd)]
        cells.append(f"{h[h.season <= 2016].ls_market.mean():.3f} / {h[h.season >= 2017].ls_market.mean():.3f}")
    cells.append(f"{g[g.season <= 2016].ls_market.mean():.3f} / {g[g.season >= 2017].ls_market.mean():.3f}")
    print(" mkt  | " + " | ".join(cells))

    d.to_csv(configured_path("outputs") / "tau_experiment.csv", index=False)


if __name__ == "__main__":
    main()
