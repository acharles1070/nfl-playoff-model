"""Print the simulator's forecast next to the market for one season and round.

Usage:
    python -m src.simulation.report --season 2025 --round WC
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import configured_path
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.simulation.backtest import (
    FIRST_MODEL_SEASON, GameModel, alive_by_round, build_table, round_snapshot,
)
from src.simulation.bracket import simulate_bracket
from src.simulation.seeding import reconstruct_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", default="WC", choices=["WC", "DIV", "CON"])
    args = parser.parse_args()

    processed = configured_path("processed")
    games = pd.read_parquet(processed / "games.parquet")
    played = games.loc[games["is_played"]]
    ratings = build_team_ratings(
        pd.read_parquet(processed / "team_game_epa.parquet"), played, RatingParams(),
        pd.read_parquet(processed / "qb_game.parquet"),
    )
    matchups = pd.read_parquet(processed / "historical_matchups_oa.parquet")
    futures = pd.read_parquet(processed / "futures_odds.parquet")

    train = attach_ratings_to_matchups(
        matchups.loc[matchups["season"].between(FIRST_MODEL_SEASON, args.season - 1)], ratings
    )
    model = GameModel(train)

    post = games.loc[games["season"].eq(args.season) & games["is_postseason"] & games["is_played"]]
    fields = reconstruct_field(games, args.season)
    alive = alive_by_round(post, fields)

    if args.round == "WC":
        teams = [t for s in fields.values() for t in s]
        use_fields, use_alive = fields, None
    else:
        use_alive = alive[args.round]
        teams = [t for v in use_alive.values() for t in v]
        use_fields = {l: [t for t in fields[l] if t in teams] for l in fields}

    table = build_table(round_snapshot(ratings, post, args.round, teams), model)
    sim = simulate_bracket(use_fields, table, n_draws=16384, seed=args.season, start=args.round, alive=use_alive)

    mkt = futures.loc[
        futures["season"].eq(args.season) & futures["market"].eq("sb") & futures["round_prior"].eq(args.round)
    ].set_index("team")

    out = sim.set_index("team")[["conference", "seed", "p_reach_con", "p_win_conf", "p_win_sb", "sb_mc_se"]]
    out["market_sb"] = mkt["p_market"]
    out["market_odds"] = mkt["american_odds"]
    out["won"] = mkt["won_market"].map({True: "CHAMPION", False: ""})
    out = out.sort_values("p_win_sb", ascending=False)

    print(f"{args.season} Super Bowl odds prior to the {args.round} round "
          f"(seeds reconstructed; model fit on seasons < {args.season}; market = BetMGM, de-vigged)\n")
    print(out.round(3).to_string())
    print(f"\nMonte Carlo standard error on a title probability is about {out['sb_mc_se'].max():.4f} at most.")


if __name__ == "__main__":
    main()
