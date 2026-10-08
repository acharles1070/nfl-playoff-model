"""Does filtering garbage time (win prob < 10% or > 90%) sharpen the ratings?

The rating filter is unchanged; only its EPA observations are recomputed from
competitive plays. Pre-registered rule: adopt only if the 2017-2025 holdout
improvement's paired-bootstrap 95% interval excludes zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.pbp import load_pbp
from src.features.epa import prepare_scrimmage_plays
from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import (
    HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, TUNE_SEASONS, Context,
)
from src.models.walk_forward import run_walk_forward


def competitive_offense(threshold_lo: float, threshold_hi: float, hard: bool = True) -> pd.DataFrame:
    """Per team-game offensive EPA and play weight using only competitive plays.

    hard=True keeps plays with wp in [lo, hi]; hard=False weights every play by
    4*wp*(1-wp) (1 at 50/50, falling to 0 as the game is decided). The filter
    only consumes offense EPA + play counts (defense = the opponent's offense).
    """
    plays = prepare_scrimmage_plays(load_pbp(regular_season_only=False))
    plays = plays.loc[plays["wp"].notna()].copy()

    if hard:
        plays["w"] = plays["wp"].between(threshold_lo, threshold_hi).astype(float)
    else:
        plays["w"] = 4.0 * plays["wp"] * (1.0 - plays["wp"])

    plays["we"] = plays["w"] * plays["epa"]

    g = plays.groupby(["game_id", "posteam"]).agg(we=("we", "sum"), w=("w", "sum")).reset_index()
    g = g.rename(columns={"posteam": "team"})
    g["c_off_epa"] = g["we"] / g["w"].replace(0, np.nan)
    g["c_off_plays"] = g["w"]
    return g[["game_id", "team", "c_off_epa", "c_off_plays"]]


def main() -> None:
    ctx = Context()
    base_tg = ctx.team_game

    variants = {"current (all plays)": base_tg}

    for label, (lo, hi, hard) in {
        "wp in [0.10, 0.90]": (0.10, 0.90, True),
        "wp in [0.05, 0.95]": (0.05, 0.95, True),
        "leverage-weighted 4*wp*(1-wp)": (0, 1, False),
    }.items():
        tg = base_tg.merge(competitive_offense(lo, hi, hard), on=["game_id", "team"], how="left")

        # a game with (almost) no competitive plays falls back to all plays
        usable = tg["c_off_plays"].gt(5)
        tg["off_epa_per_play"] = np.where(usable, tg["c_off_epa"], tg["off_epa_per_play"])
        tg["off_plays"] = np.where(usable, tg["c_off_plays"], tg["off_plays"])
        variants[label] = tg

    preds = {}
    for name, tg in variants.items():
        ratings = build_team_ratings(tg, ctx.played, RatingParams(), ctx.qb_game)
        data = attach_ratings_to_matchups(ctx.matchups, ratings)
        data = data.loc[data["season"].between(2010, 2025)]
        preds[name] = run_walk_forward(data, feature_set_name=name, feature_columns=PRIMARY_FEATURES)

    ref = preds["current (all plays)"].set_index("game_id")

    def ll(pred, seasons):
        p = pred.loc[pred["season"].isin(list(seasons))]
        return per_game_log_loss(p["home_win"].astype(int).to_numpy(), p["p_home_win"].to_numpy()).mean()

    print(f"{'variant':<34}| tune LL | holdout LL | holdout vs current [95% CI]")
    for name, pred in preds.items():
        h = pred.loc[pred["season"] >= HOLDOUT_FIRST_SEASON].set_index("game_id")
        y = h["home_win"].astype(int).to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, h["p_home_win"].to_numpy(), ref.loc[h.index, "p_home_win"].to_numpy())
        flag = "  <-- adopt" if hi < 0 and name != "current (all plays)" else ""
        print(f"{name:<34}| {ll(pred, TUNE_SEASONS):.4f}  | {ll(pred, range(HOLDOUT_FIRST_SEASON, 2026)):.4f}     | {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")


if __name__ == "__main__":
    main()
