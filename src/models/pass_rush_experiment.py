"""Separate pass and rush rating filters (different noise and persistence).

Pass efficiency persists far more than rush efficiency and has ~3x the true
team-to-team spread, so one blended offense rating is mis-calibrated for at
least one of them. Each channel reuses the same Kalman filter with its own EPA
column; only the pass channel carries the quarterback layer.

Selection: parameters are tuned on the DEV window only; the 2017-2025 holdout
is reported to confirm.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.features.ratings import RatingParams, attach_ratings_to_matchups, build_team_ratings
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import (
    FIRST_MODEL_SEASON, HOLDOUT_FIRST_SEASON, PRIMARY_FEATURES, TUNE_SEASONS, Context,
)
from src.data.schedules import last_complete_season
from src.models.walk_forward import run_walk_forward


PASS = RatingParams(
    carryover_off=0.30, carryover_def=0.70, season_var=0.005, drift_var=0.0008,
    obs_var=0.050, prior_var=0.015,
    carryover_qb=0.85, qb_season_var=0.001, qb_drift_var=0.0002, qb_prior_var=0.008,
)
RUSH = RatingParams(
    carryover_off=0.40, carryover_def=0.40, season_var=0.0015, drift_var=0.0002,
    obs_var=0.0175, prior_var=0.005,
)

SPLIT_FEATURES = ["diff_pass_off", "diff_pass_def", "diff_rush_off", "diff_rush_def"]


def channel_frame(team_game: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = team_game[["game_id", "team", "season"]].copy()
    out["off_epa_per_play"] = team_game[f"off_{kind}_epa_per_play"]
    out["off_plays"] = team_game[f"off_{kind}_plays"]
    return out


def split_features(ctx: Context, pass_params: RatingParams, rush_params: RatingParams) -> pd.DataFrame:
    base = ctx.matchups

    p = build_team_ratings(channel_frame(ctx.team_game, "pass"), ctx.played, pass_params, ctx.qb_game)
    r = build_team_ratings(channel_frame(ctx.team_game, "rush"), ctx.played, rush_params, None)

    dp = attach_ratings_to_matchups(base, p)[["game_id", "diff_rating_off_total", "diff_rating_def"]]
    dr = attach_ratings_to_matchups(base, r)[["game_id", "diff_rating_off", "diff_rating_def"]]

    out = base.merge(dp.rename(columns={"diff_rating_off_total": "diff_pass_off", "diff_rating_def": "diff_pass_def"}), on="game_id")
    out = out.merge(dr.rename(columns={"diff_rating_off": "diff_rush_off", "diff_rating_def": "diff_rush_def"}), on="game_id")
    return out.loc[out["season"].between(FIRST_MODEL_SEASON, last_complete_season())]


def dev_loss(data: pd.DataFrame, features: list[str]) -> float:
    pred = run_walk_forward(data, feature_set_name="t", feature_columns=features)
    p = pred.loc[pred["season"].isin(list(TUNE_SEASONS))]
    return float(per_game_log_loss(p["home_win"].astype(int).to_numpy(), p["p_home_win"].to_numpy()).mean())


GRIDS = {
    "pass": {"carryover_off": [0.2, 0.4, 0.6], "carryover_def": [0.5, 0.7, 0.85], "obs_var": [0.035, 0.05, 0.075],
             "season_var": [0.003, 0.005, 0.009], "carryover_qb": [0.75, 0.85, 0.93]},
    "rush": {"carryover_off": [0.2, 0.4, 0.6], "carryover_def": [0.2, 0.4, 0.6], "obs_var": [0.012, 0.0175, 0.025],
             "season_var": [0.0008, 0.0015, 0.003]},
}


def main() -> None:
    ctx = Context()
    current = ctx.featurize(RatingParams(), use_qb=True)

    params = {"pass": PASS, "rush": RUSH}
    best = dev_loss(split_features(ctx, params["pass"], params["rush"]), SPLIT_FEATURES)
    print(f"split filter, untuned scaled defaults: dev loss {best:.5f}   (current model dev loss {dev_loss(current, PRIMARY_FEATURES):.5f})")

    for channel, grid in GRIDS.items():
        for name, values in grid.items():
            scored = []
            for v in values:
                trial = dict(params, **{channel: replace(params[channel], **{name: v})})
                scored.append((dev_loss(split_features(ctx, trial["pass"], trial["rush"]), SPLIT_FEATURES), v))
            loss, v = min(scored)
            if loss < best:
                best, params[channel] = loss, replace(params[channel], **{name: v})
            print(f"  {channel} {name:<14} best={v:<7g} | " + "  ".join(f"{vv:g}:{l:.5f}" for l, vv in sorted(scored, key=lambda t: t[1])))

    print(f"\ntuned split filter dev loss {best:.5f}")

    data = split_features(ctx, params["pass"], params["rush"]).merge(
        current[["game_id", "diff_rating_off_total", "diff_rating_def"]], on="game_id", suffixes=("", "_cur"))
    variants = {
        "current (blended, 2 features)": (current, PRIMARY_FEATURES),
        "pass/rush split (4 features)": (data, SPLIT_FEATURES),
    }
    preds = {k: run_walk_forward(d, feature_set_name=k, feature_columns=f) for k, (d, f) in variants.items()}
    ref = preds["current (blended, 2 features)"].set_index("game_id")

    print(f"\n{'variant':<32}| dev LL  | holdout LL | vs current on DEV [95% CI]      | vs current on HOLDOUT [95% CI]")
    for name, pred in preds.items():
        row = [name]
        for label, mask in (("dev", pred["season"].isin(list(TUNE_SEASONS))), ("hold", pred["season"] >= HOLDOUT_FIRST_SEASON)):
            g = pred.loc[mask].set_index("game_id")
            y = g["home_win"].astype(int).to_numpy()
            d, lo, hi = paired_bootstrap_delta(y, g["p_home_win"].to_numpy(), ref.loc[g.index, "p_home_win"].to_numpy())
            row += [per_game_log_loss(y, g["p_home_win"].to_numpy()).mean(), d, lo, hi]
        print(f"{row[0]:<32}| {row[1]:.4f}  | {row[5]:.4f}     | {row[2]:+.4f} [{row[3]:+.4f},{row[4]:+.4f}]  | {row[6]:+.4f} [{row[7]:+.4f},{row[8]:+.4f}]")
    print("\ntuned params:", {k: {f: getattr(v, f) for f in ('carryover_off','carryover_def','obs_var','season_var','carryover_qb')} for k, v in params.items()})


if __name__ == "__main__":
    main()
