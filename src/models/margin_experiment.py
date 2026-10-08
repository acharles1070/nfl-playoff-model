"""Does the point-margin rating add information beyond EPA + QB ratings?

Tune MarginParams on 2011-2016 only, then report 2017-2025 untouched.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from src.features.margin_ratings import (
    MarginParams,
    attach_margin_ratings,
    build_margin_ratings,
)
from src.features.ratings import RatingParams
from src.models.compare import paired_bootstrap_delta, per_game_log_loss
from src.models.ratings_experiment import (
    HOLDOUT_FIRST_SEASON,
    PRIMARY_FEATURES,
    TUNE_SEASONS,
    Context,
)
from src.models.walk_forward import run_walk_forward


WITH_PTS = [*PRIMARY_FEATURES, "diff_pts_rating"]
ONLY_PTS = ["diff_pts_rating"]


def _ll(pred: pd.DataFrame, seasons) -> float:
    p = pred.loc[pred["season"].isin(list(seasons))]
    return float(per_game_log_loss(p["home_win"].astype(int).to_numpy(), p["p_home_win"].to_numpy()).mean())


def main() -> None:
    ctx = Context()
    base = ctx.featurize(RatingParams(), use_qb=True)
    scored = ctx.played

    def with_margin(mp: MarginParams) -> pd.DataFrame:
        return attach_margin_ratings(base, build_margin_ratings(scored, mp))

    grid = itertools.product([0.45, 0.6, 0.75], [3.0, 6.0, 12.0], [0.2, 0.4, 0.8])
    results = []

    for carry, season_var, drift in grid:
        mp = replace(MarginParams(), carryover=carry, season_var=season_var, drift_var=drift)
        pred = run_walk_forward(with_margin(mp), feature_set_name="t", feature_columns=WITH_PTS)
        results.append((_ll(pred, TUNE_SEASONS), carry, season_var, drift))

    results.sort()
    print("best 5 margin-filter settings on the TUNE window (log loss, carry, season_var, drift):")
    for r in results[:5]:
        print("  %.5f  carry=%.2f season_var=%.0f drift=%.1f" % r)

    best = replace(MarginParams(), carryover=results[0][1], season_var=results[0][2], drift_var=results[0][3])
    data = with_margin(best)

    variants = {
        "EPA + QB (current)": PRIMARY_FEATURES,
        "EPA + QB + margin rating": WITH_PTS,
        "margin rating only": ONLY_PTS,
    }
    preds = {k: run_walk_forward(data, feature_set_name=k, feature_columns=v) for k, v in variants.items()}

    ref = preds["EPA + QB (current)"].set_index("game_id")
    print("\nvariant                     | tune-window LL | holdout LL | holdout vs current (95% CI)")
    for name, pred in preds.items():
        h = pred.loc[pred["season"] >= HOLDOUT_FIRST_SEASON].set_index("game_id")
        y = h["home_win"].astype(int).to_numpy()
        d, lo, hi = paired_bootstrap_delta(y, h["p_home_win"].to_numpy(), ref.loc[h.index, "p_home_win"].to_numpy())
        print(f"{name:<27} | {_ll(pred, TUNE_SEASONS):.4f}         | {_ll(pred, range(HOLDOUT_FIRST_SEASON, 2026)):.4f}     | {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    post = {k: p.loc[(p["season"] >= HOLDOUT_FIRST_SEASON) & p["is_postseason"]].set_index("game_id") for k, p in preds.items()}
    y = post["EPA + QB (current)"]["home_win"].astype(int).to_numpy()
    d, lo, hi = paired_bootstrap_delta(
        y, post["EPA + QB + margin rating"]["p_home_win"].to_numpy(),
        post["EPA + QB (current)"]["p_home_win"].to_numpy())
    print(f"\npostseason only (n={len(y)}): +margin vs current = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nchosen margin params:", best)


if __name__ == "__main__":
    main()
